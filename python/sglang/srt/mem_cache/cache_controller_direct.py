import logging
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, get_hash_str
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.storage import StorageBackendFactory

logger = logging.getLogger(__name__)


def get_hash_list(
    token_ids: List[int], prior_hash: Optional[str] = None, page_size: int = 128
) -> List[str]:
    assert len(token_ids) % page_size == 0
    hashes = []
    last_hash = prior_hash
    token_groups = (
        token_ids[i : i + page_size] for i in range(0, len(token_ids), page_size)
    )
    for group in token_groups:
        last_hash = get_hash_str(group, last_hash)
        hashes.append(last_hash)
    return hashes


class LoadStorageOperation:
    counter = 0

    def __init__(
        self,
        request_id: str,
        device_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        page_size: int = 128,
    ):
        self.request_id = request_id
        self.device_indices = device_indices

        self.token_ids = token_ids
        self.last_hash = last_hash
        self.hash_keys = get_hash_list(token_ids, last_hash, page_size)

        self.id = LoadStorageOperation.counter
        LoadStorageOperation.counter += 1


@dataclass
class _AsyncStoreTask:
    keys: List[str]
    ptrs: List[int]
    sizes: List[int]
    # (start_page, num_pages) slice in the IO buffer to free after IO completes
    io_slice: Tuple[int, int]
    # timing / debug
    enqueue_ts: float
    pack_ms: float


class HiCacheControllerDirect:
    # Fire-and-forget async STORE queue size (per process).
    # If full, STORE is dropped (logged) to avoid blocking decode.
    ASYNC_STORE_QUEUE_MAXSIZE = 1024

    # Default async STORE IO buffer capacity (in pages). Can be overridden by CLI.
    DEFAULT_ASYNC_STORE_IO_BUFFER_PAGES = 4096

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        storage_backend: str,
        device_id: int = 0,
        model_name: Optional[str] = None,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        if self.mem_pool_device_allocator:
            self.device = self.mem_pool_device_allocator.device
        else:
            self.device = torch.device("cpu")
        # self.kv_layer_ptrs: every layer ptr
        # self.kv_layer_nbytes: the byte length of each layer
        # self.kv_page_nbytes: the page byte length of each layer
        self.kv_layer_ptrs, self.kv_layer_nbytes, self.kv_page_nbytes = (
            self.mem_pool_device.get_contiguous_buf_infos()
        )

        self.page_size = page_size
        self.device_id = device_id
        self.is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # MVP: only MHA is supported for packed page-first mode.
        # MLA keeps using the existing scatter/gather path.
        self.use_packed_page_first = not self.is_mla_model

        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = self.is_mla_model and self.tp_rank != 0

        self.model_name = model_name
        # Namespace keys by model + KV geometry to avoid collisions across different models/configs.
        # This is critical for Mooncake because it enforces fixed value size per key.
        dtype_name = str(
            getattr(self.mem_pool_device, "dtype", None)
            or getattr(self.mem_pool_device, "store_dtype", "")
        )
        if self.use_packed_page_first:
            head_num = int(self.mem_pool_device.head_num)
            head_dim = int(self.mem_pool_device.head_dim)
        else:
            head_num = -1
            head_dim = -1
        self.key_namespace = (
            f"{self.model_name or 'default'}"
            f"_ps{self.page_size}"
            f"_L{int(self.mem_pool_device.layer_num)}"
            f"_H{head_num}"
            f"_D{head_dim}"
            f"_dt{dtype_name}"
        )

        self.storage_config = HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            is_mla_model=self.is_mla_model,
            is_page_first_layout=self.use_packed_page_first,
            model_name=self.model_name,
            extra_config={"device_id": device_id},
        )
        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, None
            )
        except ValueError as e:
            raise ValueError(f"Failed to create storage backend: {e}") from e

        self.storage_backend.register_mem_pool_device(self.mem_pool_device)

        self.load_tokens_threshold = 128
        # granularity of batch storage IO operations, in number of pages
        self.storage_batch_size = 256

        # NOTE: We intentionally do NOT allocate a separate packed scratch buffer.
        # Async STORE packs directly into a slice of the pre-registered IO buffer.

        # create a new communication group for synchronizing storage operations across TP workers
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.load_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )

        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        self.load_queue: List[LoadStorageOperation] = []

        # Async STORE (IO-only in background):
        # - pack happens synchronously directly into a slice of a single pre-registered IO buffer
        # - background thread performs only batch_set() from that IO buffer slice
        self._store_queue: Queue = Queue(maxsize=self.ASYNC_STORE_QUEUE_MAXSIZE)
        self._store_thread = None
        self._store_stop = False

        # IO buffer allocator state
        self._async_store_io_buf = None
        self._async_store_io_capacity_pages = int(
            getattr(token_to_kv_pool_allocator, "async_store_io_buffer_pages", 0) or 0
        )
        if self._async_store_io_capacity_pages <= 0:
            self._async_store_io_capacity_pages = (
                self.DEFAULT_ASYNC_STORE_IO_BUFFER_PAGES
            )

        self._async_store_io_free: List[Tuple[int, int]] = [
            (0, self._async_store_io_capacity_pages)
        ]
        self._async_store_io_inflight_pages = 0
        self._async_store_io_drop_pages = 0

        # NOTE: avoid importing threading at module import time; do it lazily
        import threading  # local import

        self._async_store_io_lock = threading.Lock()

        if self.use_packed_page_first:
            # IO buffer layout matches packed scratch: [2, pages, L, page_size, H, D]
            head_num = int(self.mem_pool_device.head_num)
            head_dim = int(self.mem_pool_device.head_dim)
            layer_num = int(self.mem_pool_device.layer_num)
            dtype = self.mem_pool_device.store_dtype
            self._async_store_io_buf = torch.empty(
                (
                    2,
                    self._async_store_io_capacity_pages,
                    layer_num,
                    self.page_size,
                    head_num,
                    head_dim,
                ),
                dtype=dtype,
                device=self.device,
            )
            self.storage_backend.register_device_buffer(
                self._async_store_io_buf.data_ptr(), self._async_store_io_buf.nbytes
            )
            io_gib = self._async_store_io_buf.nbytes / (1024**3)
            logger.info(
                "hicache_direct async STORE IO buffer allocated: pages=%d size=%.3f GiB (registered)",
                self._async_store_io_capacity_pages,
                io_gib,
            )
        else:
            # MLA path currently uses scatter/gather; async-store is not enabled there.
            self._async_store_io_buf = None

        self._store_thread = threading.Thread(
            target=self._store_thread_func, daemon=True
        )
        self._store_thread.start()

    def reset(self):
        self.load_queue.clear()

    def close(self):
        # Best-effort stop; do not block shutdown.
        self._store_stop = True
        try:
            if self._store_thread is not None:
                self._store_thread.join(timeout=0.1)
        except Exception:
            pass

    def write(self, hash_keys: List[str], device_indices: torch.Tensor) -> int:
        """
        Async STORE (MHA packed page-first):
        - pack sync directly into a slice of a pre-registered IO buffer
        - enqueue IO-only task (batch_set) to background thread
        """
        if self.backup_skip:
            return 0

        # Only packed page-first path is supported for async-store MVP.
        if not self.use_packed_page_first or self._async_store_io_buf is None:
            # Best-effort fallback to sync (keeps behavior for MLA / non-packed)
            try:
                succ_pages_num = self._memcpy_between_device_and_storage(
                    hash_keys, device_indices, "write"
                )
                if self.tp_world_size > 1 and self.is_mla_model is False:
                    succ_pages_num = self._allreduce_results(succ_pages_num)
                return succ_pages_num * self.page_size
            except Exception:
                return 0

        token_len = int(device_indices.shape[0])
        if token_len == 0:
            return 0
        assert token_len % self.page_size == 0
        num_pages = token_len // self.page_size
        assert num_pages == len(hash_keys)

        # Allocate IO-buffer slice (pages)
        io_slice = self._async_store_io_alloc(num_pages)
        if io_slice is None:
            self._async_store_io_drop_pages += num_pages
            with self._async_store_io_lock:
                free_pages = sum(l for _, l in self._async_store_io_free)
                inflight_pages = self._async_store_io_inflight_pages
                cap_pages = self._async_store_io_capacity_pages
            logger.warning(
                "hicache_direct STORE dropped: async IO buffer full, pages=%d dropped_total_pages=%d inflight_pages=%d free_pages=%d cap_pages=%d",
                num_pages,
                self._async_store_io_drop_pages,
                inflight_pages,
                free_pages,
                cap_pages,
            )
            return 0

        start_page, slice_pages = io_slice
        assert slice_pages == num_pages

        # Pack directly into IO buffer slice (sync)
        t_pack0 = time.perf_counter()
        try:
            keys, ptrs, sizes = self._pack_pages_for_async_store(
                hash_keys=hash_keys,
                device_indices=device_indices,
                io_start_page=start_page,
                io_pages=slice_pages,
            )
        except Exception:
            self._async_store_io_free_slice(start_page, slice_pages)
            raise
        t_pack1 = time.perf_counter()
        pack_ms = (t_pack1 - t_pack0) * 1000.0

        task = _AsyncStoreTask(
            keys=keys,
            ptrs=ptrs,
            sizes=sizes,
            io_slice=io_slice,
            enqueue_ts=time.perf_counter(),
            pack_ms=pack_ms,
        )
        try:
            self._store_queue.put_nowait(task)
            return num_pages * self.page_size
        except Full:
            self._async_store_io_free_slice(start_page, slice_pages)
            self._async_store_io_drop_pages += num_pages
            with self._async_store_io_lock:
                free_pages = sum(l for _, l in self._async_store_io_free)
                inflight_pages = self._async_store_io_inflight_pages
                cap_pages = self._async_store_io_capacity_pages
            logger.warning(
                "hicache_direct STORE dropped: async queue full maxsize=%d pages=%d dropped_total_pages=%d inflight_pages=%d free_pages=%d cap_pages=%d",
                self.ASYNC_STORE_QUEUE_MAXSIZE,
                num_pages,
                self._async_store_io_drop_pages,
                inflight_pages,
                free_pages,
                cap_pages,
            )
            return 0

    def _store_thread_func(self):
        while not self._store_stop:
            try:
                task: _AsyncStoreTask = self._store_queue.get(timeout=0.1)
            except Empty:
                continue

            t_io0 = time.perf_counter()
            succ_pages = 0
            try:
                # IO-only: batch_set from pre-registered IO buffer slice
                succ_raw = self.storage_backend.batch_set(
                    keys=task.keys, target_locations=task.ptrs, target_sizes=task.sizes
                )
                t_io1 = time.perf_counter()
                io_ms = (t_io1 - t_io0) * 1000.0

                if isinstance(succ_raw, bool):
                    succ_keys = len(task.keys) if succ_raw else 0
                elif isinstance(succ_raw, list):
                    succ_keys = 0
                    for r in succ_raw:
                        if r != 0:
                            break
                        succ_keys += 1
                else:
                    succ_keys = int(succ_raw)

                succ_pages = succ_keys // 2
                if self.tp_world_size > 1 and self.is_mla_model is False:
                    succ_pages = self._allreduce_results(succ_pages)

                e2e_ms = (time.perf_counter() - task.enqueue_ts) * 1000.0

                with self._async_store_io_lock:
                    free_pages = sum(l for _, l in self._async_store_io_free)
                    inflight_pages = self._async_store_io_inflight_pages
                    cap_pages = self._async_store_io_capacity_pages

                logger.info(
                    "hicache_direct async STORE timing: pages=%d succ_pages=%d pack=%.3fms io=%.3fms e2e=%.3fms inflight_pages=%d free_pages=%d cap_pages=%d dropped_total_pages=%d",
                    task.io_slice[1],
                    succ_pages,
                    task.pack_ms,
                    io_ms,
                    e2e_ms,
                    inflight_pages,
                    free_pages,
                    cap_pages,
                    self._async_store_io_drop_pages,
                )
            except Exception as e:
                logger.exception("hicache_direct async STORE failed: %s", e)
            finally:
                # Always free IO slice
                try:
                    self._async_store_io_free_slice(task.io_slice[0], task.io_slice[1])
                except Exception:
                    pass
                try:
                    self._store_queue.task_done()
                except Exception:
                    pass

    def load(
        self,
        rid,
        new_input_tokens,
        device_indices,
        last_hash: Optional[str] = None,
    ):
        """
        Load KV caches from L3 storage to device memory.
        """
        self.load_queue.append(
            LoadStorageOperation(rid, device_indices, new_input_tokens, last_hash)
        )
        device_indices, free_device_indices = self.start_loading()
        return device_indices, free_device_indices

    def start_loading(self) -> tuple[Tensor | None, Tensor | None]:
        if len(self.load_queue) == 0:
            return None, None

        assert len(self.load_queue) == 1
        # producer_id = self.layer_done_counter.update_producer()
        op = self.load_queue[0]
        self.load_queue.clear()
        # producer_event = self.layer_done_counter.events[producer_id]
        # producer_event.start_event.record()

        try:
            hit_hash_len = self._storage_hit_query(op)
            if self.tp_world_size > 1:
                hit_hash_len = self._allreduce_results(hit_hash_len)

            hit_token_len = hit_hash_len * self.page_size
            if hit_token_len < self.load_tokens_threshold:
                # not to load storage if not enough benefits
                logger.debug(
                    f"Revoking Load operation for request {op.request_id} due to insufficient hits ({hit_token_len})."
                )
                return None, op.device_indices
            else:
                hit_hash_keys = op.hash_keys[:hit_hash_len]
                device_indices = op.device_indices[:hit_token_len]
                succ_pages_num = self._memcpy_between_device_and_storage(
                    hit_hash_keys, device_indices, "load"
                )
                if self.tp_world_size > 1:
                    succ_pages_num = self._allreduce_results(succ_pages_num)

                token_len = succ_pages_num * self.page_size
                hit_device_indices = op.device_indices[:token_len]
                free_device_indices = op.device_indices[token_len:]

                logger.debug(
                    f"success load {token_len} tokens for request {op.request_id}"
                )
                return hit_device_indices, free_device_indices
        except Empty:
            logger.error(
                f"Failed load storage {len(op.hash_keys)} pages for request {op.request_id}"
            )
            return None, op.device_indices

    def _allreduce_results(self, result: int) -> int:
        result_tensor = torch.tensor(result, dtype=torch.int)
        torch.distributed.all_reduce(
            result_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.load_tp_group,
        )

        return result_tensor.item()

    def _storage_hit_query(self, operation: LoadStorageOperation) -> int:
        if not operation.hash_keys:
            return 0

        # Debug: help diagnose partial hits across restarts
        try:
            hk = operation.hash_keys
            logger.info(
                "hicache_direct hit_query: rid=%s tp=%s pages=%s last_hash=%s hk0=%s hk1=%s hk_last2=%s hk_last1=%s",
                operation.request_id,
                self.tp_rank,
                len(hk),
                operation.last_hash,
                hk[0] if len(hk) > 0 else None,
                hk[1] if len(hk) > 1 else None,
                hk[-2] if len(hk) > 1 else None,
                hk[-1] if len(hk) > 0 else None,
            )
            if self.use_packed_page_first and len(hk) > 0:
                logger.info(
                    "hicache_direct hit_query probe example: %s",
                    f"{hk[0]}_{self.tp_rank}_k",
                )
        except Exception:
            pass

        total_len = len(operation.hash_keys)
        total_hit_num = 0
        for start in range(0, total_len, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_len)
            batch_hashes = operation.hash_keys[start:end]

            if self.use_packed_page_first:
                # Packed mode stores 2 keys per page: {key}_{tp}_k and {key}_{tp}_v.
                # For hit probing we only check the K keys (assume V exists if K exists).
                probe_keys = [
                    f"{self.key_namespace}_{k}_{self.tp_rank}_k" for k in batch_hashes
                ]
                hit_num = self.storage_backend.batch_exists(probe_keys)
            else:
                hit_num = self.storage_backend.batch_exists(batch_hashes)

            total_hit_num += hit_num
            if hit_num < len(batch_hashes):
                break

        logger.info(
            "hicache_direct hit_query result: rid=%s hit_pages=%s/%s",
            operation.request_id,
            total_hit_num,
            total_len,
        )
        return total_hit_num

    def _async_store_io_alloc(self, pages: int) -> Optional[Tuple[int, int]]:
        if pages <= 0:
            return None
        with self._async_store_io_lock:
            for i, (start, length) in enumerate(self._async_store_io_free):
                if length >= pages:
                    alloc = (start, pages)
                    # shrink or remove free segment
                    if length == pages:
                        self._async_store_io_free.pop(i)
                    else:
                        self._async_store_io_free[i] = (start + pages, length - pages)
                    self._async_store_io_inflight_pages += pages
                    return alloc
        return None

    def _memcpy_between_device_and_storage(
        self,
        hash_keys: List[str],
        device_indices: torch.Tensor,
        direction: str,
    ) -> int:
        assert hash_keys

        if self.use_packed_page_first:
            return self._memcpy_packed_page_first(hash_keys, device_indices, direction)

        batch_memcpy = None
        if direction == "write":
            batch_memcpy = self.storage_backend.batch_set
        elif direction == "load":
            batch_memcpy = self.storage_backend.batch_get
        assert batch_memcpy is not None

        total_elements = len(hash_keys)
        ptr_list, element_size_list = self._get_page_buffer_meta(device_indices)
        assert total_elements == len(ptr_list)
        assert total_elements == len(element_size_list)
        total_succ_num = 0
        for start in range(0, total_elements, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_elements)
            batch_hashes = hash_keys[start:end]
            target_locations = ptr_list[start:end]
            target_sizes = element_size_list[start:end]
            succ_raw = batch_memcpy(
                keys=batch_hashes,
                target_locations=target_locations,
                target_sizes=target_sizes,
            )
            # Normalize return type to int (number of successful pages)
            if isinstance(succ_raw, bool):
                succ_num = len(batch_hashes) if succ_raw else 0
            elif isinstance(succ_raw, list):
                succ_num = 0
                for r in succ_raw:
                    ok = (
                        (r == 0) if direction == "write" else (r is not None and r >= 0)
                    )
                    if not ok:
                        break
                    succ_num += 1
            else:
                succ_num = int(succ_raw)

            total_succ_num += succ_num
            if succ_num < len(batch_hashes):
                break

        return total_succ_num

    def _async_store_io_free_slice(self, start_page: int, pages: int) -> None:
        if pages <= 0:
            return
        with self._async_store_io_lock:
            self._async_store_io_inflight_pages -= pages
            self._async_store_io_free.append((start_page, pages))
            # coalesce
            self._async_store_io_free.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int]] = []
            for s, l in self._async_store_io_free:
                if not merged:
                    merged.append((s, l))
                    continue
                ps, pl = merged[-1]
                if ps + pl == s:
                    merged[-1] = (ps, pl + l)
                else:
                    merged.append((s, l))
            self._async_store_io_free = merged

    def _memcpy_packed_page_first(
        self,
        hash_keys: List[str],
        device_indices: torch.Tensor,
        direction: str,
    ) -> int:
        """
        Packed page-first mode (MHA only):
        - pack device KVCache pages into contiguous (K,V) buffers on GPU
        - store/load 2 keys per page: {hash}_{tp}_k and {hash}_{tp}_v
        """
        assert not self.is_mla_model, "Packed page-first mode is MHA-only in MVP"
        assert hasattr(self.mem_pool_device, "head_num") and hasattr(
            self.mem_pool_device, "head_dim"
        ), "Packed page-first mode expects MHA KVCache"

        if direction == "write":
            batch_memcpy = self.storage_backend.batch_set
        elif direction == "load":
            batch_memcpy = self.storage_backend.batch_get
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        token_len = int(device_indices.shape[0])
        assert token_len % self.page_size == 0
        num_pages = token_len // self.page_size

        # Build page->token indices [num_pages, page_size]
        page_token_idx = device_indices.view(num_pages, self.page_size)

        # [num_pages, layer_num, page_size, head_num, head_dim]
        head_num = int(self.mem_pool_device.head_num)
        head_dim = int(self.mem_pool_device.head_dim)
        layer_num = int(self.mem_pool_device.layer_num)

        # Prepare 2 keys per page and pointers/sizes using a pre-registered scratch buffer.
        total_elements = len(hash_keys)
        assert total_elements == num_pages
        assert (
            self._async_store_io_buf is not None
        ), "Async STORE IO buffer must be initialized"

        pack_ms_total = 0.0
        io_ms_total = 0.0
        unpack_ms_total = 0.0
        pages_total = 0
        total_succ_pages = 0

        t_op0 = time.perf_counter()
        for start in range(0, total_elements, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_elements)
            batch_hashes = hash_keys[start:end]
            batch_pages = end - start
            pages_total += batch_pages

            # Views into IO buffer (direct pack target)
            io_k = self._async_store_io_buf[0, :batch_pages]
            io_v = self._async_store_io_buf[1, :batch_pages]

            batch_token_idx = page_token_idx[start:end]
            flat_idx = batch_token_idx.reshape(-1)

            t_pack0 = time.perf_counter()
            if direction == "write":
                for layer_id in range(layer_num):
                    k_layer = self.mem_pool_device._get_key_buffer(layer_id)
                    v_layer = self.mem_pool_device._get_value_buffer(layer_id)
                    io_k[:, layer_id] = k_layer.index_select(0, flat_idx).view(
                        batch_pages, self.page_size, head_num, head_dim
                    )
                    io_v[:, layer_id] = v_layer.index_select(0, flat_idx).view(
                        batch_pages, self.page_size, head_num, head_dim
                    )
            t_pack1 = time.perf_counter()
            pack_ms_total += (t_pack1 - t_pack0) * 1000.0

            keys_kv: List[str] = []
            ptrs: List[int] = []
            sizes: List[int] = []
            for local_page_i, base_key in enumerate(batch_hashes):
                namespaced = f"{self.key_namespace}_{base_key}"
                keys_kv.append(f"{namespaced}_{self.tp_rank}_k")
                keys_kv.append(f"{namespaced}_{self.tp_rank}_v")

                k_page = io_k[local_page_i]
                v_page = io_v[local_page_i]
                ptrs.append(int(k_page.data_ptr()))
                ptrs.append(int(v_page.data_ptr()))
                sizes.append(int(k_page.nbytes))
                sizes.append(int(v_page.nbytes))

            t_io0 = time.perf_counter()
            succ_raw = batch_memcpy(
                keys=keys_kv, target_locations=ptrs, target_sizes=sizes
            )
            t_io1 = time.perf_counter()
            io_ms_total += (t_io1 - t_io0) * 1000.0

            if isinstance(succ_raw, bool):
                succ_keys = len(keys_kv) if succ_raw else 0
            elif isinstance(succ_raw, list):
                succ_keys = 0
                for r in succ_raw:
                    ok = (
                        (r == 0) if direction == "write" else (r is not None and r >= 0)
                    )
                    if not ok:
                        break
                    succ_keys += 1
            else:
                succ_keys = int(succ_raw)

            # Mooncake legacy batch_get returns pages (k/v pairs), while batch_set returns keys.
            # Normalize to "pages" here.
            if direction == "load" and succ_keys <= batch_pages:
                succ_pages = succ_keys
            else:
                succ_pages = succ_keys // 2
            total_succ_pages += succ_pages

            t_unpack0 = time.perf_counter()
            if direction == "load" and succ_pages > 0:
                loaded_flat_idx = batch_token_idx[:succ_pages].reshape(-1)
                for layer_id in range(layer_num):
                    k_layer = self.mem_pool_device._get_key_buffer(layer_id)
                    v_layer = self.mem_pool_device._get_value_buffer(layer_id)
                    k_layer.index_copy_(
                        0,
                        loaded_flat_idx,
                        io_k[:succ_pages, layer_id].reshape(-1, head_num, head_dim),
                    )
                    v_layer.index_copy_(
                        0,
                        loaded_flat_idx,
                        io_v[:succ_pages, layer_id].reshape(-1, head_num, head_dim),
                    )
            t_unpack1 = time.perf_counter()
            unpack_ms_total += (t_unpack1 - t_unpack0) * 1000.0

            if succ_pages < batch_pages:
                break

        t_op1 = time.perf_counter()
        total_ms = (t_op1 - t_op0) * 1000.0
        denom_pages = float(pages_total) if pages_total > 0 else 1.0

        logger.info(
            "hicache_direct %s timing: pages=%d succ_pages=%d pack=%.3fms (%.4fms/page) io=%.3fms (%.4fms/page) unpack=%.3fms (%.4fms/page) total=%.3fms (%.4fms/page)",
            "STORE" if direction == "write" else "LOAD",
            pages_total,
            total_succ_pages,
            pack_ms_total,
            pack_ms_total / denom_pages,
            io_ms_total,
            io_ms_total / denom_pages,
            unpack_ms_total,
            unpack_ms_total / denom_pages,
            total_ms,
            total_ms / denom_pages,
        )

        return total_succ_pages

    def _pack_pages_for_async_store(
        self,
        hash_keys: List[str],
        device_indices: torch.Tensor,
        io_start_page: int,
        io_pages: int,
    ) -> tuple[List[str], List[int], List[int]]:
        """
        Pack pages directly into IO buffer slice, and return (keys, ptrs, sizes) for batch_set().
        """
        assert self.use_packed_page_first
        assert self._async_store_io_buf is not None
        assert io_pages == len(hash_keys)

        token_len = int(device_indices.shape[0])
        assert token_len % self.page_size == 0
        num_pages = token_len // self.page_size
        assert num_pages == io_pages

        page_token_idx = device_indices.view(num_pages, self.page_size)

        head_num = int(self.mem_pool_device.head_num)
        head_dim = int(self.mem_pool_device.head_dim)
        layer_num = int(self.mem_pool_device.layer_num)

        # Pack directly into IO buffer slice (no extra scratch buffer).
        for chunk_start in range(0, io_pages, self.storage_batch_size):
            chunk_end = min(chunk_start + self.storage_batch_size, io_pages)
            chunk_pages = chunk_end - chunk_start

            io_k = self._async_store_io_buf[
                0, io_start_page + chunk_start : io_start_page + chunk_end
            ]
            io_v = self._async_store_io_buf[
                1, io_start_page + chunk_start : io_start_page + chunk_end
            ]

            chunk_flat_idx = page_token_idx[chunk_start:chunk_end].reshape(-1)

            for layer_id in range(layer_num):
                k_layer = self.mem_pool_device._get_key_buffer(layer_id)
                v_layer = self.mem_pool_device._get_value_buffer(layer_id)
                io_k[:, layer_id] = k_layer.index_select(0, chunk_flat_idx).view(
                    chunk_pages, self.page_size, head_num, head_dim
                )
                io_v[:, layer_id] = v_layer.index_select(0, chunk_flat_idx).view(
                    chunk_pages, self.page_size, head_num, head_dim
                )

        # build keys/ptrs/sizes (2 keys per page) from the IO buffer slice
        io_k_all = self._async_store_io_buf[0, io_start_page : io_start_page + io_pages]
        io_v_all = self._async_store_io_buf[1, io_start_page : io_start_page + io_pages]

        keys_kv: List[str] = []
        ptrs: List[int] = []
        sizes: List[int] = []
        for local_page_i, base_key in enumerate(hash_keys):
            namespaced = f"{self.key_namespace}_{base_key}"
            keys_kv.append(f"{namespaced}_{self.tp_rank}_k")
            keys_kv.append(f"{namespaced}_{self.tp_rank}_v")

            k_page = io_k_all[local_page_i]
            v_page = io_v_all[local_page_i]
            ptrs.append(int(k_page.data_ptr()))
            ptrs.append(int(v_page.data_ptr()))
            sizes.append(int(k_page.nbytes))
            sizes.append(int(v_page.nbytes))

        return keys_kv, ptrs, sizes

    def _parse_success_hashes_from_l3_results(
        self,
        hash_keys: List[str],
        results: List[int],
    ) -> int:
        # for each key
        hit_hash_len = 0
        for h, r in zip(hash_keys, results):
            if r == 1:
                hit_hash_len += 1
            else:
                break

        return hit_hash_len

    def _get_page_buffer_meta(
        self, device_indices: torch.Tensor
    ) -> tuple[List[List[int]], List[List[int]]]:
        # 1. concatenate device index tensors
        token_len = device_indices.shape[0]
        assert token_len % self.page_size == 0

        # 2. compute page indices
        group_first_indices = device_indices[:: self.page_size]
        page_indices = group_first_indices // self.page_size

        # 3. Translate layer_ptrs and page_nbytes into tensors
        kv_layer_ptrs_tensor = torch.tensor(
            self.kv_layer_ptrs, dtype=torch.int64, device=device_indices.device
        )
        kv_page_nbytes_tensor = torch.tensor(
            self.kv_page_nbytes, dtype=torch.int64, device=device_indices.device
        )

        # 4. compute the pointers and sizes for all layers
        # Expand page_indices to shape [M, 1] and broadcast with [L] to shape [M, L]
        page_indices_expanded = page_indices.unsqueeze(1)
        ptr_tensor = (
            kv_layer_ptrs_tensor + page_indices_expanded * kv_page_nbytes_tensor
        )
        element_size_tensor = kv_page_nbytes_tensor.unsqueeze(0).expand_as(ptr_tensor)

        # 6. translate to a list
        ptr_list = ptr_tensor.tolist()
        element_size_list = element_size_tensor.tolist()

        return ptr_list, element_size_list
