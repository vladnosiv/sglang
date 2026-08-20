"""Fixed-ring CUDA transient buffers for buffer-only HiCache + MoonCake."""

from __future__ import annotations

import itertools
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

from sglang.srt.managers.cache_controller import HiCacheAck, StorageOperation
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.buffer_mode.transient_buffer import (
    TransientBufferBackend,
    TransientBufferLease,
    TransientRestore,
)
from sglang.srt.mem_cache.gpu_transient import (
    RegisteredGpuRing,
    build_gpu_payload_layout,
)
from sglang.srt.mem_cache.gpu_transient.kernels import (
    copy_opaque_pages,
    copy_token_rows,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

logger = logging.getLogger(__name__)


class _ThreadEvent:
    """HiCacheAck-compatible completion event for a CPU worker terminal state."""

    def __init__(self, *, completed: bool = False) -> None:
        self._event = threading.Event()
        if completed:
            self._event.set()

    def set(self) -> None:
        self._event.set()

    def query(self) -> bool:
        return self._event.is_set()

    def synchronize(self) -> None:
        self._event.wait()

    def elapsed_time(self, other: _ThreadEvent) -> float:
        return 0.0


@dataclass
class _GpuOperationState:
    operation_id: int
    direction: str
    num_tokens: int
    device_indices: Optional[torch.Tensor] = None
    pool_device_indices: dict[PoolName, torch.Tensor] = field(default_factory=dict)
    io_specs: tuple[_PoolIoSpec, ...] = ()
    prefetch_operation: Optional[Any] = None
    success: bool = False
    done: bool = False
    cancelled: bool = False
    release_requested: bool = False
    committed: bool = False
    released: bool = False
    enqueued_at: float = 0.0
    worker_started_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_event: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _GpuWorkItem:
    state: _GpuOperationState
    run: Callable[[], None]


@dataclass(frozen=True)
class _GpuTransientBufferLease(TransientBufferLease):
    state: _GpuOperationState


@dataclass(frozen=True)
class _PoolIoSpec:
    name: PoolName
    keys: tuple[str, ...]
    device_indices: torch.Tensor
    index_page_size: int


@dataclass
class _PoolTiming:
    pages: int = 0
    payload_bytes: int = 0
    ring_wait_ms: float = 0.0
    cuda_copy_ms: float = 0.0
    storage_ms: float = 0.0


def validate_gpu_transient_buffer_stack(
    *,
    tree_components: set,
    sidecar_pool_specs: list,
    controller,
) -> None:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    allowed_trees = (
        {ComponentType.FULL},
        {ComponentType.FULL, ComponentType.SWA},
    )
    if tree_components not in allowed_trees:
        raise ValueError(
            "--hicache-storage-io-mode gpu_transient supports flat FULL and "
            "DeepSeek-V4 FULL+SWA trees only."
        )
    if controller.storage_backend_type != "mooncake":
        raise ValueError("GPU-transient HiCache requires the MoonCake backend.")
    if controller.page_size != 256:
        raise ValueError("GPU-transient HiCache requires complete 256-token pages.")
    if controller.mem_pool_device_allocator.page_size != controller.page_size:
        raise ValueError(
            "GPU-transient HiCache requires the page-aligned device allocator."
        )
    if getattr(controller, "has_draft", False) or getattr(
        controller, "has_mtp_draft", False
    ):
        raise ValueError("GPU-transient HiCache does not yet support draft KV pools.")
    layout = build_gpu_payload_layout(controller.mem_pool_device, controller.page_size)
    expected_sidecars = {
        obj.pool_name
        for obj in layout.objects
        if obj.pool_name not in (PoolName.KV, PoolName.SWA)
    }
    actual_sidecars = {spec.pool_name for spec in sidecar_pool_specs}
    if actual_sidecars != expected_sidecars:
        raise ValueError(
            "GPU-transient sidecar/layout mismatch: "
            f"expected={sorted(str(x) for x in expected_sidecars)}, "
            f"actual={sorted(str(x) for x in actual_sidecars)}."
        )


class GpuTransientBufferBackend(TransientBufferBackend):
    """Bounded GPU I/O queue with one physical worker and scheduler publication."""

    # DSV4 can publish roughly one hundred tree-node intents when a 256k
    # request finishes. Keep two such bursts admissible while the independent
    # token cap remains the primary bound on pinned device memory.
    _MAX_ADMITTED_PUT_OPS = 256
    _PUT_LOCK_FRACTION = 0.10

    def __init__(
        self,
        *,
        cache,
        controller,
        wave_pages: int,
        ring_depth: int,
        max_active_ops: int,
    ) -> None:
        if max_active_ops != 1:
            raise ValueError("Initial GPU-transient backend requires max_active_ops=1.")
        self._cache = cache
        self._controller = controller
        self._allocator = controller.mem_pool_device_allocator
        self._full_allocator = getattr(
            self._allocator, "full_attn_allocator", self._allocator
        )
        self._storage = controller.storage_backend
        self._page_size = controller.page_size
        self._wave_pages = wave_pages
        self._layout = build_gpu_payload_layout(
            controller.mem_pool_device, self._page_size
        )
        self._device = torch.device(controller.mem_pool_device.device)
        self._tx_ring = RegisteredGpuRing(
            direction="tx",
            depth=ring_depth,
            wave_pages=wave_pages,
            layout=self._layout,
            device=self._device,
        )
        self._rx_ring = RegisteredGpuRing(
            direction="rx",
            depth=ring_depth,
            wave_pages=wave_pages,
            layout=self._layout,
            device=self._device,
        )
        self._storage.register_device_buffers(
            [self._tx_ring.tensor, self._rx_ring.tensor]
        )
        self._pack_stream = torch.cuda.Stream(device=self._device)
        self._unpack_stream = torch.cuda.Stream(device=self._device)
        # The rings are still consumed by exactly one physical operation at a
        # time. A bounded producer/consumer queue lets that worker immediately
        # start the next operation instead of waiting for another scheduler
        # round to observe the previous operation's synthetic storage ack.
        self._work_condition = threading.Condition()
        self._put_work_queue: deque[_GpuWorkItem] = deque()
        self._get_work_queue: deque[_GpuWorkItem] = deque()
        self._running_state: Optional[_GpuOperationState] = None
        self._admitted_put_ops = 0
        self._admitted_put_tokens = 0
        self._admitted_get_ops = 0
        self._put_op_cap = self._MAX_ADMITTED_PUT_OPS
        min_put_tokens = 2 * ring_depth * self._wave_pages * self._page_size
        self._put_token_cap = min(
            self._allocator.size_full,
            max(
                min_put_tokens,
                int(self._allocator.size_full * self._PUT_LOCK_FRACTION),
            ),
        )
        self._states: dict[int, _GpuOperationState] = {}
        self._prefetch_states: dict[int, _GpuOperationState] = {}
        self._cleanup_queue: queue.Queue[_GpuOperationState] = queue.Queue()
        self._operation_ids = itertools.count()
        self._closed = False
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="hicache-gpu-transient",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(
            "Initialized GPU-transient HiCache: layout=%s wave_pages=%d "
            "ring_depth=%d fixed_bytes=%d put_op_cap=%d put_token_cap=%d",
            self._layout.compatibility_id,
            wave_pages,
            ring_depth,
            self._tx_ring.nbytes + self._rx_ring.nbytes,
            self._put_op_cap,
            self._put_token_cap,
        )

    def _gpu_lease(self, lease: TransientBufferLease) -> _GpuTransientBufferLease:
        if not isinstance(lease, _GpuTransientBufferLease):
            raise TypeError(
                "GpuTransientBufferBackend received a lease from another backend."
            )
        return lease

    def _try_enqueue_work(self, item: _GpuWorkItem) -> bool:
        state = item.state
        with self._work_condition:
            if self._closed:
                return False
            if state.direction == "put":
                if self._admitted_put_ops >= self._put_op_cap:
                    return False
                next_tokens = self._admitted_put_tokens + state.num_tokens
                # One oversized operation is allowed so a valid tree node can
                # never wedge permanently at the queue head.
                if self._admitted_put_tokens and next_tokens > self._put_token_cap:
                    return False
                self._admitted_put_ops += 1
                self._admitted_put_tokens = next_tokens
                work_queue = self._put_work_queue
            elif state.direction == "get":
                # A GET owns private destination pages until scheduler commit.
                # Preserve the original one-prefetch budget while allowing it
                # to jump ahead of queued PUTs.
                if self._admitted_get_ops >= 1:
                    return False
                self._admitted_get_ops += 1
                work_queue = self._get_work_queue
            else:
                raise ValueError(f"Unknown GPU-transient direction {state.direction}.")
            state.enqueued_at = time.perf_counter()
            self._states[state.operation_id] = state
            if state.prefetch_operation is not None:
                self._prefetch_states[state.prefetch_operation.id] = state
            work_queue.append(item)
            self._work_condition.notify_all()
            return True

    def _worker_loop(self) -> None:
        while True:
            with self._work_condition:
                while (
                    not self._closed
                    and not self._get_work_queue
                    and not self._put_work_queue
                ):
                    self._work_condition.wait()
                if (
                    self._closed
                    and not self._get_work_queue
                    and not self._put_work_queue
                ):
                    return
                # Do not preempt a running PUT, but service a newly queued GET
                # before the next PUT.
                work_queue = (
                    self._get_work_queue
                    if self._get_work_queue
                    else self._put_work_queue
                )
                item = work_queue.popleft()
                self._running_state = item.state
                item.state.worker_started_at = time.perf_counter()
            try:
                item.run()
            except Exception:
                logger.exception(
                    "Unhandled GPU-transient worker failure op=%d direction=%s",
                    item.state.operation_id,
                    item.state.direction,
                )
                self._finish_state(item.state, False)
                if item.state.prefetch_operation is not None:
                    item.state.prefetch_operation.pool_transfers_done = True
            finally:
                self._mark_worker_terminal(item.state)
                with self._work_condition:
                    self._running_state = None
                    self._work_condition.notify_all()

    def _mark_worker_terminal(self, state: _GpuOperationState) -> None:
        state.terminal_event.set()
        with state.lock:
            enqueue_cleanup = state.release_requested and not state.released
        if enqueue_cleanup:
            self._cleanup_queue.put(state)

    def _is_cancelled(self, state: _GpuOperationState) -> bool:
        with state.lock:
            return state.cancelled

    def _finish_state(self, state: _GpuOperationState, success: bool) -> None:
        with state.lock:
            state.success = success and not state.cancelled
            state.done = True
        if not state.success and state.prefetch_operation is not None:
            state.prefetch_operation.mark_terminate()

    def _copy_wave(
        self,
        *,
        ring: RegisteredGpuRing,
        ring_lease,
        pool_name: PoolName,
        device_indices: torch.Tensor,
        index_page_size: int,
        q_pages: int,
        stream: torch.cuda.Stream,
        pack: bool,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        page_starts = device_indices.reshape(-1, index_page_size)[:, 0]
        slot = ring.slot_view(ring_lease)
        with torch.cuda.stream(stream):
            started = torch.cuda.Event(enable_timing=True)
            started.record(stream)
            for obj in self._layout.objects_for_pool(pool_name):
                if obj.codec == "token_rows":
                    copy_token_rows(
                        obj,
                        page_starts,
                        slot,
                        q_pages,
                        self._layout.combined_page_bytes,
                        pack=pack,
                    )
                elif obj.codec == "opaque_pages":
                    copy_opaque_pages(
                        obj,
                        page_starts,
                        slot,
                        q_pages,
                        self._layout.combined_page_bytes,
                        pack=pack,
                    )
                else:
                    raise ValueError(f"Unsupported GPU-transient codec {obj.codec}.")
            done = torch.cuda.Event(enable_timing=True)
            done.record(stream)
        return started, done

    @staticmethod
    def _all_pages_succeeded(
        results: dict[PoolName, list[bool]], pool_name: PoolName, expected_pages: int
    ) -> bool:
        page_results = results.get(pool_name)
        if page_results is None:
            page_results = results.get(pool_name.value)
        return (
            isinstance(page_results, (list, tuple))
            and len(page_results) == expected_pages
            and all(bool(result) for result in page_results)
        )

    def _pool_page_bytes(self, pool_name: PoolName) -> int:
        if not hasattr(self._layout, "objects_for_pool"):
            return 0
        return sum(
            obj.page_payload_bytes for obj in self._layout.objects_for_pool(pool_name)
        )

    @staticmethod
    def _cuda_elapsed_ms(started, done) -> float:
        if hasattr(started, "elapsed_time"):
            return float(started.elapsed_time(done))
        return 0.0

    def _build_io_specs(
        self,
        *,
        primary_indices: torch.Tensor,
        hash_values: list[str],
        pool_transfers: Optional[list[PoolTransfer]],
        pool_device_indices: Optional[dict[PoolName, torch.Tensor]] = None,
    ) -> tuple[_PoolIoSpec, ...]:
        transfers = {transfer.name: transfer for transfer in pool_transfers or ()}
        source_indices = {PoolName.KV: primary_indices}
        source_keys = {PoolName.KV: tuple(hash_values)}
        if any(obj.indices_from_pool == PoolName.SWA for obj in self._layout.objects):
            swa = transfers.get(PoolName.SWA)
            swa_indices = (pool_device_indices or {}).get(PoolName.SWA)
            if swa_indices is None and swa is not None:
                swa_indices = swa.device_indices
            if swa is not None:
                if swa_indices is None:
                    raise ValueError(
                        "GPU-transient DeepSeek V4 SWA transfer has no device indices."
                    )
                swa_objects = [
                    obj
                    for obj in self._layout.objects
                    if obj.indices_from_pool == PoolName.SWA
                ]
                swa_page_size = swa_objects[0].index_page_size
                if len(swa_indices) % swa_page_size != 0:
                    raise ValueError("GPU-transient SWA indices are not page-aligned.")
                swa_pages = len(swa_indices) // swa_page_size
                swa_keys = tuple((swa.keys or hash_values[-swa_pages:])[-swa_pages:])
                if len(swa_keys) != swa_pages:
                    raise ValueError(
                        "GPU-transient SWA key/device page count mismatch."
                    )
                source_indices[PoolName.SWA] = swa_indices
                source_keys[PoolName.SWA] = swa_keys

        specs = []
        for pool_name in self._layout.pool_names:
            objects = self._layout.objects_for_pool(pool_name)
            source_pool = objects[0].indices_from_pool
            index_page_size = objects[0].index_page_size
            if any(
                obj.indices_from_pool != source_pool
                or obj.index_page_size != index_page_size
                for obj in objects
            ):
                raise ValueError(
                    f"GPU-transient pool {pool_name} has inconsistent index geometry."
                )
            # DSV4 drops SWA component state outside the live sliding window.
            # Such nodes still carry FULL-backed C4/C128 payloads and must be
            # publishable; omit only pools sourced from the absent component.
            if source_pool not in source_indices:
                continue
            indices = source_indices[source_pool]
            keys = source_keys[source_pool]
            if len(indices) != len(keys) * index_page_size:
                raise ValueError(
                    f"GPU-transient {pool_name} key/device page count mismatch: "
                    f"keys={len(keys)}, indices={len(indices)}, "
                    f"page_size={index_page_size}."
                )
            specs.append(
                _PoolIoSpec(
                    name=pool_name,
                    keys=keys,
                    device_indices=indices,
                    index_page_size=index_page_size,
                )
            )
        return tuple(specs)

    def _run_pool_put(
        self, state: _GpuOperationState, spec: _PoolIoSpec, timing: _PoolTiming
    ) -> tuple[bool, str]:
        pending_waves = deque()
        next_start = 0
        num_pages = len(spec.keys)
        try:
            while next_start < num_pages or pending_waves:
                if self._is_cancelled(state):
                    return False, "cancelled"
                while (
                    next_start < num_pages and len(pending_waves) < self._tx_ring.depth
                ):
                    start = next_start
                    q_pages = min(self._wave_pages, num_pages - start)
                    token_start = start * spec.index_page_size
                    token_end = token_start + q_pages * spec.index_page_size
                    wait_start = time.perf_counter()
                    ring_lease = self._tx_ring.acquire()
                    timing.ring_wait_ms += (time.perf_counter() - wait_start) * 1000
                    try:
                        copy_result = self._copy_wave(
                            ring=self._tx_ring,
                            ring_lease=ring_lease,
                            pool_name=spec.name,
                            device_indices=spec.device_indices[token_start:token_end],
                            index_page_size=spec.index_page_size,
                            q_pages=q_pages,
                            stream=self._pack_stream,
                            pack=True,
                        )
                        if isinstance(copy_result, tuple):
                            copy_started, copy_done = copy_result
                        else:
                            copy_started = copy_done = copy_result
                    except Exception:
                        self._tx_ring.release(ring_lease)
                        raise
                    pending_waves.append(
                        (start, q_pages, ring_lease, copy_started, copy_done)
                    )
                    next_start += q_pages

                start, q_pages, ring_lease, copy_started, copy_done = (
                    pending_waves.popleft()
                )
                try:
                    copy_done.synchronize()
                    timing.cuda_copy_ms += self._cuda_elapsed_ms(
                        copy_started, copy_done
                    )
                    if self._is_cancelled(state):
                        return False, "cancelled"
                    transfer = PoolTransfer(
                        name=spec.name,
                        keys=list(spec.keys[start : start + q_pages]),
                    )
                    storage_start = time.perf_counter()
                    results = self._storage.batch_set_v2_device(
                        [transfer], self._tx_ring.regions(ring_lease, q_pages)
                    )
                    timing.storage_ms += (time.perf_counter() - storage_start) * 1000
                    if not self._all_pages_succeeded(results, spec.name, q_pages):
                        return False, f"storage_put:{spec.name.value}"
                    timing.pages += q_pages
                    timing.payload_bytes += q_pages * self._pool_page_bytes(spec.name)
                finally:
                    self._tx_ring.release(ring_lease)
        finally:
            while pending_waves:
                _start, _q_pages, ring_lease, copy_started, copy_done = (
                    pending_waves.popleft()
                )
                try:
                    copy_done.synchronize()
                    timing.cuda_copy_ms += self._cuda_elapsed_ms(
                        copy_started, copy_done
                    )
                finally:
                    self._tx_ring.release(ring_lease)
        return True, "ok"

    def _run_pool_get(
        self,
        state: _GpuOperationState,
        operation: Any,
        spec: _PoolIoSpec,
        timing: _PoolTiming,
    ) -> tuple[bool, str]:
        pending_waves = deque()
        try:
            for start in range(0, len(spec.keys), self._wave_pages):
                if self._is_cancelled(state) or operation.is_terminated():
                    return False, "cancelled"
                if len(pending_waves) >= self._rx_ring.depth:
                    ring_lease, copy_started, copy_done = pending_waves.popleft()
                    try:
                        copy_done.synchronize()
                        timing.cuda_copy_ms += self._cuda_elapsed_ms(
                            copy_started, copy_done
                        )
                    finally:
                        self._rx_ring.release(ring_lease)
                q_pages = min(self._wave_pages, len(spec.keys) - start)
                token_start = start * spec.index_page_size
                token_end = token_start + q_pages * spec.index_page_size
                wait_start = time.perf_counter()
                ring_lease = self._rx_ring.acquire()
                timing.ring_wait_ms += (time.perf_counter() - wait_start) * 1000
                unpack_queued = False
                try:
                    transfer = PoolTransfer(
                        name=spec.name,
                        keys=list(spec.keys[start : start + q_pages]),
                    )
                    storage_start = time.perf_counter()
                    results = self._storage.batch_get_v2_device(
                        [transfer], self._rx_ring.regions(ring_lease, q_pages)
                    )
                    timing.storage_ms += (time.perf_counter() - storage_start) * 1000
                    if not self._all_pages_succeeded(results, spec.name, q_pages):
                        return False, f"storage_get:{spec.name.value}"
                    copy_result = self._copy_wave(
                        ring=self._rx_ring,
                        ring_lease=ring_lease,
                        pool_name=spec.name,
                        device_indices=spec.device_indices[token_start:token_end],
                        index_page_size=spec.index_page_size,
                        q_pages=q_pages,
                        stream=self._unpack_stream,
                        pack=False,
                    )
                    if isinstance(copy_result, tuple):
                        copy_started, copy_done = copy_result
                    else:
                        copy_started = copy_done = copy_result
                    pending_waves.append((ring_lease, copy_started, copy_done))
                    unpack_queued = True
                    timing.pages += q_pages
                    timing.payload_bytes += q_pages * self._pool_page_bytes(spec.name)
                finally:
                    if not unpack_queued:
                        self._rx_ring.release(ring_lease)
            while pending_waves:
                ring_lease, copy_started, copy_done = pending_waves.popleft()
                try:
                    copy_done.synchronize()
                    timing.cuda_copy_ms += self._cuda_elapsed_ms(
                        copy_started, copy_done
                    )
                finally:
                    self._rx_ring.release(ring_lease)
        finally:
            while pending_waves:
                ring_lease, copy_started, copy_done = pending_waves.popleft()
                try:
                    copy_done.synchronize()
                    timing.cuda_copy_ms += self._cuda_elapsed_ms(
                        copy_started, copy_done
                    )
                finally:
                    self._rx_ring.release(ring_lease)
        return True, "ok"

    def _log_timing(
        self,
        *,
        state: _GpuOperationState,
        reason: str,
        total_ms: float,
        source_wait_ms: float,
        timings: dict[PoolName, _PoolTiming],
    ) -> None:
        payload = {
            "event": "hicache_gpu_transient_io",
            "direction": state.direction,
            "operation_id": state.operation_id,
            "success": state.success,
            "reason": reason,
            "layout": self._layout.compatibility_id,
            "anchor_pages": state.num_tokens // self._page_size,
            "queue_wait_ms": round(
                max(0.0, state.worker_started_at - state.enqueued_at) * 1000, 3
            ),
            "total_ms": round(total_ms, 3),
            "source_wait_ms": round(source_wait_ms, 3),
            "pools": {
                name.value: {
                    "pages": timing.pages,
                    "payload_bytes": timing.payload_bytes,
                    "ring_wait_ms": round(timing.ring_wait_ms, 3),
                    "cuda_copy_ms": round(timing.cuda_copy_ms, 3),
                    "storage_ms": round(timing.storage_ms, 3),
                }
                for name, timing in timings.items()
            },
        }
        logger.info("GPU_TRANSIENT_TIMING %s", json.dumps(payload, sort_keys=True))

    @staticmethod
    def _publish_pool_hit_pages(
        operation: Any, io_specs: Sequence[_PoolIoSpec]
    ) -> None:
        """Publish actual device-transfer completions for hybrid validation.

        ``batch_exists_v2`` records a trailing pool's usable anchor prefix, while
        the scheduler's terminal validation expects the number of pages that
        were actually fetched for that pool.  Host-backed I/O replaces those
        values from ``batch_get_v2``; GPU-transient I/O must do the same before
        publishing ``completed_tokens``.
        """
        fetched_pages = {spec.name: len(spec.keys) for spec in io_specs}
        hit_pages = operation.pool_storage_result.extra_pool_hit_pages
        for transfer in operation.pool_transfers or ():
            if transfer.name in fetched_pages:
                hit_pages[transfer.name] = fetched_pages[transfer.name]

    def _run_put(
        self,
        state: _GpuOperationState,
        *,
        node_id: int,
        device_indices: torch.Tensor,
        hash_values: list[str],
        pool_transfers: Optional[list[PoolTransfer]],
        source_ready: torch.cuda.Event,
    ) -> None:
        success = False
        terminal_reason = "unknown"
        timings: dict[PoolName, _PoolTiming] = {}
        total_start = time.perf_counter()
        source_wait_ms = 0.0
        try:
            with torch.cuda.device(self._device):
                wait_start = time.perf_counter()
                source_ready.synchronize()
                source_wait_ms = (time.perf_counter() - wait_start) * 1000
                if self._controller.backup_skip:
                    success = True
                    terminal_reason = "replicated_mla_non_owner"
                else:
                    state.io_specs = self._build_io_specs(
                        primary_indices=device_indices,
                        hash_values=hash_values,
                        pool_transfers=pool_transfers,
                    )
                    success = True
                    for spec in state.io_specs:
                        timing = timings.setdefault(spec.name, _PoolTiming())
                        success, terminal_reason = self._run_pool_put(
                            state, spec, timing
                        )
                        if not success:
                            break
        except Exception:
            success = False
            terminal_reason = "exception"
            logger.exception(
                "GPU-transient PUT failed op=%d node=%d layout=%s",
                state.operation_id,
                node_id,
                self._layout.compatibility_id,
            )
        finally:
            self._finish_state(state, success)
            event = _ThreadEvent(completed=True)
            self._controller.ack_write_queue.append(
                HiCacheAck(
                    start_event=event,
                    finish_event=event,
                    node_ids=[node_id],
                    num_tokens=state.num_tokens,
                    num_tokens_by_pool={PoolName.KV.value: state.num_tokens},
                    num_bytes=sum(t.payload_bytes for t in timings.values()),
                )
            )
            logger.info(
                "GPU-transient PUT terminal op=%d pages=%d layout=%s success=%s "
                "reason=%s",
                state.operation_id,
                state.num_tokens // self._page_size,
                self._layout.compatibility_id,
                state.success,
                terminal_reason,
            )
            self._log_timing(
                state=state,
                reason=terminal_reason,
                total_ms=(time.perf_counter() - total_start) * 1000,
                source_wait_ms=source_wait_ms,
                timings=timings,
            )

    def _run_get(
        self,
        state: _GpuOperationState,
        *,
        hash_values: list[str],
    ) -> None:
        success = False
        completion_published = False
        terminal_reason = "unknown"
        operation = state.prefetch_operation
        assert operation is not None and state.device_indices is not None
        timings: dict[PoolName, _PoolTiming] = {}
        total_start = time.perf_counter()
        try:
            with torch.cuda.device(self._device):
                if not state.io_specs:
                    state.io_specs = (
                        _PoolIoSpec(
                            name=PoolName.KV,
                            keys=tuple(hash_values),
                            device_indices=state.device_indices,
                            index_page_size=self._page_size,
                        ),
                    )
                success = True
                for spec in state.io_specs:
                    timing = timings.setdefault(spec.name, _PoolTiming())
                    success, terminal_reason = self._run_pool_get(
                        state, operation, spec, timing
                    )
                    if not success:
                        break
                if success:
                    self._publish_pool_hit_pages(operation, state.io_specs)
                    # Publish backend terminal state before completed_tokens.
                    # The scheduler treats completed_tokens as readiness and
                    # may immediately terminate/finalize the operation.
                    self._finish_state(state, True)
                    completion_published = True
                    success = state.success and operation.increment(state.num_tokens)
                    if not success:
                        with state.lock:
                            state.success = False
                        terminal_reason = "cancelled_at_commit"
                    else:
                        terminal_reason = "ok"
        except Exception:
            success = False
            terminal_reason = "exception"
            if completion_published:
                with state.lock:
                    state.success = False
                operation.mark_terminate()
            logger.exception(
                "GPU-transient GET failed op=%d request=%s layout=%s",
                state.operation_id,
                operation.request_id,
                self._layout.compatibility_id,
            )
        finally:
            operation.pool_transfers_done = True
            if not completion_published:
                self._finish_state(state, success)
            logger.info(
                "GPU-transient GET terminal op=%d pages=%d layout=%s success=%s "
                "reason=%s",
                state.operation_id,
                state.num_tokens // self._page_size,
                self._layout.compatibility_id,
                state.success,
                terminal_reason,
            )
            self._log_timing(
                state=state,
                reason=terminal_reason,
                total_ms=(time.perf_counter() - total_start) * 1000,
                source_wait_ms=0.0,
                timings=timings,
            )

    def backup_fits(
        self, primary_tokens: int, pool_transfers: Optional[list[PoolTransfer]]
    ) -> bool:
        if primary_tokens <= 0 or primary_tokens % self._page_size != 0:
            return False
        swa = next((t for t in pool_transfers or () if t.name == PoolName.SWA), None)
        if swa is not None:
            if swa.device_indices is not None:
                page_size = self._controller.mem_pool_host.entry_map[
                    PoolName.SWA
                ].host_pool.page_size
                if len(swa.device_indices) % page_size != 0:
                    return False
        return True

    def backup_live_cap(self) -> int:
        with self._work_condition:
            return 0 if self._closed else self._put_token_cap

    def requires_write_drain_before_device_eviction(self) -> bool:
        # Admitted work pins source pages, but metadata-only pending intents do
        # not. Drain at an eviction dependency boundary so those intents are
        # either admitted or found stale before page reuse.
        return True

    def wait_for_progress(self) -> None:
        # Publication stays scheduler-owned. Wait for the oldest admitted PUT
        # to finish physical I/O; check_hicache_events() performs tree mutation
        # and TP collectives afterwards.
        with self._work_condition:
            state = next(
                (
                    state
                    for state in self._states.values()
                    if state.direction == "put" and not state.released
                ),
                None,
            )
        if state is not None:
            state.terminal_event.wait()

    def backup_blocked(self, pool_transfers: Optional[list[PoolTransfer]]) -> bool:
        del pool_transfers
        with self._work_condition:
            return self._closed or self._admitted_put_ops >= self._put_op_cap

    def stage_backup(
        self,
        device_indices: torch.Tensor,
        *,
        node_id: int,
        pool_transfers: Optional[list[PoolTransfer]],
        token_ids: Sequence[int],
        hash_values: list[str],
        prefix_keys: Optional[list[str]],
    ) -> Optional[TransientBufferLease]:
        del token_ids, prefix_keys
        if not self.backup_fits(len(device_indices), pool_transfers):
            return None
        if len(hash_values) * self._page_size != len(device_indices):
            raise ValueError("GPU-transient PUT hash/device page count mismatch.")
        operation_id = next(self._operation_ids)
        state = _GpuOperationState(
            operation_id=operation_id,
            direction="put",
            num_tokens=len(device_indices),
        )
        try:
            with torch.cuda.device(self._device):
                source_ready = torch.cuda.Event()
                source_ready.record(torch.cuda.current_stream(self._device))
            device_indices_ref = device_indices
            hash_values_ref = list(hash_values)
            pool_transfers_ref = list(pool_transfers or ())
            work = _GpuWorkItem(
                state=state,
                run=lambda: self._run_put(
                    state,
                    node_id=node_id,
                    device_indices=device_indices_ref,
                    hash_values=hash_values_ref,
                    pool_transfers=pool_transfers_ref,
                    source_ready=source_ready,
                ),
            )
            if not self._try_enqueue_work(work):
                return None
        except Exception:
            logger.exception("Failed to submit GPU-transient PUT op=%d", operation_id)
            return None
        return _GpuTransientBufferLease(
            num_tokens=len(device_indices),
            accounted_tokens=0,
            pool_names=tuple(dict.fromkeys((PoolName.KV, *self._layout.pool_names))),
            state=state,
        )

    def submit_storage_write(
        self,
        lease: TransientBufferLease,
        *,
        token_ids: Sequence[int],
        hash_values: list[str],
        prefix_keys: Optional[list[str]],
    ) -> int:
        gpu_lease = self._gpu_lease(lease)
        state = gpu_lease.state
        with state.lock:
            if not state.done:
                raise RuntimeError("GPU-transient PUT ack arrived before terminal I/O.")
            success = state.success
        operation = StorageOperation(
            None,
            list(token_ids),
            hash_value=list(hash_values),
            prefix_keys=prefix_keys,
        )
        operation.completed_tokens = lease.num_tokens if success else 0
        self._controller.ack_backup_queue.put(operation)
        return operation.id

    def _reserve_prefetch_pages(self, num_tokens: int) -> Optional[torch.Tensor]:
        available = self._full_allocator.available_size()
        if available < num_tokens:
            needed = num_tokens - available
            evicted = self._cache.evict(EvictParams(num_tokens=needed))
            if evicted.num_tokens_evicted < needed:
                return None
        return self._full_allocator.alloc(num_tokens)

    def _reserve_swa_prefetch_pages(self, num_tokens: int) -> Optional[torch.Tensor]:
        entry = self._controller.mem_pool_host.entry_map.get(PoolName.SWA)
        if entry is None or entry.device_alloc_fn is None:
            return None
        indices = entry.device_alloc_fn(num_tokens)
        if indices is None and entry.device_evict_fn is not None:
            entry.device_evict_fn(num_tokens)
            indices = entry.device_alloc_fn(num_tokens)
        return indices

    def _free_pool_device_indices(
        self, pool_device_indices: dict[PoolName, torch.Tensor]
    ) -> None:
        for pool_name, indices in pool_device_indices.items():
            if indices is None or indices.numel() == 0:
                continue
            entry = self._controller.mem_pool_host.entry_map.get(pool_name)
            if entry is not None and entry.device_free_fn is not None:
                entry.device_free_fn(indices)

    def _prepare_prefetch_transfers(self, operation: Any) -> None:
        if not operation.pool_transfers:
            return
        self._controller._sync_trailing_keys(
            operation.pool_transfers,
            operation.hash_value,
            len(operation.hash_value),
        )
        self._controller._resolve_sidecar_derived_pool_transfers(operation)

    def try_start_prefetch(self, operation: Any) -> Optional[TransientBufferLease]:
        self.drain_completions()
        num_tokens = operation.storage_hit_count
        if num_tokens <= 0 or num_tokens % self._page_size != 0:
            return None
        with self._work_condition:
            if self._closed or self._admitted_get_ops >= 1:
                return None
        device_indices = self._reserve_prefetch_pages(num_tokens)
        if device_indices is None:
            return None
        operation.hash_value = operation.hash_value[: num_tokens // self._page_size]
        self._prepare_prefetch_transfers(operation)
        pool_device_indices: dict[PoolName, torch.Tensor] = {}
        if any(obj.indices_from_pool == PoolName.SWA for obj in self._layout.objects):
            swa_transfer = next(
                (t for t in operation.pool_transfers or () if t.name == PoolName.SWA),
                None,
            )
            if swa_transfer is None:
                self._full_allocator.free(device_indices)
                raise ValueError("GPU-transient DSV4 prefetch is missing SWA keys.")
            swa_entry = self._controller.mem_pool_host.entry_map[PoolName.SWA]
            swa_tokens = len(swa_transfer.keys or ()) * swa_entry.host_pool.page_size
            swa_indices = self._reserve_swa_prefetch_pages(swa_tokens)
            if swa_indices is None:
                self._full_allocator.free(device_indices)
                return None
            pool_device_indices[PoolName.SWA] = swa_indices
        operation.pool_transfers_done = False
        operation_id = next(self._operation_ids)
        state = _GpuOperationState(
            operation_id=operation_id,
            direction="get",
            num_tokens=num_tokens,
            device_indices=device_indices,
            pool_device_indices=pool_device_indices,
            prefetch_operation=operation,
        )
        state.io_specs = self._build_io_specs(
            primary_indices=device_indices,
            hash_values=list(operation.hash_value),
            pool_transfers=operation.pool_transfers,
            pool_device_indices=pool_device_indices,
        )
        try:
            # PREFETCH preparation allocated metadata-only Host indices for
            # SWA key geometry. Device-ring I/O owns real destinations now.
            if operation.pool_transfers:
                self._controller.append_host_mem_release(
                    extra_pools=operation.pool_transfers
                )
            for transfer in operation.pool_transfers or ():
                if transfer.indices_from_pool is None:
                    transfer.host_indices = None
            hash_values_ref = list(operation.hash_value)
            work = _GpuWorkItem(
                state=state,
                run=lambda: self._run_get(state, hash_values=hash_values_ref),
            )
            if not self._try_enqueue_work(work):
                self._full_allocator.free(device_indices)
                self._free_pool_device_indices(pool_device_indices)
                operation.mark_terminate()
                operation.pool_transfers_done = True
                return None
        except Exception:
            self._full_allocator.free(device_indices)
            self._free_pool_device_indices(pool_device_indices)
            operation.mark_terminate()
            operation.pool_transfers_done = True
            logger.exception("Failed to submit GPU-transient GET op=%d", operation_id)
            return None
        return _GpuTransientBufferLease(
            num_tokens=num_tokens,
            accounted_tokens=0,
            pool_names=tuple(dict.fromkeys((PoolName.KV, *self._layout.pool_names))),
            state=state,
        )

    def terminate_prefetch(self, operation: Any) -> tuple[int, list[str]]:
        state = self._prefetch_states.get(operation.id)
        operation.mark_terminate()
        if state is not None:
            with state.lock:
                state.cancelled = not state.done
        return operation.completed_tokens, operation.hash_value

    def finalize_prefetch(
        self,
        lease: TransientBufferLease,
        *,
        usable_tokens: int,
        completed_tokens: int,
    ) -> TransientBufferLease:
        gpu_lease = self._gpu_lease(lease)
        state = gpu_lease.state
        if not 0 <= usable_tokens <= completed_tokens <= lease.num_tokens:
            raise ValueError(
                "Invalid GPU-transient prefetch bounds: "
                f"usable={usable_tokens}, completed={completed_tokens}, "
                f"allocated={lease.num_tokens}."
            )
        if completed_tokens == 0:
            # Timeout/cancellation can return MISS while the worker drains.
            # release() retains private pages until terminal I/O.
            self.release(lease)
            return _GpuTransientBufferLease(
                num_tokens=0,
                accounted_tokens=0,
                pool_names=gpu_lease.pool_names,
                state=state,
            )
        with state.lock:
            if not state.done or not state.success or state.device_indices is None:
                raise RuntimeError("Cannot finalize an incomplete GPU-transient GET.")
            tail = state.device_indices[usable_tokens:]
            state.device_indices = state.device_indices[:usable_tokens]
        if tail.numel() > 0:
            getattr(self, "_full_allocator", self._allocator).free(tail)
        return _GpuTransientBufferLease(
            num_tokens=usable_tokens,
            accounted_tokens=0,
            pool_names=gpu_lease.pool_names,
            state=state,
        )

    def discard_prefetch(
        self, lease: TransientBufferLease, *, completed_tokens: int
    ) -> None:
        del completed_tokens
        self.release(lease)

    def release_unstarted_prefetch(
        self, pool_transfers: Optional[list[PoolTransfer]]
    ) -> None:
        self._controller.append_host_mem_release(extra_pools=pool_transfers)

    def restore(
        self, lease: TransientBufferLease, *, operation_id: int
    ) -> Optional[TransientRestore]:
        del operation_id
        state = self._gpu_lease(lease).state
        with state.lock:
            if not state.done or not state.success or state.device_indices is None:
                return None
            device_indices = state.device_indices
            pool_device_indices = dict(state.pool_device_indices)
        return TransientRestore(
            device_indices=device_indices,
            pool_device_indices=pool_device_indices,
            requires_ack=False,
        )

    def commit_restore(self, lease: TransientBufferLease) -> None:
        state = self._gpu_lease(lease).state
        with state.lock:
            if state.direction != "get" or state.device_indices is None:
                raise RuntimeError(
                    "Only a completed GPU-transient GET can be committed."
                )
            state.committed = True

    def restore_needs_device_allocation(self, lease: TransientBufferLease) -> bool:
        self._gpu_lease(lease)
        return False

    def storage_write_succeeded(self, lease: TransientBufferLease) -> Optional[bool]:
        state = self._gpu_lease(lease).state
        with state.lock:
            return state.success if state.done else False

    def _release_state(self, state: _GpuOperationState) -> None:
        device_indices = None
        pool_device_indices = {}
        with state.lock:
            if state.released or not state.done:
                return
            if state.direction == "get" and not state.committed:
                device_indices = state.device_indices
                pool_device_indices = state.pool_device_indices
            state.device_indices = None
            state.pool_device_indices = {}
            state.released = True
        if device_indices is not None and device_indices.numel() > 0:
            getattr(self, "_full_allocator", self._allocator).free(device_indices)
        self._free_pool_device_indices(pool_device_indices)
        if state.prefetch_operation is not None:
            prefetch_id = state.prefetch_operation.id
        else:
            prefetch_id = None
        with self._work_condition:
            if prefetch_id is not None:
                self._prefetch_states.pop(prefetch_id, None)
            removed = self._states.pop(state.operation_id, None)
            if removed is not None:
                if state.direction == "put":
                    self._admitted_put_ops -= 1
                    self._admitted_put_tokens -= state.num_tokens
                else:
                    self._admitted_get_ops -= 1
            self._work_condition.notify_all()

    def release(self, lease: TransientBufferLease) -> None:
        state = self._gpu_lease(lease).state
        with state.lock:
            if state.released:
                return
            if not state.done:
                state.cancelled = True
                state.release_requested = True
                if state.prefetch_operation is not None:
                    state.prefetch_operation.mark_terminate()
                return
        self._release_state(state)

    def drain_completions(self) -> None:
        while True:
            try:
                state = self._cleanup_queue.get_nowait()
            except queue.Empty:
                break
            self._release_state(state)

    def reset(self) -> None:
        with self._work_condition:
            states = list(self._states.values())
        for state in states:
            with state.lock:
                state.cancelled = True
                state.release_requested = True
                if state.prefetch_operation is not None:
                    state.prefetch_operation.mark_terminate()
        with self._work_condition:
            while (
                self._running_state is not None
                or self._get_work_queue
                or self._put_work_queue
            ):
                self._work_condition.wait()
        self.drain_completions()
        for state in states:
            self._release_state(state)

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        with self._work_condition:
            self._closed = True
            self._work_condition.notify_all()
        self._worker_thread.join()
