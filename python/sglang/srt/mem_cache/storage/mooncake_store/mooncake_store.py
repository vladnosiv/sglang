import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min

logger = logging.getLogger(__name__)


def _parse_global_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int
    check_server: bool

    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.value
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        if "master_server_address" not in config:
            raise ValueError("master_server_address is required in config file")

        return MooncakeStoreConfig(
            local_hostname=config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get("master_server_address"),
            master_metrics_port=config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # other required environment variables...
        if not envs.MOONCAKE_MASTER.is_set():
            raise ValueError("The environment variable 'MOONCAKE_MASTER' is not set.")

        # Special handling for local_hostname: try MOONCAKE_LOCAL_HOSTNAME first,
        # then fall back to LOCAL_HOSTNAME if not set.
        # This is for forward compatibility with the legacy LOCAL_HOSTNAME environment variable.
        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.value
        else:
            local_hostname = os.getenv(
                "LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default
            )

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.value,
            global_segment_size=_parse_global_segment_size(
                envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.value
            ),
            protocol=envs.MOONCAKE_PROTOCOL.value,
            device_name=envs.MOONCAKE_DEVICE.value,
            master_server_address=envs.MOONCAKE_MASTER.value,
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.value,
            check_server=envs.MOONCAKE_CHECK_SERVER.value,
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """Load config from extra_config dictionary."""
        if "master_server_address" not in extra_config:
            raise ValueError("master_server_address is required in extra_config")

        return MooncakeStoreConfig(
            local_hostname=extra_config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=extra_config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config["master_server_address"],
            master_metrics_port=extra_config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=extra_config.get(
                "check_server", envs.MOONCAKE_CHECK_SERVER.default
            ),
        )


class MooncakeStore(HiCacheStorage):

    def __init__(self, storage_config: HiCacheStorageConfig = None):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
                "to run SGLang with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()

            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            # Load configuration with master_server_address prioritized from extra_config if available
            if (
                extra_config is not None
                and extra_config.get("master_server_address") is not None
            ):
                # Load from extra_config
                self.config = MooncakeStoreConfig.load_from_extra_config(extra_config)
                logger.info(
                    "Mooncake Configuration loaded from extra_config successfully."
                )
            elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
                # Load from config file
                self.config = MooncakeStoreConfig.from_file()
                logger.info("Mooncake Configuration loaded from file successfully.")
            else:
                # Load from environment variables
                self.config = MooncakeStoreConfig.load_from_env()
                logger.info("Mooncake Configuration loaded from env successfully.")

            tp_scale_factor = 1 if storage_config is None else storage_config.tp_size

            per_tp_global_segment_size = (
                self.config.global_segment_size // tp_scale_factor
            )

            # Check if extra_backend_tag should be passed to MooncakeDistributedStore
            self.extra_backend_tag = None
            if extra_config and "extra_backend_tag" in extra_config:
                self.extra_backend_tag = extra_config["extra_backend_tag"]
                logger.info(f"Using extra_backend_tag: {self.extra_backend_tag}")

            # Check server status
            if self.config.check_server:
                self.check_server()

            # Handle JSON device_name configuration
            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        # Try both integer and string keys since JSON parsing may convert keys
                        device_name = device_config.get(tp_rank, "")
                        if not device_name:
                            device_name = device_config.get(str(tp_rank), "")
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"Failed to parse device_name as JSON: {device_name}"
                    )
                    device_name = ""

            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                per_tp_global_segment_size,
                DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                self.config.protocol,
                device_name,
                self.config.master_server_address,
            )
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.local_rank = storage_config.tp_rank
            else:
                self.is_mla_backend = False
                self.local_rank = 0

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    def check_server(self):
        master_server_ip = self.config.master_server_address.split(":")[0]
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"
        start_time = time.perf_counter()

        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if check_segments_resp.text == "":
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            logger.info("Mooncake store server started successfully.")
            check_result = True
            break

        if not check_result:
            logger.error("Launch mooncake store server timeout")
            raise ValueError("Launch mooncake store server timeout")

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB
        assert self.store.put(warmup_key, warmup_value) == 0
        assert self.store.is_exist(warmup_key) == 1
        assert self.store.get(warmup_key) == warmup_value

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
            "page_head",
        ], "mooncake store storage backend only support page first or page first direct layout"
        buffer = self.mem_pool_host.kv_buffer
        try:
            buffer_ptr = buffer.data_ptr()
            buffer_size = buffer.numel() * buffer.element_size()
            ret_code = self.store.register_buffer(buffer_ptr, buffer_size)
            if ret_code:
                logger.error(f"Failed to register buffer, error code: {ret_code}")
                raise RuntimeError(
                    f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
                )
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

    def register_mem_pool_device(self, mem_pool_device: KVCache):
        super().register_mem_pool_device(mem_pool_device)
        # Register device buffers for RDMA access
        # Similar to host buffer registration, but for device memory
        try:
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                mem_pool_device.get_contiguous_buf_infos()
            )
            for ptr, len, _ in zip(kv_data_ptrs, kv_data_lens, kv_item_lens):
                self.register_device_buffer(ptr, len)

            logger.info("Device memory buffers registered for Mooncake direct mode")
        except (AttributeError, TypeError) as err:
            logger.error("Failed to register device buffers to Mooncake Store: %s", err)
            raise RuntimeError(
                f"Mooncake Store Device Buffer Registration Error: {err}"
            ) from err

    def register_device_buffer(self, ptr: int, size: int) -> None:
        ret_code = self.store.register_buffer(int(ptr), int(size))
        if ret_code:
            logger.error(f"Failed to register buffer, error code: {ret_code}")
            raise RuntimeError(
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
            )

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.local_rank}_k")
            key_list.append(f"{key_}_{self.local_rank}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: List[int], is_set_operate=False):
        """
        refer to https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/pybind_client.h
        for batch_get_into, results is Vector of integers,
            where each element is the number of bytes read on success, or a negative value on error
        for batch_put_from, results is Vector of integers,
            where each element is 0 on success, or a negative value on error
        """
        if self.is_mla_backend:
            return [k_res == 0 if is_set_operate else k_res > 0 for k_res in results]
        else:
            kv_pairs = zip(results[::2], results[1::2])
            return [
                (
                    (k_res == 0 and v_res == 0)
                    if is_set_operate
                    else (k_res > 0 and v_res > 0)
                )
                for k_res, v_res in kv_pairs
            ]

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Apply extra_backend_tag prefix if available
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Apply extra_backend_tag prefix if available
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        exist_result = self._batch_exist(key_strs)

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        for i in range(len(key_strs)):
            if exist_result[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
            else:
                set_results[i] = 0

        # Only set non-existing keys to storage
        if len(set_keys) > 0:
            put_results = self._put_batch_zero_copy_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
        assert target_location is not None and target_sizes is not None
        exist_result = self._batch_exist([key])
        if exist_result[0] == 1:
            return True
        put_result = self._put_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0

    def _apply_extra_backend_tag(self, keys: List[str]) -> List[str]:
        if self.extra_backend_tag is None:
            return keys
        prefix = self.extra_backend_tag
        return [f"{prefix}_{key}" for key in keys]

    def _is_scatter_gather_locations(self, target_locations: Any) -> bool:
        # direct-controller passes List[List[int]] (pages x buffers)
        return (
            isinstance(target_locations, list)
            and len(target_locations) > 0
            and isinstance(target_locations[0], list)
        )

    def _explode_scatter_gather(
        self,
        keys: List[str],
        target_locations: List[List[int]],
        target_sizes: List[List[int]],
    ) -> tuple[List[str], List[int], List[int], int]:
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return [], [], [], 0

        num_bufs = len(target_locations[0])
        for i in range(len(keys)):
            assert len(target_locations[i]) == num_bufs
            assert len(target_sizes[i]) == num_bufs

        # Always include tp rank in key to avoid collisions across TP workers.
        # buf_idx corresponds to the contiguous buffer index returned by
        # KVCache.get_contiguous_buf_infos() (MHA: 2*layers, MLA: layers).
        flat_keys: List[str] = []
        flat_ptrs: List[int] = []
        flat_sizes: List[int] = []
        for page_idx, base_key in enumerate(keys):
            for buf_idx in range(num_bufs):
                flat_keys.append(f"{base_key}_{self.local_rank}_b{buf_idx}")
                flat_ptrs.append(int(target_locations[page_idx][buf_idx]))
                flat_sizes.append(int(target_sizes[page_idx][buf_idx]))

        return flat_keys, flat_ptrs, flat_sizes, num_bufs

    def _count_consecutive_full_pages(self, ok_flags: List[bool], num_bufs: int) -> int:
        if num_bufs <= 0:
            return 0
        assert len(ok_flags) % num_bufs == 0
        pages = len(ok_flags) // num_bufs
        for p in range(pages):
            start = p * num_bufs
            end = start + num_bufs
            if not all(ok_flags[start:end]):
                return p
        return pages

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool | int:
        # Only support zero copy set for now
        assert target_locations is not None and target_sizes is not None

        if len(keys) == 0:
            return False

        # Apply extra_backend_tag prefix if available
        keys = self._apply_extra_backend_tag(keys)

        # direct-controller path: scatter/gather pointers per page
        if self._is_scatter_gather_locations(target_locations):
            assert isinstance(target_locations, list) and isinstance(target_sizes, list)
            flat_keys, flat_ptrs, flat_sizes, num_bufs = self._explode_scatter_gather(
                keys, target_locations, target_sizes
            )
            if not flat_keys:
                return 0

            exist_result = self._batch_exist(flat_keys)

            set_keys: List[str] = []
            set_ptrs: List[int] = []
            set_sizes: List[int] = []
            set_indices: List[int] = []
            for i in range(len(flat_keys)):
                if exist_result[i] != 1:
                    set_keys.append(flat_keys[i])
                    set_ptrs.append(flat_ptrs[i])
                    set_sizes.append(flat_sizes[i])
                    set_indices.append(i)

            if len(set_keys) > 0:
                put_result = self._put_batch_zero_copy_impl(
                    set_keys, set_ptrs, set_sizes
                )
                for j, flat_i in enumerate(set_indices):
                    if put_result[j] == 0:
                        exist_result[flat_i] = 1

            ok_flags = [x == 1 for x in exist_result]
            return self._count_consecutive_full_pages(ok_flags, num_bufs)

        # legacy path: one key -> one ptr/size
        assert len(keys) == len(target_locations) == len(target_sizes)

        for i in range(len(keys)):
            if (
                keys[i] is None
                or target_locations[i] is None
                or target_sizes[i] is None
            ):
                return False

        exist_result = self._batch_exist(keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(keys)):
            if exist_result[i] != 1:
                set_keys.append(keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes[i])
                set_indices.append(i)
        # Only set non-existing keys to storage
        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        success_count = 0
        for i in range(len(keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        # Return the number of consecutive successful operations from the start.
        return success_count

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        get_result = self._get_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return get_result[0] >= 0

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert target_locations is not None and target_sizes is not None
        if len(keys) == 0:
            return 0

        # Apply extra_backend_tag prefix if available
        keys = self._apply_extra_backend_tag(keys)

        # direct-controller path: scatter/gather pointers per page
        if self._is_scatter_gather_locations(target_locations):
            assert isinstance(target_locations, list) and isinstance(target_sizes, list)
            flat_keys, flat_ptrs, flat_sizes, num_bufs = self._explode_scatter_gather(
                keys, target_locations, target_sizes
            )
            if not flat_keys:
                return 0

            get_result = self._get_batch_zero_copy_impl(
                flat_keys, flat_ptrs, flat_sizes
            )
            ok_flags = [r >= 0 for r in get_result]
            return self._count_consecutive_full_pages(ok_flags, num_bufs)

        # legacy path: one key -> one ptr/size
        assert len(keys) == len(target_locations) == len(target_sizes)
        get_result = self._get_batch_zero_copy_impl(
            keys, target_locations, target_sizes
        )
        # legacy semantics: MHA stores k/v as two keys per page
        key_multiplier = 1 if self.is_mla_backend else 2
        for i in range(len(keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(keys) // key_multiplier

    def exists(self, key) -> bool:
        exist_result = self._batch_exist([key])
        return exist_result[0] == 1

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if not keys:
            return 0

        # Apply extra_backend_tag prefix if available
        keys = self._apply_extra_backend_tag(keys)

        # If caller already provides explicit KV keys (packed/legacy mode),
        # do NOT expand into b{buf} keys. Just check as-is.
        # Examples:
        # - MHA packed/legacy: "{hash}_{tp}_k" / "{hash}_{tp}_v"
        # - MLA legacy: "{hash}_k"
        if any(
            isinstance(k, str)
            and (
                k.endswith("_k")
                or k.endswith("_v")
                or f"_{self.local_rank}_k" in k
                or f"_{self.local_rank}_v" in k
            )
            for k in keys
        ):
            exist_result = self._batch_exist(keys)
            for i, r in enumerate(exist_result):
                if r != 1:
                    return i
            return len(keys)

        # direct-controller scatter/gather mode: one key per page, but each page maps to multiple buffers.
        # Implement "pages hit" semantics by probing buffer indices until the first miss.
        if self.mem_pool_device is not None:
            kv_data_ptrs, _, _ = self.mem_pool_device.get_contiguous_buf_infos()
            num_bufs = len(kv_data_ptrs)
        else:
            num_bufs = 0

        if num_bufs > 0:
            query_keys: List[str] = []
            for key in keys:
                for buf_idx in range(num_bufs):
                    query_keys.append(f"{key}_{self.local_rank}_b{buf_idx}")

            exist_result = self._batch_exist(query_keys)
            ok_flags = [x == 1 for x in exist_result]
            return self._count_consecutive_full_pages(ok_flags, num_bufs)

        # Fallback: legacy v1 semantics (page keys -> k/v keys)
        if self.is_mla_backend:
            query_keys = [f"{key}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            for key in keys:
                query_keys.append(f"{key}_{self.local_rank}_k")
                query_keys.append(f"{key}_{self.local_rank}_v")
            key_multiplier = 2

        exist_result = self._batch_exist(query_keys)
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def clear(self) -> None:
        self.store.remove_all()

    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        # сейчас сюда приходят keys по числу страниц, а buffer_ptrs len(keys_strs),
        # но внутри у каждого ключа лежат 2 * Layers элементов, нужно учитывать при записи
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def _batch_exist(self, key_strs: List[str]) -> List[int]:
        return self.store.batch_is_exist(key_strs)
