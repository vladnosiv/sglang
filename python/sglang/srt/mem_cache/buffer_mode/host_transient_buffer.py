"""Host-backed transient buffers for HiCache buffer-only mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.srt.managers.cache_controller import HICACHE_WRITE_STAGING_POOL_FRACTION
from sglang.srt.mem_cache.buffer_mode.transient_buffer import (
    TransientBufferBackend,
    TransientBufferLease,
    TransientRestore,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.unified_cache.components import SWAComponent


def validate_host_transient_buffer_stack(
    sidecar_pool_specs: list, swa_component: Optional[SWAComponent]
) -> None:
    """Validate configurations supported by the Host transient backend."""
    if sidecar_pool_specs:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "sidecar storage pools (DeepSeek-V4 compressed regions)."
        )
    swa = swa_component
    if swa is not None and swa._swa_kv_pool_host is None:
        # Only reachable on SWA models with the unified_kv layout (SWA as
        # a device-only ring): without a host pool the window can neither
        # stage for writes nor fetch for load-backs.
        raise ValueError(
            "--hicache-host-memory-mode buffer_only on SWA models "
            "requires an SWA host staging pool; the unified_kv layout "
            "keeps SWA as a device-only ring."
        )
    if swa is not None and swa._swa_kv_pool_host is not None:
        # Below two windows the pool cannot hold a staging write AND the
        # loads-priority reserve (_aux_loads_margin floors at one window).
        window_tokens = swa.full_window_pages * swa._swa_kv_pool_host.page_size
        if swa._swa_kv_pool_host.size < 2 * window_tokens:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only requires an SWA "
                f"host pool of at least two trailing windows "
                f"({2 * window_tokens} tokens; got "
                f"{swa._swa_kv_pool_host.size}): one staging a write "
                "while one stays reserved for prefetch window allocs."
            )


@dataclass(frozen=True)
class _HostTransientBufferLease(TransientBufferLease):
    host_indices: torch.Tensor
    pool_transfers: tuple[PoolTransfer, ...]


class HostTransientBufferBackend(TransientBufferBackend):
    """Adapter preserving the original Host-backed buffer-only data path."""

    def __init__(
        self,
        controller: HybridCacheController,
        *,
        swa_window_pages: int,
    ) -> None:
        self._controller = controller
        self._swa_window_pages = swa_window_pages

    def _host_lease(self, lease: TransientBufferLease) -> _HostTransientBufferLease:
        if not isinstance(lease, _HostTransientBufferLease):
            raise TypeError(
                "HostTransientBufferBackend received a lease from another backend."
            )
        return lease

    def _pool_names(
        self, pool_transfers: Optional[list[PoolTransfer]]
    ) -> tuple[PoolName, ...]:
        return (PoolName.KV, *(t.name for t in pool_transfers or ()))

    def _aux_loads_margin(self, host_pool) -> int:
        return max(
            self._swa_window_pages * host_pool.page_size,
            host_pool.size // 10,
        )

    def backup_fits(
        self, primary_tokens: int, pool_transfers: Optional[list[PoolTransfer]]
    ) -> bool:
        group = self._controller.mem_pool_host
        if primary_tokens > group.size:
            return False
        for transfer in pool_transfers or ():
            entry = group.entry_map.get(transfer.name)
            if entry is None:
                continue
            needed = len(transfer.keys or ()) * entry.host_pool.page_size
            if needed > entry.host_pool.size - self._aux_loads_margin(entry.host_pool):
                return False
        return True

    def backup_live_cap(self) -> int:
        controller = self._controller
        pool_tokens = controller.mem_pool_host.size
        return max(
            int(HICACHE_WRITE_STAGING_POOL_FRACTION * pool_tokens),
            pool_tokens - controller.prefetch_tokens_occupied - pool_tokens // 10,
        )

    def backup_blocked(self, pool_transfers: Optional[list[PoolTransfer]]) -> bool:
        group = self._controller.mem_pool_host
        for transfer in pool_transfers or ():
            entry = group.entry_map.get(transfer.name)
            if entry is None:
                continue
            needed = len(transfer.keys or ()) * entry.host_pool.page_size
            headroom = entry.host_pool.available_size() - self._aux_loads_margin(
                entry.host_pool
            )
            if needed > headroom:
                return True
        return False

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
        host_indices = self._controller.write(
            device_indices,
            node_id=node_id,
            extra_pools=pool_transfers,
        )
        if host_indices is None:
            return None
        transfers = tuple(pool_transfers or ())
        return _HostTransientBufferLease(
            num_tokens=len(host_indices),
            accounted_tokens=len(host_indices),
            pool_names=self._pool_names(pool_transfers),
            host_indices=host_indices,
            pool_transfers=transfers,
        )

    def _aux_window_keys(
        self, hash_values: list[str], transfer: PoolTransfer
    ) -> Optional[list[str]]:
        if transfer.host_indices is None or transfer.host_indices.numel() == 0:
            return None
        if transfer.indices_from_pool is not None:
            return None
        entry = self._controller.mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            return None
        num_keys = len(transfer.host_indices) // entry.host_pool.page_size
        if num_keys == 0 or num_keys > len(hash_values):
            return None
        return hash_values[-num_keys:]

    def submit_storage_write(
        self,
        lease: TransientBufferLease,
        *,
        token_ids: Sequence[int],
        hash_values: list[str],
        prefix_keys: Optional[list[str]],
    ) -> int:
        host_lease = self._host_lease(lease)
        storage_transfers: list[PoolTransfer] = []
        for staged in host_lease.pool_transfers:
            keys = self._aux_window_keys(hash_values, staged)
            if keys is None:
                continue
            storage_transfers.append(
                PoolTransfer(
                    name=staged.name,
                    host_indices=staged.host_indices,
                    keys=keys,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        return self._controller.write_storage(
            host_lease.host_indices,
            list(token_ids),
            hash_values,
            prefix_keys,
            extra_pools=storage_transfers or None,
        )

    def try_start_prefetch(self, operation: Any) -> Optional[TransientBufferLease]:
        if self._controller.prefetch_rate_limited():
            return None
        num_tokens = operation.storage_hit_count
        host_indices = self._controller.mem_pool_host.alloc(num_tokens)
        if host_indices is None:
            return None
        operation.hash_value = operation.hash_value[
            : num_tokens // self._controller.page_size
        ]
        operation.host_indices = host_indices
        transfers = tuple(operation.pool_transfers or ())
        lease = _HostTransientBufferLease(
            num_tokens=num_tokens,
            accounted_tokens=num_tokens,
            pool_names=self._pool_names(list(transfers)),
            host_indices=host_indices,
            pool_transfers=transfers,
        )
        self._controller.prefetch_buffer.put(operation)
        return lease

    def terminate_prefetch(self, operation: Any) -> tuple[int, list[str]]:
        return self._controller.terminate_prefetch(operation)

    def finalize_prefetch(
        self,
        lease: TransientBufferLease,
        *,
        usable_tokens: int,
        completed_tokens: int,
    ) -> TransientBufferLease:
        host_lease = self._host_lease(lease)
        if not 0 <= usable_tokens <= completed_tokens <= host_lease.num_tokens:
            raise ValueError(
                "Invalid Host transient prefetch bounds: "
                f"usable={usable_tokens}, completed={completed_tokens}, "
                f"allocated={host_lease.num_tokens}."
            )
        self._controller.append_host_mem_release(
            host_lease.host_indices[usable_tokens:completed_tokens]
        )
        return _HostTransientBufferLease(
            num_tokens=usable_tokens,
            accounted_tokens=host_lease.accounted_tokens,
            pool_names=host_lease.pool_names,
            host_indices=host_lease.host_indices[:usable_tokens],
            pool_transfers=host_lease.pool_transfers,
        )

    def discard_prefetch(
        self, lease: TransientBufferLease, *, completed_tokens: int
    ) -> None:
        host_lease = self._host_lease(lease)
        if not 0 <= completed_tokens <= host_lease.num_tokens:
            raise ValueError(
                "Invalid Host transient completed prefix: "
                f"completed={completed_tokens}, allocated={host_lease.num_tokens}."
            )
        self._controller.append_host_mem_release(
            host_indices=host_lease.host_indices[:completed_tokens],
            extra_pools=list(host_lease.pool_transfers),
        )

    def release_unstarted_prefetch(
        self, pool_transfers: Optional[list[PoolTransfer]]
    ) -> None:
        self._controller.append_host_mem_release(extra_pools=pool_transfers)

    def restore(
        self, lease: TransientBufferLease, *, operation_id: int
    ) -> Optional[TransientRestore]:
        host_lease = self._host_lease(lease)
        pool_transfers = list(host_lease.pool_transfers)
        device_indices = self._controller.load(
            host_indices=host_lease.host_indices,
            node_id=operation_id,
            extra_pools=pool_transfers or None,
        )
        if device_indices is None:
            return None
        return TransientRestore(
            device_indices=device_indices,
            pool_device_indices={
                transfer.name: transfer.device_indices
                for transfer in pool_transfers
                if transfer.device_indices is not None
                and transfer.device_indices.numel() > 0
            },
        )

    def prefetch_swa_tokens_to_allocate(self, lease: TransientBufferLease) -> int:
        host_lease = self._host_lease(lease)
        return sum(
            int(transfer.host_indices.numel())
            for transfer in host_lease.pool_transfers
            if transfer.name == PoolName.SWA and transfer.host_indices is not None
        )

    def release(self, lease: TransientBufferLease) -> None:
        host_lease = self._host_lease(lease)
        controller = self._controller
        if host_lease.host_indices.numel() > 0:
            controller.mem_pool_host.free(host_lease.host_indices)
        for transfer in host_lease.pool_transfers:
            if (
                transfer.host_indices is None
                or transfer.host_indices.numel() == 0
                or transfer.indices_from_pool is not None
            ):
                continue
            entry = controller.mem_pool_host.entry_map.get(transfer.name)
            if entry is not None:
                entry.host_pool.free(transfer.host_indices)
