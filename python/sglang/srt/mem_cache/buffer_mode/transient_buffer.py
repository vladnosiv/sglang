"""Memory-location-neutral transient-buffer contracts for HiCache buffer-only mode.

Concrete Host and GPU backends live in sibling modules. The scheduler pipeline
depends only on the contracts defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer


@dataclass(frozen=True)
class TransientBufferLease:
    """Backend-owned payload memory held by one logical buffer-mode operation.

    ``num_tokens`` is the usable anchor-pool span. ``accounted_tokens`` is the
    scheduler occupancy charged to the operation; it can be larger when a
    storage fetch was allocated before its TP-synchronized usable prefix was
    known. The concrete memory representation remains backend-private.
    """

    num_tokens: int
    accounted_tokens: int
    pool_names: tuple[PoolName, ...]


@dataclass(frozen=True)
class TransientRestore:
    """Device allocation produced when a transient buffer is materialized."""

    device_indices: torch.Tensor
    pool_device_indices: dict[PoolName, torch.Tensor]
    requires_ack: bool = True


class TransientBufferBackend(ABC):
    """Memory-location-neutral operations required by ``BufferModePipeline``."""

    @abstractmethod
    def backup_fits(
        self, primary_tokens: int, pool_transfers: Optional[list[PoolTransfer]]
    ) -> bool:
        """Whether this logical backup can ever fit in the backend."""

    @abstractmethod
    def backup_live_cap(self) -> int:
        """Current anchor-pool budget available to staged writes."""

    @abstractmethod
    def backup_blocked(self, pool_transfers: Optional[list[PoolTransfer]]) -> bool:
        """Whether transient pressure should defer this backup now."""

    @abstractmethod
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
        """Allocate a transient buffer and launch device-to-buffer staging."""

    @abstractmethod
    def submit_storage_write(
        self,
        lease: TransientBufferLease,
        *,
        token_ids: Sequence[int],
        hash_values: list[str],
        prefix_keys: Optional[list[str]],
    ) -> int:
        """Submit storage writes backed by ``lease`` and return the operation id."""

    @abstractmethod
    def try_start_prefetch(self, operation: Any) -> Optional[TransientBufferLease]:
        """Acquire a receive buffer and submit a queried storage hit."""

    @abstractmethod
    def terminate_prefetch(self, operation: Any) -> tuple[int, list[str]]:
        """Stop a submitted prefetch and return its completed logical prefix."""

    @abstractmethod
    def finalize_prefetch(
        self,
        lease: TransientBufferLease,
        *,
        usable_tokens: int,
        completed_tokens: int,
    ) -> TransientBufferLease:
        """Trim a completed fetch to the rank-synchronized usable prefix."""

    @abstractmethod
    def discard_prefetch(
        self, lease: TransientBufferLease, *, completed_tokens: int
    ) -> None:
        """Release the completed portion of an unusable or aborted fetch."""

    @abstractmethod
    def release_unstarted_prefetch(
        self, pool_transfers: Optional[list[PoolTransfer]]
    ) -> None:
        """Release resources prepared before a storage hit was submitted."""

    @abstractmethod
    def restore(
        self, lease: TransientBufferLease, *, operation_id: int
    ) -> Optional[TransientRestore]:
        """Allocate destination pages and launch buffer-to-device restore."""

    @abstractmethod
    def release(self, lease: TransientBufferLease) -> None:
        """Return a terminal lease to the backend."""

    def commit_restore(self, lease: TransientBufferLease) -> None:
        """Transfer restored device-page ownership to the radix tree."""

    def restore_needs_device_allocation(self, lease: TransientBufferLease) -> bool:
        """Whether admission must make room before ``restore()`` allocates."""
        return True

    def prefetch_device_tokens_reserved(self, lease: TransientBufferLease) -> int:
        """FULL tokens already removed from device availability by this lease."""
        return 0

    def prefetch_swa_tokens_to_allocate(self, lease: TransientBufferLease) -> int:
        """SWA tokens that consuming this lease will allocate on device."""
        return 0

    def storage_write_succeeded(self, lease: TransientBufferLease) -> Optional[bool]:
        """Return terminal storage status, or None for legacy ack semantics."""
        return None

    def requires_write_drain_before_device_eviction(self) -> bool:
        """Whether queued backups must drain before source pages are evicted."""
        return False

    def wait_for_progress(self) -> None:
        """Wait for the backend's active operation to reach a terminal state."""

    def drain_completions(self) -> None:
        """Run scheduler-thread cleanup queued by asynchronous backends."""

    def reset(self) -> None:
        """Cancel and reclaim backend-private operations during cache reset."""

    def close(self) -> None:
        """Release backend-owned executors and fixed buffers at shutdown."""
        self.reset()
