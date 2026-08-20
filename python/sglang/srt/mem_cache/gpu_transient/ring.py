from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.gpu_transient.layout import GpuPayloadLayout
from sglang.srt.mem_cache.hicache_storage import DeviceRegion, PoolName


@dataclass(frozen=True)
class RingLease:
    slot: int
    generation: int


class RegisteredGpuRing:
    """Fixed registered ring with generation-checked slot credits."""

    def __init__(
        self,
        *,
        direction: str,
        depth: int,
        wave_pages: int,
        layout: GpuPayloadLayout,
        device: torch.device,
    ) -> None:
        self.direction = direction
        self.depth = depth
        self.wave_pages = wave_pages
        self.layout = layout
        self.tensor = torch.empty(
            (depth, wave_pages, layout.combined_page_bytes),
            dtype=torch.uint8,
            device=device,
        )
        self._credits: queue.Queue[int] = queue.Queue(maxsize=depth)
        self._generations = [0] * depth
        self._in_use = [False] * depth
        self._lock = threading.Lock()
        for slot in range(depth):
            self._credits.put_nowait(slot)

    @property
    def nbytes(self) -> int:
        return self.tensor.nbytes

    def acquire(self) -> RingLease:
        slot = self._credits.get()
        with self._lock:
            self._generations[slot] += 1
            self._in_use[slot] = True
            generation = self._generations[slot]
        return RingLease(slot=slot, generation=generation)

    def _validate_lease(self, lease: RingLease) -> None:
        if not 0 <= lease.slot < self.depth:
            raise RuntimeError(
                f"Invalid {self.direction} ring slot {lease.slot}; depth={self.depth}."
            )
        if self._generations[lease.slot] != lease.generation:
            raise RuntimeError(
                f"Stale {self.direction} ring lease: slot={lease.slot}, "
                f"generation={lease.generation}, "
                f"current={self._generations[lease.slot]}."
            )
        if not self._in_use[lease.slot]:
            raise RuntimeError(
                f"Released {self.direction} ring lease reused: slot={lease.slot}, "
                f"generation={lease.generation}."
            )

    def release(self, lease: RingLease) -> None:
        with self._lock:
            self._validate_lease(lease)
            self._in_use[lease.slot] = False
        self._credits.put_nowait(lease.slot)

    def slot_view(self, lease: RingLease) -> torch.Tensor:
        with self._lock:
            self._validate_lease(lease)
        return self.tensor[lease.slot]

    def regions(
        self, lease: RingLease, q_pages: int
    ) -> dict[PoolName, list[DeviceRegion]]:
        if not 0 < q_pages <= self.wave_pages:
            raise ValueError(
                f"Invalid {self.direction} wave size {q_pages}; "
                f"capacity={self.wave_pages}."
            )
        base = int(self.slot_view(lease).data_ptr())
        regions: dict[PoolName, list[DeviceRegion]] = {
            pool_name: [] for pool_name in self.layout.pool_names
        }
        # MoonCake key order is page-major, then object suffix within a pool.
        for page in range(q_pages):
            page_base = base + page * self.layout.combined_page_bytes
            for obj in self.layout.objects:
                regions[obj.pool_name].append(
                    DeviceRegion(
                        ptr=page_base + obj.ring_page_offset,
                        size=obj.page_payload_bytes,
                    )
                )
        return regions
