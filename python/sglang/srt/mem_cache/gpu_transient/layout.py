from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool


@dataclass(frozen=True)
class GpuPayloadObject:
    pool_name: PoolName
    codec: Literal["token_rows", "opaque_pages"]
    layer_ptrs: torch.Tensor
    local_layers: int
    row_bytes: int
    index_page_size: int
    page_payload_bytes: int
    ring_page_offset: int
    object_suffix_ordinal: int
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    indices_from_pool: PoolName = PoolName.KV


@dataclass(frozen=True)
class GpuPayloadLayout:
    page_size: int
    objects: tuple[GpuPayloadObject, ...]
    combined_page_bytes: int
    compatibility_id: str

    @property
    def pool_names(self) -> tuple[PoolName, ...]:
        return tuple(dict.fromkeys(obj.pool_name for obj in self.objects))

    def objects_for_pool(self, pool_name: PoolName) -> tuple[GpuPayloadObject, ...]:
        return tuple(obj for obj in self.objects if obj.pool_name == pool_name)


def _row_bytes(buffer: torch.Tensor) -> int:
    if buffer.ndim < 2 or not buffer.is_contiguous():
        raise ValueError(
            "GPU-transient token_rows requires contiguous [token, ...] buffers."
        )
    return int(buffer[0].nbytes)


def _validate_layer_ptrs(
    layer_ptrs: torch.Tensor, buffers: list[torch.Tensor], *, label: str
) -> None:
    if not layer_ptrs.is_contiguous():
        raise ValueError(f"GPU-transient {label} layer pointers must be contiguous.")
    if layer_ptrs.dtype not in (torch.int64, torch.uint64):
        raise ValueError(
            f"GPU-transient {label} layer pointers must use 64-bit addresses."
        )
    if layer_ptrs.numel() != len(buffers) or not buffers:
        raise ValueError(
            f"GPU-transient {label} layer pointer/buffer count mismatch: "
            f"pointers={layer_ptrs.numel()}, buffers={len(buffers)}."
        )
    if any(buffer.device != buffers[0].device for buffer in buffers) or (
        layer_ptrs.device != buffers[0].device
    ):
        raise ValueError(
            f"GPU-transient {label} buffers and layer pointers must share a device."
        )


def _finish_layout(page_size: int, objects: list[GpuPayloadObject]) -> GpuPayloadLayout:
    offset = sum(obj.page_payload_bytes for obj in objects)
    identity = {
        "page_size": page_size,
        "objects": [
            {
                "pool": obj.pool_name.value,
                "codec": obj.codec,
                "layers": obj.local_layers,
                "row_bytes": obj.row_bytes,
                "index_page_size": obj.index_page_size,
                "page_bytes": obj.page_payload_bytes,
                "ordinal": obj.object_suffix_ordinal,
                "hit_policy": obj.hit_policy.value,
                "indices_from_pool": obj.indices_from_pool.value,
            }
            for obj in objects
        ],
    }
    compatibility_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return GpuPayloadLayout(
        page_size=page_size,
        objects=tuple(objects),
        combined_page_bytes=offset,
        compatibility_id=compatibility_id,
    )


def _build_primary_kv_payload_layout(kv_pool, page_size: int) -> GpuPayloadLayout:
    """Build the exact page_first MoonCake object layout for plain MLA/MHA KV."""
    if page_size != 256:
        raise ValueError(
            f"GPU-transient payload layout requires page_size=256, got {page_size}."
        )
    if getattr(kv_pool, "layer_shard_enabled", False):
        raise ValueError(
            "GPU-transient payloads do not yet support layer-sharded KV pools."
        )

    object_specs: list[tuple[torch.Tensor, torch.Tensor]]
    if type(kv_pool) is MLATokenToKVPool:
        if not kv_pool.kv_buffer:
            raise ValueError("GPU-transient MLA payload has no local KV layers.")
        _validate_layer_ptrs(kv_pool.data_ptrs, kv_pool.kv_buffer, label="MLA")
        object_specs = [(kv_pool.data_ptrs, kv_pool.kv_buffer[0])]
    elif type(kv_pool) is MHATokenToKVPool:
        if getattr(kv_pool, "is_quantized_kv_cache", False):
            raise ValueError(
                "GPU-transient MHA does not yet support quantized KV payloads."
            )
        if kv_pool.kv_cache_layout != "nhd":
            raise ValueError(
                "GPU-transient MHA requires the plain NHD token-row layout, got "
                f"{kv_pool.kv_cache_layout!r}."
            )
        if not kv_pool.k_buffer or not kv_pool.v_buffer:
            raise ValueError("GPU-transient MHA payload has no local KV layers.")
        _validate_layer_ptrs(kv_pool.k_data_ptrs, kv_pool.k_buffer, label="MHA K")
        _validate_layer_ptrs(kv_pool.v_data_ptrs, kv_pool.v_buffer, label="MHA V")
        if _row_bytes(kv_pool.k_buffer[0]) != _row_bytes(kv_pool.v_buffer[0]):
            raise ValueError(
                "GPU-transient MHA currently requires equal K/V row sizes to "
                "preserve the existing page_first MoonCake ABI."
            )
        object_specs = [
            (kv_pool.k_data_ptrs, kv_pool.k_buffer[0]),
            (kv_pool.v_data_ptrs, kv_pool.v_buffer[0]),
        ]
    else:
        raise ValueError(
            "GPU-transient HiCache currently supports only plain "
            f"MLATokenToKVPool/MHATokenToKVPool, got {type(kv_pool).__name__}."
        )

    objects = []
    offset = 0
    for ordinal, (layer_ptrs, representative) in enumerate(object_specs):
        local_layers = int(layer_ptrs.numel())
        row_bytes = _row_bytes(representative)
        page_payload_bytes = page_size * local_layers * row_bytes
        objects.append(
            GpuPayloadObject(
                pool_name=PoolName.KV,
                codec="token_rows",
                layer_ptrs=layer_ptrs,
                local_layers=local_layers,
                row_bytes=row_bytes,
                index_page_size=page_size,
                page_payload_bytes=page_payload_bytes,
                ring_page_offset=offset,
                object_suffix_ordinal=ordinal,
            )
        )
        offset += page_payload_bytes

    return _finish_layout(page_size, objects)


def _opaque_page_object(
    *,
    pool_name: PoolName,
    buffers: list[torch.Tensor],
    index_page_size: int,
    ring_page_offset: int,
    hit_policy: PoolHitPolicy,
    indices_from_pool: PoolName,
) -> GpuPayloadObject:
    if not buffers:
        raise ValueError(f"GPU-transient {pool_name} payload has no local layers.")
    if any(buffer.ndim != 2 or not buffer.is_contiguous() for buffer in buffers):
        raise ValueError(
            f"GPU-transient {pool_name} requires contiguous [page, bytes] buffers."
        )
    row_bytes = int(buffers[0][0].nbytes)
    if any(int(buffer[0].nbytes) != row_bytes for buffer in buffers):
        raise ValueError(
            f"GPU-transient {pool_name} device page rows must have equal sizes."
        )
    layer_ptrs = torch.tensor(
        [buffer.data_ptr() for buffer in buffers],
        dtype=torch.uint64,
        device=buffers[0].device,
    )
    _validate_layer_ptrs(layer_ptrs, buffers, label=str(pool_name))
    return GpuPayloadObject(
        pool_name=pool_name,
        codec="opaque_pages",
        layer_ptrs=layer_ptrs,
        local_layers=len(buffers),
        row_bytes=row_bytes,
        index_page_size=index_page_size,
        page_payload_bytes=len(buffers) * row_bytes,
        ring_page_offset=ring_page_offset,
        object_suffix_ordinal=0,
        hit_policy=hit_policy,
        indices_from_pool=indices_from_pool,
    )


def _state_page_views(state_pools: list, pool_name: PoolName) -> list[torch.Tensor]:
    views = []
    expected_page_bytes = None
    for pool in state_pools:
        if pool is None:
            raise ValueError(f"GPU-transient {pool_name} state pool is missing.")
        state = pool.kv_score_buffer.kv_score
        if not state.is_contiguous():
            raise ValueError(
                f"GPU-transient {pool_name} state tensor is not contiguous."
            )
        page_bytes = int(pool.ring_size * state[0].nbytes)
        if expected_page_bytes is None:
            expected_page_bytes = page_bytes
        elif page_bytes != expected_page_bytes:
            raise ValueError(
                f"GPU-transient {pool_name} state page sizes do not match."
            )
        state_bytes = state.view(torch.uint8).reshape(state.shape[0], -1)
        usable_slots = (state.shape[0] // pool.ring_size) * pool.ring_size
        views.append(state_bytes[:usable_slots].reshape(-1, page_bytes))
    return views


def _build_deepseek_v4_payload_layout(kv_pool, page_size: int) -> GpuPayloadLayout:
    if getattr(kv_pool, "_unified_kv", False):
        raise ValueError(
            "GPU-transient DeepSeek V4 does not support the unified_kv layout."
        )
    if page_size != 256:
        raise ValueError(
            f"GPU-transient DeepSeek V4 requires page_size=256, got {page_size}."
        )
    stage_items = kv_pool.layer_mapping[kv_pool.start_layer : kv_pool.end_layer]
    c4_global_layers = [
        kv_pool.start_layer + i
        for i, item in enumerate(stage_items)
        if item.compress_ratio == 4
    ]

    specs: list[tuple[PoolName, list[torch.Tensor], int, PoolHitPolicy, PoolName]] = [
        (
            PoolName.SWA,
            list(kv_pool.swa_kv_pool.kv_buffer),
            kv_pool.swa_page_size,
            PoolHitPolicy.TRAILING_PAGES,
            PoolName.SWA,
        ),
        (
            PoolName.DEEPSEEK_V4_C4,
            list(kv_pool.c4_kv_pool.kv_buffer),
            page_size,
            PoolHitPolicy.ALL_PAGES,
            PoolName.KV,
        ),
        (
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            list(kv_pool.c4_indexer_kv_pool.index_k_with_scale_buffer),
            page_size,
            PoolHitPolicy.ALL_PAGES,
            PoolName.KV,
        ),
        (
            PoolName.DEEPSEEK_V4_C4_STATE,
            _state_page_views(
                [kv_pool.compress_state_pools[i] for i in c4_global_layers],
                PoolName.DEEPSEEK_V4_C4_STATE,
            ),
            kv_pool.swa_page_size,
            PoolHitPolicy.TRAILING_PAGES,
            PoolName.SWA,
        ),
        (
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            _state_page_views(
                [kv_pool.indexer_compress_state_pools[i] for i in c4_global_layers],
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            ),
            kv_pool.swa_page_size,
            PoolHitPolicy.TRAILING_PAGES,
            PoolName.SWA,
        ),
        (
            PoolName.DEEPSEEK_V4_C128,
            list(kv_pool.c128_kv_pool.kv_buffer),
            page_size,
            PoolHitPolicy.ALL_PAGES,
            PoolName.KV,
        ),
    ]
    objects = []
    offset = 0
    for pool_name, buffers, index_page_size, hit_policy, indices_from_pool in specs:
        if not buffers:
            continue
        obj = _opaque_page_object(
            pool_name=pool_name,
            buffers=buffers,
            index_page_size=index_page_size,
            ring_page_offset=offset,
            hit_policy=hit_policy,
            indices_from_pool=indices_from_pool,
        )
        objects.append(obj)
        offset += obj.page_payload_bytes
    return _finish_layout(page_size, objects)


def build_gpu_payload_layout(kv_pool, page_size: int) -> GpuPayloadLayout:
    """Build the registered-ring ABI for the active device KV stack."""
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    if isinstance(kv_pool, DeepSeekV4TokenToKVPool):
        return _build_deepseek_v4_payload_layout(kv_pool, page_size)
    return _build_primary_kv_payload_layout(kv_pool, page_size)


def build_primary_kv_payload_layout(kv_pool, page_size: int) -> GpuPayloadLayout:
    """Compatibility wrapper retained for flat Stage-2 callers/tests."""
    return _build_primary_kv_payload_layout(kv_pool, page_size)
