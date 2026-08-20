from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _copy_token_rows_kernel(
    layer_ptrs,
    page_starts,
    ring,
    q_pages,
    PAGE_SIZE: tl.constexpr,
    LOCAL_LAYERS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    RING_PAGE_STRIDE: tl.constexpr,
    RING_OBJECT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
    PACK: tl.constexpr,
):
    page = tl.program_id(0)
    layer = tl.program_id(1)
    slab = tl.program_id(2)

    off = slab * BLOCK + tl.arange(0, BLOCK)
    layer_bytes = PAGE_SIZE * ROW_BYTES
    mask = (page < q_pages) & (off < layer_bytes)
    token = off // ROW_BYTES
    inner = off - token * ROW_BYTES

    layer_base = tl.load(layer_ptrs + layer).to(tl.pointer_type(tl.uint8))
    slot0 = tl.load(page_starts + page)
    pool_ptr = layer_base + (slot0 + token) * ROW_BYTES + inner
    # Existing Host page_first ABI is [token, local_layer, row_bytes].
    ring_ptr = (
        ring
        + page * RING_PAGE_STRIDE
        + RING_OBJECT_OFFSET
        + (token * LOCAL_LAYERS + layer) * ROW_BYTES
        + inner
    )

    if PACK:
        value = tl.load(pool_ptr, mask=mask, other=0)
        tl.store(ring_ptr, value, mask=mask)
    else:
        value = tl.load(ring_ptr, mask=mask, other=0)
        tl.store(pool_ptr, value, mask=mask)


def copy_token_rows(
    layer_ptrs,
    page_starts,
    ring_slot,
    q_pages: int,
    ring_page_stride: int,
    *,
    page_payload_bytes: int,
    local_layers: int,
    row_bytes: int,
    ring_page_offset: int,
    pack: bool,
) -> None:
    block = 16_384
    bytes_per_layer_page = page_payload_bytes // local_layers
    grid = (
        q_pages,
        local_layers,
        triton.cdiv(bytes_per_layer_page, block),
    )
    _copy_token_rows_kernel[grid](
        layer_ptrs,
        page_starts,
        ring_slot,
        q_pages,
        PAGE_SIZE=bytes_per_layer_page // row_bytes,
        LOCAL_LAYERS=local_layers,
        ROW_BYTES=row_bytes,
        RING_PAGE_STRIDE=ring_page_stride,
        RING_OBJECT_OFFSET=ring_page_offset,
        BLOCK=block,
        PACK=pack,
        num_warps=8,
    )


@triton.jit
def _copy_opaque_pages_kernel(
    layer_ptrs,
    page_starts,
    ring,
    q_pages,
    LOCAL_LAYERS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    INDEX_PAGE_SIZE: tl.constexpr,
    RING_PAGE_STRIDE: tl.constexpr,
    RING_OBJECT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
    PACK: tl.constexpr,
):
    page = tl.program_id(0)
    layer = tl.program_id(1)
    slab = tl.program_id(2)

    inner = slab * BLOCK + tl.arange(0, BLOCK)
    mask = (page < q_pages) & (inner < ROW_BYTES)
    layer_base = tl.load(layer_ptrs + layer).to(tl.pointer_type(tl.uint8))
    slot0 = tl.load(page_starts + page)
    row = slot0 // INDEX_PAGE_SIZE
    pool_ptr = layer_base + row * ROW_BYTES + inner
    # Existing DSV4 Host page_first ABI is [local_layer, opaque_page_bytes].
    ring_ptr = (
        ring + page * RING_PAGE_STRIDE + RING_OBJECT_OFFSET + layer * ROW_BYTES + inner
    )
    if PACK:
        value = tl.load(pool_ptr, mask=mask, other=0)
        tl.store(ring_ptr, value, mask=mask)
    else:
        value = tl.load(ring_ptr, mask=mask, other=0)
        tl.store(pool_ptr, value, mask=mask)


def copy_opaque_pages(
    layer_ptrs,
    page_starts,
    ring_slot,
    q_pages: int,
    ring_page_stride: int,
    *,
    local_layers: int,
    row_bytes: int,
    index_page_size: int,
    ring_page_offset: int,
    pack: bool,
) -> None:
    block = 16_384
    grid = (q_pages, local_layers, triton.cdiv(row_bytes, block))
    _copy_opaque_pages_kernel[grid](
        layer_ptrs,
        page_starts,
        ring_slot,
        q_pages,
        LOCAL_LAYERS=local_layers,
        ROW_BYTES=row_bytes,
        INDEX_PAGE_SIZE=index_page_size,
        RING_PAGE_STRIDE=ring_page_stride,
        RING_OBJECT_OFFSET=ring_page_offset,
        BLOCK=block,
        PACK=pack,
        num_warps=8,
    )
