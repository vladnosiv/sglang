# RFC: Device-Selectable Transient Buffers for HiCache

- **Status:** Draft
- **Base:** `hicache-gpu-buffer-mode` at `182b3a583dda99a6507635382a883f441384b530`
- **Initial storage backend:** MoonCake on CUDA
- **Compatibility default:** Host transient buffers

## Summary

HiCache `buffer_only` currently uses Host memory as an operation-owned payload
buffer between the GPU KV cache and L3 storage. It does not retain that memory
as a cache tier, but the payload allocation is still proportional to active
backups and storage hits.

The proposed implementation makes this transient buffer device-selectable:

```text
                         +-----------------------+
UnifiedRadixCache -----> | BufferModePipeline    |
                         | ordering / visibility |
                         +-----------+-----------+
                                     |
                         TransientBufferBackend
                              /             \
                    Host transient       GPU transient
                  HostPoolGroup + L2    fixed TX/RX rings
                              \             /
                              L3 storage ABI
```

The Host implementation preserves the existing path and remains the default.
The GPU implementation will use small, fixed, registered TX/RX rings so KV
payload bytes do not pass through Host RAM. This is primarily a memory-topology
change, not a requirement to beat the Host path in isolated I/O latency.

## Refactor PR Boundary

The first PR introduces no GPU behavior and no new user-facing flag. It:

- adds `TransientBufferBackend` and opaque `TransientBufferLease` objects;
- keeps the neutral contract in `transient_buffer.py` and the concrete Host
  implementation in `host_transient_buffer.py`, leaving a symmetric module
  boundary for the future GPU implementation;
- moves Host allocation, D2H/H2D copies, storage submission, pressure gates,
  partial-prefetch trimming, and terminal release into
  `HostTransientBufferBackend`;
- makes `BufferModePipeline` track leases instead of raw `host_indices` and
  auxiliary Host transfers;
- stores an optional transient lease in in-flight prefetch state, so the
  scheduler no longer allocates the primary Host bounce directly;
- keeps tree ownership, request admission, TP synchronization, cache
  visibility, keys, and storage object bytes unchanged.

Normal HiCache `cache` mode remains outside this interface and keeps its
persistent Host L2 behavior.

The interface follows the existing state machine rather than introducing a
second scheduler:

```python
class TransientBufferBackend(ABC):
    def backup_fits(...) -> bool: ...
    def backup_live_cap(...) -> int: ...
    def backup_blocked(...) -> bool: ...
    def stage_backup(...) -> BufferLease | None: ...
    def submit_storage_write(...) -> int: ...

    def try_start_prefetch(...) -> BufferLease | None: ...
    def terminate_prefetch(...) -> tuple[int, list[str]]: ...
    def finalize_prefetch(...) -> BufferLease: ...
    def discard_prefetch(...) -> None: ...

    def restore(...) -> TransientRestore | None: ...
    def release(...) -> None: ...
```

`BufferLease` is intentionally opaque to the pipeline. It exposes only logical
token accounting and pool membership; the concrete Host slots or GPU ring slot
remain backend-private.

## GPU Backend Follow-up

The next PR adds a disabled-by-default `GpuTransientBufferBackend` with:

- one fixed registered TX ring and one fixed registered RX ring per rank;
- bounded waves, initially 64 pages with depth two per direction;
- one active logical operation per rank initially;
- GPU pack/unpack streams and slot generations;
- MoonCake batch PUT/GET from registered device-region slices;
- private destination GPU pages and one final scheduler-thread publication.

Configuration is expected to be:

```text
--hicache-storage-io-mode host|gpu_transient     # default: host
--hicache-gpu-transient-wave-pages 64
--hicache-gpu-transient-ring-depth 2
--hicache-gpu-transient-max-active-ops 1
```

For wave size `Q`, TX/RX depths `D_tx` and `D_rx`, and serialized bytes per
logical page `B_page`, fixed transient GPU memory is:

```text
ring_bytes = (D_tx + D_rx) * Q * B_page
```

It is independent of the total cache-hit length.

## Runtime Payload Plan

GPU mode needs one immutable payload plan per rank, built from active device
pools and storage sidecars. For every physical stored object it defines:

- logical pool and source index domain;
- object order, key suffix order, and hit policy;
- codec (`token_rows` or `opaque_page` initially);
- bytes per page, physical strides, alignment, and ring offset.

The existing Host serializer defines the storage ABI. Host and GPU modes must
produce identical keys, object ordering, sizes, and bytes. DeepSeek V4 must be
handled as a runtime collection of physical sidecars, not by model name.

The first GPU slice should enable only layouts for which every required object
has an exact codec. Multi-pool component preparation that currently allocates
Host buffers must move to this runtime plan before such a layout can be enabled
in GPU mode.

## Correctness Contracts

1. **Required-object closure:** a logical page is usable only when every
   required physical object succeeded.
2. **Visibility:** restored pages are not visible to a request, radix tree, or
   attention until all GETs and unpack kernels complete.
3. **Lifetime:** source pages, destination pages, and ring slots cannot be
   freed or reused while submitted transfer or kernel work can access them.
4. **Scheduler ownership:** tree mutation and publication occur only on the
   scheduler thread; rank-dependent outcomes are synchronized before they
   affect visible state.
5. **Failure degradation:** allocation, transfer, timeout, cancellation, or
   stale-generation failure publishes nothing and becomes a cache miss.
6. **Storage ABI:** Host-written objects are GPU-readable and GPU-written
   objects are Host-readable.

MoonCake completion status remains authoritative. The proposal does not add a
second production checksum layer.

## Initial Scope

Enable `gpu_transient` only for CUDA, MoonCake, `UnifiedRadixCache`, complete
pages, `page_size == 256`, rank-local payloads, and a fully validated runtime
payload plan. Reject unsupported configurations at startup.

Initially reject DCP, cross-rank payload gathering, Mamba, unified-KV SWA
rings, partial pages, unknown opaque layouts, and any required physical pool
without a codec.

## Validation

The refactor PR must preserve the existing Host roundtrips, SWA trailing-window
semantics, partial-hit trimming, abort cleanup, occupancy accounting, and
storage object ABI.

The GPU follow-up additionally requires:

- Host/GPU byte-identity and bidirectional interoperability tests;
- source and destination page IDs that intentionally differ;
- no publication when any required object or later wave fails;
- timeout/cancellation tests proving stale work cannot touch reused memory;
- repeated large restore/failure/eviction cycles with no leaked lease or page;
- proof that GPU mode performs no Host payload allocation proportional to L1
  size or cache-hit length;
- serving measurements for Host bytes, fixed ring bytes, shared-L3 hit rate,
  TTFT, ITL/TPOT, throughput, phase time, credit waits, and terminal failures.

The production decision should be based on eliminated Host memory, increased
shared-L3 useful capacity and hit rate, and acceptable serving interference.
