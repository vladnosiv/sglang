"""Buffer-only mode transfer pipelines for the unified radix cache.

``BufferModePipeline`` owns all buffer-mode state and the two pipelines that
move KV through an operation-owned transient buffer:

- backup (write path): admission-gated FIFO intents, head-of-line staging
  launches, storage writes at the staging ack, buffers freed at the
  storage ack;
- load back (read path): completed storage fetches parked as op-owned buffers,
  consumed at prefill admission via a device alloc + backend restore + plain
  tree insert, buffer freed at the restore ack.

The pipeline is an intimate collaborator of ``UnifiedRadixCache``: it is
constructed by ``init_hicache`` only when ``--hicache-host-memory-mode
buffer_only`` is active, and it drives tree/controller operations (insert,
match, evict, lock refs, cache actions) through the owning cache. All
buffer-mode-only state lives here; the cache dispatches to this object at
its mode branches.

TP-lockstep contract: every mutation runs on the scheduler thread at
rank-synchronized points (insert walks, rank-MIN-reduced drains, ack
drains), so per-rank state never diverges. There is no runtime
verification; a violation surfaces as an unexplained collective hang.
"""

from __future__ import annotations

import logging
from array import array
from collections import deque
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.buffer_mode.transient_buffer import (
    TransientBufferBackend,
    TransientBufferLease,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import RebuildFullToSWAMapping
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import NodeId, UnifiedTreeNode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class _UnifiedBackupIntent(msgspec.Struct):
    """Buffer-mode backup intent, unpinned while queued.

    Snapshots node identity at enqueue time: a split rewrites the node's
    key/hash in place while these copies stay intact, so
    ``node.hash_value != hash_values`` doubles as split detection and a None
    FULL device value as eviction detection (``_backup_intent_stale``).
    """

    node: UnifiedTreeNode
    node_id: int
    hash_values: list[str]
    key: RadixKey
    prefix_keys: Optional[list[str]] = None


class _UnifiedBufferBackupEntry(msgspec.Struct):
    """A buffer-mode backup after staging: intent + transient lease."""

    intent: _UnifiedBackupIntent
    buffer: TransientBufferLease
    lock_params: DecLockRefParams


class _StagedPrefetch(msgspec.Struct):
    """Completed fetch parked until prefill admission as an opaque lease."""

    req_id: str
    key_tokens: list[int]
    extra_key: Optional[str]
    matched_len: int
    num_tokens: int
    buffer: TransientBufferLease
    hash_values: list[str]
    operation_id: int


class _OngoingBufferLoadBack(msgspec.Struct):
    """A restore awaiting its ack after the span became tree-resident."""

    req_id: str
    num_tokens: int
    buffer: TransientBufferLease
    hash_values: list[str]


def _track_content_refs(refs: dict[str, int], hash_values: list[str]) -> None:
    """Add one content ref per page hash (at staging launch). Refcounted,
    not a flag: several launched entries can carry the same content
    (duplicate staging of republished spans)."""
    for h in hash_values:
        refs[h] = refs.get(h, 0) + 1


def _untrack_content_refs(refs: dict[str, int], hash_values: list[str]) -> None:
    """Drop one content ref per page hash (at storage-ack)."""
    for h in hash_values:
        n = refs.get(h, 0) - 1
        if n <= 0:
            refs.pop(h, None)
        else:
            refs[h] = n


class BufferModePipeline:
    """All buffer-mode state plus the backup and load-back pipelines.

    Constructed by ``UnifiedRadixCache.init_hicache`` when host memory mode
    is ``buffer_only``; ``cache.buffer_pipeline is None`` elsewhere, which
    the cache's mode branches use as the dispatch test.
    """

    def __init__(
        self,
        cache: UnifiedRadixCache,
        transient_buffer: TransientBufferBackend,
        write_backlog_cap: int,
    ):
        self._cache = cache
        self.transient_buffer = transient_buffer
        # Metadata-only pending-write backlog cap; beyond it new intents
        # are dropped at admission (re-trigger on a later hit).
        self.write_backlog_cap = write_backlog_cap
        self.reset()

    def reset(self) -> None:
        # Load pipeline: hits awaiting a staging grant (park-and-retry),
        # enqueue-time prefix context, completed prefetches staged until
        # prefill admission, and load-backs in flight (keyed by synthetic
        # negative ack id).
        self.pending_hit_allocs: deque = deque()
        self._prefetch_prefix_ctx: dict[str, list[int]] = {}
        self.staged_prefetches: dict[str, _StagedPrefetch] = {}
        self.ongoing_buffer_load_back: dict[int, _OngoingBufferLoadBack] = {}
        # Backup pipeline: FIFO intents awaiting a transient slot, node ids
        # anywhere in flight (dedupes re-triggers), and a content refcount
        # of every page hash between staging launch and storage-ack — admission
        # skips content covered by beliefs + launched writes.
        self.pending_write_queue: deque[_UnifiedBackupIntent] = deque()
        self.inflight_backup_node_ids: set[int] = set()
        self.inflight_backup_hashes: dict[str, int] = {}
        # Backups between staging launch and staging ack (keyed by node id), then
        # between storage-write launch and storage ack (keyed by operation
        # id). Mirrors the cache-mode ongoing_write_through/ongoing_backup
        # stages, with buffer entries.
        self.ongoing_write_through: dict[int, _UnifiedBufferBackupEntry] = {}
        self.ongoing_backup: dict[int, _UnifiedBufferBackupEntry] = {}
        self.write_staged_tokens_ = 0
        self.write_backlog_tokens_ = 0
        self._backlog_cap_hits = 0

    def is_idle(self) -> bool:
        """No queued writes, staged prefetches, or storage writes in flight
        (all of which hold transient memory or would re-trigger IO)."""
        return not (
            self.pending_write_queue or self.staged_prefetches or self.ongoing_backup
        )

    # ---- backup pipeline (device -> staging -> storage) ----

    def _backup_parent_covered(self, node: UnifiedTreeNode) -> bool:
        """Only admit a node whose parent is stored/in-flight: writing above
        a dropped parent creates a permanent longest-prefix hole."""
        parent = node.parent
        if (
            parent is self._cache.root_node
            or parent.id in self.inflight_backup_node_ids
        ):
            return True
        last_hash = parent.get_last_hash_value()
        return last_hash is not None and self._cache.storage_existence_cache.contains(
            PoolName.KV, last_hash
        )

    def _log_backup_dropped(self, num_tokens: int) -> None:
        cache = self._cache
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_backup_dropped_tokens(num_tokens)

    def enqueue_backup_intent(self, node: UnifiedTreeNode) -> None:
        """Snapshot a backup intent and commit it to the write queue.
        Admission gates: belief skip, parent-cover, backlog cap, oversize.
        Drops are silent; the node re-triggers on a later hit."""
        if not self._cache.enable_storage or not node.hash_value:
            return
        if node.id in self.inflight_backup_node_ids:
            return
        # Admission cover: beliefs plus content past its D2H launch. The
        # launched cover keeps republished content (fill inserts under new
        # node ids) from re-writing while the original write drains.
        if self._cache.storage_existence_cache.covers_all(
            PoolName.KV, node.hash_value, extra_cover=self.inflight_backup_hashes
        ):
            return
        intent_tokens = len(node.hash_value) * self._cache.page_size
        if self.write_backlog_tokens_ >= self.write_backlog_cap:
            # The cap sits at 2x the intrinsic live-backlog ceiling (see
            # init_hicache), so reaching it means leaked accounting or a
            # broken stale sweep — a bug, not load.
            self._backlog_cap_hits += 1
            if self._backlog_cap_hits <= 3 or self._backlog_cap_hits % 1000 == 0:
                logger.error(
                    "HiCache write backlog cap hit (occurrence %d): "
                    "backlog=%d cap=%d queue=%d. Live backlog is bounded "
                    "by the device pool span, so this indicates a "
                    "stale-sweep or accounting leak.",
                    self._backlog_cap_hits,
                    self.write_backlog_tokens_,
                    self.write_backlog_cap,
                    len(self.pending_write_queue),
                )
            self._log_backup_dropped(intent_tokens)
            return
        # A span larger than any pool's whole staging capacity can never
        # stage; admitting it would wedge the head-of-line queue forever.
        if not self._backup_parent_covered(node) or self._backup_oversize(
            node, intent_tokens
        ):
            self._log_backup_dropped(intent_tokens)
            return

        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self._cache.hicache_storage_pass_prefix_keys
            else None
        )
        intent = _UnifiedBackupIntent(
            node=node,
            node_id=node.id,
            hash_values=list(node.hash_value),
            key=node.key,
            prefix_keys=prefix_keys,
        )
        self.pending_write_queue.append(intent)
        self.inflight_backup_node_ids.add(node.id)
        self.write_backlog_tokens_ += intent_tokens

    def _build_aux_staging_transfers(
        self, node: UnifiedTreeNode
    ) -> Optional[list[PoolTransfer]]:
        """Keys-only aux transfers mirroring what BACKUP_STORAGE would write;
        sizes the per-pool oversize gate (beliefs do not consult these)."""
        transfers: list[PoolTransfer] = []
        if ComponentType.SWA in self._cache.components:
            cd = node.component_data[ComponentType.SWA]
            if cd.value is not None:
                num_pages = len(cd.value) // self._cache.page_size
                if num_pages > 0:
                    transfers.append(
                        PoolTransfer(
                            name=PoolName.SWA,
                            keys=node.hash_value[-num_pages:],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES,
                        )
                    )
        return transfers or None

    def _backup_oversize(
        self,
        node: UnifiedTreeNode,
        intent_tokens: int,
        aux_xfers: Optional[list[PoolTransfer]] = None,
    ) -> bool:
        """True if any pool's staging need exceeds that pool's write-usable
        capacity (total for KV, total minus the loads-priority margin for aux
        pools — matching ``_aux_budget_blocked``'s admission ceiling): such an
        intent could never stage and would wedge the FIFO head."""
        if aux_xfers is None:
            aux_xfers = self._build_aux_staging_transfers(node)
        return not self.transient_buffer.backup_fits(intent_tokens, aux_xfers)

    def _backup_intent_stale(self, intent: _UnifiedBackupIntent) -> bool:
        # Arena-lookup failure = deleted, hash mismatch vs the enqueue-time
        # snapshot = split, a None FULL device value = evicted. Stale
        # intents drop silently; the node re-triggers on a later hit.
        node = intent.node
        try:
            self._cache.tree_core.node_by_id(intent.node_id)
        except KeyError:
            return True
        return (
            node.component_data[BASE_COMPONENT_TYPE].value is None
            or node.hash_value != intent.hash_values
        )

    def _sweep_stale_backup_intents(self) -> None:
        """Cancel stale intents anywhere in the queue, not just at the head:
        a dead intent would otherwise inflate the backlog accounting and
        hold FIFO position ahead of live segments."""
        if not self.pending_write_queue:
            return
        page_size = self._cache.page_size
        survivors: deque[_UnifiedBackupIntent] = deque()
        for intent in self.pending_write_queue:
            if self._backup_intent_stale(intent):
                self.inflight_backup_node_ids.discard(intent.node_id)
                self.write_backlog_tokens_ -= len(intent.hash_values) * page_size
                continue
            survivors.append(intent)
        self.pending_write_queue = survivors

    def flush_pending_writes(self) -> None:
        """Launch staging for admitted intents, head-of-line: source locks and
        transient slots are taken only here, when capacity allows."""
        if not self.pending_write_queue:
            return
        self._sweep_stale_backup_intents()
        live_cap = self.transient_buffer.backup_live_cap()
        while self.pending_write_queue:
            intent = self.pending_write_queue[0]
            intent_tokens = len(intent.hash_values) * self._cache.page_size
            if not self._backup_parent_covered(intent.node) or self._backup_oversize(
                intent.node, intent_tokens
            ):
                # Unwritable intent (dropped parent or unstageable size):
                # cascade the drop down the chain rather than creating a
                # permanent storage hole / stalling the head-of-line queue.
                self.pending_write_queue.popleft()
                self.inflight_backup_node_ids.discard(intent.node_id)
                self.write_backlog_tokens_ -= intent_tokens
                self._log_backup_dropped(intent_tokens)
                continue
            if self.write_staged_tokens_ >= live_cap:
                # Yield to live fetch demand; retry next round.
                break
            if self._aux_budget_blocked(intent):
                # An aux pool lacks staging headroom: yield at the gate
                # instead of failing the alloc inside cc.write; acks free
                # aux staging, retry next round.
                break
            if not self._launch_backup_intent(intent):
                # Pool full of in-flight staging and nothing reclaimable
                # (the tree never holds host values in buffer mode):
                # defer, head-of-line; pending acks will free slots.
                break
            self.pending_write_queue.popleft()

    def _launch_backup_intent(self, intent: _UnifiedBackupIntent) -> bool:
        """Launch one admitted intent's backend staging and source lock; the
        caller removes it from pending_write_queue. Returns False when staging
        cannot be allocated. From a successful launch the intent always reaches
        its storage-ack, so its content joins the LAUNCHED admission cover."""
        cache = self._cache
        node = intent.node
        # Build aux transfers from the node's CURRENT state: a SWA span
        # tombstoned since admission backs up FULL-only, as in cache mode.
        device_value, comp_xfers = cache.tree_core.build_backup_spec(node.id)
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        buffer = self.transient_buffer.stage_backup(
            device_value,
            node_id=node.id,
            pool_transfers=aux_xfers or None,
        )
        if buffer is None:
            return False
        _track_content_refs(self.inflight_backup_hashes, intent.hash_values)
        # NOTE: no commit_backup — transient payload must never become
        # tree-owned cache state; the lease lives in the pipeline entry.
        lock_params = cache.inc_lock_ref(node.id).to_dec_params()
        self.ongoing_write_through[node.id] = _UnifiedBufferBackupEntry(
            intent=intent,
            buffer=buffer,
            lock_params=lock_params,
        )
        self.write_staged_tokens_ += buffer.num_tokens
        self.write_backlog_tokens_ -= len(intent.hash_values) * cache.page_size
        return True

    def _aux_budget_blocked(self, intent: _UnifiedBackupIntent) -> bool:
        """True when an aux pool cannot stage this intent right now (free
        minus the loads-priority margin falls short of the need): defer at
        the gate instead of failing inside backend staging and blocking
        pure-KV intents behind an unallocatable head. The margin enforces
        loads-have-priority on aux pools the way live_cap does on the KV
        pool; avail already reflects prefetch-held slots, so no occupancy
        subtraction here."""
        return self.transient_buffer.backup_blocked(
            self._build_aux_staging_transfers(intent.node)
        )

    def finish_backup_ack(self, ack_id: int) -> None:
        """Staging confirmed: unlock source pages and enqueue storage write."""
        entry = self.ongoing_write_through.pop(ack_id)
        intent = entry.intent
        self._cache.dec_lock_ref(intent.node_id, entry.lock_params)

        operation_id = self.transient_buffer.submit_storage_write(
            entry.buffer,
            token_ids=intent.key.token_ids,
            hash_values=intent.hash_values,
            prefix_keys=intent.prefix_keys,
        )
        self.ongoing_backup[operation_id] = entry

    def finish_storage_write_ack(self, operation_id: int) -> None:
        """Storage write acked (rank-synced drain): free the entry's staging
        outright. Existence entries are added unconditionally
        (completed_tokens can diverge across ranks under backend failure) to
        keep admission decisions TP-deterministic. No-op for operations this
        pipeline does not own (e.g. acks for already-reset state)."""
        entry = self.ongoing_backup.pop(operation_id, None)
        if entry is None:
            return
        intent = entry.intent
        self._cache.storage_existence_cache.add(PoolName.KV, intent.hash_values)
        self.transient_buffer.release(entry.buffer)
        self.write_staged_tokens_ -= entry.buffer.num_tokens
        self.inflight_backup_node_ids.discard(entry.intent.node_id)
        _untrack_content_refs(self.inflight_backup_hashes, intent.hash_values)

    # ---- load back pipeline (storage -> staging -> device) ----

    def set_prefix_ctx(self, req_id: str, matched_prefix_tokens) -> None:
        """Record the device-matched prefix at prefetch enqueue; consumed at
        staging commit to build the full-span tree key."""
        self._prefetch_prefix_ctx[req_id] = list(matched_prefix_tokens or [])

    def pop_prefix_ctx(self, req_id: str) -> None:
        self._prefetch_prefix_ctx.pop(req_id, None)

    def has_staged(self, req_id: str) -> bool:
        return req_id in self.staged_prefetches

    def try_start_prefetch(self, operation) -> Optional[TransientBufferLease]:
        """Acquire a backend-owned receive buffer and submit a known hit."""
        return self.transient_buffer.try_start_prefetch(operation)

    def terminate_prefetch(self, operation) -> tuple[int, list[str]]:
        return self.transient_buffer.terminate_prefetch(operation)

    def finalize_prefetch(
        self,
        buffer: TransientBufferLease,
        *,
        usable_tokens: int,
        completed_tokens: int,
    ) -> TransientBufferLease:
        return self.transient_buffer.finalize_prefetch(
            buffer,
            usable_tokens=usable_tokens,
            completed_tokens=completed_tokens,
        )

    def discard_prefetch(
        self, buffer: TransientBufferLease, *, completed_tokens: int
    ) -> None:
        self.transient_buffer.discard_prefetch(
            buffer, completed_tokens=completed_tokens
        )

    def release_unstarted_prefetch(
        self, pool_transfers: Optional[list[PoolTransfer]]
    ) -> None:
        self.transient_buffer.release_unstarted_prefetch(pool_transfers)

    def stage_completed_prefetch(
        self,
        req_id: str,
        num_tokens: int,
        hash_value: list[str],
        buffer: TransientBufferLease,
    ) -> bool:
        """Park the completed fetch as a held buffer; the scheduler surfaces
        it as host_hit_length and the adder consumes it via init_load_back.
        Always returns True (ready is a stable, revisited state)."""
        cache = self._cache
        (
            _anchor,
            prefetch_key,
            _host_indices,
            operation,
            _lock_params,
            _comp_xfers,
            tracked_buffer,
        ) = cache.ongoing_prefetch.pop(req_id)
        cc = cache.cache_controller
        prefix_tokens = self._prefetch_prefix_ctx.pop(req_id, None)
        if tracked_buffer is not buffer:
            raise RuntimeError(
                "Completed prefetch lease no longer matches request state."
            )

        if num_tokens == 0 or prefix_tokens is None:
            # Nothing usable fetched: recompute.
            self.transient_buffer.release(buffer)
            cc.prefetch_tokens_occupied -= buffer.accounted_tokens
            cache.prefetch_loaded_tokens_by_reqid[req_id] = 0
            return True

        staged_pages = num_tokens // cache.page_size
        staged_hashes = hash_value[:staged_pages]
        # Feed existence beliefs from the storage-fetched pages: the fetch
        # itself is the evidence, so feeding is sound even if this staged
        # prefetch is later dropped unconsumed.
        cache.storage_existence_cache.add(PoolName.KV, list(staged_hashes))
        self.staged_prefetches[req_id] = _StagedPrefetch(
            req_id=req_id,
            key_tokens=prefix_tokens + list(prefetch_key[:num_tokens].token_ids),
            extra_key=prefetch_key.extra_key,
            matched_len=len(prefix_tokens),
            num_tokens=num_tokens,
            buffer=buffer,
            hash_values=staged_hashes,
            operation_id=operation.id,
        )
        cache.prefetch_loaded_tokens_by_reqid[req_id] = num_tokens
        return True

    def staged_prefetch_tokens(self, req_id: str) -> int:
        """Tokens a staged prefetch would splice (0 = no hold); surfaced by the
        scheduler as the request's host_hit_length."""
        f = self.staged_prefetches.get(req_id)
        return f.num_tokens if f is not None else 0

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, NodeId]:
        """Buffer-mode branch of init_load_back: consume the staged prefetch
        at prefill admission — device alloc (evict-before-alloc), backend
        restore, and a plain insert so downstream sees ordinary tree state.
        Misaligned or alloc-failed holds drop; the request recomputes."""
        cache = self._cache
        req = params.req
        assert req is not None
        empty = cache.tree_core.empty_match_result.device_indices
        unchanged = (empty, req.last_node)
        f = self.staged_prefetches.pop(req.rid, None)
        if f is None:
            return unchanged
        cc = cache.cache_controller

        def _drop() -> tuple[torch.Tensor, NodeId]:
            self.transient_buffer.release(f.buffer)
            cc.prefetch_tokens_occupied -= f.buffer.accounted_tokens
            return unchanged

        # Splice-validity: the span only fits if the device prefix still
        # ends exactly at the enqueue-time matched_len.
        if len(req.prefix_indices) != f.matched_len:
            # Prefix moved while held (leaf eviction or sibling extension):
            # drop and recompute.
            return _drop()

        # Evict-before-restore (mirrors _load_back_transfers): the budget gate
        # counts evictable pages, but the backend needs free destination slots.
        if cache.supports_swa():
            avail = cache.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = cache.token_to_kv_pool_allocator.available_size()
        if avail < f.num_tokens:
            needed = f.num_tokens - avail
            evicted = cache.evict(EvictParams(num_tokens=needed))
            if evicted.num_tokens_evicted < needed:
                # Genuinely no room (locked pages): recompute.
                return _drop()

        load_back_id = -(f.operation_id) - 1
        restore = self.transient_buffer.restore(
            f.buffer,
            operation_id=load_back_id,
        )
        if restore is None:
            # Transient allocator shortfall despite the evict: recompute
            # (init_load_back's degrade contract).
            return _drop()
        device_indices = restore.device_indices

        swa_dev = restore.pool_device_indices.get(PoolName.SWA)
        if swa_dev is not None:
            # Register the trailing window's FULL->SWA translation NOW: the
            # admitted request's attention reads the window through this
            # mapping during the layer-gated forward.
            cache._apply_cache_action(
                RebuildFullToSWAMapping([device_indices[-len(swa_dev) :]], [swa_dev])
            )

        # Publish via a plain insert under the admission lock choreography;
        # the caller's request lock then pins the span (load_back pattern).
        key = RadixKey(
            array("q", f.key_tokens),
            extra_key=f.extra_key,
            is_bigram=cache.tree_core.is_eagle,
        ).page_aligned(cache.page_size)
        span_end = f.matched_len + f.num_tokens
        cache.insert(
            InsertParams(
                key=key,
                value=torch.cat([req.prefix_indices, device_indices]),
                prev_prefix_len=f.matched_len,
                swa_evicted_seqlen=(
                    max(0, span_end - len(swa_dev)) if swa_dev is not None else 0
                ),
            )
        )
        self.ongoing_buffer_load_back[load_back_id] = _OngoingBufferLoadBack(
            req_id=f.req_id,
            num_tokens=f.num_tokens,
            buffer=f.buffer,
            hash_values=f.hash_values,
        )
        m = cache.match_prefix(MatchPrefixParams(key=key))
        if len(m.device_indices) < span_end:
            # The insert walk did not adopt the full span (should not happen
            # for a locked prefix); the slots are tree-owned/evictable — do
            # not splice, the request recomputes.
            return unchanged
        return device_indices, m.last_device_node

    def try_finish_load_back(self, ack_id: int) -> bool:
        """Fill ack: release the transient buffer and return True when the ack is
        a buffer-mode load-back. The span was published at admission; the
        ack never touches the tree (existence beliefs were fed from the
        storage-fetched pages at staging commit)."""
        f = self.ongoing_buffer_load_back.pop(ack_id, None)
        if f is None:
            return False
        cache = self._cache
        cc = cache.cache_controller

        self.transient_buffer.release(f.buffer)

        cc.prefetch_tokens_occupied -= f.buffer.accounted_tokens
        logger.info(
            "HiCache prefetch fill committed req=%s filled=%d occupied=%d",
            f.req_id,
            f.num_tokens,
            cc.prefetch_tokens_occupied,
        )
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_prefetched_tokens(f.num_tokens)
        return True

    def release_aborted_staged(self, rid: str) -> bool:
        """Free an aborted request's staged prefetch. Returns whether one existed."""
        staged = self.staged_prefetches.pop(rid, None)
        if staged is None:
            return False
        self.transient_buffer.release(staged.buffer)
        self._cache.cache_controller.prefetch_tokens_occupied -= (
            staged.buffer.accounted_tokens
        )
        return True
