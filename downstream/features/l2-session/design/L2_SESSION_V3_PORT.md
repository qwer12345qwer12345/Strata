# Persistent conversation L2 port

Baseline: upstream `main` at `6e38f588a646c0ee6948a74afbe670eb75896106` (0.2.18 / Engine 0.1.64). This is a new implementation on the current Engine session-file and spill APIs; the old L2 branch is reference material, not a merge base.

## Implemented in this branch

The existing `ConversationSpill` remains process-scoped by default. A separate opt-in `open_persistent()` mode now provides a restart-persistent full-session tier:

- a stable namespace per `SessionFileIdentity` (model + config), protected by the existing process lock;
- payload session files plus metadata-only session-file sidecars (token IDs, image keys, checkpoint prefixes and steering mode);
- startup index reconstruction from sidecars without reading K/V payloads into RAM;
- stale sidecar/payload cleanup, orphan cleanup after an interrupted two-file publish, and oldest-first cap enforcement;
- close releases the lock but retains persistent entries; take/drop/clear remove both payload and sidecar;
- the default ephemeral `open()` path is unchanged.

The metadata sidecar deliberately uses the current session-file format and identity validation rather than introducing a second ad-hoc metadata serializer.

Activation: set `STRATA_L2_PERSISTENT=1` for the Engine serve process. The existing serve loop computes the current `SessionFileIdentity` before opening the tier, uses the stable persistent namespace, and continues to use the existing background writer and `ConversationCache::set_evict_sink()` path. When unset, the process-scoped SSD spill behavior remains unchanged.

## Why this is staged

The older branch's V2/V3 archive and L2 control code was coupled to its older `conversation_file.cpp` internals. Upstream main has a different session-file implementation and already has a process-scoped SSD spill tier with a writer thread and `ConversationCache::set_evict_sink()`.

This branch now establishes durable ownership and restart-safe indexing in the live serve loop using the current APIs. It does **not** yet claim the old branch's incremental V3 behavior.

## Remaining work

1. Add the immutable base + ordered delta archive (V3) behind the current `session_file_write/read` interfaces.
2. Add SAVEINC and RESTOREPP with bounded prefix reads and checkpoint selection.
3. Route the existing eviction sink and background writer to the persistent tier without changing guarded/pinned-prefix admission rules.
4. Add a persistent-store policy/budget and explicit user controls; preserve the current ephemeral tier as the default.
5. Add crash/fault-injection coverage for payload publish, metadata publish, manifest publish, ENOSPC and restart recovery.
6. Wire the persistent store into the Engine serve lifecycle, including model/config identity changes and orderly shutdown.

## Tests

`conversation_spill_test` now includes a close/reopen round trip that checks metadata-only reindexing, prefix matching and full payload restore. The test is registered in the Engine conversation-test block. Build and runtime tests still need to be run on the target toolchain before this branch is considered ready to merge.
