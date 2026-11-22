# Shutdown Persistent Cache

Patch B of the L2 patch split (see `patch-plans/shutdown-persistent-cache/`): keeps the RAM conversation cache's
parked conversations across a **normal engine restart**. When the serve loop ends orderly (the server closed stdin
or sent QUIT), every parked image is written as an official session file plus a small sidecar; the next start
indexes the sidecars and reads a record back only when a prompt matches it.

The runtime spill tier (`ConversationSpill`) is unchanged and stays the runtime SSD tier for evictions; the two are
independent — this package neither scans nor moves another process's spill files, and it does not depend on
`runtime-kv-storage` (Patch A).

## Behavior

- **Write (shutdown only)**: at the end of the request loop, while the RAM cache still holds its entries and nothing
  is in flight, each parked conversation goes to `<data>/conversation-persist/<model>-<config>/`, oldest first, so
  the conversations the client used last win the budget. A layer-split serve (stage images) never writes.
- **Index (startup / first matching request)**: valid sidecars rebuild the prefix index without reading any K/V
  payload. A sidecar that fails its checks — magic, version, reserved fields, header or stub hash, end marker,
  identity, stub shape, or a payload that is missing, linked, or of another size — is ignored, alone.
- **Restore**: a record this prompt matches better than the session, any batch slot and every RAM entry (and when
  the runtime SSD tier did not win the request) is read whole under the session RESTORE's limits and RAM floor. The
  record is **not consumed**: it stays until a new snapshot of the same conversation replaces it or the budget
  drops it, so it answers the next restart too.
- **Budget**: `used` never passes `cap`. A write that does not fit evicts whole oldest records (never one checkpoint
  or a delta of one); an image larger than the whole budget is refused. A lowered cap retires records at the scan.
- **Failure behavior**: a full disk, a failed write or an unreadable record costs only the cache hit it would have
  been — the prompt is read again. Nothing of a refused record is left on the disk; a record whose payload no
  longer reads is dropped with its sidecar.
- **Isolation**: one folder per `SessionFileIdentity` (model x config), so two models or two configurations never
  hit each other's records. A namespace lock (held for one shutdown flush) serializes concurrent engines: an engine
  that cannot take it writes nothing and logs it.

Not claimed: no save on a crash or power cut (only the orderly-exit code writes), no cross-identity reuse, no
second session-file format, no runtime write-amplification change (that is Patch A's scope).

## Activation

Off by default. Opt-in with:

| Variable | Meaning |
|---|---|
| `STRATA_SHUTDOWN_CACHE_MIB=N` | on, with a budget of N MiB (`N >= 1`; unset or 0: off) |
| `STRATA_SHUTDOWN_CACHE_DIR=<dir>` | optional persist root (default: the data folder `--pack` sits in, i.e. the parent of `<data>/packs/<tag>`; without it: off) |

The model fingerprint is taken once, on the serve thread, when the tier first opens — the same one-time cost the
runtime spill tier pays before its first write.

## Scope limits

- Single-stage serve only, like the runtime spill tier: a layer split's stage images are not persisted.
- Only the RAM cache's parked entries are persisted; the runtime spill tier's files and the writer's in-flight
  images are not scanned or moved.
- Records hold one session file each (`session_file_write` / `session_file_read`); restore goes through the
  official `conversation_snapshot_validate` path with the engine's usual read limits.
