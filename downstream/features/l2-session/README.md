# Persistent Conversation Spill (L2)

## Scope

Adds an opt-in restart-persistent mode to the existing ConversationSpill tier while preserving the process-scoped default behavior.

The implementation provides:

- a stable namespace per model/config identity;
- a process lock for the persistent namespace;
- metadata sidecars used to rebuild the prefix index without loading K/V payloads;
- stale/orphan cleanup and oldest-first budget enforcement;
- close/reopen persistence while take/drop/clear remove payload and sidecar together.

## Activation

```text
STRATA_L2_PERSISTENT=1
```

Default: off; the existing ephemeral spill path remains unchanged.

## Patch ownership

- `engine/include/strata/core/conversation_spill.hpp`
- `engine/src/program/generate.cpp`

The updated ConversationSpill regression test is stored in this package's test overlay.

## Scope boundary

This is a restart-persistent full-session L2 foundation using the current upstream session-file format. It does not implement the older branch's V3 immutable base/delta archive, SAVEINC/RESTOREPP or incremental K/V writes.

## Tests

See [tests/README.md](tests/README.md).
