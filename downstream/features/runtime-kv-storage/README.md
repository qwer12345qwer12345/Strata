# Runtime KV Storage

Status: active Feature Package; Patch 1 (Session Delta Core) only.

This package implements Patch A from the runtime KV storage plan. It starts from the official SessionFile serialization layer and leaves the runtime ConversationSpill lifecycle unchanged in this first patch.

## What Patch 1 adds

- A separate versioned STRSDLT file format. Existing STRSESS v1 Base files remain unchanged.
- A Delta header bound to model/config identity, an opaque chain ID, exact parent generation, next generation, payload size and header checksum.
- A complete non-KV state image for the new generation.
- Five opaque KV parts per layer: K, V, K scale, V scale and pooled/indexer. Each part is represented as a caller-supplied keep prefix plus a changed tail.
- Atomic Delta publication using the existing SessionFile temporary-file, flush and publish machinery.
- Bounded parsing, identity/generation checks, payload hash validation, layer/layout compatibility checks and checked keep/tail lengths.
- A pure-host test target. Its test source is a test overlay, not part of the production patch.

## API

- `session_delta_write()`: writes one immutable Delta for generation = parent_generation + 1.
- `session_delta_apply()`: applies one exact child generation to an already-read Base image.

The caller supplies per-part keep byte lengths. They are not derived from token counts in this patch. The future Snapshot provenance adapter must obtain exact boundaries from the official Snapshot part-size/layout API. A layout/format change or uncertain lineage starts a new Base.

A parse or validation failure does not mutate the Base image. An allocation/copy failure during application can leave the in-progress image partially materialized; callers must discard it and treat the chain as a cache miss.

## What this patch does not change yet

- It does not call the Delta API from ConversationSpill::put() or take_serial().
- It does not preserve spill entries across Restore or add chain leases.
- It does not reduce runtime SSD writes until the later Snapshot provenance and ConversationSpill integration patches are implemented.
- It does not implement compaction or shutdown persistence.

## Tests

The test target is enabled only when the test overlay is present, so ordinary builds with STRATA_BUILD_CONVERSATION_TESTS=ON do not acquire a missing test-source dependency.

See tests/README.md and validation/README.md.

Design and staged implementation plan: [DESIGN.md](../../patch-plans/runtime-kv-storage/DESIGN.md)
