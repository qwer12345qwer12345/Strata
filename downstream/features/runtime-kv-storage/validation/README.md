# Validation — Runtime KV Storage

Status: Patch 1 (Session Delta Core) and Patch 2 (Snapshot provenance adapter) are written. Patch 2's acceptance
gate was compiled and run on a real CUDA device; Patch 1's host suite passes; the runtime spill integration
(Patch 3) and compaction (Patch 4) are not started, so no runtime write-amplification change is claimed.

## Verified while preparing the package

- The patches are based on the pinned upstream commit 6e38f588a646c0ee6948a74afbe670eb75896106.
- Patch 1 touches only the SessionFile header/implementation and an optional pure-host CMake test registration.
- Patch 2 adds one header-only host module (conversation_delta_keep.hpp) and an optional guarded registration of
  the CUDA round-trip test in strata_gpu_engine.cmake. No upstream behavior changes.
- Patch 1's host suite (conversation_delta_test) passed locally with the MSVC toolchain: 30 checks, 0 failures -
  the Delta format cases and the adapter's decision table (keeps and every refusal).
- Patch 2's acceptance gate (conversation_delta_roundtrip_test) compiled with nvcc/cl and ran on a real CUDA
  device: 360 checks passed across F16, INT8, Q4_0 and K8V4 hybrid, resident, streamed and ring modes. For an
  append boundary the materialized Base+Delta image is byte for byte the full capture of the grown state and goes
  through the Snapshot's own validate, restore and authoritative draft read-back; boundaries without a complete
  unchanged page (a rewrite below one page, a full rewrite) fall back to a new Base, as the design requires.
- Patch 2's keep lengths come from the layers' own recorded metadata and buffer sizes (the same values a session
  file stores) and the kernels' qsa_real_shapes block; nothing derives bytes from token counts.
- Patch SHA256 values, the standalone tree and the full stack tree are recorded in feature.json / stack.json.

These were run outside downstream/scripts/test.py (which needs CMake on the host). The recorded tree objects are
metadata calculations; downstream/scripts/verify.py is the gate the CI runs.

## Pending

1. Run downstream/scripts/test.py for both targets on a machine with CMake (the runner's overlay/configure path
   was verified for Patch 1; the CUDA target needs a GPU machine with CMake).
2. Patch 3 - ConversationSpill integration: spill entries hold Base + Delta refs, non-destructive Restore, chain
   lineage carried on the reuse (SessionDeltaLineage), per-chain writer serialization, whole-chain budget and
   eviction. The adapter and the tail-source narrowing this package adds are its inputs.
3. Patch 4 - Compaction and qualification (atomic new-Base switch, churn soak, write-amplification metrics).
4. The layer-split stage reuse is refused by the adapter by design; whole-session session files still reject
   stage_parts, so no claim is made there.
