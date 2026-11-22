# Runtime KV Storage tests

Two CTest targets. The test runner copies the source overlays into the prepared source tree before configuring CMake; the production patches register the targets only when their overlays exist.

From the repository root, using an already prepared source tree:

```bash
# the pure-host suite (always)
python downstream/scripts/test.py --feature runtime-kv-storage   --source ../Strata-prepared --build ../Strata-build-runtime-kv

# the CUDA round-trip gate (needs a CUDA device and STRATA_ENABLE_CUDA)
python downstream/scripts/test.py --feature runtime-kv-storage   --source ../Strata-prepared --build ../Strata-build-runtime-kv --with-cuda
```

`conversation_delta_test` (pure host) covers the Session Delta Core's file format - a complete Base, two chained append Deltas, a keep=0 replacement/rewind, all five KV parts, generation and chain mismatches, identity mismatch, invalid keep lengths, truncation and payload checksum failure - and the provenance adapter's decision table (`conversation_delta_keep.hpp`): what a restored image lends as keep lengths (complete pages of cells below the unchanged boundary, complete indexer rows, never the moving spare row), and every refusal (no provenance, no complete unchanged unit, a rewind, a changed layout, a layer-count or stage mismatch, a part whose bytes do not divide into its recorded units).

`conversation_delta_roundtrip_test` (CUDA, skips with 77 without a device) is Patch 2's acceptance gate: for F16, INT8 (with scale parts), Q4_0, K8V4 hybrid, across resident, streamed and ring modes, a restore -> grow -> dirty run drives Base -> Delta -> materialize -> Snapshot validate/restore, and the materialized image is byte for byte the full capture of the grown state; a boundary without a complete unchanged page falls back to a new Base.
