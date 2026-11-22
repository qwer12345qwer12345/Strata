# Live Memory tests

This package carries four test overlays:

- `live_memory_policy_test.cpp`: CPU-only request parsing and capacity-policy checks, plus PR4's terminal
  ACK / CAPACITY line shapes and the pending request's supersede rule.
- `live_ram_blocks_test.cpp` (PR4 Phase 2): CPU-only block store - step sizing, the fill plan, grow's
  publish-or-rollback under injected allocation and copy failures, the pinned-with-pageable-fallback
  allocator, and release's free-the-last-block-first order with entries out of reach before the free.
- `expert_cache_segmented_test.cpp`: CUDA VMM prefix, transactional grow/shrink rollback, resident-tail
  protection, live-prefix admission/pointer bounds and driver-refusal checks.
- `tools/test_settings_register.py`: Settings scanner regression for `std::from_chars`.

PR4's server half is tested in `terminal/serve` (part of the terminal's own suites, not this package's
runner): `test_live_memory_policy.py` (the pure policy) and `test_live_memory_protocol.py` (the engine
command's ACK parsing, the CAPACITY routing, the /v1/memory route and Vision's unified owner).

Run CPU policy, RAM-blocks and Settings tests:

```bash
python downstream/scripts/test.py --feature live-memory   --source ../Strata-prepared --build ../Strata-build-live-memory
```

Run the CUDA VMM test as well:

```bash
python downstream/scripts/test.py --feature live-memory   --source ../Strata-prepared --build ../Strata-build-live-memory-cuda --with-cuda
```

The CUDA test requires a supported CUDA VMM device/driver. The test runner overlays package test sources
temporarily and restores them afterward.
