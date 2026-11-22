# Live Memory tests

This package carries three test overlays:

- `live_memory_policy_test.cpp`: CPU-only request parsing and capacity-policy checks.
- `expert_cache_segmented_test.cpp`: CUDA VMM prefix, transactional grow/shrink rollback, resident-tail protection and driver-refusal checks.
- `tools/test_settings_register.py`: Settings scanner regression for `std::from_chars`.

Run CPU policy and Settings tests:

```bash
python downstream/scripts/test.py --feature live-memory \
  --source ../Strata-prepared --build ../Strata-build-live-memory
```

Run the CUDA VMM test as well:

```bash
python downstream/scripts/test.py --feature live-memory \
  --source ../Strata-prepared --build ../Strata-build-live-memory-cuda --with-cuda
```

The CUDA test requires a supported CUDA VMM device/driver. The test runner overlays package test sources temporarily and restores them afterward.
