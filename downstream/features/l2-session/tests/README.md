# Persistent L2 tests

The package carries the updated `conversation_spill_test.cpp` in `tests/overlay`. The test runner overlays it temporarily onto the pinned source tree and uses the existing `conversation_spill_test` CMake target.

```bash
python downstream/scripts/test.py --feature l2-session \
  --source ../Strata-prepared --build ../Strata-build-l2-session
```

Configure with `STRATA_BUILD_CONVERSATION_TESTS=ON`. The regression covers close/reopen persistence, metadata-only reindexing, prefix matching and payload restoration. Additional crash/fault-injection coverage remains future work.
