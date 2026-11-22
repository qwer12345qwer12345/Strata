# Runtime KV Storage tests

The Session Delta Core test is a pure-host CTest target. The test runner copies its source overlay into the prepared source tree before configuring CMake; the production patch registers the target only when that overlay exists.

From the repository root:

```bash
python downstream/scripts/prepare.py --feature runtime-kv-storage --target ../Strata-runtime-kv-prepared
python downstream/scripts/test.py --feature runtime-kv-storage \
  --source ../Strata-runtime-kv-prepared --build ../Strata-build-runtime-kv
```

The current test covers a complete Base, two chained append Deltas, a keep=0 replacement/rewind, all five KV parts, generation and chain mismatches, identity mismatch, invalid keep lengths, truncation and payload checksum failure.
