# TQ CPU AVX-VNNI

## Scope

Opt-in VEX-encoded AVX-VNNI `vpdpwssd` pair-dot path for TQK6/TQK7 rows. The existing AVX2 implementation remains the fallback when the feature is disabled or runtime AVX-VNNI detection fails.

## Activation

```text
STRATA_TQ_AVXVNNI=1
```

Default: off.

## Patch ownership

- `engine/src/kernels/cpu/tq_avx2.cpp`
- CPU parity test registration in `engine/CMakeLists.txt`

The parity implementation itself is already present in the pinned upstream source as `src/kernels/cpu/tq_cpu_parity.cpp`; this package only registers the baseline and opt-in test invocations.

## Tests

See [tests/README.md](tests/README.md). The test runner builds and runs `tq_cpu_parity` and `tq_cpu_parity_avxvnni`.

## Compatibility

This patch was extracted from PR #1 and targets upstream commit `6e38f588a646c0ee6948a74afbe670eb75896106`. No AVX-VNNI hardware qualification is claimed by this migration.
