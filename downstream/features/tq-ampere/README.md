# TQ Ampere paired Gate/Up Kernel

## Scope

Opt-in CUDA paired Gate/Up TQ decode path. The implementation is restricted to the supported Ampere/SM86 route and keeps the upstream path for other devices and formats.

## Activation

```text
STRATA_TQ_AMPERE_ROWS=2
```

Default: off.

## Patch ownership

- `engine/src/kernels/cuda/iq_kernels.cu`
- Synthetic parity test registration in `engine/CMakeLists.txt`

The tests reuse the existing `native_expert_parity` executable and cover TQK6/TQK7 and TQ2_T.

The combined Ampere/Gyro implementation rationale and test inventory are preserved in [AMPERE_GYRO_REIMPLEMENTATION.md](../../design/AMPERE_GYRO_REIMPLEMENTATION.md).

## Tests

See [tests/README.md](tests/README.md). Run on an SM86 CUDA device to qualify the optimized path.

## Compatibility

Extracted from PR #1 against `c6f6427d9de9016fd92d13e29740f65625261df7`. No CUDA hardware result or performance claim is included in this migration.
