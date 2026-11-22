# K8V4 Q*K split-INT8 IMMA

## Scope

Opt-in SM86 K8V4 prompt-attention Q*K path using split INT8 query fragments and IMMA. The existing Q4_0 V gather/dequantization path is unchanged, and the upstream K8V4 route remains the fallback.

## Activation

```text
STRATA_K8V4_QK_IMMA=1
```

Default: off.

## Patch ownership

- `engine/include/strata/kernels/qsa_prompt_attn.hpp`
- `engine/src/kernels/cuda/qsa_prompt_attn.cu`
- CUDA parity target registration in `engine/cmake/backends/cuda.cmake`

The parity test source is kept in this package's `tests/overlay`, not in the production patch.

The combined Ampere/Gyro implementation rationale and test inventory are preserved in [AMPERE_GYRO_REIMPLEMENTATION.md](../../design/AMPERE_GYRO_REIMPLEMENTATION.md).

## Tests

See [tests/README.md](tests/README.md). The K8V4 IMMA test skips with code 77 unless the active CUDA device is SM86.

## Compatibility

Extracted from PR #1 against `c6f6427d9de9016fd92d13e29740f65625261df7`. Hardware parity and latency qualification remain outstanding.
