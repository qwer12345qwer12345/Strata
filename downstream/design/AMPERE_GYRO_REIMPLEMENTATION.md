# Ampere / Gyro reimplementation

Baseline: upstream `main` at `6e38f588a646c0ee6948a74afbe670eb75896106` (0.2.18 / Engine 0.1.64). The kernels and dispatch are adapted to the current Engine sources; the old branch is used only as a numerical/reference guide.

## Opt-in paths

| Path | Opt-in | Hardware | Default |
|---|---|---|---|
| TQ paired gate/up rows | `STRATA_TQ_AMPERE_ROWS=2` | CUDA SM86 (RTX 30) | off |
| K8V4 Q*K split-INT8 IMMA | `STRATA_K8V4_QK_IMMA=1` | CUDA SM86 | off |
| TQ AVX-VNNI pair-dot | `STRATA_TQ_AVXVNNI=1` | x86 AVX-VNNI | off |

The TQ paired kernel computes gate and up together while preserving the existing per-row partial-sum and reduction order. It is selected only for the supported trellis gate/up formats and SM86; every other device and format stays on the upstream kernel.

The K8V4 path uses a three-part signed/unsigned INT8 query representation and IMMA for Q*K. The q4_0 V gather/dequantization path is unchanged. It is selected only on SM86; the upstream K8V4 implementation remains the fallback.

The CPU path uses VEX-encoded `vpdpwssd` for TQK6/TQK7 row pairs when the runtime AVX-VNNI probe succeeds. It is opt-in and otherwise uses the existing AVX2 implementation. Non-GNU/Clang builds retain the AVX2 fallback.

## Regression tests

- `tq_cpu_parity`
- `tq_cpu_parity_avxvnni` (environment: `STRATA_TQ_AVXVNNI=1`)
- `native_expert_parity_tqk6_tqk7_ampere_pair`
- `native_expert_parity_tq2_t_ampere_pair`
- `qsa_prompt_attn_parity_k8v4` (skips with code 77 unless the active device is SM86)

All opt-in flags are deliberately off by default. The new tests are registered but have not yet been run on a CUDA SM86 host or an AVX-VNNI CPU in this work session. Performance claims require the benchmark suite and real-model latency measurements on the target hardware.
