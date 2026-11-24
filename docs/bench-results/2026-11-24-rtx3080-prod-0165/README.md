# Gyro-S on RTX 3080 12 GB and 32 GiB DDR5, November 24, 2026

This report separates a measured production-engine ablation from older reference runs and diagnostics. The host was a Linux headless system with an Intel Core Ultra 7 270K Plus. The [numeric extract](results.json) contains configuration values and measurements, without prompts, replies, host paths or raw logs.

## 0.1.65 production-engine PR ablation

See the [five-arm table and implementation notes](../../l2-pr-ablation.md). The test used a common frozen expert profile, K8V4, BF16 PLE, DFlash2 spec 4, prior 0.5 and tail skip 0. Each arm received the same 1,472-token prompt and produced 320 tokens. All five used 5,540 expert slots, the full RAM complement and zero resident-mode expert blob reads. No new swap was observed.

Keeping adaptive fetch at 2 reduced mean decode window time from 51.18 to 27.48 ms with roughly the same tokens per window. When persistent HC was turned off, throughput rose from 76.6 to 87.1 tok/s because fewer windows were needed. Mean window time rose from 27.48 to 28.92 ms, so the throughput result does not measure HC kernel latency. The production branch keeps #8 enabled and sets #6, #7 and #9 to zero. That combined configuration has not yet been timed after the change.

## 0.1.65 versus 0.1.66 reference observations

The long-suite references ran in alternating order, two starts per engine. Both requested persistence off, adaptive fetch 0, tail skip 7, DFlash2 spec 4 and the same other reference settings. The installed binaries did not emit the routing marker required by the harness, so these figures are from completed requests in runs the harness rejected. One 0.1.65 long-output request ended at 2,186 of 3,072 requested tokens. The table is useful for diagnosing the direction and the bottleneck, but does not compare the current production settings.

| Long-suite reference | 0.1.65 | 0.1.66 |
| --- | ---: | ---: |
| Long output, decode tok/s | 121.5–129.6 | 108.3–109.4 |
| Long context, decode tok/s | 104.9–111.8 | 93.3–95.0 |
| VRAM expert slots | 5,542 | 5,316 |
| Long-output VRAM expert hit rate, excluding PCIe | 63.9–64.9% | 41.4–41.6% |
| Long-output CPU expert time per window | 9.97–11.65 ms | 13.89–14.30 ms |
| Long-output GPU-reach wait per window | 10.45–10.86 ms | 10.38–10.43 ms |
| Resident-mode expert blob reads from SSD | 0 | 0 |

The old reference gap is present with persistent HC requested off on both engines. The logs point to more work reaching the CPU on 0.1.66. The slot difference is only about 4%, so it cannot by itself establish the cause of the hit-rate gap. Expert placement and the generated routing sequence also matter. GPU-reach wait includes scheduling and synchronization. It is not an isolated GPU kernel timing.

Keep **0.1.65 plus adaptive fetch mode 2** for this host. Before switching to 0.1.66, compare both with tail 0, prior 0.5, adaptive fetch 2, persistence off, the same frozen profile and the same explicit prompt/batching settings. Match expert slot counts for a second pass if the automatic allocations differ. Measure first content, window time and output per window as well as hit rate and long-turn cache reuse.

## Other observations from the headless run

- **Interpretation of the 180 arms:** 172 produced both benchmark cases. 170 were rejected for the Ampere dispatch marker, eight reached a time or memory guard, and only two passed the harness. The two passes both used tail skip 7, which is unsuitable for the production conversation workload on this host. See the [separately published run and dense sweep](https://github.com/qwer12345qwer12345/Strata/tree/45fb3cb720e5bc9ed28d1b9d32a4b496f6b31651/downstream/features/ampere-tc-dispatch/validation/headless-2026-11-24-rtx3080) for the full sanitized measurements.
- **Spec and window sizes:** On the older 0.1.66 tail-0/fetch-2 test, spec 3/4/6 reached approximately 76/87/91 tok/s for long output. Spec 4 gave about 45 ms stream-gap p99, versus about 60 ms for spec 6. These arms also lack verified dispatch markers. In the separate tail/fetch diagnostic, the spec 4, tail 0, fetch 2 arms verified at widths of at least 9 on about 0.5% of long-output windows and 0.2% of long-context windows. Tune widths 1–4 first for these prompts.
- **Quality and adaptive fetch:** A separate 0.1.66 diagnostic found about 85–86 tok/s with fetch 0 and 86–87 tok/s with fetch 2 for tail-0 long output. The large fetch effect in the 0.1.65 production-engine test does not carry directly across builds and settings. All eight runs in the separate diagnostic failed at least one objective quality task. The 128 synthetic tool probes passed, without executing a tool, but they do not establish production reliability or make tail 7 safe. See the [published tail/fetch evidence](https://github.com/qwer12345qwer12345/Strata/tree/5b28d748a64fa213772aabc6efab7e1003c349e5/downstream/features/ampere-tc-dispatch/validation/tail-fetch-quality-2026-11-24).
- **Vision lease:** The older headless vision check recorded one lend and one restore, with 1,610 MiB released and 1,608 expert slots restored in 311.3 ms. First content took about 2.97 s for a cold image and 66 ms for the same cached image. The following text request emitted all 3,072 requested tokens with zero resident-mode expert blob reads. The same missing dispatch marker kept the overall test from passing. This demonstrates the observed lease round trip and does not validate vision on the updated 0.1.65 production settings.
- **Correctness versus speed:** The older synthetic checks passed for the selected CUDA parity tests, both TQ prefill modes, the dense layout sweep and tool repetition recovery. This does not establish that each optional kernel is faster. Compare fixed-width layers and end-to-end request windows separately.
