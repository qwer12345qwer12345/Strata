# RTX 3080 patch review: interrupted first run

The November 25 headless run used Strata 0.1.65, Gyro-S, an RTX 3080 12 GB and 32 GiB DDR5. Eight of 20 groups saved reports in 31m 45s, then the machine stopped during group 9. The [numeric extract](results.json) retains the 31 saved arms and whatever could be recovered from group 9's engine logs. The recovered entries have no API stream timings and are kept separate. All 122 recorded cases in the eight group reports showed zero decode expert reads from disk.

The 1,536-token code request produced two distinct results with the same production engine, settings, seed and expert profile:

| Identical production settings | Ordinary code output | Repeated code sequence |
| --- | ---: | ---: |
| Full-length samples | 9 | 6 |
| Decode throughput, median | 100.4 tok/s | 138.9 tok/s |
| Output tokens / verification window | 2.93 | 7.46 |
| Verification window time | 29.21 ms | 53.68 ms |
| Number of verification windows | 524 | 206 |
| Stream content gap p99 | 37.63 ms | 105.25 ms |
| VRAM expert hit rate, excluding PCIe | 53.1% | 53.0% |

All six fast baseline outputs have the same output digest; none of the nine ordinary outputs share it. In the remaining group 9 logs, a 112.3 tok/s baseline accepted 1,101 suffix tokens versus three in a 102.6 tok/s baseline; both used the same configuration. These logs support the mechanism: once the code looks like a sequence already seen in that request, suffix lookup can propose a longer chunk and the main model can verify many tokens at once. Average token rate rises even though each window takes longer and the stream arrives in larger bursts. The logs do not establish which near-tie or scheduling event first caused the output to diverge. A different suffix regime is not evidence that the kernel or GPU became faster.

Results that guide the next decision:

- The trimmed binary's 101.0 tok/s sits in the ordinary production range. Its cold prefill was 11.40 s and hot prefill was 168 ms, versus a typical 11.31 s and 161 ms across the original binary's baselines. This comparison establishes neither a 26% regression nor a clear speed gain from the trimming.
- Turning off fused TQ raised 29K-token cold prefill to 16.40 s in this trial, versus 11.31–11.41 s for the same group's two production runs. This is the clearest candidate for retaining the existing setting. The short completion of the unrelated hot case does not invalidate the cold measurement.
- Turning off the BF16 upload or changing the DFlash2 projection path showed no clear cold-prefill win. Spec 6 showed 108.2 tok/s on this code request, with a 50.83 ms stream-gap p99 versus roughly 38 ms in the ordinary spec 4 regime. Do not promote it from this sample.
- The dense-shape variant reached 128.4 tok/s in group 9, but also accepted 1,073 suffix tokens. The other group 9 production run stopped after the short-context decode before its cold and hot cases could finish. The dense patch is still undecided.

Eight arms finished their hot-prefix answer naturally after 42–49 tokens; another baseline ended its long decode after 1,033. These triggered the old 75% length rule despite confirmed zero expert disk reads. Short completions can still measure prefix reuse and first content but should not be ranked with long decodes.

The revised [boot suite](../../boot-patch-review.md) runs only two production baselines in total and marks comparability for each workload. It also records suffix counts so future analysis can compare like with like.

## Latency decisions for the next boot

Use wire first-content time to answer how long a user waits, prompt time to locate input work, and p99 inter-chunk gaps to assess stalls during streaming. These loopback trials run one request at a time; the measured first-content time does not include the production gateway's queue. In the saved production baselines, the 29K cold prompt took 11.34 s to first content (median; prefill 11.31 s). Reusing 29,154 of 29,175 tokens brought this to 185 ms (prefill 161 ms). Cold-row streaming p99 was about 39 ms. The long-code fast and slow regimes above must be compared by source and window width rather than by a single median over both.

| Keep current production choice | Remove from the next boot | Why the current evidence is sufficient for this decision |
| --- | --- | --- |
| Fused TQ INT8 on | Unfused TQ; fused TQ F16 | Disabling fusion moves cold first content from 11.34–11.45 s to 16.43 s. The GPU timeline gains a separate 2.80 s dequant pass and 1.66 s down multiplication. The F16 arm increases the gate/up stage from 3.27 s to 3.65 s and cold first content to 11.52 s; it offers no lower latency on this long prompt. |
| K8V4 decode TC off | #6 on | Two short retests put verification at 26.83/26.84 ms with TC, versus 26.47/26.56 ms without. On the matched cold-long workload, window time is 26.02 ms versus 24.79–25.69 ms, and p99 is 39.95 versus 38.44–38.84 ms. The fast short-context code result came with longer windows and a 108 ms streaming p99. This decision does not retire the separate prompt IMMA test. |
| DFlash2 shape table 2 and QKV TC off | Old shape table; QKV TC on | Cold and hot first content are essentially unchanged. On the ordinary decode regime, keeping the tuned table costs about 1.25 ms of draft work per window versus 1.39 ms with the old table; the latter also has a roughly 1 ms longer full window. The corrected, fixed-input SM86 GPU shape sweep already preferred the tuned DP4A QKV path to the earlier TC path. Neither application arm supplies a reason to replace it. |
| Spec 4 as the low-latency default | Spec 6 | Spec 6 raised cold-long streaming p99 to 51.69 ms versus 39.09–39.31 ms in the local spec 4 reference pair. It reached 108.2 tok/s on the short-input code test, but lengthened the window and the cold, warm and short-input stream gaps. Spec 3 remains in the suite because its 34.47 ms cold-long p99 is a plausible latency/throughput tradeoff. |
| Adaptive fetch 2 | Fetch 0 alone | Fetch 0 already ran in multiple two-way tests: the most recent two lost roughly 3 percentage points of VRAM expert hits without a clear per-window latency benefit. Fetch 1 and the resident-overlap bundle are distinct alternatives and remain queued. |

HC persistence and tail skip 7 were already excluded. BF16 pack upload, paired rows, AVX-VNNI, lookup width and batching, Dense short-window dispatch, and the trimmed build stay in the queue: their saved arms either show small first-content differences or mix different draft regimes. The 128.4 tok/s Dense result, for example, is a suffix-rich run and does not isolate Dense. No production setting was promoted or reverted.

The next boot keeps only two production baselines and attempts all 31 remaining candidates in one sequence. The reports retain first content, prefill, prefix hits and streaming p95/p99, plus verification p50/p95/p99 with sample counts keyed by exact width and draft source.

## Completed headless run

The second run attempted all 18 groups and saved 33 arms including two baselines. Thirty completed. The 4,096 prompt chunk, upstream-only elastic vision and final baseline triggered the host memory guard. The final baseline reached 520 MiB of new swap and its long-code windows included stalls. Its 78.7 tok/s is not a clean second baseline, so the short-window comparisons below also use unaffected arms running in the same draft regime. The complete run was still on the original Gyro-S weights.

| Comparison | Evidence | Decision on this host |
| --- | --- | --- |
| GPU expert rows and CPU dots | Paired to single changed the 3-row neural p95 from 33.67 to 33.10 ms. Disabling AVX-VNNI gave 33.48 ms. The three share the same suffix acceptance and 53.0% VRAM hit. | No standalone speed patch justified for the new release. |
| Dense shape dispatch | On/off had 138.4/138.8 tok/s and suffix-16 p95 of 106.6/108.6 ms. Similar cold and hot first content. | Leave the extra table off. |
| Q8 hyper-connections | Disabling the Q8 path raised the cold 3-row neural p95 from 33.4 to 34.5 ms and lowered long-decode VRAM hit from 53% to 51%. Both new GGUFs retain Q8_0 hyper-connection up matrices. | Keep the ordinary Q8 path and the upstream shared-memory read fix. Keep persistent reads off. |
| Profile pinning and resident overlap | Disabling the 25% pin increased VRAM hits from 53.0% to 62.7% but raised the 3-row neural p95 from 33.7 to 56.9 ms. Its 137.1 tok/s had 1,252 accepted suffix tokens. The resident-overlap bundle lowered hits to 47.7%. | Keep the pin and resident Fetch 2. |
| Prompt processing | Disabling GDN TC raised 29K-token cold first content from 11.37 to 11.66 s. CPU share at 3,072 raised sub-1K first content from 1.32 to 1.66 s and 1K–3K from 1.32 to 1.91 s. The 1,024 configuration also lost on these requests. | Keep GDN TC at NP=2 and CPU-share off. Keep fused TQ INT8 from the first run. |
| BF16 table upload | Compact/off changed first content from 11.37 to 11.34 s. Disabling the separate copy stream gave 11.33 s with the same suffix regime and 53% hit. | Keep the existing lossless compact path on 0.1.65. Drop the independent copy stream override and omit the local BF16 GPU upload patch from the new release. |
| Drafting | Lookup width 16 accepted 1,153 suffix tokens; width 8 accepted 19 on this request. The suffix-16 median was 102.3 ms per window. Spec 3 reduced cold streaming p99 from 38.4 to 33.2 ms while long-output throughput was 93.6 tok/s. | Keep neural spec 4. Compare lookup width on the new weights and retain a narrow check of NOWAIT. |

The 0.1.69 candidate retains V2/V3 archives, the ChatML boundary and unpublished-repeat recovery, resident adaptive fetch, and the on-demand vision lease. It leaves out fork PRs 6, 7 and 9 and the additional expert-row and Dense dispatch patches. It uses the upstream wide-window implementation and Q8 shared-memory read correction. The port also fixes an L2 pin that could be lost when a longer snapshot superseded it.

The single-card comparison holds the model, expert seed, PLE, resident KV window, draft and API workloads constant. Version 0.1.66 began waiting for cuBLAS warm-up before measuring the free VRAM available to experts. To distinguish that timing change from the operator's reserve, 0.1.69 has one arm with warm-up disabled at 448 MiB reserve and another with warm-up enabled at 384 MiB. The comparison records pre-cache free VRAM, expert slots, post-load free VRAM, first content, conditional verification latency, and expert reads. The older 226-slot difference alone cannot explain the hit-rate gap.

The November 26 desktop startup probe used the original Gyro-S, a 32K context, a fixed resident KV setting and a 6 GiB pinned expert RAM budget on the same RTX 3080. It ran one 12-token Chat Completions request per start. Downloads shared the desktop, so these starts measure capacity and readiness rather than full-workload speed.

| Engine and startup policy | VRAM free before expert-cache sizing | Expert slots | VRAM free at ready |
| --- | ---: | ---: | ---: |
| 0.1.65, warm-up on, reserve 448 MiB | 6.86 GiB | 5,934 | 369 MiB |
| 0.1.69, warm-up on, reserve 448 MiB | 6.64 GiB | 5,708 | 619 MiB |
| 0.1.69, warm-up off, reserve 448 MiB | 6.86 GiB | 5,930 | 433 MiB |
| 0.1.69, warm-up on, reserve 384 MiB | 6.64 GiB | 5,772 | 555 MiB |

Disabling the pre-load warm-up returned 222 of the 226 slots lost across versions. That identifies the cause of the slot-count change in this setup. The 64 MiB reserve adjustment returned 64 slots. Long-output expert hit rate and tail latency still need the same-model headless comparison.

[Source patch for Strata v0.2.22](strata-0.2.22-gyro.patch). Apply it at upstream commit `b28ca95e` with `git apply --index` and build CUDA for SM86 with `STRATA_WINDOW_MAX_T=16`. No models, executables or host boot configuration are included. The patched build passed its conversation file, L2 pin and vision lease CTests, plus 48 mocked Python protocol and recovery tests. The desktop capacity probe used that build.
