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
