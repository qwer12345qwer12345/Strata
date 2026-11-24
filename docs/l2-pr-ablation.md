# Gyro-S: interpreting the 0.1.65 boot ablation

The November 24, 2026 run used one RTX 3080 12 GB, a 32 GB host, the same saved expert profile, DFlash2 spec 4, route prior 0.5 and tail skip 0. Every arm had 5,540 GPU expert slots and the resident RAM complement. There were zero resident-mode expert blob reads from disk. The timed input had 1,472 tokens and every arm produced 320 tokens. Each PR row turned just that switch off from the all-on baseline.

| Trial | Prompt ms | Decode tok/s | ms/window | Output tokens/window | Accepted/offered drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| All enabled | 1306 | 76.6 | 27.48 | 2.11 | 168/187 |
| #6 decode attention TC off | 1302 | 77.5 | 28.48 | 2.21 | 175/202 |
| #7 prefill TQ F16 off | 1262 | 76.6 | 28.42 | 2.18 | 173/203 |
| #8 adaptive fetch off | 3231 | 42.0 | 51.18 | 2.15 | 171/210 |
| #9 persistent HC off | 1308 | 87.1 | 28.92 | 2.52 | 193/208 |

- **#6:** Only the Q×K score calculation changes. The INT8 tensor core path packs queries in three parts and exchanges partial scores in shared memory. Value reads, softmax, merge and expert work remain. Compare the two kernels at fixed numbers of attended cells and fixed verify widths before adding a dispatch threshold. This short application run cannot establish one.
- **#7:** The TQK6/TQK7 prefill path keeps the trellis decoder and uses F16 tensor core products instead of INT8 products. Setting this switch also makes TQ eligible for the wide expert batches when requested. On this host the time spent getting experts to the GPU can hide a faster matrix multiply, and F16 is already competing with an INT8 tensor core path. Compare long prompt prefill by phase, including expert copy wait, before choosing a default for that workload.
- **#8:** When a PCIe fetch has a candidate worth keeping, mode 2 can use that fetch to place it directly in a GPU cache slot. The evicted expert is copied back to keep the RAM complement valid, and the ownership change commits after the copies finish. The adaptive tier still handles work this path cannot take. This targets redundant transfers and future CPU misses when the GPU holds only part of the model. Output per window was similar with and without it. Keep mode 2; the single run does not establish how much of the prompt-time difference it caused.
- **#9:** A persistent read saves kernel launches at the cost of global barriers across an 80-block grid. For supported Q8 rows it also bypasses the separate, fast down projection and INT8 tensor core up projection selected by default on Ampere. The residency check only establishes eligibility. Turning persistence off improved total throughput in this run because more drafts were accepted and fewer windows were needed. The mean window did not get faster. A fixed-input, fixed-width HC comparison is needed to isolate kernel speed; fixed-prompt replay across more inputs is needed to characterize draft acceptance.

The host's production selection keeps #8 at `STRATA_ADAPT_FETCH=2` and sets #6, #7 and #9 to `0`. The four code paths remain available; the next boot ablation flips each switch from its actual production value. The follow-up headless boot measured the combined selection at 82.8 tok/s on the same timed prompt.

See the [RTX 3080 headless report](bench-results/2026-11-24-rtx3080-prod-0165/README.md) for the numeric extract, the older engine references and the vision lease observations.
