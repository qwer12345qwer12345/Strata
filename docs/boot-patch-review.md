# Strata 0.1.65 headless patch review

The GRUB Strata entry attempts the whole suite in one boot before exposing the production API. Each batch starts and ends with the current production settings. Between those two runs it changes one option at a time, or one named bundle where the settings are coupled. It runs a synthetic long decode, cold long prompt, and shared-prefix continuation. The vision batch uses a generated color-panel image. Reports record first content, stream gaps, verification-window costs, expert hits excluding PCIe, and expert file reads. The tests reuse one frozen expert profile for the entire suite. They run serially on loopback with their own scratch directories and do not write production archives or the expert profile.

| Area | Covered |
| --- | --- |
| Attention and experts | K8V4 decode TC and prompt IMMA, fused TQ INT8 and F16, GPU paired rows, CPU AVX-VNNI |
| PLE | Lossless BF16 upload and its independent copy stream; the BF16 table stays fixed |
| DFlash2 and Dense | Projection table, QKV TC, context batching, wide lookup, spec, Dense shape dispatch, Q8 HC |
| Expert movement | Fetch 0/1/2, NOWAIT, profile pinning, lag, resident overlap with its required settings |
| Prefill | Embedding scratch accounting, HC scratch, KV prefetch, group scratch, fused batch size, chunk size, CPU sharing and its threshold, GDN modes, QFUSE and HC upmix |
| Vision | GPU expert lease against upstream elastic reader behavior; cold image, reread of that image and text afterward |
| Trimmed build | A separately built candidate compared with the current engine. A difference here is an aggregate result; individual arms locate the cause. |

Keep session V2/V3, conversation persistence and tool-boundary/repetition protection out of this performance comparison. Do not enable route tail skipping or Q8 HC persistence in the test series. Tail skipping caused repetition in use. Q8 HC persistence lost about 7–8% throughput in two retests; in both, tokens per verification window remained almost the same. The older Live Memory design was set aside in favor of the vision lease. It is not linked into this production engine. A switch with no reader in the tested source (for example STRATA_DF2_LOOKUP_SKIP) is not a benchmark arm. Unsupported or unwired packages must not be counted as measured simply because an environment variable was set.

The deployed profile runs one pass in a single boot: 20 groups, 38 candidates and 78 serial engine starts. Each trial has a 125-second watchdog plus cleanup; the suite has no shared deadline that would cut off later groups. Failed arms are recorded and the next arm is attempted. Cases with insufficient output, nonresident expert fallback, or a memory guard trip do not qualify as throughput evidence. Bring a promising patch back only after a representative replay shows a repeatable improvement beyond baseline drift, without worse memory behavior or a practical quality regression. The suite does not promote settings automatically.

In patch-review mode, reports are in startup-patch-review/ under the configured benchmark output directory. The startup-patch-review-latest.json summary is updated after each group. The startup-patch-review-progress.json file marks the series once every group has been attempted, so later boots can start normally. An interrupted run leaves its finished group reports in place. Changing the series starts a new full run; an optional second pass runs in reverse order during the same boot.
