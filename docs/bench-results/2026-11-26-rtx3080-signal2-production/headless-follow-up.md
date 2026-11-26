# Signal 2 headless follow-up

The first Signal 2 boot used engine 0.1.69, CUDA SM86, RTX 3080 12 GB, the BF16 table, a frozen expert profile and route prior 0.5. Each trial started with 4,706 GPU expert slots and a 21.62 GiB page-locked RAM complement.

| Arm | Long-code decode | GPU expert hits | Streaming p99 | 29K cold first content |
| --- | ---: | ---: | ---: | ---: |
| Production baseline | 138.5 tok/s | 92.5% | 30.36 ms | 11.09 s |
| Single-row Q8 off | 135.3 tok/s | 92.2% | 30.87 ms | 11.20 s |

Both long-code samples generated 1,536 tokens and reported zero expert file reads. The Q8-off arm completed, while the baseline stopped when available host RAM fell to 1,020 MiB at the 1,024 MiB guard. Four-row neural window p95 was 29.03 ms on baseline and 28.11 ms with Q8 off. The matched timings point in different directions, so the existing Q8 default stays and this arm leaves the next boot. The mid-coverage adaptation-off arm stopped during decode with a generic streamed error. Its original native error is unknown because the trial log was more temporary than its JSON report.

## DFlash2

The production engine reported `dflash2: draft rows out of range` at 18:28:47 and restarted. The eight-row drafter rejects its input before allocating or launching a graph when `n_new > 8`. The suffix verifier supports up to 16 rows and the commit-to-draft path passed those committed taps straight to the eight-row entry point. Some 16-row verifier windows complete without incident when they commit at most eight. Logged pre-capture VRAM reached 148 MiB, and the driver log had no CUDA allocation or GPU fault at this failure.

[Upstream PR #7015](https://github.com/Niko1221/Strata/pull/7015) uses a synchronous bridge. It commits the prefix in narrow batches, copies the last row to a fixed landing buffer and runs the regular graph on that row. The new candidate captures the same context batches and the eight-row block on the original verifier buffer. Each attention layer waits for the final context chunk's K/V signal. The host waits at draft completion. Context-only updates also support 16 committed rows, including the idle DFlash2 drafter when auto mode selects MTP.

The candidate reuses the current SM86 shape kernels, K/V buffers and context scratch. It reserves eight extra position entries and 22.5 KiB of pinned host staging for the ring log. CUDA Graphs for widths above eight are created on first use. A wide graph allocation failure switches that width to the same kernels queued without a graph. The model keeps its eight-row block and the target verifier still checks every proposed token.

Sixteen-row verification is expensive for a short acceptance streak. In the earlier Signal 2 baseline one 16-row suffix window took 68.6 ms and accepted one draft. Four-row neural windows had a median of 24.0 ms. The next run tests upstream `STRATA_DF2_LOOKUP_SKIP=8` for long suffix matches. The existing draft policy still decides whether to use the suffix. When it does, the engine updates the DFlash2 ring and skips a neural draft. Trace timing includes deferred context and fallback drafting work. The report counts both outcomes by request.

## Save latency and host memory

Nine production auto-saves took 30.0 seconds each and returned HTTP 503. A full save sampled 1,394–1,824 MiB available and needed approximately 1,891–1,893 MiB, including the 1,536 MiB RAM floor. The gateway's synchronous save kept the conversation lock while the backend polled for memory. The session directory had no completed archive. Some long requests also spent a separate 30–37 seconds blocked on PLE reads; that is prompt work and independent of the 30-second save retry.

The follow-up allows an automatic save to attempt native cache reclamation once and return a typed RAM refusal immediately if it still cannot admit the write. Explicit session saves retain the existing settling behavior. A refusal keeps any prior archive. This bounds the observed blocking wait. A successful archive still needs sufficient RAM.

The tentative memory arm limits context to 221,184 tokens and halves the BF16 PLE row cache to 524,288 entries. The production 262,144-token configuration allocated 2.39 GiB of pinned host K/V beyond the 20,480 GPU-resident cells. Reducing the reserved range and halving the PLE row cache could recover about half a GiB. The benchmark records save return status, timing, RAM and bytes written on an isolated temporary archive. The production context and PLE settings are untouched.

## Second headless run and production promotion

The real Q4 pack passed draft-token, probability and K/V ring parity across every width from 9 to 16, plus narrow rounds, ring wrap, graph invalidation and context-only updates. In 24 warm pairs at width 16, draft p50 fell from 1.103 to 1.063 ms and p95 fell from 1.320 to 1.080 ms. These timings cover the drafter and its context update. They do not include target verification.

After the trials, the original production engine served a 93,471-token prompt. It exited twice after a 16-row verifier window with `dflash2: draft rows out of range`. The verifier reported 202 MiB free before capture and the kernel log had no GPU fault or OOM event. The gateway started a fresh engine on the next request each time. The same checksum that passed the Q4 check is now installed in the normal boot path with the 16-row limit and prior 0.5. The attempted benchmark fingerprint was carried forward because all six arms already ran on that checksum.

| Arm | Min available RAM | Added swap | Long decode | Expert file reads | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline, prior 0.5 | 1,023 MiB | 0 MiB | 131.6 tok/s | 0 | Memory guard stopped the suite before the rewrite and archive probe |
| Lookup skip 8 | 965 MiB | 157 MiB | 129.3 tok/s | 0 | Memory guard |
| Mid-coverage adaptation off | 1,245 MiB | 13 MiB | 121.8 tok/s | 0 | Complete |
| Prior 0.25 | 1,166 MiB | 4 MiB | 126.8 tok/s | 0 | Complete |
| Prior 0 | 1,133 MiB | 3 MiB | 110.5 tok/s | 0 | Complete |
| Context 221,184 and PLE row cache 524,288 | 1,726 MiB | 2 MiB | 130.9 tok/s | 0 | Complete |

The harness stops below 1,024 MiB of available RAM or above 512 MiB of additional swap. The earlier engine exit and this harness stop are independent observations. The `lookup-skip=8` arm skipped three drafts and ran 22 deferred drafts on the long-code case. It skipped none and ran 98 deferred drafts in the rewrite. The baseline never reached the rewrite. Default lookup deferral stays off, and the measured prior 0.5 remains the production setting.

The memory arm has evidence for both sustained decoding and a successful save. It wrote 571,223,919 bytes in 650.8 ms at 1,735 MiB available. The old 262,144-token profile reserves 2.39 GiB of pinned host KV in addition to 21.62 GiB of page-locked experts. The smaller context alone removes about 415 MiB of that KV reservation. The full arm raised observed minimum available RAM by 703 MiB. Its 29K prompt prefill rose from 11.14 to 11.33 seconds. The two changes were tested together, and larger prompts may have a different PLE cache hit rate.

The production context stays at 262,144 tokens. Automatic saves use the bounded admission path that can return a typed RAM refusal promptly while keeping the previous archive. The verified 221,184-token profile is the option when reliable saving under pressure matters more than the extra 40,960 tokens. Keeping more RAM free by putting experts on SSD would change the zero-file-read result. The post-promotion production request remains to be checked after the next headless boot. The desktop GPU was occupied at installation.

## Bounded archive memory at 256K context

The old full-save admission estimated 355–357 MiB of additional host RAM. It counted two copies of the selected checkpoint and one capture of live state. The checkpoint was about 113 MiB in the observed failing saves. Incremental planning also made its first fresh-checkpoint copy before RAM admission.

The synchronous writer now borrows validated checkpoints for the duration of the session lock. Full and incremental archives stream those fields without cloning them. The admission estimate retains the live-state capture, the K/V source directory and the write buffer. The two eliminated copies account for about 225 MiB at the measured geometry. Roughly 130 MiB remains in the write-side estimate. These are buffer-accounting estimates. Host high-water marks and save latency at 256K still need a model run.

V2/V3 size passes now sum K/V extents without reading the underlying bytes. The actual write reads them directly into the output block. V3 had two size passes before a full segment write, so the change removes two reads of the same K/V. Streaming readers keep one field reader alive at a time and hash each block in its original read buffer. Optional read-ahead buffers are included in the full-read RAM estimate. The disk layout and checksum rules stay compatible with existing archives.

The archive file tests passed 375 checks with serial reading and 375 with read-ahead depth four. They compare V1/V2 write bytes, test V2/V3 checkpoint rotation and restore, and assert that a V3 full write reads exactly the K/V payload once. The memory tests passed 121 reclaim checks and 45 admission checks. The startup framework passed eight tests and a dry run proving that ordinary comparisons use the resolved production executable, server source and cache capacity.

A full headless validation needs a save and a subsequent disk restore after reaching the 262,144-token capacity. Record the archive bytes, peak available RAM, native capture and write times, and reread tokens, with zero expert file reads and the same resolved profile. The running profile and its RAM floor stay at the previous production settings until that validation. The write budget and the remaining reserve for the OS are independent quantities.

## SSD spill exhaustion and model disconnect

A subsequent production boot completed 313 requests and processed a prompt of 196,392 tokens. The previous DFlash2 row-limit error did not recur. About 49 minutes into this boot, V3 saves began refusing new generations for lack of disk space. The gateway then logged native process exit code -7 (SIGBUS) and stopped relaunching it after three failures in ten minutes. The kernel log showed no OOM kill or NVIDIA Xid during that boot. No crash stack was available, so the exact SIGBUS instruction is unconfirmed.

Temporary SSD spill consumed about 121 GiB in 117 V3 directories even though its nominal budget was 32 GiB. V3 snapshots are directory trees. The spill tier used a single-entry unlink and released its accounting when the nonempty directory remained. The restart path also checked free space before reaping the crashed process's temporary namespace, so subsequent starts on the full disk left it behind. Removing the one unlocked temporary namespace raised available space from 3.7 GiB to about 125 GiB. Five committed V3 archives were retained and passed read-only structural verification.

Temporary spill now uses a single file (V2 when enabled); formal L2 archives retain V3 incremental generations. Eviction removes both files and legacy directory trees, releases the budget only after removal and leaves in-flight reads charged until their file goes away. Startup reaps dead temporary namespaces before measuring disk capacity, and each spill write checks the configured free-space floor. Linux disk-backed park buffers now reserve disk blocks before touching the mapping. Refusal falls back to the existing anonymous allocation path. A sparse mapping running out of disk blocks was one possible source of SIGBUS and has not been proven as the original crash site.

Six cache tests passed, including spill operation with V3 enabled, legacy directory removal, disk reserve refusal, and preservation of a durable archive. V2/V3 read-ahead passed 375 checks. A Btrfs allocation smoke test passed and an injected ENOSPC verified that the anonymous fallback retained the written bytes. The SM86 build is installed at the headless boot path. Prior remains 0.5 and the context ceiling remains 262,144 tokens. The desktop session was left running; headless 256K save and restore validation is pending.

## Production prior and the next headless run

The production and headless baseline now use route prior 0.25 for both boot profiles. The host setting lives in `$XDG_CONFIG_HOME/strata/headless.json` (or `~/.config/strata/headless.json`) as `{"route_prior": 0.25}`, applied after GRUB selects its model profile and before either the tests or the server start. The prior comparison still tests 0 and 0.5 once each; the 0.25 case is the baseline.

The benchmark records minimum available RAM and maximum added swap without interrupting any arm at a memory threshold. The time limits and native session admission rules stay the same. In a simulated trial at 256 MiB available and 1 GiB added swap, the timed request still completed and both figures reached the report. This verifies the harness policy. Any new throughput, headroom or quality claim still needs the next actual headless run.

[Follow-up source patch](signal2-follow-up.patch) applies on top of the existing 0.2.22 production source.
