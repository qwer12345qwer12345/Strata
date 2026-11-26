# Signal 2 headless follow-up

The first Signal 2 boot used engine 0.1.69, CUDA SM86, RTX 3080 12 GB, the BF16 table, a frozen expert profile and route prior 0.5. Each trial started with 4,706 GPU expert slots and a 21.62 GiB page-locked RAM complement.

| Arm | Long-code decode | GPU expert hits | Streaming p99 | 29K cold first content |
| --- | ---: | ---: | ---: | ---: |
| Production baseline | 138.5 tok/s | 92.5% | 30.36 ms | 11.09 s |
| Single-row Q8 off | 135.3 tok/s | 92.2% | 30.87 ms | 11.20 s |

Both long-code samples generated 1,536 tokens and reported zero expert file reads. The Q8-off arm completed, while the baseline stopped when available host RAM fell to 1,020 MiB at the 1,024 MiB guard. Four-row neural window p95 was 29.03 ms on baseline and 28.11 ms with Q8 off. The matched timings point in different directions, so the existing Q8 default stays and this arm leaves the next boot. The mid-coverage adaptation-off arm stopped during decode with a generic streamed error. Its original native error is unknown because the trial log was more temporary than its JSON report.

## DFlash2

The production engine reported `dflash2: draft rows out of range` at 18:28:47 and restarted. The eight-row drafter rejects its input before allocating or launching a graph when `n_new > 8`. The suffix verifier supports up to 16 rows and the commit-to-draft path passed those committed taps straight to the eight-row entry point. Some 16-row verifier windows complete without incident when they commit at most eight. Logged pre-capture VRAM reached 148 MiB, and the driver log had no CUDA allocation or GPU fault at this failure.

The follow-up restores the wide handoff from the previous work. It writes the first committed rows to the DFlash2 context ring in narrow batches, copies the final row to the existing landing buffer and launches the same eight-row draft block. It keeps the narrow CUDA graphs bound to their original taps. The first headless check uses the real Q4 drafter pack to compare narrow and wide calls against explicit context updates, including 9, 12 and 16 rows, graph invalidation and ring wrap.

## Save latency and host memory

Nine production auto-saves took 30.0 seconds each and returned HTTP 503. A full save sampled 1,394–1,824 MiB available and needed approximately 1,891–1,893 MiB, including the 1,536 MiB RAM floor. The gateway's synchronous save kept the conversation lock while the backend polled for memory. The session directory had no completed archive. Some long requests also spent a separate 30–37 seconds blocked on PLE reads; that is prompt work and independent of the 30-second save retry.

The follow-up allows an automatic save to attempt native cache reclamation once and return a typed RAM refusal immediately if it still cannot admit the write. Explicit session saves retain the existing settling behavior. A refusal keeps any prior archive. This bounds the observed blocking wait. A successful archive still needs sufficient RAM.

The tentative memory arm limits context to 221,184 tokens and halves the BF16 PLE row cache to 524,288 entries. The production 262,144-token configuration allocated 2.39 GiB of pinned host K/V beyond the 20,480 GPU-resident cells. Reducing the reserved range and halving the PLE row cache could recover about half a GiB. The benchmark records save return status, timing, RAM and bytes written on an isolated temporary archive. The production context and PLE settings are untouched.

## Planned boot

Signal 2 runs one baseline at prior 0.5, then prior 0.25 and 0, the interrupted mid-coverage adaptation comparison, and one archive-memory arm. All matched arms use the same staged engine, frozen expert profile, input workload and seed. Compare by neural or suffix source and exact verify width, together with disk expert reads and memory pressure. The completed production binary is still the original 0.1.69 artifact. The follow-up CUDA build and 57 Python protocol tests passed. The real-pack parity test is queued for the next headless boot.

[Follow-up source patch](signal2-follow-up.patch) applies on top of the existing 0.2.22 production source.
