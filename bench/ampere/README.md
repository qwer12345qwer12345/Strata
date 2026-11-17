# Gyro expert decode on RTX 3080

This report covers the opt-in CUDA SM86 gate/up kernel that assigns two output rows to the same eight threads. Those rows share the activation reads, and each row keeps its original accumulation order. The tested engine uses Strata 0.2.14 / engine 0.1.60 with [the upstream session persistence PR #5500](https://github.com/Niko1221/Strata/pull/5500). The local patches also cover CPU AVX-VNNI, CUDA K8V4 INT8, disk-backed conversation parking, and ChatML tool boundaries. The gate/up idea was adapted for the Gyro trellis formats after examining [Speedy's CUDA expert kernels](https://github.com/DevDock-AI/Speedy/blob/36e827fe2a3153efa451e5a4e8196eb478bb5ad2/src/kernels/cuda/iq_kernels.cu).

## Measured environment

| Item | Setting |
| --- | --- |
| GPU | GeForce RTX 3080, 12 GiB VRAM, SM86 |
| CPU and memory | Core Ultra 7 270K Plus, 32 GB DDR5, 15 Strata workers for API trials |
| OS and toolchain | CachyOS Linux, kernel 7.3.3-1-cachyos, NVIDIA 620.70.06, NVCC 13.4.92, GCC 16.2.1, Python 3.14 |
| PCIe | Engine measured 25.2 GB/s host to device and selected a 0.55 share in the headless run |
| API model | Qwen3.8-Flash-Next Gyro-S TQ1_0, K8V4 KV, one request at a time |
| Headless settings | 262,144-token context limit, 20,480 resident KV tokens, 448 MiB VRAM reserve, auto expert cache and prefill, prior 0.5 |
| Current 0.1.60 run | DFlash2 with spec 6 and confidence 0.8, fused TQ prefill enabled, 5,548 GPU expert slots and 20.82 GiB page-locked RAM experts in a representative paired run |

The synthetic CUDA timing ran in an idle KDE session. The API datasets include a separate earlier KDE run at 64K context and two headless comparisons. The timing passes ran after compilation stopped. The [example profile](profile.example.json) captures the current headless benchmark parameters with portable paths. Disk parking and V2/V3 support are present. Conversation snapshots and profile writeback were disabled in the benchmark, and each server start read the same frozen expert profile. Published artifacts contain aggregate timings, synthetic output checksums, and generic settings. The tested runtime source is commit `bcaf61f6` with tree `8dc44f9ec4aae1effdddaa63ddd951d3c6ea8eac`. The timing driver adds instrumentation to the benchmark executable only. The publication branch inherits the CI snapshot workflow from the fork's `main`.

## GPU kernel latency

The new [CUDA results](results/kernel.json) contain 16 fresh processes in two ABBA/BAAB blocks. Each process measured 200 individual CUDA event intervals per phase and shape after 64 warm-up calls. They cover TQ2_T and TQK6 gate/up against TQK7 down, 8/16/24/48 GPU-resident expert groups, and 1/2/4/8 entries per group plus a repeating mixed pattern with zero-entry groups. Hidden and expert dimensions are 2,560 and 640. Each row below shows the median of eight process-level p50 or p95 measurements. Full grouped expert timing includes gate/up, SwiGLU, quantization of the intermediate output, and down. Its input Q8_1 activations and expert blobs are already on the GPU.

| Gate/up type and shape | Phase | Single p50 (µs) | Paired p50 (µs) | Single p95 (µs) | Paired p95 (µs) |
| --- | --- | ---: | ---: | ---: | ---: |
| TQK6 8 × 1 | Gate/up | 27.66 | 28.67 | 28.67 | 29.18 |
| TQK6 8 × 1 | Full expert | 49.15 | 49.15 | 49.68 | 50.18 |
| TQK6 16 × 2 | Full expert | 96.77 | 84.42 | 97.71 | 85.52 |
| TQK6 24 × 2 | Gate/up | 87.94 | 66.16 | 88.10 | 66.56 |
| TQK6 24 × 2 | Full expert | 134.14 | 114.69 | 135.68 | 118.78 |
| TQK6 24 × 4 | Full expert | 174.08 | 151.55 | 175.62 | 153.60 |
| TQK6 48 × 4 | Full expert | 335.87 | 278.53 | 336.90 | 280.06 |

The grouped expert outputs passed the finite-value check and had matching hashes for all 40 fixtures across all trials. On 8 × 1 of TQK6 the full-expert p50 matched within the clock's measurement granularity. On 24 × 2 paired rows saved 18.84 µs in the median of adjacent pair-minus-single p50 differences. Each fixture uses the same weights, token indices, group shape and inputs on both paths. The microbenchmark has no CPU experts, host-to-device transfers within the timed call, model sampling or file reads. GPU clock and temperature were allowed to vary. The JSON retains per-process p50, p95 and p99 and the associated paired differences.

## Full model measurements

The [long-output headless runs](results/headless-long-output.json) used the same 0.1.60 engine, single → paired → paired → single, with `STRATA_VERIFY_PROFILE=1`. All four runs reached their output caps and reported zero decode expert file reads.

| Synthetic request | Single runs (token/s) | Paired runs (token/s) |
| --- | --- | --- |
| 3,072 output tokens after a short input | 90.3, 100.2 | 107.0, 100.8 |
| 1,536 output tokens after 28,750 input tokens | 84.0, 85.3 | 87.9, 97.8 |

The four response hashes differ for each request. DFlash2 produced 2.22 and 2.41 tokens per verify window for the long input in the single-row arms, and 2.51 and 3.12 in the paired arms. This changes both the amount of decoding work and the routed expert groups. Per-window GPU timings and request throughput therefore include workload variation. The next headless protocol uses the opposite order, two 3,072-token output workloads and the 28K input workload, with verifier stage profiling disabled. It records time to first content, output length and hashes, and loopback SSE content-gap percentiles. One SSE event can carry several model tokens. Full-model SSE results using this protocol are pending.

The [earlier headless runs](results/headless-drafters.json) used shorter outputs and include MTP4 and a one-run 0.1.60 candidate. On the 768-token output case, earlier-engine medians were 104.05 token/s for single-row DFlash2, 103.95 for paired DFlash2 and 97.5 for paired MTP4. The 0.1.60 candidate measured 105.2 token/s once. Its 28,707-token input took 10.92 seconds to prefill, versus 11.67 seconds for the earlier paired DFlash2 median. The [earlier KDE dataset](results/desktop-history.json) includes desktop results, a short quality smoke test and an 8 × 1 measurement with substantial drift. The historical binaries used different engine revisions from this branch. The first Gyro-M trial stopped during argument validation. Its residency and performance remain unmeasured here.

## Reproducing the comparisons

Build on CUDA with SM86. Finish the build before measuring.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=ON
cmake --build build --target tq_pascal_bench strata -j 4
python bench/ampere/kernel_latency.py build/tq_pascal_bench --output kernel-results.json
```

The CUDA benchmark needs no model files. For Chat Completions, install the Gyro-S GGUF, compatible pack and DFlash2 drafter using the [Gyro guide](../../docs/GYRO.md) and [source setup](../../docs/SOURCE_INSTALL.md). Use a fresh local copy of an expert profile, such as the shipped `data/expert-profile.bin`, for every process. The published API runs used a different frozen profile, so the example can reproduce the protocol and settings while routing and exact rates can vary. Set the paths in a copy of `profile.example.json`, create `bench-run/parked`, and keep the API on loopback. Confirm the logs show DFlash2 at spec 6/confidence 0.8 and page-locked expert RAM. Use a different log for each run.

Run the profile from the checkout root using `python strata.py serve --profile ./profile.json`. Set `STRATA_TQ_AMPERE_ROWS` in the profile env to `2`, `0`, `0`, `2` for four separate starts. For each ready process, run the matching command below, with a new output filename. Stop the process before starting the next arm. The paired engine logs `TQ Ampere paired-row GU enabled` when that CUDA path first executes.

```sh
python bench/ampere/chat_latency.py --mode pair --port 8095 --output pair-1.json
python bench/ampere/chat_latency.py --mode single --port 8095 --output single-1.json
```

Record the engine's decode expert file counter with each run. Compare runs using the actual output lengths, hashes, drafter acceptance and content-gap distributions. More memory or a different expert profile can change the split between CPU and GPU work.
