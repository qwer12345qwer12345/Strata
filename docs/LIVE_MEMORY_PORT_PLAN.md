# Live Memory port plan (Linux, single NVIDIA GPU)

Status: research/design note. The pure C++ request/budget helpers and their CPU-only unit test are now committed; no engine or server runtime behavior has changed yet.

Target branch: `feature/live-memory-cuda`, created from this repository's `main` at `c77572806996349b49995e4f98d41b5294d36a21` (Strata 0.2.15 / engine 0.1.60).

## 1. Scope and baseline decision

Implement an opt-in Live Memory mode for one CUDA GPU. The initial acceptance machine is Linux, RTX 3080 12 GB, 32 GB system RAM.

Do not rebase this work onto upstream `Niko1221/Strata/main` as part of this feature. At the time of research, that branch compared as 728 commits ahead of this fork's base. The live-memory reference branch is also based on an older integration point. Port the small set of required changes onto this repository's current interfaces instead of importing either branch wholesale.

GitHub Issues are disabled in this repository, so this document records the proposed scope for review before the engine changes begin.

## 2. What this repository already has

- `ExpertCache::open_segmented` in `src/core/expert_cache.cpp` reserves a CUDA VMM virtual-address range, but maps every segment during startup. Its existing `shrink()` and `grow()` operations can leave a partial prefix after a driver refusal. This is suitable for the current request-boundary `--vram-elastic` feature, but it is not the transactional, partially committed arena required by Live Memory.
- `FileExpertSource::pin_cache_complement` in `src/core/expert_source.cpp` builds a startup-sized RAM complement. It is not a collection of independently releasable blocks.
- `--resident-budget-gib` already enables mmap-backed, profile-ranked RAM residency. It remains a startup budget today.
- The current `conversation_memory` implementation is more advanced than the older reference branch: `HostMemoryReading` accounts for `MemAvailable`, cgroup v1/v2 limits and usage, reclaimable clean file cache, and the operator cap. Reuse this implementation for admission checks; do not replace it with the older memory sampler from the reference branch.
- The server has a `VRAM` control path and on-demand Vision lifecycle, but no `MEMORY` acknowledgement protocol or automatic RAM/VRAM policy yet. Vision currently asks for VRAM through the existing `VRAM` command.
- Linux CUDA CI already compiles with CUDA 12.9 and 13.0 and includes `sm_86` in its architecture matrix. The `cuda-compile` preset has tests disabled, and these CI jobs do not execute on an NVIDIA GPU.

## 3. Reference implementation findings

The main reference is [PR #1493](https://github.com/Niko1221/Strata/pull/1493), still a Draft, and its [LIVE_MEMORY design and test report](https://github.com/midhatn/Strata/blob/codex/adaptive-background-draft/docs/LIVE_MEMORY.md). [PR #726](https://github.com/Niko1221/Strata/pull/726) is the earlier implementation; its closure was associated with the upstream history reset, not a technical rejection.

The useful design pieces are:

1. Reserve the full expert-cache virtual address range, but map only the active prefix. Use independently allocated VMM blocks (the reference uses 32 MiB blocks) so startup does not have to commit the full expert cache.
2. Make resize transactional. On a failed shrink, restore mappings already removed; on failed growth, release only the new blocks. Do not report the requested capacity as the actual capacity.
3. Store RAM experts in independent 32 MiB blocks. A block can be page-locked with CUDA host allocation or fall back to pageable memory. Freeing a block must remove all expert pointers that refer to it before the block is released.
4. Use a request-ID protocol, `MEMORY <id> <resident_mib> <vram_reserve_mib>`, with explicit terminal acknowledgements and periodic `CAPACITY` readings. Apply changes only after outstanding expert readers, lookahead, staged copies and GPU work have drained.
5. Keep the process, model, KV cache and conversation state alive. Preserve the guarded prefill-borrowing path and its minimum retained cache floor; a control received during borrowed prefill must wait for the completed prompt/refill boundary.

The reference uses 32 MiB RAM/VMM blocks, 128 MiB GPU capacity steps and a bounded service loop. These are reasonable initial quanta for the 12 GB target, subject to measurement.

### Important performance caveat

The reference report contains real-model Windows measurements on an RTX 5080 16 GB with 64 GB RAM, not on Linux or an RTX 3080. Its first separate-prompt-buffer experiment regressed materially (mean decode 35.13 to 22.79 tokens/s; prefill 1,021.55 to 318.31 tokens/s). The later guarded-borrowing implementation measured 44.52 versus 45.87 decode tokens/s and 1,310.17 versus 1,327.36 prefill tokens/s in two-request samples. The report explicitly does not claim a general speedup.

Therefore, do not ship a default path that silently disables prompt borrowing. Port and test the guarded borrowing/floor behavior, and treat the reference measurements as design evidence only—not as performance expectations for the 3080.

## 4. Proposed implementation sequence

### Phase A — native capacity owner

- Add a Live VMM open path that reserves the full profile-sized address range and initially maps only the active prefix. Reuse this repository's CUDA driver-entry-point wrapper instead of copying the reference's direct `cuMem*` linkage.
- Add atomic live resize and explicit accounting for mapped and physically committed bytes. Keep the current segmented `--vram-elastic` behavior unchanged.
- Add independently releasable RAM expert blocks to `FileExpertSource`, with pinned/pageable fallback and failure-safe pointer ownership.
- Reuse `sample_host_memory()` / `available_host_bytes()` and `conversation_memory_admit()` for every RAM growth step. The native engine must be the final admission authority even if the server policy has stale telemetry.
- Add the native `MEMORY` request/ack protocol and a `CAPACITY` heartbeat that reports actual GPU/RAM capacity and a fresh cgroup-aware host-available reading.

### Phase B — safe-point and prefill integration

- Drain every asynchronous reader/copy before changing a RAM block or VMM mapping.
- Retain guarded prefill borrowing. Queue capacity changes during a borrowed prompt and apply them only after its refill boundary.
- Keep the minimum cache floor explicit. If the requested reserve cannot be reached without crossing that floor, return an honest limitation/error with actual capacity.
- Add fault injection for VMM map/unmap/release, pinned allocation, pageable allocation, copy failure and rollback.

### Phase C — server policy and controls

- Add an `Engine.request_memory()` path that routes `MEMORY` acknowledgements and `CAPACITY` telemetry away from token/batch output channels.
- Add a conservative, opt-in policy with fresh-sample requirements, pressure debounce, slower growth, cooldown and bounded steps. Prefer the engine's cgroup-aware host-available reading over host-wide `psutil` figures.
- Expose status and manual target changes through a small memory API. Do not enable automatic resizing by default.
- Do not run Live Memory and `--vram-elastic` as concurrent cache owners.

### Phase D — Vision shares the same owner

The reference server's on-demand Vision path still calls `Engine.vram()`, which sends `VRAM` and requires `--vram-elastic`. It does not coordinate Vision's reserve with Live Memory. In this port, Vision start/idle-stop must request/release its reserve through the same `MEMORY` owner, serialized with policy requests. Until that is implemented and tested, reject or clearly disable the incompatible combination rather than implying that VRAM was reserved.

## 5. Initial support boundary

The reference's initial boundary is deliberately narrow: single-GPU CUDA serving, mmap-backed native expert pack, static expert profile and graph residency. It excludes HIP, multi-GPU/layer splits, peer/remote caches, per-layer admission, batch serving, `--vram-elastic`, `--adapt-async`, diagnostic dumps and positive `STRATA_FS_SLOTS`.

Keep this boundary for the first port. Broader backends and serving modes need separate designs and tests.

## 6. RTX 3080 / 32 GB validation gates

1. Build the Linux CUDA target with `sm_86` under the repository's CUDA 12.9 and 13.0 toolchains.
2. Run the native live-memory test on the actual card. At startup, check the driver's `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`; do not infer runtime VMM support from the GPU name alone.
3. Test both pageable and pinned RAM blocks. On Linux, pinned host allocations are not constrained in the same way as Windows WDDM, but pinned pages are still non-reclaimable; start with pageable memory and enable pinning only after measuring it.
4. Use a conservative initial host headroom of 6–8 GiB on a 32 GB machine and at least a 1.5 GiB VRAM reserve as a test envelope, not as a permanent default. Keep the resident budget bounded until the actual model and other processes are known.
5. Real-model acceptance must verify: same engine PID across resize; successful shrink and regrowth; no invalid device access; no lost KV prefix on the next turn; no false-success acknowledgement; no OOM/cgroup kill; and correct Vision reserve restoration.
6. Record `nvidia-smi`, driver/toolkit versions, `MemAvailable`, cgroup limits (if any), and the actual model/profile/pack format with every hardware run.

## 7. Required model compatibility check

The reference Live Memory mode requires a **native expert pack** in addition to mmap-backed experts and a static expert profile. The ordinary resident-budget path can also read GGUF shards, but that does not make those packs eligible for the reference Live Memory implementation. Confirm the model's pack format before treating a real-model run as an acceptance test.
