# Live Memory implementation plan

Status: the request grammar and CPU-only policy tests are in place. ExpertCache supports an opt-in partially committed VMM prefix at startup and now has `resize_transactional()` with rollback on partial VMM failures, a resident-tail guard, and CUDA fault-injection coverage. Engine MEMORY/ACK/CAPACITY handling, independently releasable RAM blocks, and Server policy are still not implemented. This branch is opt-in and does not change the default allocator.

Baseline: `Niko1221/Strata/main` at `6e38f588a646c0ee6948a74afbe670eb75896106` (0.2.18 / Engine 0.1.64). This implementation is being built against the current Engine interfaces, not by merging the older reference branch.

## 1. Initial support boundary

First acceptance target: Linux, one NVIDIA CUDA GPU, one Engine process, static native expert profile and mmap-backed expert source. Keep HIP, multi-GPU/layer splits, peer/remote caches, batch serving, `--vram-elastic`, `--adapt-async`, and concurrent Vision reserve changes out of the first implementation.

The feature remains opt-in until the full engine/server protocol and hardware acceptance tests pass.

## 2. Upstream facilities to reuse

- `engine/src/core/expert_cache.cpp`: `ExpertCache::open_segmented`, `shrink` and `grow` use CUDA VMM. The new initial-prefix option reserves the full VA range but maps only the requested prefix. The default still maps every segment.
- `engine/src/core/conversation_memory.cpp`: `sample_host_memory()`, cgroup v1/v2 accounting, reclaimable clean file cache and operator memory limits are already more complete than the older reference sampler. Reuse them for every RAM growth admission.
- `engine/include/strata/core/conversation_cache.hpp`: the RAM cache already exposes an eviction sink. Do not replace this cache or bypass its guarded/pinned-prefix rules.
- `engine/include/strata/core/conversation_spill.hpp`: upstream has a process-scoped SSD spill tier with a writer thread. It is temporary across process lifetimes; it is not the persistent Live Memory owner.
- The Server already has a `VRAM` control path and an on-demand Vision lifecycle. There is no acknowledged `MEMORY` protocol or automatic RAM/VRAM policy yet.

## 3. Policy contract already added

Header: `engine/include/strata/core/live_memory.hpp`

Initial VRAM commit:
- call `ExpertCache::set_initial_segment_bytes(bytes)` before `open/open_sized`, or set `STRATA_LIVE_MEMORY_INITIAL_MIB=<positive MiB>`;
- the environment path uses 32 MiB segments unless the caller already selected a segment size;
- only the initial prefix is mapped and zeroed; the full address range remains reserved;
- the ordinary path is unchanged when the option is absent;
- devices without CUDA VMM are refused rather than silently allocating the full arena.

Control request grammar:

```text
MEMORY <positive-id> <resident-MiB> <VRAM-reserve-MiB>
```

The helper layer currently covers:
- strict parsing into a temporary value (malformed requests cannot partially update state);
- superseding only when the new request releases capacity in both tiers;
- latching RAM-shrink completion for the whole request;
- preventing RAM pressure relief from implicitly authorizing GPU growth;
- quantized GPU budget calculation with an extra growth margin.

CPU-only test: `engine/tests/core/live_memory_policy_test.cpp`, registered as `live_memory_policy_test`.

The VMM prefix behavior is covered by `engine/tests/core/expert_cache_segmented_test.cpp`, now registered as `expert_cache_segmented_test` in CUDA test builds.

These helpers are not a runtime implementation. They do not allocate, resize, acknowledge, or report actual capacity.

## 4. Native implementation sequence

### Phase A — capacity owner

1. Add a Live VMM open path that reserves the profile-sized address range but maps only the active prefix. The first version is in place; next, connect it to the Engine's capacity owner and make runtime changes safe at reader/graph boundaries.
2. **Implemented at the ExpertCache primitive level:** `resize_transactional()` checks resident tail slots, synchronizes GPU work, attempts grow/shrink, and rolls partial driver progress back to the previous mapped prefix. If rollback itself fails, it reports actual mapped bytes. The caller must still drain host readers and prompt loans before calling it; Engine safe-point integration remains Phase B.
3. Add independently releasable RAM expert blocks to `FileExpertSource`, with pageable-first allocation and an optional pinned fallback. Remove every expert pointer into a block before freeing it.
4. Reuse the current cgroup-aware `sample_host_memory()` and `conversation_memory_admit()` logic for each RAM growth step.
5. **VMM map/unmap/release fault injection is implemented** in the segmented-cache test. RAM allocation, copy failure and RAM-block rollback injection remain outstanding.

### Phase B — protocol and safe points

1. Add Engine handling for `MEMORY` requests with request IDs and terminal ACK states.
2. Add a periodic `CAPACITY` report containing actual RAM resident bytes, actual VRAM committed bytes, reserve, and a fresh cgroup-aware available-memory reading.
3. Apply changes only after outstanding expert readers, lookahead, staged copies and GPU work have drained.
4. Queue capacity changes during guarded prefill borrowing and apply them only after the prompt/refill boundary. Preserve the minimum cache floor.
5. If the requested reserve cannot be reached without crossing that floor, report the limitation and actual capacity instead of false success.

### Phase C — Server policy

1. Add `Engine.request_memory()` and route MEMORY ACKs and CAPACITY telemetry away from token/batch output channels.
2. Add conservative, opt-in pressure policy: fresh samples, debounce, slower growth, cooldown and bounded steps. The Engine remains the final admission authority.
3. Expose manual targets and actual status through a small memory API. Do not enable automatic resizing by default.
4. Do not run Live Memory and `--vram-elastic` as concurrent cache owners.

### Phase D — Vision shares the same owner

Vision start/idle-stop must request and release its VRAM reserve through the same serialized MEMORY owner. Until that is implemented and tested, reject or clearly disable the incompatible combination rather than implying a reserve was obtained.

## 5. Validation gates

- Build Linux CUDA with `sm_86` under the supported CUDA toolchains.
- At runtime, query `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`; do not infer VMM support from the GPU name.
- Test pageable and pinned RAM blocks independently.
- Inject failures into every map/unmap/release/allocation/copy step and verify rollback plus truthful capacity reporting.
- Verify the Engine PID, model, KV cache and conversation state remain alive across resize.
- Verify a subsequent turn reuses the correct KV prefix and guarded prefill borrowing still works.
- Verify Vision reserve acquisition/release is serialized with policy requests.
- Record driver/toolkit, GPU, model/profile/pack format, `MemAvailable`, cgroup limits and actual capacity with every hardware run.

The earlier reference measurements were collected on a different Windows GPU/RAM configuration and are design evidence only, not performance expectations for this port.
