# Live Memory — the acknowledged capacity owner

## Scope

This package migrates the Live Memory capacity-owner work in four steps (PR #1-#4):

- PR1: request grammar and capacity-policy helpers in `live_memory.hpp`; an opt-in initial committed prefix
  for the ExpertCache CUDA VMM arena; the `STRATA_LIVE_MEMORY_INITIAL_MIB` environment entry point; Settings
  scanner support for `std::from_chars`.
- PR2: transactional VMM grow/shrink (`resize_transactional`) with rollback, exact mapped-byte reporting,
  resident-tail protection, and driver-refusal fault injection.
- PR3: live-slot guards so admission and device pointers cannot escape the mapped prefix.
- PR4 Phase 2: **independently releasable RAM expert blocks** in `FileExpertSource` (`ram_expert_blocks.hpp`)
  - bounded blocks, allocated **pageable-first** with an optional **pinned** fallback per block, each growth
  step admitted against a fresh cgroup-aware reading (`sample_host_memory` + `conversation_memory_admit`),
  every failed step rolled back whole, and a release that publishes a block's entries out of reach BEFORE
  its memory is freed.
- PR4 Phase 3: the Engine's **MEMORY / ACK / CAPACITY protocol** - `MEMORY <id> <resident-mib>
  <vram-reserve-mib>` at the VRAM command's safe points (a request that arrives mid-conversation waits in a
  one-slot pending queue and is superseded only by one that releases capacity in both tiers), exactly one
  terminal ACK per request (OK / PARTIAL / REFUSED) reporting the ACTUAL capacity, and CAPACITY reports
  after every ACK/VRAM command, on demand, and periodically between requests.
- PR4 Phase 4: the Server's **opt-in automatic policy** (`serve/live_memory_policy.py`: fresh samples,
  hysteresis, debounce, cooldown with REFUSED backoff, bounded steps - growth never a side effect), a small
  memory API (`POST/GET /v1/memory`, `StrataEngine.request_memory()`), and **Vision riding the same
  serialized owner** (Phase D: an image job's reserve goes through `request_memory`, never a second owner).

## Activation

```text
STRATA_LIVE_MEMORY_INITIAL_MIB=<positive whole MiB>   # PR1: the startup committed prefix
STRATA_LIVE_MEMORY=1                                   # PR4: the MEMORY protocol + the RAM blocks tier
STRATA_LIVE_MEMORY_PIN_RAM=1                           # PR4: try page-locked blocks (pageable fallback)
STRATA_LIVE_MEMORY_MAX_BLOCK_MIB=<MiB, default 512>    # PR4: one growth step's cap
STRATA_LIVE_MEMORY_CAPACITY_S=<s, default 10>          # PR4: the between-requests CAPACITY cadence (0: demand)
STRATA_LIVE_MEMORY_POLICY=1                            # PR4: the server's automatic policy thread
```

Config (server): `"live_memory": true | {"pin_ram": .., "max_block_mib": .., "capacity_s": ..}` (implies
`vram_elastic`) and `"live_memory_policy": true | {tuning}`. Manual control: `POST /v1/memory`. Defaults
keep every existing behavior: without the switches the engine never takes or releases a block and never
resizes on its own.

## Patch ownership

- `engine/include/strata/core/live_memory.hpp` — grammar, policy helpers, ACK/CAPACITY lines, pending queue
- `engine/include/strata/core/ram_expert_blocks.hpp` — the block store and its pure policy helpers (PR4)
- `engine/include/strata/core/expert_source.hpp` / `engine/src/core/expert_source.cpp` — the blocks tier
- `engine/include/strata/core/expert_cache.hpp` / `engine/src/core/expert_cache.cpp` — the VMM side
- `engine/src/program/generate.cpp` — the MEMORY/ACK/CAPACITY handling and safe points (PR4)
- `engine/CMakeLists.txt`, `tools/settings_register.py` — test registration and the Settings scanner

The test sources and Settings unit test are stored under this package's test overlay; the server's policy
module and its tests live in the `terminal/serve` part of the same patches.

## Scope boundary

The stack stays opt-in until the hardware acceptance gates in [PORT_PLAN.md](design/PORT_PLAN.md) pass on
a Linux CUDA machine (the migration machine's CPU tests and an RTX 4060 build do not qualify the VMM and
safe-point paths). The earlier `feature/live-memory-cuda` branch remains retained as a separate historical
research/implementation reference.

## Tests

See [tests/README.md](tests/README.md) and [validation/README.md](validation/README.md).
