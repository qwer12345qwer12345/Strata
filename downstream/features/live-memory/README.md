# Live Memory — initial VMM commit and policy helpers

## Scope

This package migrates the first Live Memory capacity-owner step:

- request grammar and capacity-policy helpers in `live_memory.hpp`;
- an opt-in initial committed prefix for the ExpertCache CUDA VMM arena;
- the `STRATA_LIVE_MEMORY_INITIAL_MIB` environment entry point;
- Settings scanner support for `std::from_chars`;
- CPU policy and CUDA segmented-cache regression tests.

## Activation

```text
STRATA_LIVE_MEMORY_INITIAL_MIB=<positive whole MiB>
```

Default: unset. When unset, ExpertCache retains its existing full-map behavior. The environment path chooses 32 MiB segments unless a caller already selected a segment size.

## Patch ownership

- `engine/include/strata/core/live_memory.hpp`
- `engine/include/strata/core/expert_cache.hpp`
- `engine/src/core/expert_cache.cpp`
- `tools/settings_register.py`
- Live Memory test registration in `engine/CMakeLists.txt`

The test sources and Settings unit test are stored under this package's test overlay.

## Scope boundary

This is not the complete dynamic Live Memory implementation. Runtime MEMORY/ACK/CAPACITY handling, transactional safe-point resizing, independently releasable RAM expert blocks and Server policy remain unimplemented. The feature stays opt-in.

The complete staged implementation plan, support boundary and hardware validation gates are preserved in [PORT_PLAN.md](design/PORT_PLAN.md). The earlier `feature/live-memory-cuda` branch remains retained as a separate historical research/implementation reference.

## Tests

See [tests/README.md](tests/README.md).
