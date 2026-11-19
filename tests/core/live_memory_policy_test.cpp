#include "strata/core/live_memory.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

using strata::core::LiveMemoryRequest;

static void parse_requests() {
    LiveMemoryRequest request{9, 8, 7};
    assert(strata::core::parse_live_memory_request("MEMORY 1 8192 1536", request));
    assert(request.id == 1 && request.resident_mib == 8192 && request.vram_reserve_mib == 1536);

    const LiveMemoryRequest before = request;
    for (const char* bad : {
             "VRAM 1", "MEMORY", "MEMORY 0 1 1", "MEMORY -1 1 1",
             "MEMORY 1 -1 1", "MEMORY 1 1 -1", "MEMORY 1 1 1 extra",
             "MEMORY 18446744073709551616 1 1", "MEMORY 1 1048577 1",
             "MEMORY 1 1 1048577"}) {
        assert(!strata::core::parse_live_memory_request(bad, request));
        assert(request.id == before.id && request.resident_mib == before.resident_mib &&
               request.vram_reserve_mib == before.vram_reserve_mib);
    }
}

static void superseding_only_releases() {
    const LiveMemoryRequest old{1, 8192, 1024};
    assert(strata::core::live_memory_supersedes({2, 4096, 1536}, old, 8192ull << 20));
    assert(!strata::core::live_memory_supersedes({2, 12288, 1536}, old, 8192ull << 20));
    assert(!strata::core::live_memory_supersedes({2, 4096, 512}, old, 8192ull << 20));
    assert(!strata::core::live_memory_supersedes({1, 4096, 1536}, old, 8192ull << 20));
    assert(!strata::core::live_memory_supersedes({2, 4096, 1536}, old, 2048ull << 20));
}

static void ram_completion_is_latched() {
    bool done = false;
    int calls = 0;
    assert(strata::core::live_memory_ram_step(done, [&](bool& reached) {
        ++calls;
        reached = true;
        return true;
    }));
    assert(done && calls == 1);
    assert(strata::core::live_memory_ram_step(done, [&](bool&) {
        ++calls;
        return true;
    }));
    assert(calls == 1);

    bool failed_done = false;
    assert(!strata::core::live_memory_ram_step(failed_done, [](bool&) { return false; }));
    assert(!failed_done);
}

static void growth_is_not_a_side_effect_of_pressure_relief() {
    const uint64_t resident = 4096ull << 20;
    assert(strata::core::live_memory_gpu_growth_allowed({1, 4096, 1024}, resident, 1536));
    assert(!strata::core::live_memory_gpu_growth_allowed({2, 2048, 1024}, resident, 1536));
    assert(!strata::core::live_memory_gpu_growth_allowed({3, 4096, 2048}, resident, 1536));
}

static void gpu_budget_respects_reserve_and_quantum() {
    constexpr uint64_t mib = 1ull << 20;
    constexpr uint64_t q = 128 * mib;
    assert(strata::core::live_memory_gpu_budget(512 * mib, 4096 * mib, 1536 * mib, q, true)
           == 3072 * mib);
    assert(strata::core::live_memory_gpu_budget(2048 * mib, 4096 * mib, 1024 * mib, q, false)
           == 4096 * mib);
    assert(strata::core::live_memory_gpu_budget(2048 * mib, 4096 * mib, 1024 * mib, q, true)
           == 4992 * mib);
    assert(strata::core::live_memory_gpu_budget(512 * mib, 2048 * mib, 4096 * mib, q, true) == 0);
    assert(strata::core::live_memory_gpu_budget(2048 * mib, 4096 * mib, 1024 * mib, 0, true)
           == 4096 * mib);
}

int main() {
    parse_requests();
    superseding_only_releases();
    ram_completion_is_latched();
    growth_is_not_a_side_effect_of_pressure_relief();
    gpu_budget_respects_reserve_and_quantum();
    return 0;
}
