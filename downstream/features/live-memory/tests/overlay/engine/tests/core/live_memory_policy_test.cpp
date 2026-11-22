#include "strata/core/live_memory.hpp"

#include <cassert>
#include <cstdint>
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

// ---- PR4 Phase 3: the terminal ACK, the CAPACITY report, the one pending request slot ----

static void ack_line_reports_actual_capacity() {
    using strata::core::LiveMemoryAck;
    assert(strata::core::live_memory_ack_line(7, LiveMemoryAck::Ok, 4096, 3584, 1536, "")
           == "ACK 7 OK ram_resident_mib=4096 vram_committed_mib=3584 vram_reserve_mib=1536");
    assert(strata::core::live_memory_ack_line(9, LiveMemoryAck::Partial, 2048, 3584, 1536, "the_RAM_release_stopped_early")
           == "ACK 9 PARTIAL ram_resident_mib=2048 vram_committed_mib=3584 vram_reserve_mib=1536 "
              "note=the_RAM_release_stopped_early");
    assert(strata::core::live_memory_ack_line(3, LiveMemoryAck::Refused, 0, 4096, 1024, "")
           == "ACK 3 REFUSED ram_resident_mib=0 vram_committed_mib=4096 vram_reserve_mib=1024");
}

static void capacity_line_shape() {
    assert(strata::core::live_memory_capacity_line(1024, 8192, 3584, 6144, 1536)
           == "CAPACITY ram_resident_mib=1024 ram_available_mib=8192 vram_committed_mib=3584 "
              "vram_free_mib=6144 vram_reserve_mib=1536");
}

static void pending_waits_supersedes_and_refuses() {
    strata::core::LiveMemoryPending pending;
    assert(!pending.active());
    strata::core::LiveMemoryRequest taken;
    assert(!pending.take(taken));

    std::string why;
    assert(pending.push({1, 8192, 1024}, 8192ull << 20, why) && why.empty());
    assert(pending.active() && pending.request.id == 1);

    // a newer request that releases capacity in both tiers replaces the waiting one
    assert(pending.push({2, 4096, 1536}, 8192ull << 20, why) && why.empty() && pending.request.id == 2);
    // one that grows either tier is refused at once (the sender learns now, not after the effects)
    assert(!pending.push({3, 12288, 1536}, 8192ull << 20, why) && !why.empty());
    assert(!pending.push({3, 4096, 512}, 8192ull << 20, why) && !why.empty());
    assert(pending.request.id == 2);   // the waiting request is unchanged by a refusal

    assert(pending.take(taken) && taken.id == 2 && taken.resident_mib == 4096 && taken.vram_reserve_mib == 1536);
    assert(!pending.active() && !pending.take(taken));
}

int main() {
    parse_requests();
    superseding_only_releases();
    ram_completion_is_latched();
    growth_is_not_a_side_effect_of_pressure_relief();
    gpu_budget_respects_reserve_and_quantum();
    ack_line_reports_actual_capacity();
    capacity_line_shape();
    pending_waits_supersedes_and_refuses();
    return 0;
}
