#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace strata::core {

// A single, acknowledged request to change the two expert-cache capacity limits.
// Values are whole MiB on the pipe; the native engine reports actual rounded capacity.
struct LiveMemoryRequest {
    uint64_t id = 0;
    uint64_t resident_mib = 0;
    uint64_t vram_reserve_mib = 0;
};

// A newer request may replace a pending request only when it releases capacity in both
// tiers. This prevents a pressure-relief request from being turned into a growth request.
inline bool live_memory_supersedes(const LiveMemoryRequest& next, const LiveMemoryRequest& old,
                                   uint64_t resident_bytes) {
    return next.id != old.id && next.resident_mib <= old.resident_mib &&
           next.resident_mib <= (resident_bytes >> 20) &&
           next.vram_reserve_mib >= old.vram_reserve_mib &&
           (next.resident_mib < old.resident_mib || next.vram_reserve_mib > old.vram_reserve_mib);
}

// Completion belongs to the whole request. A whole RAM block can take capacity below an
// arbitrary MiB target; once that shrink is complete, later GPU steps must not refill the gap.
template<class Resize>
bool live_memory_ram_step(bool& done, Resize resize) {
    if (done) return true;
    bool reached = false;
    if (!resize(reached)) return false;
    done = reached;
    return true;
}

// A RAM shrink or a higher VRAM reserve is pressure relief, not permission to grow the GPU
// cache with VRAM that becomes free as a side effect of releasing RAM.
inline bool live_memory_gpu_growth_allowed(const LiveMemoryRequest& request, uint64_t resident_bytes,
                                           uint64_t previous_reserve_mib) {
    return request.resident_mib >= (resident_bytes >> 20) &&
           request.vram_reserve_mib <= previous_reserve_mib;
}

// Calculate the target committed GPU bytes from a fresh free-memory sample. Growth leaves one
// extra quantum of margin for allocator/driver accounting; shrink does not add that margin.
inline uint64_t live_memory_gpu_budget(uint64_t free_bytes, uint64_t committed, uint64_t reserve,
                                       uint64_t quantum, bool allow_growth) {
    if (quantum == 0) return committed;
    const uint64_t available = free_bytes > std::numeric_limits<uint64_t>::max() - committed
        ? std::numeric_limits<uint64_t>::max() : free_bytes + committed;
    const uint64_t target = available > reserve ? (available - reserve) / quantum * quantum : 0;
    if (target <= committed) return target;
    if (!allow_growth) return committed;
    const uint64_t spare = free_bytes > reserve && free_bytes - reserve > quantum
        ? free_bytes - reserve - quantum : 0;
    const uint64_t growth = spare / quantum * quantum;
    return growth > std::numeric_limits<uint64_t>::max() - committed
        ? std::numeric_limits<uint64_t>::max() : committed + growth;
}

// Exact pipe grammar: MEMORY <positive-id> <resident-MiB> <VRAM-reserve-MiB>.
// Parse into a temporary so a malformed line never partially changes the caller's request.
inline bool parse_live_memory_request(const std::string& line, LiveMemoryRequest& out) {
    std::istringstream in(line);
    std::string verb, id, ram, vram, extra;
    if (!(in >> verb >> id >> ram >> vram) || verb != "MEMORY" || (in >> extra)) return false;
    auto number = [](const std::string& s, uint64_t& value) {
        if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
        const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
        return result.ec == std::errc{} && result.ptr == s.data() + s.size();
    };
    LiveMemoryRequest next;
    if (!number(id, next.id) || !number(ram, next.resident_mib) ||
        !number(vram, next.vram_reserve_mib) || next.id == 0 ||
        next.resident_mib > (1ull << 20) || next.vram_reserve_mib > (1ull << 20)) return false;
    out = next;
    return true;
}

} // namespace strata::core
