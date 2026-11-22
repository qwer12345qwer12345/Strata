// tests/core/live_ram_blocks_test.cpp - Live Memory PR4 Phase 2: the independently releasable RAM expert
// blocks (header-only, CPU-only): block step sizing, the fill plan, grow's publish-or-rollback on allocation
// and copy failures, the pinned-with-pageable-fallback allocator, and release's free-the-last-block-first
// order with its entries out of reach before the memory goes back.
#include "strata/core/ram_expert_blocks.hpp"

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

using strata::core::RamExpertBlocks;
using strata::core::live_ram_block_bytes;
using strata::core::live_ram_fill_plan;

namespace {

// Records every allocation and free; `fail_alloc_left` refuses that many allocations (the tests' allocation
// gate), and `freed` remembers the order the memory went back in.
struct RecordingAllocator final : RamExpertBlocks::Allocator {
    size_t live = 0;
    int fail_alloc_left = 0;
    bool fail_alloc_throw = false;
    std::vector<void*> freed_order;
    std::map<void*, size_t> sizes;

    void* alloc(size_t bytes, bool& pinned) override {
        if (fail_alloc_left > 0) {
            --fail_alloc_left;
            if (fail_alloc_throw) throw std::bad_alloc();
            pinned = false;
            return nullptr;
        }
        void* p = std::malloc(bytes);
        if (p != nullptr) {
            sizes[p] = bytes;
            ++live;
        }
        pinned = false;
        return p;
    }
    void free(void* p, bool pinned) noexcept override {
        (void) pinned;
        if (p == nullptr) return;
        sizes.erase(p);
        --live;
        freed_order.push_back(p);
    }
};

// 3 layers x 4 experts, layer 1's blobs twice as big: the plan walk and the block layout cover a native
// pack's differing sizes.
struct Fixture {
    static constexpr int64_t kLayers = 3, kExperts = 4;
    uint64_t blob_bytes(int64_t layer) const { return layer == 1 ? 32 : 16; }
    std::vector<std::pair<int32_t, int32_t>> rank;   // hottest first, filled by the tests
    RecordingAllocator pageable;
    RamExpertBlocks store{pageable, 0};
    Fixture() {
        store.set_blob_sizes(std::vector<uint64_t>{16, 32, 16}.data(), 3);
    }
    static size_t index(int64_t layer, int64_t expert) { return (size_t) layer * (size_t) kExperts + (size_t) expert; }
};

void block_step_is_capped() {
    assert(live_ram_block_bytes(0, 512) == 0);
    assert(live_ram_block_bytes(1024, 0) == 1024);      // uncapped when the caller gave no cap
    assert(live_ram_block_bytes(1024, 512) == 512);     // one bounded step, never more
    assert(live_ram_block_bytes(300, 512) == 300);      // under the cap: the request's own size
}

void fill_plan_skips_held_and_fits_the_budget() {
    const std::vector<std::pair<int32_t, int32_t>> rank = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 3}};
    Fixture f;
    // (0,0) is held somewhere else (the GPU cache, the complement); (1,0) does not fit what is left.
    auto held = [&](size_t layer, size_t expert) { return layer == 0 && expert == 0; };
    auto bytes = [&](int64_t layer, int64_t) { return f.blob_bytes(layer); };
    std::vector<std::pair<int32_t, int32_t>> out;
    // budget 16: only (0,1) fits (16); (1,0) needs 32.
    assert(live_ram_fill_plan(rank, 16, held, bytes, out) == 16);
    assert((out == std::vector<std::pair<int32_t, int32_t>>{{0, 1}}));
    // budget 64: (0,1) 16 + (1,0) 32 + (1,1) 32? No: 16+32 = 48, (1,1) does not fit the 16 left, but
    // (2,3) does (16) - a smaller blob behind a bigger one is not dropped (a native pack's sizes differ).
    assert(live_ram_fill_plan(rank, 64, held, bytes, out) == 16 + 32 + 16);
    assert((out == std::vector<std::pair<int32_t, int32_t>>{{0, 1}, {1, 0}, {2, 3}}));
    // budget 0: nothing.
    assert(live_ram_fill_plan(rank, 0, held, bytes, out) == 0 && out.empty());
}

void grow_publishes_and_lays_blobs_back_to_back() {
    Fixture f;
    const std::vector<std::pair<int32_t, int32_t>> rank = {{0, 0}, {1, 0}};
    auto held = [](size_t, size_t) { return false; };
    auto plan = [&](std::vector<std::pair<int32_t, int32_t>>& entries) {
        return live_ram_fill_plan(rank, 64, held,
                                  [&](int64_t layer, int64_t) { return f.blob_bytes(layer); }, entries);
    };
    auto filler = [&](uint8_t* base, const std::vector<std::pair<int32_t, int32_t>>& entries, uint64_t bytes) {
        assert(bytes == 48);   // 16 + 32
        uint64_t at = 0;
        for (const auto& [layer, expert] : entries) {
            std::fill(base + at, base + at + f.blob_bytes(layer), (uint8_t) (layer + 1));
            at += f.blob_bytes(layer);
        }
        return true;
    };
    uint64_t added = 0;
    bool pinned = true;   // the store reports what the allocator said, not what was hoped
    std::string err;
    assert(f.store.grow(64, plan, filler, Fixture::index, added, pinned, err));
    assert(added == 48 && pinned == false && err.empty());
    const auto s = f.store.stats();
    assert(s.blocks == 1 && s.entries == 2 && s.bytes == 48);
    assert(f.store.holds(Fixture::index(0, 0)) && f.store.holds(Fixture::index(1, 0)));
    assert(!f.store.holds(Fixture::index(2, 2)));
    const uint8_t* a = f.store.blob(Fixture::index(0, 0));
    const uint8_t* b = f.store.blob(Fixture::index(1, 0));
    assert(a != nullptr && b != nullptr && b == a + 16);   // back to back in plan order
    assert(a[0] == 1 && b[0] == 2);
}

void grow_rolls_back_on_allocation_and_copy_failures() {
    Fixture f;
    const std::vector<std::pair<int32_t, int32_t>> rank = {{0, 0}};
    auto held = [](size_t, size_t) { return false; };
    auto plan = [&](std::vector<std::pair<int32_t, int32_t>>& entries) {
        return live_ram_fill_plan(rank, 64, held,
                                  [&](int64_t layer, int64_t) { return f.blob_bytes(layer); }, entries);
    };
    auto filler = [&](uint8_t*, const std::vector<std::pair<int32_t, int32_t>>&, uint64_t) { return true; };
    uint64_t added = 7;
    bool pinned = false;
    std::string err;

    // The host refuses the block: nothing published, nothing freed twice, the store untouched.
    f.pageable.fail_alloc_left = 1;
    assert(!f.store.grow(64, plan, filler, Fixture::index, added, pinned, err));
    assert(added == 0 && !err.empty());
    assert(f.store.stats().blocks == 0 && f.store.stats().entries == 0);
    assert(f.pageable.live == 0 && f.pageable.freed_order.empty());

    // bad_alloc out of the allocator: the same rollback.
    f.pageable.fail_alloc_left = 1;
    f.pageable.fail_alloc_throw = true;
    assert(!f.store.grow(64, plan, filler, Fixture::index, added, pinned, err));
    assert(added == 0 && f.store.stats().blocks == 0 && f.pageable.live == 0);
    f.pageable.fail_alloc_throw = false;

    // The copy fails half way: the block is freed, no entry is reachable.
    auto bad_filler = [&](uint8_t*, const std::vector<std::pair<int32_t, int32_t>>&, uint64_t) { return false; };
    assert(!f.store.grow(64, plan, bad_filler, Fixture::index, added, pinned, err));
    assert(added == 0 && !f.store.holds(Fixture::index(0, 0)));
    assert(f.store.stats().blocks == 0 && f.pageable.live == 0 && f.pageable.freed_order.size() == 1);

    // A plan bigger than the step (the caller ignored the budget): refused before any copy.
    auto greedy_plan = [&](std::vector<std::pair<int32_t, int32_t>>& entries) {
        entries.assign({{0, 0}, {1, 0}});
        return 48;   // over the 32-byte step below
    };
    assert(!f.store.grow(32, greedy_plan, filler, Fixture::index, added, pinned, err));
    assert(added == 0 && f.store.stats().blocks == 0);
}

void release_frees_the_last_block_first_and_unpublishes_before_the_free() {
    RecordingAllocator alloc;
    RamExpertBlocks store(alloc, 0);
    store.set_blob_sizes(std::vector<uint64_t>{16, 32, 16}.data(), 3);
    auto held = [](size_t, size_t) { return false; };
    auto bytes = [&](int64_t layer) { return layer == 1 ? 32 : 16; };
    auto grow_one = [&](int32_t layer, int32_t expert, uint64_t budget) {
        const std::vector<std::pair<int32_t, int32_t>> rank = {{layer, expert}};
        auto plan = [&](std::vector<std::pair<int32_t, int32_t>>& entries) {
            return live_ram_fill_plan(rank, budget, held,
                                      [&](int64_t l, int64_t) { return bytes(l); }, entries);
        };
        auto filler = [&](uint8_t*, const std::vector<std::pair<int32_t, int32_t>>&, uint64_t) { return true; };
        uint64_t added = 0;
        bool pinned = false;
        std::string err;
        return store.grow(budget, plan, filler,
                          [](int64_t l, int64_t e) { return Fixture::index(l, e); }, added, pinned, err) &&
               added == bytes(layer);
    };
    assert(grow_one(0, 0, 16));   // block 0
    assert(grow_one(1, 0, 32));   // block 1
    assert(grow_one(2, 0, 16));   // block 2
    const auto s0 = store.stats();
    assert(s0.blocks == 3 && s0.bytes == 64);
    const uint8_t* mid = store.blob(Fixture::index(0, 0));

    // Drop 32: blocks go WHOLE, the last grown first - block 2 (16) is not enough, block 1 (32) goes too:
    // 48 bytes out, block 0 (the oldest) stays reachable.
    uint64_t dropped = 0, remaining = 0;
    std::string err;
    assert(store.release(32, dropped, err) && dropped == 48);
    remaining = store.stats().bytes;   // (FileExpertSource reports this as the release's remaining)
    assert(remaining == 16);
    assert(store.holds(Fixture::index(0, 0)));
    assert(!store.holds(Fixture::index(1, 0)) && !store.holds(Fixture::index(2, 0)));
    assert(store.blob(Fixture::index(2, 0)) == nullptr && store.blob(Fixture::index(1, 0)) == nullptr);
    assert(store.blob(Fixture::index(0, 0)) != nullptr);   // an earlier block's pointer still holds

    // Dropping more than is resident frees every block and leaves the tier empty.
    assert(store.release(4096, dropped, err) && dropped == 16);
    remaining = store.stats().bytes;
    assert(remaining == 0);
    assert(store.stats().blocks == 0 && store.stats().entries == 0 && store.stats().bytes == 0);
    assert(store.blob(Fixture::index(1, 0)) == nullptr);
    // every block's memory went back exactly once, entries out of reach first
    assert(alloc.freed_order.size() == 3);
    assert(alloc.live == 0);

    // A release on an empty tier is a no-op that succeeds.
    assert(store.release(1, dropped, err) && dropped == 0);
}

void pinned_fallback_reported_never_assumed() {
    RecordingAllocator pageable;
    static void* last_pin = nullptr;
    struct PinCount {
        int ok = 0, refused = 0, freed = 0;
    };
    static PinCount counts;
    counts = PinCount{};
    auto pin_alloc = [](size_t bytes) -> void* {
        if (counts.ok >= 1) { ++counts.refused; return nullptr; }   // the driver refuses after the first
        void* p = std::malloc(bytes);
        if (p != nullptr) { last_pin = p; ++counts.ok; }
        return p;
    };
    auto pin_free = [](void* p) noexcept {
        if (p == last_pin) { ++counts.freed; last_pin = nullptr; }
        std::free(p);
    };
    RamExpertBlocks::PinnedFallbackAllocator pinned(pageable, pin_alloc, pin_free);
    RamExpertBlocks store(pinned, 0);
    store.set_blob_sizes(std::vector<uint64_t>{16}.data(), 1);
    auto held = [](size_t, size_t) { return false; };
    auto plan = [&](std::vector<std::pair<int32_t, int32_t>>& entries) {
        const std::vector<std::pair<int32_t, int32_t>> rank = {{0, 0}};
        return live_ram_fill_plan(rank, 32, held, [](int64_t, int64_t) { return 16; }, entries);
    };
    auto filler = [&](uint8_t*, const std::vector<std::pair<int32_t, int32_t>>&, uint64_t) { return true; };
    uint64_t added = 0;
    bool was_pinned = false;
    std::string err;
    // The first block comes back page-locked and is reported so.
    assert(store.grow(32, plan, filler, Fixture::index, added, was_pinned, err));
    assert(was_pinned && added == 16 && store.stats().pinned_bytes == 16);
    // The second is refused and falls back to pageable: still grows, reported as not pinned.  A different
    // flat index (expert 1): a store's index names one resident blob and the two blocks are disjoint.
    assert(store.grow(32, plan, filler,
                      [](int64_t l, int64_t e) { return (size_t) l * 16 + (size_t) e + 1; }, added, was_pinned, err));
    assert(!was_pinned && added == 16 && store.stats().pinned_bytes == 16);
    // The release routes each block to its own free (the pinned one through cudaFreeHost's stand-in).
    uint64_t dropped = 0;
    assert(store.release(4096, dropped, err) && dropped == 32 && counts.freed == 1);
}

}  // namespace

int main() {
    block_step_is_capped();
    fill_plan_skips_held_and_fits_the_budget();
    grow_publishes_and_lays_blobs_back_to_back();
    grow_rolls_back_on_allocation_and_copy_failures();
    release_frees_the_last_block_first_and_unpublishes_before_the_free();
    pinned_fallback_reported_never_assumed();
    return 0;
}
