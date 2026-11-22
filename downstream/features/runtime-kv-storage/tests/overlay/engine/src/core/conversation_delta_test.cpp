// Pure-host tests for the immutable Base + independent Delta file format.
#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}

void set_buffer(ConversationBuffer& b, const std::vector<uint8_t>& bytes) {
    b.resize(bytes.size());
    if (!bytes.empty()) {
        const bool ok = b.visit(0, bytes.size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, bytes.data() + at, n);
            return true;
        });
        if (!ok) std::abort();
    }
}

std::vector<uint8_t> get_buffer(const ConversationBuffer& b) {
    std::vector<uint8_t> out(b.size());
    if (!out.empty() && !b.read(out.data(), 0, out.size())) std::abort();
    return out;
}

void append_buffer(ConversationBuffer& b, const std::vector<uint8_t>& tail) {
    const size_t at = b.size();
    b.resize_for_overwrite(at + tail.size());
    if (!tail.empty()) {
        const bool ok = b.visit(at, tail.size(), [&](uint8_t* p, size_t n, size_t offset) {
            std::memcpy(p, tail.data() + offset - at, n);
            return true;
        });
        if (!ok) std::abort();
    }
}

SavedConversation image(uint8_t seed) {
    SavedConversation s;
    s.geometry[0] = 123;
    s.layer_lo = 0;
    s.layer_hi = 1;
    s.cvec = true;
    s.live.ids = {1, 2, 3};
    s.live.gdn = {seed, uint8_t(seed + 1), uint8_t(seed + 2)};
    s.live.ple = {uint8_t(seed + 3)};
    ConversationCheckpoint cp;
    cp.ids = {1, 2};
    cp.gdn = {uint8_t(seed + 4), uint8_t(seed + 5)};
    cp.used = 17;
    s.checkpoints.push_back(std::move(cp));

    s.kv.resize(1);
    auto& k = s.kv[0];
    k.format = 16;
    k.cells = 4;
    k.heads = 2;
    k.head_dim = 4;
    k.page_size = 4;
    k.pooled_rows = 1;
    k.idx_dim = 2;
    set_buffer(k.k, {1, 2, 3, 4});
    set_buffer(k.v, {5, 6, 7, 8, 9});
    set_buffer(k.k_scale, {10, 11});
    set_buffer(k.v_scale, {12, 13, 14});
    set_buffer(k.pooled, {15, 16, 17, 18});
    return s;
}

bool same_image(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi ||
        a.cvec != b.cvec || a.live.ids != b.live.ids || a.live.imgs.size() != b.live.imgs.size() ||
        a.live.gdn != b.live.gdn || a.live.ple != b.live.ple || a.live.tails != b.live.tails ||
        a.live.dead != b.live.dead || a.live.block_pos != b.live.block_pos ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size()) return false;
    if (!a.checkpoints.empty()) {
        const auto& x = a.checkpoints[0];
        const auto& y = b.checkpoints[0];
        if (x.ids != y.ids || x.gdn != y.gdn || x.ple != y.ple || x.tails != y.tails ||
            x.dead != y.dead || x.block_pos != y.block_pos || x.used != y.used) return false;
    }
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i];
        const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            get_buffer(x.k) != get_buffer(y.k) || get_buffer(x.v) != get_buffer(y.v) ||
            get_buffer(x.k_scale) != get_buffer(y.k_scale) || get_buffer(x.v_scale) != get_buffer(y.v_scale) ||
            get_buffer(x.pooled) != get_buffer(y.pooled)) return false;
    }
    return true;
}

SessionKvSource tails_for(const ConversationKv& target, const std::array<uint64_t, 5>& keep) {
    const std::array<const ConversationBuffer*, 5> parts =
        {&target.k, &target.v, &target.k_scale, &target.v_scale, &target.pooled};
    auto data = std::make_shared<std::array<std::vector<uint8_t>, 5>>();
    SessionKvSource source;
    source.format = target.format;
    source.cells = target.cells;
    source.heads = target.heads;
    source.head_dim = target.head_dim;
    source.page_size = target.page_size;
    source.pooled_rows = target.pooled_rows;
    source.idx_dim = target.idx_dim;
    for (size_t i = 0; i < 5; ++i) {
        const auto bytes = get_buffer(*parts[i]);
        if (keep[i] > bytes.size()) std::abort();
        (*data)[i].assign(bytes.begin() + (size_t) keep[i], bytes.end());
        source.sizes[i] = (*data)[i].size();
    }
    source.read = [data](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > (*data)[part].size() || n > (*data)[part].size() - offset) return false;
        if (n) std::memcpy(dst, (*data)[part].data() + offset, n);
        return true;
    };
    return source;
}

bool write_delta(const fs::path& path, const SavedConversation& target,
                 const std::array<uint64_t, 5>& keep, const SessionFileIdentity& id,
                 uint64_t chain, uint64_t parent, uint64_t generation, std::string& error) {
    SavedConversation meta = target;
    meta.kv.clear();
    std::vector<SessionKvSource> tails;
    tails.push_back(tails_for(target.kv[0], keep));
    SessionDeltaKeep table{keep};
    size_t bytes = 0;
    return session_delta_write(path.string(), meta, tails, table, id, chain, parent, generation,
                               bytes, error);
}
} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "strata-session-delta-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    const fs::path base_path = root / "base.session";
    const fs::path d1_path = root / "delta-1";
    const fs::path d2_path = root / "delta-2";
    const fs::path d3_path = root / "delta-3";
    const fs::path bad_path = root / "delta-bad";
    const SessionFileIdentity id{101, 202};
    constexpr uint64_t chain = 0x123456789abcdef0ull;
    std::string error;
    size_t bytes = 0;

    SavedConversation base = image(20);
    SessionStatus status;
    check(session_file_write(base_path.string(), base, id, bytes, error, {}, &status), "write complete Base");
    SavedConversation restored;
    check(session_file_read(base_path.string(), id, restored, bytes, error, {}, &status), "read complete Base");

    // Delta 1: append to all five official KV parts while replacing the complete non-KV state.
    SavedConversation target1 = base;
    target1.live.ids.push_back(4);
    target1.live.gdn.push_back(99);
    append_buffer(target1.kv[0].k, {30, 31});
    append_buffer(target1.kv[0].v, {32});
    append_buffer(target1.kv[0].k_scale, {33, 34});
    append_buffer(target1.kv[0].v_scale, {35});
    append_buffer(target1.kv[0].pooled, {36, 37});
    target1.kv[0].cells = 8;
    target1.kv[0].pooled_rows = 2;
    const std::array<uint64_t, 5> keep1 = {4, 5, 2, 3, 4};
    check(write_delta(d1_path, target1, keep1, id, chain, 0, 1, error), "write Delta generation 1");
    uint64_t generation = 0;
    check(session_delta_apply(d1_path.string(), id, chain, 0, restored, generation, bytes, error, {}, &status) &&
          generation == 1 && same_image(restored, target1), "Base + Delta 1 materializes byte-for-byte");

    // Delta 2: another append, chained to the exact first generation.
    SavedConversation target2 = target1;
    target2.live.ids.push_back(5);
    target2.live.gdn.push_back(100);
    append_buffer(target2.kv[0].k, {40});
    append_buffer(target2.kv[0].v, {41, 42});
    append_buffer(target2.kv[0].k_scale, {43});
    append_buffer(target2.kv[0].v_scale, {44, 45});
    append_buffer(target2.kv[0].pooled, {46});
    target2.kv[0].cells = 12;
    target2.kv[0].pooled_rows = 3;
    const std::array<uint64_t, 5> keep2 = {6, 6, 4, 4, 6};
    check(write_delta(d2_path, target2, keep2, id, chain, 1, 2, error), "write Delta generation 2");
    check(session_delta_apply(d2_path.string(), id, chain, 1, restored, generation, bytes, error, {}, &status) &&
          generation == 2 && same_image(restored, target2), "Base + Delta 1 + Delta 2 materializes byte-for-byte");

    // Delta 3: rewind/replacement uses keep=0 and carries each complete target part.
    SavedConversation target3 = image(70);
    target3.live.ids = {7, 8};
    target3.live.gdn = {71, 72, 73, 74};
    target3.kv[0].cells = 4;
    target3.kv[0].pooled_rows = 1;
    const std::array<uint64_t, 5> keep3 = {0, 0, 0, 0, 0};
    check(write_delta(d3_path, target3, keep3, id, chain, 2, 3, error), "write full-replacement Delta");
    check(session_delta_apply(d3_path.string(), id, chain, 2, restored, generation, bytes, error, {}, &status) &&
          generation == 3 && same_image(restored, target3), "keep=0 replaces all five parts and restores rewind");

    // Parent and chain mismatches are rejected before the in-progress Base image is changed.
    SavedConversation unchanged = restored;
    check(!session_delta_apply(d3_path.string(), id, chain, 1, restored, generation, bytes, error, {}, &status) &&
          same_image(restored, unchanged), "wrong parent generation is rejected without mutation");
    check(!session_delta_apply(d3_path.string(), SessionFileIdentity{101, 203}, chain, 2,
                               restored, generation, bytes, error, {}, &status) &&
          same_image(restored, unchanged), "wrong model/config identity is rejected without mutation");
    check(!session_delta_apply(d3_path.string(), id, chain + 1, 2,
                               restored, generation, bytes, error, {}, &status) &&
          same_image(restored, unchanged), "wrong chain ID is rejected without mutation");

    // A keep length outside the parent part is refused before any part is modified.
    SavedConversation bad_target = target3;
    SavedConversation bad_meta = bad_target;
    bad_meta.kv.clear();
    const std::array<uint64_t, 5> full_keep_zero = {0, 0, 0, 0, 0};
    std::vector<SessionKvSource> bad_tails{tails_for(bad_target.kv[0], full_keep_zero)};
    SessionDeltaKeep bad_table{{UINT64_MAX, 0, 0, 0, 0}};
    check(session_delta_write(bad_path.string(), bad_meta, bad_tails, bad_table, id, chain, 3, 4,
                              bytes, error), "write malformed keep-table fixture");
    check(!session_delta_apply(bad_path.string(), id, chain, 3, restored, generation, bytes, error, {}, &status) &&
          same_image(restored, unchanged), "out-of-range keep length is rejected without mutation");

    // Truncation is detected by the bounded parser before the Base image is touched.
    fs::copy_file(d3_path, root / "truncated", fs::copy_options::overwrite_existing, ec);
    fs::resize_file(root / "truncated", fs::file_size(root / "truncated") - 1, ec);
    check(!session_delta_apply((root / "truncated").string(), id, chain, 2,
                               restored, generation, bytes, error, {}, &status) &&
          same_image(restored, unchanged), "truncated Delta is rejected without mutation");

    fs::copy_file(d3_path, root / "corrupt", fs::copy_options::overwrite_existing, ec);
    {
        std::fstream f(root / "corrupt", std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(96 + 40);   // first payload byte after the one-layer keep table
        char value = 0;
        f.read(&value, 1);
        value ^= 0x40;
        f.seekp(96 + 40);
        f.write(&value, 1);
    }
    check(!session_delta_apply((root / "corrupt").string(), id, chain, 2,
                               restored, generation, bytes, error, {}, &status) &&
          same_image(restored, unchanged), "payload checksum failure is rejected without mutation");

    fs::remove_all(root, ec);
    std::fprintf(stderr, "conversation_delta_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
