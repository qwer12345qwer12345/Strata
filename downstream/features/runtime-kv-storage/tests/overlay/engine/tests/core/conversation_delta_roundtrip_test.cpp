// Runtime KV Storage Patch 2's acceptance gate (downstream/patch-plans/runtime-kv-storage): for every officially
// supported K/V mode, a capture -> Base -> Delta (keep lengths from the provenance adapter) -> materialize ->
// Snapshot validate/restore round trip is byte for byte the full capture it was built from.  Skips with 77 where
// no CUDA device is available; the host half of the adapter's decisions is conversation_delta_test's.
#include "strata/core/conversation_delta_keep.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
template <class... A> bool snap_save(A&&... a) {
    const bool ok = conversation_snapshot_save(std::forward<A>(a)...);
    conversation_checkpoint_join(nullptr);
    return ok;
}

struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*, 5> sources{};
    std::array<size_t, 5> sizes{};

    template <class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt == kKvInt8; st.kv_q4 = fmt == kKvQ4; st.kv_hybrid = fmt == 3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = st.max_cells / 4 + 2;   // a whole 65+ token session's rows
        const size_t per = fmt == kKvQ4 ? kv_q4_bytes_per_head((int) g.head_dim)
                                        : g.head_dim * ((st.kv_int8 || st.kv_hybrid) ? 1 : 2);
        const size_t rows = st.max_cells * g.n_head_kv, slot_rows = st.n_slots * 4 * g.n_head_kv;
        sizes = {rows * per, rows * per, fmt == kKvInt8 ? rows * (g.head_dim / 64) * 2 : 0,
                 fmt == kKvInt8 ? rows * (g.head_dim / 64) * 2 : 0, (size_t) st.idx_pooled_rows * g.idx_key_dim * 4};
        if (fmt == 3) {
            sizes = {rows * per, rows * kv_q4_bytes_per_head((int) g.head_dim), rows * (g.head_dim / 64) * 2, 0,
                     (size_t) st.idx_pooled_rows * g.idx_key_dim * 4};
            alloc(st.k_q, sizes[0]); alloc(st.v_q4, sizes[1]); alloc(st.k_scale, sizes[2]);
            sources[0] = st.k_q; sources[1] = st.v_q4; sources[2] = st.k_scale;
        } else if (fmt == kKvQ4) {
            alloc(st.k_q4, slot_rows * per); alloc(st.v_q4, slot_rows * per);
            if (mode) { alloc(st.host.k_q4, sizes[0], true); alloc(st.host.v_q4, sizes[1], true); }
            sources[0] = mode ? st.host.k_q4 : st.k_q4; sources[1] = mode ? st.host.v_q4 : st.v_q4;
        } else if (fmt == kKvInt8) {
            alloc(st.k_q, slot_rows * per); alloc(st.v_q, slot_rows * per);
            alloc(st.k_scale, slot_rows * (g.head_dim / 64) * 2); alloc(st.v_scale, slot_rows * (g.head_dim / 64) * 2);
            if (mode) {
                alloc(st.host.k_q, sizes[0], true); alloc(st.host.v_q, sizes[1], true);
                alloc(st.host.k_scale, sizes[2], true); alloc(st.host.v_scale, sizes[3], true);
            }
            sources[0] = mode ? st.host.k_q : st.k_q; sources[1] = mode ? st.host.v_q : st.v_q;
            sources[2] = mode ? st.host.k_scale : st.k_scale; sources[3] = mode ? st.host.v_scale : st.v_scale;
        } else {
            alloc(st.k_pool, slot_rows * per); alloc(st.v_pool, slot_rows * per);
            if (mode) { alloc(st.host.k_pool, sizes[0], true); alloc(st.host.v_pool, sizes[1], true); }
            sources[0] = mode ? st.host.k_pool : st.k_pool; sources[1] = mode ? st.host.v_pool : st.v_pool;
        }
        alloc(st.idx_pooled, sizes[4]); sources[4] = st.idx_pooled;
        alloc(st.page_table, st.n_pages * 4);
        if (mode == 1) {
            auto& m = st.map;
            m.page_table = st.page_table; m.n_blocks = st.n_pages; m.n_slots = st.n_slots;
            alloc(m.slot_block, st.n_slots * 4); alloc(m.slot_stamp, st.n_slots * 4);
            alloc(m.slot_ref, st.n_slots * 4); alloc(m.miss_block, st.n_slots * 4);
            alloc(m.miss_slot, st.n_slots * 4); alloc(m.ctl, kKvCtlInts * 4);
        } else if (mode == 2) {
            kv_ring_table(st.page_table, st.n_pages, st.n_slots, nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i = 0; i < sources.size(); ++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j = 0; j < data.size(); ++j) data[j] = (uint8_t) (salt + i * 31 + j * 7 + j / 257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i], data.data(), data.size(), cudaMemcpyDefault));
        }
    }
    // the bytes from the first complete page or row at or above `first_dirty` change (the reuse contract's own
    // granularity: complete pages of cells, complete indexer rows)
    void fill_after(uint8_t salt, int64_t first_dirty) {
        const auto shapes = qsa_real_shapes();
        for (size_t i = 0; i < sources.size(); ++i) {
            const size_t offset = i == 4 ? size_t(first_dirty / shapes.idx_block) * g.idx_key_dim * 4
                                         : (sizes[i] / size_t(state.max_cells)) *
                                               size_t((first_dirty / shapes.page_size) * shapes.page_size);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i]) + offset, salt, sizes[i] - offset));
        }
    }
    ~Fixture() { for (void* p : device) cudaFree(p); for (void* p : host) cudaFreeHost(p); }
};

bool same_layer(const ConversationKv& a, const ConversationKv& b) {
    if (a.format != b.format || a.cells != b.cells || a.heads != b.heads || a.head_dim != b.head_dim ||
        a.page_size != b.page_size || a.pooled_rows != b.pooled_rows || a.idx_dim != b.idx_dim) return false;
    const std::array<const ConversationBuffer*, 5> pa = {&a.k, &a.v, &a.k_scale, &a.v_scale, &a.pooled};
    const std::array<const ConversationBuffer*, 5> pb = {&b.k, &b.v, &b.k_scale, &b.v_scale, &b.pooled};
    for (size_t part = 0; part < 5; ++part) {
        if (pa[part]->size() != pb[part]->size()) return false;
        std::vector<uint8_t> xa(pa[part]->size()), xb(pb[part]->size());
        if (!xa.empty() && (!pa[part]->read(xa.data(), 0, xa.size()) || !pb[part]->read(xb.data(), 0, xb.size())))
            return false;
        if (xa != xb) return false;
    }
    return true;
}

bool same_image(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        a.live.ids != b.live.ids || a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
        a.live.tails != b.live.tails || a.live.dead != b.live.dead || a.live.block_pos != b.live.block_pos ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!same_layer(a.kv[i], b.kv[i])) return false;
    return true;
}

void roundtrip(int fmt, int mode) {
    Fixture main(fmt, mode), draft(fmt == 3 ? kKvInt8 : fmt, 2);
    auto& g = main.g;
    g.n_layers = 4; g.n_expert = 256;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    SessionState ss;
    ss.max_cells = 96; ss.qsa_states = &main.state;
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g, sizes, err), "session geometry sizes");
    main.alloc(ss.gdn_state, sizes.gdn); main.alloc(ss.ple_hist, sizes.ple);
    main.alloc(main.state.idx_tail, sizes.tail); main.alloc(main.state.idx_dead, sizes.dead);
    main.alloc(main.state.idx_block_pos, sizes.block_pos);

    std::vector<int32_t> ids(65);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i + 1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill = [&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p, n] : std::vector<std::pair<void*, size_t>>{
                 {ss.gdn_state, sizes.gdn}, {ss.ple_hist, sizes.ple}, {main.state.idx_tail, sizes.tail},
                 {main.state.idx_dead, sizes.dead}, {main.state.idx_block_pos, sizes.block_pos}})
            cuda_check(cudaMemset(p, salt, n));
        // a real indexer maintains its spare pooled row as a copy of idx_dead - only where the fixture's pooled
        // buffer has that row at all (a ring's 2-row pool holds none of a 65-token conversation's spare row)
        if ((ids.size() / 4) < (size_t) main.state.idx_pooled_rows)
            cuda_check(cudaMemcpy(main.state.idx_pooled + (ids.size() / 4) * g.idx_key_dim,
                                  main.state.idx_dead, sizes.dead, cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(), ids.begin() + 3);
    check(conversation_checkpoint_save(checkpoint, ss, g, err) == true, "save the running checkpoint");
    conversation_checkpoint_join(nullptr);
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids, images, checkpoints, true};

    // the Base: one full capture, written as an official session file
    SavedConversation base;
    if (!snap_save(base, view, ss, g, draft.state, err)) {
        std::fprintf(stderr, "capture the Base failed: %s\n", err.c_str());
        std::exit(1);
    }
    check(true, "capture the Base");
    const SessionFileIdentity id{101, 202};
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("strata-delta-roundtrip-" + std::to_string(fmt) + "-" + std::to_string(mode));
    std::error_code ec;
    fs::create_directories(dir);
    const std::string base_path = (dir / "base.session").string();
    size_t bytes = 0;
    check(session_file_write(base_path, base, id, bytes, err), "write the Base session file");

    constexpr uint64_t chain = 0xa5a5a5a5a5a5a5a5ull;
    const uint64_t idx_block = qsa_real_shapes().idx_block;

    for (const int64_t dirty : {65, 3, 0}) {
        // the provenance contract: the retained K/V is the image the session actually restored
        check(conversation_snapshot_restore(base, ss, g, draft.state, err) == ConversationRestore::restored,
              "restore the Base into the session");
        ids.resize(70);
        for (size_t i = 65; i < ids.size(); ++i) ids[i] = (int32_t) i + 1;
        // everything at or above the boundary is rewritten (an append at 65, a rewrite at 3, a full rewrite at 0)
        main.fill_after(91, dirty);
        draft.fill_after(91, dirty > 0 ? dirty - 1 : 0);
        cuda_check(cudaMemset(ss.gdn_state, 91, sizes.gdn));
        if ((ids.size() / 4) < (size_t) main.state.idx_pooled_rows)
            cuda_check(cudaMemcpy(main.state.idx_pooled + (ids.size() / 4) * g.idx_key_dim,
                                  main.state.idx_dead, sizes.dead, cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());

        // the reference: a full capture of the grown state
        SavedConversation fresh;
        if (!snap_save(fresh, view, ss, g, draft.state, err)) {
            std::fprintf(stderr, "capture the grown state failed: %s\n", err.c_str());
            std::exit(1);
        }
        check(true, "capture the grown state whole");

        // the plan: the Base's own layers, the new state's official sources, the official dirty boundary
        ConversationKvReuse reuse;
        reuse.kv = base.kv;
        reuse.captured_tokens = 65;
        reuse.unchanged_tokens = dirty;
        std::vector<SessionKvSource> full;
        SavedConversation meta;
        check(conversation_snapshot_sources(meta, full, view, ss, g, draft.state, err),
              "the grown state's official K/V sources");
        const SessionDeltaKeepPlan plan = conversation_delta_keep_plan(reuse, full, idx_block);
        std::fprintf(stderr, "delta plan (fmt %d mode %d dirty %lld): ok=%d keep0=%llu/%llu/%llu/%llu/%llu of %llu "
                             "(page %lld block %llu, parent cells %lld)\n", fmt, mode, (long long) dirty,
                     plan.ok ? 1 : 0, (unsigned long long) (plan.keep.empty() ? 0 : plan.keep[0][0]),
                     (unsigned long long) (plan.keep.empty() ? 0 : plan.keep[0][1]),
                     (unsigned long long) (plan.keep.empty() ? 0 : plan.keep[0][2]),
                     (unsigned long long) (plan.keep.empty() ? 0 : plan.keep[0][3]),
                     (unsigned long long) (plan.keep.empty() ? 0 : plan.keep[0][4]),
                     (unsigned long long) full[0].sizes[0], (long long) qsa_real_shapes().page_size,
                     (unsigned long long) idx_block, (long long) base.kv[0].cells);
        check(plan.ok == (dirty >= 4), "only a boundary with a complete unchanged page plans a Delta");
        if (!plan.ok) {
            check(plan.keep.empty() && !plan.why.empty(), "a refusal leaves no table and says why");
            continue;
        }
        // the keep boundary is below the rewrite: the materialized layer must be the grown capture byte for byte
        check(plan.keep.size() == full.size() && plan.keep[0][0] > 0, "the plan keeps the leading bytes");

        std::vector<SessionKvSource> tails;
        for (size_t i = 0; i < full.size(); ++i) tails.push_back(session_delta_tail_source(full[i], plan.keep[i]));
        for (size_t i = 0; i < tails.size(); ++i)
            for (size_t part = 0; part < 5; ++part)
                check(tails[i].sizes[part] == full[i].sizes[part] - plan.keep[i][part],
                      "a tail source's size is its part minus the keep");
        const std::string delta_path = (dir / ("delta-" + std::to_string(dirty))).string();
        check(session_delta_write(delta_path, meta, tails, plan.keep, id, chain, 0, 1, bytes, err),
              "write the Delta of the grown state");

        // materialize: read the Base non-destructively, apply the exact child generation
        SavedConversation materialized;
        check(session_file_read(base_path, id, materialized, bytes, err), "read the Base back whole");
        uint64_t generation = 0;
        check(session_delta_apply(delta_path, id, chain, 0, materialized, generation, bytes, err) && generation == 1,
              "apply the Delta to the Base");
        check(same_image(materialized, fresh), "Base + Delta materializes byte for byte the full capture");

        // the materialized image goes through the Snapshot's own gates and restores
        check(conversation_snapshot_validate(materialized, ss, g, draft.state, err),
              "the materialized image validates");
        check(conversation_snapshot_restore(materialized, ss, g, draft.state, err) ==
                  ConversationRestore::restored, "the materialized image restores");
        uint64_t fingerprint = 0;
        check(conversation_kv_verify(materialized.kv.back(), draft.state, g, 70, false, fingerprint, err),
              "the draft layer's authoritative bytes read back");
        ids.resize(65);
    }
    fs::remove_all(dir, ec);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    for (int fmt : std::array<int, 4>{kKvF16, kKvInt8, kKvQ4, 3})
        for (int mode : {0, 1, 2}) {
            if (fmt == 3 && mode != 0) continue;
            roundtrip(fmt, mode);
        }
    std::printf("conversation_delta_roundtrip_test: %d checks passed\n", checks);
    return 0;
}
