// Build-only preparation and opt-in SM86 latency sweep for native Dense layouts.
// Output is JSONL; timing is kernel-only with resident rotating weights.
#include "strata/kernels/dense_i8_tc.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace {
struct Case { int type, k, n; };
constexpr Case kCases[] = {
    {8, 2560, 2560}, {8, 2560, 6144}, {8, 2560, 10240}, {8, 2560, 12288}, {8, 6144, 2560},
    {12, 2560, 2560}, {12, 2560, 6144}, {12, 2560, 10240}, {12, 2560, 12288}, {12, 6144, 2560},
    {13, 2560, 2560}, {13, 2560, 6144}, {13, 2560, 10240}, {13, 2560, 12288}, {13, 6144, 2560},
    {14, 2560, 2560}, {14, 2560, 6144}, {14, 2560, 10240}, {14, 2560, 12288}, {14, 6144, 2560},
};
constexpr int kWidths[] = {2, 3, 4, 5, 6, 8, 9, 12, 16};
constexpr std::size_t kWeightCycleBytes = 128u << 20;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_layout_sweep: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::size_t weight_bytes(const Case& c) { return K::native_mmvq_weight_bytes(c.type, c.k, c.n); }

void fill_weights(const Case& c, std::vector<uint8_t>& w, std::mt19937& rng) {
    w.resize(weight_bytes(c));
    for (auto& b : w) b = uint8_t(rng());
    const int block_elems = c.type == 8 ? 32 : 256;
    const int block_bytes = c.type == 8 ? 34 : c.type == 12 ? 144 : c.type == 13 ? 176 : 210;
    const int scale_at = c.type == 8 ? 32 : c.type == 14 ? 208 : 0;
    const std::size_t blocks = std::size_t(c.n) * std::size_t(c.k / block_elems);
    for (std::size_t b = 0; b < blocks; ++b) {
        const float d = 0.002f + float(rng() & 255u) * 0.00002f;
        const uint16_t h = __half_as_ushort(__float2half_rn(d));
        std::memcpy(w.data() + b * block_bytes + scale_at, &h, sizeof(h));
        if (c.type == 12 || c.type == 13) {
            const uint16_t dm = __half_as_ushort(__float2half_rn(d * 0.7f));
            std::memcpy(w.data() + b * block_bytes + 2, &dm, sizeof(dm));
        }
    }
}

bool has_generic(int type) { return type == 12 || type == 13 || type == 14; }
bool has_b6(int type) { return type == 12 || type == 13 || type == 14; }

void native_call(const Case& c, const std::vector<void*>& weights, int wi, const void* xq, float* y,
                 int t, cudaStream_t s) {
    K::native_mmvq(c.type, weights[size_t(wi) % weights.size()], xq, y, c.k, c.n, t, s);
}

double timed(const std::function<void()>& prepare, const std::function<void(int)>& call, int iters, cudaStream_t s) {
    prepare();
    cudaEvent_t a, b; ck(cudaEventCreate(&a), "event create"); ck(cudaEventCreate(&b), "event create");
    for (int i = 0; i < 16; ++i) call(i);
    ck(cudaStreamSynchronize(s), "warmup sync");
    ck(cudaEventRecord(a, s), "event record");
    for (int i = 0; i < iters; ++i) call(i);
    ck(cudaEventRecord(b, s), "event record"); ck(cudaEventSynchronize(b), "timing sync");
    float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "elapsed time");
    cudaEventDestroy(a); cudaEventDestroy(b);
    return double(ms) * 1000.0 / iters;
}

void compare(const float* ref, const float* got, size_t count, double& rms, double& max_abs, double& same) {
    double num = 0, den = 0; size_t bits = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) { rms = max_abs = -1; same = 0; return; }
        const double d = double(got[i]) - ref[i]; num += d * d; den += double(ref[i]) * ref[i];
        max_abs = std::max(max_abs, std::abs(d));
        if (std::memcmp(ref + i, got + i, sizeof(float)) == 0) ++bits;
    }
    rms = std::sqrt(num / std::max(den, 1e-30)); same = double(bits) / count;
}

} // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::max(16, std::atoi(argv[1])) : 128;
    const int pairs = argc > 2 ? std::max(2, std::atoi(argv[2])) : 4;
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess || major != 8 || minor != 6) {
        std::fprintf(stderr, "dense_layout_sweep: SM86 RTX 3080-class GPU required; build-only is safe on other devices.\n");
        return 77;
    }
    // No candidate defaults are changed by this program. Q6_K packing is enabled only for the explicit packed arm.
#if defined(_WIN32)
    _putenv_s("STRATA_DENSE_I8_TC", "0");
    _putenv_s("STRATA_DENSE_I8_AMPERE_SHAPES", "0");
    _putenv_s("STRATA_Q6_PACKED", "1");
    _putenv_s("STRATA_Q6P_SELFTEST", "0");
#else
    setenv("STRATA_DENSE_I8_TC", "0", 1);
    setenv("STRATA_DENSE_I8_AMPERE_SHAPES", "0", 1);
    setenv("STRATA_Q6_PACKED", "1", 1);
    setenv("STRATA_Q6P_SELFTEST", "0", 1);
#endif
    cudaStream_t s; ck(cudaStreamCreate(&s), "stream create");
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, dev), "device properties");
    if (std::strstr(prop.name, "RTX 3080") == nullptr || prop.totalGlobalMem < (11ull << 30)) {
        std::fprintf(stderr, "dense_layout_sweep: target is RTX 3080 with 12 GiB; found %s with %.2f GiB.\n",
                     prop.name, double(prop.totalGlobalMem) / double(1ull << 30));
        return 77;
    }
    auto select = [](bool exact, int b6) { K::native_mmvq_set_multi_exact(exact); K::native_mmvq_set_b6_rows(b6); };
    const std::function<void()> select_exact = [&] { select(true, 0); };
    std::printf("{\"device\":\"%s\",\"memory_gib\":%.2f,\"iters\":%d,\"pairs\":%d,\"weight_cycle_mib\":128}\n",
                prop.name, double(prop.totalGlobalMem) / double(1ull << 30), iters, pairs);
    for (const Case& c : kCases) {
        const std::size_t wb = weight_bytes(c);
        const size_t copies = std::max<size_t>(1, std::min<size_t>(32, (kWeightCycleBytes + wb - 1) / wb));
        std::mt19937 rng(9123u + unsigned(c.type * 31 + c.k + c.n));
        std::vector<uint8_t> host; fill_weights(c, host, rng);
        std::vector<void*> weights(copies, nullptr);
        bool alloc_ok = true;
        for (auto& p : weights) {
            if (cudaMalloc(&p, wb) != cudaSuccess || cudaMemcpy(p, host.data(), wb, cudaMemcpyHostToDevice) != cudaSuccess) { alloc_ok = false; break; }
        }
        if (!alloc_ok) {
            std::fprintf(stderr, "dense_layout_sweep: insufficient VRAM for type=%d K=%d N=%d weight cycle\n", c.type, c.k, c.n);
            for (void* p : weights) if (p) cudaFree(p);
            continue;
        }
        float *x = nullptr, *ref = nullptr, *got = nullptr; void* xq = nullptr; void* xil = nullptr;
        const size_t xbytes = size_t(16) * c.k * sizeof(float), ybytes = size_t(16) * c.n * sizeof(float);
        ck(cudaMalloc(&x, xbytes), "activation allocation"); ck(cudaMalloc(&xq, K::native_q8_1_bytes(c.k, 16)), "Q8_1 allocation");
        ck(cudaMalloc(&ref, ybytes), "reference allocation"); ck(cudaMalloc(&got, ybytes), "result allocation");
        std::vector<float> hx(size_t(16) * c.k);
        for (auto& v : hx) v = float(int(rng() % 20001) - 10000) * 0.0001f;
        ck(cudaMemcpy(x, hx.data(), xbytes, cudaMemcpyHostToDevice), "activation copy");
        select_exact();
        for (int first = 0; first < 16; first += 8)
            K::native_quantize_q8_1(x + size_t(first) * c.k,
                static_cast<uint8_t*>(xq) + size_t(first) * K::native_q8_1_bytes(c.k, 1), c.k, std::min(8, 16 - first), s);
        ck(cudaStreamSynchronize(s), "quantize sync");
        for (int first = 0; first < 16; first += 8)
            K::native_mmvq(c.type, weights[0], static_cast<uint8_t*>(xq) + size_t(first) * K::native_q8_1_bytes(c.k, 1),
                           ref + size_t(first) * c.n, c.k, c.n, std::min(8, 16 - first), s);
        ck(cudaStreamSynchronize(s), "reference sync");

        for (int t : kWidths) {
            const size_t count = size_t(t) * c.n;
            const std::function<void(int)> exact = [&](int i) { native_call(c, weights, i, xq, got, t, s); };
            select_exact();
            exact(0); ck(cudaStreamSynchronize(s), "exact sync");
            std::vector<float> exact_out(count); ck(cudaMemcpy(exact_out.data(), got, count * sizeof(float), cudaMemcpyDeviceToHost), "exact output");
            std::vector<float> reference_out(count); ck(cudaMemcpy(reference_out.data(), ref, count * sizeof(float), cudaMemcpyDeviceToHost), "reference output");
            double reference_rms = 0, reference_max = 0, reference_same = 0;
            compare(reference_out.data(), exact_out.data(), count, reference_rms, reference_max, reference_same);
            struct Candidate { std::string name; std::function<void()> prepare; std::function<void(int)> call; double prep_us = 0; };
            std::vector<Candidate> candidates;
            if (has_generic(c.type))
                candidates.push_back({"generic", [&] { select(false, 0); }, [&](int i) { native_call(c, weights, i, xq, got, t, s); }});
            if (has_b6(c.type))
                candidates.push_back({"b6_2row", [&] { select(true, 1); }, [&](int i) { native_call(c, weights, i, xq, got, t, s); }});
            if (t <= 8 && t >= 2 && t <= 4 && K::native_mmvq_il_supported(c.type, t, c.n)) {
                if (!xil) ck(cudaMalloc(&xil, K::native_q8_1_il_bytes(c.k, 4)), "interleaved activation allocation");
                const std::function<void(int)> interleave = [&](int) { K::native_q8_1_interleave(xq, xil, c.k, t, s); };
                const double interleave_us = timed([] {}, interleave, iters, s);
                interleave(0); ck(cudaStreamSynchronize(s), "interleave sync");
                candidates.push_back({"il_table", select_exact, [&](int i) {
                    K::native_mmvq_il(c.type, weights[size_t(i) % weights.size()], xq, xil, got, c.k, c.n, t, s); }, interleave_us});
            }
            if (t <= 16 && K::dense_i8_tc_mmvq(c.type, weights[0], xq, got, c.k, c.n, t, s)) {
                // The call above only probes eligibility and enqueues one launch; clear it before any timing.
                ck(cudaStreamSynchronize(s), "TC eligibility sync");
                candidates.push_back({"int8_tc", select_exact, [&](int i) { if (!K::dense_i8_tc_mmvq(c.type, weights[size_t(i) % weights.size()], xq, got, c.k, c.n, t, s))
                    std::fprintf(stderr, "dense_layout_sweep: TC refused a previously eligible shape\n"); }});
            }

            for (const Candidate& item : candidates) {
                std::vector<double> base, cand;
                for (int r = 0; r < pairs; ++r) {
                    if ((r & 1) == 0) { base.push_back(timed(select_exact, exact, iters, s)); cand.push_back(timed(item.prepare, item.call, iters, s)); }
                    else { cand.push_back(timed(item.prepare, item.call, iters, s)); base.push_back(timed(select_exact, exact, iters, s)); }
                }
                std::sort(base.begin(), base.end()); std::sort(cand.begin(), cand.end());
                const double base_us = (base[(base.size()-1)/2] + base[base.size()/2]) / 2.0;
                const double cand_us = (cand[(cand.size()-1)/2] + cand[cand.size()/2]) / 2.0;
                item.prepare(); item.call(0); ck(cudaStreamSynchronize(s), "candidate parity sync");
                std::vector<float> got_out(count); ck(cudaMemcpy(got_out.data(), got, count * sizeof(float), cudaMemcpyDeviceToHost), "candidate output");
                double rms = 0, max_abs = 0, same = 0; compare(exact_out.data(), got_out.data(), count, rms, max_abs, same);
                std::printf("{\"type\":%d,\"K\":%d,\"N\":%d,\"T\":%d,\"candidate\":\"%s\",\"exact_us\":%.4f,\"candidate_us\":%.4f,\"candidate_prep_us\":%.4f,\"speedup\":%.5f,\"kernel_speedup\":%.5f,\"relative_rms\":%.9g,\"max_abs\":%.9g,\"same_fraction\":%.6f,\"exact_reference_rms\":%.9g,\"exact_reference_max_abs\":%.9g,\"exact_reference_same_fraction\":%.6f,\"weight_cycle_mib\":%.2f}\n",
                    c.type, c.k, c.n, t, item.name.c_str(), base_us, cand_us, item.prep_us, base_us / (cand_us + item.prep_us), base_us / cand_us, rms, max_abs, same,
                    reference_rms, reference_max, reference_same, double(wb * copies) / 1048576.0);
                std::fflush(stdout);
            }

            // Compare packed layouts in separate arms so the exact baseline cannot silently take the packed path.
            const bool q8_pack = c.type == 8 && K::native_q8_0_packed_eligible(c.k, c.n) && wb * copies * 2 < (8ull << 30);
            const bool q6_pack = c.type == 14 && c.n >= 2048 && wb * copies * 2 < (8ull << 30);
            if (q8_pack || q6_pack) {
                std::vector<void*> packed(weights.size(), nullptr);
                std::vector<double> base, cand;
                bool pack_ok = true;
                auto pack_arm = [&]() {
                    if (q8_pack) {
                        std::vector<uint8_t> packed_host(wb); K::native_q8_0_pack_host(host.data(), packed_host.data(), c.k, c.n);
                        for (size_t i = 0; i < weights.size(); ++i) {
                            if (!packed[i]) ck(cudaMalloc(&packed[i], wb), "Q8_0 packed allocation");
                            ck(cudaMemcpy(packed[i], packed_host.data(), wb, cudaMemcpyHostToDevice), "Q8_0 packed copy");
                            K::native_q8_0_packed_register(weights[i], packed[i], c.k, c.n);
                        }
                        return true;
                    }
                    size_t done = 0;
                    for (void* p : weights) {
                        if (!K::native_q6_k_pack(p, c.k, c.n, "dense layout sweep")) {
                            for (size_t j = 0; j < done; ++j) K::native_q6_k_unpack(weights[j]);
                            return false;
                        }
                        ++done;
                    }
                    return true;
                };
                auto unreg = [&]() {
                    if (q8_pack) for (void* p : weights) K::native_q8_0_packed_unregister(p);
                    else for (void* p : weights) K::native_q6_k_unpack(p);
                };
                const std::function<void(int)> packed_call = [&](int i) { native_call(c, weights, i, xq, got, t, s); };
                for (int r = 0; r < pairs; ++r) {
                    auto measure_base = [&]() { base.push_back(timed(select_exact, exact, iters, s)); };
                    auto measure_packed = [&]() {
                        pack_ok = pack_arm();
                        if (pack_ok) cand.push_back(timed(select_exact, packed_call, iters, s));
                        unreg();
                    };
                    if ((r & 1) == 0) { measure_base(); measure_packed(); }
                    else { measure_packed(); measure_base(); }
                }
                if (pack_ok && !cand.empty()) {
                    std::sort(base.begin(), base.end()); std::sort(cand.begin(), cand.end());
                    const double base_us = (base[(base.size()-1)/2] + base[base.size()/2]) / 2.0;
                    const double cand_us = (cand[(cand.size()-1)/2] + cand[cand.size()/2]) / 2.0;
                    pack_arm(); select_exact(); packed_call(0); ck(cudaStreamSynchronize(s), "packed parity sync");
                    std::vector<float> packed_out(count); ck(cudaMemcpy(packed_out.data(), got, count * sizeof(float), cudaMemcpyDeviceToHost), "packed output");
                    double rms = 0, max_abs = 0, same = 0; compare(exact_out.data(), packed_out.data(), count, rms, max_abs, same);
                    std::printf("{\"type\":%d,\"K\":%d,\"N\":%d,\"T\":%d,\"candidate\":\"packed\",\"exact_us\":%.4f,\"candidate_us\":%.4f,\"candidate_prep_us\":0,\"speedup\":%.5f,\"kernel_speedup\":%.5f,\"relative_rms\":%.9g,\"max_abs\":%.9g,\"same_fraction\":%.6f,\"exact_reference_rms\":%.9g,\"exact_reference_max_abs\":%.9g,\"exact_reference_same_fraction\":%.6f,\"weight_cycle_mib\":%.2f,\"packed_extra_mib\":%.2f}\n",
                        c.type, c.k, c.n, t, base_us, cand_us, base_us / cand_us, base_us / cand_us, rms, max_abs, same,
                        reference_rms, reference_max, reference_same,
                        double(wb * copies) / 1048576.0, double(wb * copies) / 1048576.0);
                    std::fflush(stdout);
                    unreg();
                }
                for (void* p : packed) if (p) cudaFree(p);
            }
        }
        if (xil) cudaFree(xil); cudaFree(x); cudaFree(xq); cudaFree(ref); cudaFree(got);
        for (void* p : weights) cudaFree(p);
    }
    select_exact();
    cudaStreamDestroy(s);
    return 0;
}
