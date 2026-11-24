// src/kernels/hc_persist_parity.cpp - the STRATA_HC_PERSIST=1 hyper-connection read (one persistent kernel) against the
// default multi read (fused_gr_read_multi).
//
// Real geometry (n_embd 2560, hc 4, hc_lr 320), every T = 1..8, the pending write applied and the inject rows on, as the
// verifier calls it. The persistent read sums in another order, so it is bounded, not bitwise: every output (the mixed
// row, the inject outputs, the written R) must be finite and within 1e-4 of the default read relative to its largest
// magnitude, and two runs of the persistent read must agree bit for bit. Where the card does not hold the 80 blocks at
// once (or on CUDA, where the read is not built) the read reports that it did not run and the test passes as a skip.
// `--off`: with STRATA_HC_PERSIST unset the persistent read must never run (the default path is untouched).
// GPU, synthetic, no model.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_WIN32)   // MSVC has no setenv / unsetenv
#define setenv(k, v, o) _putenv_s((k), (v))
#define unsetenv(k) _putenv_s((k), "")
#endif

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = k::kFusedGrWideT;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return (T*) p;
}
}  // namespace

int main(int argc, char** argv) {
    const bool off = argc > 1 && std::strcmp(argv[1], "--off") == 0;
    if (off) unsetenv("STRATA_HC_PERSIST");
    else setenv("STRATA_HC_PERSIST", "1", 1);
    k::fused_gr_check();
    k::fused_gr_persist_init();
    cudaStream_t quant_stream;
    ck(cudaStreamCreate(&quant_stream), "quant stream");

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float* dR = dev<float>(R.size());
    float* dbo = up(bo);
    float* dinj = up(inj);
    float* dwn = up(wn);
    uint16_t* dwd = up(wd);
    uint16_t* dwu = up(wu);
    uint16_t* dwi = up(wi);
    float* dxn = dev<float>((size_t) TM * D);
    float* dlo = dev<float>((size_t) TM * LR);
    float* drs = dev<float>((size_t) TM * HC);
    float* dio = dev<float>((size_t) TM * HC);
    float* dmix = dev<float>((size_t) TM * N);

    int bad = 0, ran = 0;
    for (int T = 1; T <= k::kFusedGrMaxT; ++T) {
        k::FusedGrArgs a[TM];
        for (int t = 0; t < T; ++t) {
            a[t] = k::FusedGrArgs{};
            a[t].R = dR + (size_t) t * D;
            a[t].R_out = dR + (size_t) t * D;
            a[t].apply = true;
            a[t].bo_prev = dbo + (size_t) t * N;
            a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn;
            a[t].w_down = dwd;
            a[t].w_up = dwu;
            a[t].w_inject = dwi;
            a[t].eps = 1e-6f;
            a[t].lo = dlo + (size_t) t * LR;
            a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = dio + (size_t) t * HC;
            a[t].mixed = dmix + (size_t) t * N;
        }
        // 0: the default read, 1 and 2: the persistent read twice; each output is mixed | inject | R
        std::vector<float> out[3];
        bool used = true;
        for (int w = 0; w < 3; ++w) {
            ck(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
            ck(cudaMemset(dmix, 0xff, (size_t) TM * N * 4), "mix");
            ck(cudaMemset(dio, 0xff, (size_t) TM * HC * 4), "io");
            ck(cudaDeviceSynchronize(), "sync");
            if (w == 0) k::fused_gr_read_multi(a, T, dxn, nullptr);
            else used = k::fused_gr_read_persist(a, T, nullptr) && used;
            ck(cudaDeviceSynchronize(), "read");
            out[w].resize((size_t) T * (N + HC + D));
            ck(cudaMemcpy(out[w].data(), dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost), "d2h mix");
            ck(cudaMemcpy(out[w].data() + (size_t) T * N, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost), "d2h io");
            ck(cudaMemcpy(out[w].data() + (size_t) T * (N + HC), dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost), "d2h R");
            if (w == 1 && !used) break;
        }
        if (off) {
            if (used) {
                std::printf("T=%d: the persistent read ran with STRATA_HC_PERSIST unset  FAIL\n", T);
                ++bad;
            }
            continue;
        }
        if (!used) {
            std::printf("T=%d: the persistent read did not run on this card (skip)\n", T);
            continue;
        }
        ++ran;
        double mref = 0.0, md = 0.0;
        bool finite = true;
        for (size_t i = 0; i < out[0].size(); ++i) {
            finite = finite && std::isfinite(out[1][i]);
            mref = std::fmax(mref, std::fabs((double) out[0][i]));
            md = std::fmax(md, std::fabs((double) out[1][i] - out[0][i]));
        }
        const double rel = md / (mref > 0 ? mref : 1.0);
        const bool same = std::memcmp(out[1].data(), out[2].data(), out[1].size() * 4) == 0;
        const bool ok = finite && rel < 1e-4 && same;
        std::printf("T=%d: max relative difference %.3g against the default read, %s, two runs %s  %s\n", T, rel,
                    finite ? "finite" : "NON-FINITE", same ? "bitwise equal" : "DIFFER", ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    }
    if (!off) {
        // Q8_0 weights are produced by the engine's packer so this covers its stored int8/half-scale layout, the default
        // Q8 (including the TC dispatch when enabled), and the persistent path. QFUSE bytes are checked against the
        // native quantizer for each path's own mixed values; numerical differences are reported for later acceptance work.
        constexpr size_t Q8_0_BYTES = 34, Q8_1_BYTES = 36;
        const size_t q8d_blocks = wd.size() / 32, q8u_blocks = wu.size() / 32, q8i_blocks = wi.size() / 32;
        const size_t q8row_bytes = (size_t) (N / 32) * Q8_1_BYTES;
        uint8_t* dq8d = dev<uint8_t>(q8d_blocks * Q8_0_BYTES);
        uint8_t* dq8u = dev<uint8_t>(q8u_blocks * Q8_0_BYTES);
        uint8_t* dq8i = dev<uint8_t>(q8i_blocks * Q8_0_BYTES);
        uint8_t* dqmix = dev<uint8_t>((size_t) TM * q8row_bytes);
        uint8_t* dqref = dev<uint8_t>((size_t) TM * q8row_bytes);
        unsigned* dqcnt = dev<unsigned>(N / 32);
        k::fused_gr_quantize_q8_0(dwd, dq8d, (int64_t) q8d_blocks, nullptr);
        k::fused_gr_quantize_q8_0(dwu, dq8u, (int64_t) q8u_blocks, nullptr);
        k::fused_gr_quantize_q8_0(dwi, dq8i, (int64_t) q8i_blocks, nullptr);
        ck(cudaDeviceSynchronize(), "quantize Q8_0 fixtures");

        auto make_q8_args = [&](int T) {
            std::vector<k::FusedGrArgs> a((size_t) T);
            for (int t = 0; t < T; ++t) {
                a[(size_t) t] = k::FusedGrArgs{};
                a[(size_t) t].R = dR + (size_t) t * D;
                a[(size_t) t].R_out = dR + (size_t) t * D;
                a[(size_t) t].apply = true;
                a[(size_t) t].bo_prev = dbo + (size_t) t * N;
                a[(size_t) t].inj_prev = dinj + (size_t) t * HC;
                a[(size_t) t].w_norm = dwn;
                a[(size_t) t].q8_down = dq8d;
                a[(size_t) t].q8_up = dq8u;
                a[(size_t) t].q8_inject = dq8i;
                a[(size_t) t].eps = 1e-6f;
                a[(size_t) t].lo = dlo + (size_t) t * LR;
                a[(size_t) t].rs = drs + (size_t) t * HC;
                a[(size_t) t].inject_out = dio + (size_t) t * HC;
                a[(size_t) t].mixed = dmix + (size_t) t * N;
                a[(size_t) t].q8_mixed = dqmix + (size_t) t * q8row_bytes;
                a[(size_t) t].q8_cnt = dqcnt;
            }
            return a;
        };
        auto capture_q8 = [&](int T, std::vector<float>& out, std::vector<uint8_t>& qimage) {
            out.resize((size_t) T * (N + HC + D));
            qimage.resize((size_t) T * q8row_bytes);
            ck(cudaMemcpy(out.data(), dmix, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToHost), "Q8 mixed");
            ck(cudaMemcpy(out.data() + (size_t) T * N, dio, (size_t) T * HC * sizeof(float), cudaMemcpyDeviceToHost), "Q8 inject");
            ck(cudaMemcpy(out.data() + (size_t) T * (N + HC), dR, (size_t) T * D * sizeof(float), cudaMemcpyDeviceToHost), "Q8 R");
            ck(cudaMemcpy(qimage.data(), dqmix, qimage.size(), cudaMemcpyDeviceToHost), "QFUSE image");
        };
        auto q8_read = [&](bool persist, int T, const std::vector<k::FusedGrArgs>& a, std::vector<float>& out,
                           std::vector<uint8_t>& qimage, bool& used) {
            ck(cudaMemcpy(dR, R.data(), R.size() * sizeof(float), cudaMemcpyHostToDevice), "Q8 reset R");
            ck(cudaMemset(dqcnt, 0, (N / 32) * sizeof(unsigned)), "QFUSE counters");
            ck(cudaDeviceSynchronize(), "Q8 pre-read sync");
            used = persist ? k::fused_gr_read_persist(a.data(), T, nullptr)
                           : k::fused_gr_read_multi(a.data(), T, dxn, nullptr);
            if (!used) return true;
            ck(cudaDeviceSynchronize(), persist ? "Q8 persistent read" : "Q8 default read");
            capture_q8(T, out, qimage);
            k::native_quantize_q8_1(dmix, dqref, N, T, quant_stream);
            ck(cudaDeviceSynchronize(), "native QFUSE reference");
            std::vector<uint8_t> ref(qimage.size());
            ck(cudaMemcpy(ref.data(), dqref, ref.size(), cudaMemcpyDeviceToHost), "QFUSE reference image");
            return std::memcmp(qimage.data(), ref.data(), ref.size()) == 0;
        };

        int q8_ran = 0;
        for (int T = 1; T <= k::kFusedGrMaxT; ++T) {
            const auto a = make_q8_args(T);
            std::vector<float> q8_default, q8_persist, q8_repeat;
            std::vector<uint8_t> q8_default_img, q8_persist_img, q8_repeat_img;
            bool default_used = false, persist_used = false, repeat_used = false;
            const bool default_qfuse = q8_read(false, T, a, q8_default, q8_default_img, default_used);
            const bool persist_qfuse = q8_read(true, T, a, q8_persist, q8_persist_img, persist_used);
            if (!default_used || !default_qfuse) {
                std::printf("Q8 T=%d: default QFUSE image differs from native q8_1  FAIL\n", T);
                ++bad;
                continue;
            }
            if (!persist_used) {
                std::printf("Q8 T=%d: persistent kernel not resident on this card (skip)\n", T);
                continue;
            }
            if (!persist_qfuse || !q8_read(true, T, a, q8_repeat, q8_repeat_img, repeat_used) || !repeat_used) {
                std::printf("Q8 T=%d: persistent QFUSE image differs from native q8_1  FAIL\n", T);
                ++bad;
                continue;
            }
            const bool deterministic = q8_persist.size() == q8_repeat.size() && q8_persist_img == q8_repeat_img &&
                                       std::memcmp(q8_persist.data(), q8_repeat.data(), q8_persist.size() * sizeof(float)) == 0;
            bool finite = true;
            double mag = 0.0, delta = 0.0;
            for (size_t i = 0; i < q8_persist.size(); ++i) {
                finite = finite && std::isfinite(q8_persist[i]);
                mag = std::fmax(mag, std::fabs((double) q8_default[i]));
                delta = std::fmax(delta, std::fabs((double) q8_persist[i] - q8_default[i]));
            }
            const double rel = delta / (mag > 0.0 ? mag : 1.0);
            std::printf("Q8 T=%d: default (TC when enabled) vs persistent max relative difference %.3g; finite %s, repeat %s, QFUSE exact yes\n",
                        T, rel, finite ? "yes" : "NO", deterministic ? "yes" : "NO");
            if (!finite || !deterministic) ++bad;
            ++q8_ran;
        }

        // The persistent kernel is an 8-row kernel. A wider window must use the existing Q8 multi-read slices, including
        // QFUSE output for the tail rows; compare its automatic split with an explicit 8 + 1 call.
        if constexpr (TM > k::kFusedGrMaxT) {
            constexpr int T = k::kFusedGrMaxT + 1;
            const auto a = make_q8_args(T);
            std::vector<float> wide, split;
            std::vector<uint8_t> wide_q, split_q;
            bool wide_used = false;
            const bool wide_qfuse = q8_read(false, T, a, wide, wide_q, wide_used);
            const bool persist_wide = k::fused_gr_read_persist(a.data(), T, nullptr);
            if (!wide_used || !wide_qfuse || persist_wide) {
                std::printf("Q8 T=%d: wide read/QFUSE or persistent fallback contract failed  FAIL\n", T);
                ++bad;
            } else {
                ck(cudaMemcpy(dR, R.data(), R.size() * sizeof(float), cudaMemcpyHostToDevice), "split reset R");
                ck(cudaMemset(dqcnt, 0, (N / 32) * sizeof(unsigned)), "split QFUSE counters");
                if (!k::fused_gr_read_multi(a.data(), k::kFusedGrMaxT, dxn, nullptr) ||
                    !k::fused_gr_read_multi(a.data() + k::kFusedGrMaxT, 1,
                                            dxn + (size_t) k::kFusedGrMaxT * D, nullptr)) {
                    std::printf("Q8 T=%d: explicit 8 + tail slices declined  FAIL\n", T);
                    ++bad;
                } else {
                    ck(cudaDeviceSynchronize(), "wide explicit slices");
                    capture_q8(T, split, split_q);
                    k::native_quantize_q8_1(dmix, dqref, N, T, quant_stream);
                    ck(cudaDeviceSynchronize(), "wide QFUSE reference");
                    std::vector<uint8_t> ref(wide_q.size());
                    ck(cudaMemcpy(ref.data(), dqref, ref.size(), cudaMemcpyDeviceToHost), "wide QFUSE reference image");
                    const bool same = wide == split && wide_q == split_q;
                    const bool fused = split_q == ref;
                    std::printf("Q8 T=%d: automatic vs explicit 8 + tail slicing %s; QFUSE exact %s\n", T,
                                same ? "equal" : "DIFFERS", fused ? "yes" : "NO");
                    if (!same || !fused) ++bad;
                }
            }
        }
        if (q8_ran == 0) std::printf("Q8 persistent read: no candidate cases ran on this card\n");
        cudaFree(dq8d); cudaFree(dq8u); cudaFree(dq8i); cudaFree(dqmix); cudaFree(dqref); cudaFree(dqcnt);
    }
    if (off) std::printf("%s: STRATA_HC_PERSIST unset, the persistent read never ran\n", bad ? "FAIL" : "PASS");
    else if (ran == 0) std::printf("SKIP: the persistent read is not used on this card\n");
    else std::printf("%s: %d token counts checked\n", bad ? "FAIL" : "PASS", ran);
    ck(cudaStreamDestroy(quant_stream), "destroy quant stream");
    return bad ? 1 : 0;
}
