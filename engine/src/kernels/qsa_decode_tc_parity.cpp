// Parity harness for the K8V4 decode attention's tensor-core score phase (STRATA_ATTN_DECODE_TC=1, Ampere or
// newer) against a host FP64 mathematical reference and the default warp scoring. GPU, synthetic data, no model
// file. Both GPU paths run chunks + merge:
// daily_chunk=0 is the default scoring, daily_chunk=2 names the tensor-core chunk kernel (TC compiles only for
// K8V4 pools with warp scoring - see the static_assert in qsa_decode_attn.cu), so the harness covers the kernel
// itself and, through the batch entry under the environment, the dispatch that selects it.
//
// The host reference decodes the stored K8V4 values and computes dot products, softmax, and the Q4_0 value mix in
// double. Report each GPU path's error against that reference; apply prompt i8v3's two checks: TC's FP64 error is
// at most 4x the default path's error (with a 1e-6 * reference-scale floor), and TC-vs-default is at most 1e-4 of
// output scale. The production batch dispatch is also checked against the selected named path.
//
// Returns 77 (ctest SKIP_RETURN_CODE) with no CUDA device, or on a device below sm_80 where the tensor-core
// kernel traps: a non-Ampere run is not evidence of a pass.  The batch entry's dispatch reads the environment
// once per process, so ctest runs the same binary twice (default, and STRATA_ATTN_DECODE_TC=1).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include "strata/core/geometry.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

namespace {

using strata::kernels::QsaAttnPools;
using strata::kernels::QsaShapes;
using strata::kernels::block_q4_0;

uint32_t g_st = 12345677u;
uint32_t rnd(void) { g_st = g_st * 1664525u + 1013904223u; return g_st >> 8; }

struct Case {
    const char* name;
    int cap;          // cells per query
    int width;        // selected cells (steps[q][kStepWidth])
    int page_size;
    bool shuffle;     // page_table shuffled: the paged gather goes through the table
    bool mask_pages;  // some page_table entries are -1: the streamed slots that were never written
    int n_q;
};

int failures = 0;

void report(const char* name, const char* check, double error, double tol) {
    if (error > tol) {
        std::fprintf(stderr, "case %s: %s %.3e exceeds limit %.3e\n", name, check, error, tol);
        ++failures;
    } else {
        std::printf("case %s: %s %.3e (limit %.3e) ok\n", name, check, error, tol);
    }
}

bool all_finite(const std::vector<float>& v, const char* name) {
    for (float x : v)
        if (!std::isfinite(x)) { std::fprintf(stderr, "case %s: non-finite output\n", name); return false; }
    return true;
}

double half_to_double(uint16_t bits) {
    return (double) __half2float(__ushort_as_half(bits));
}

void host_double_reference(const Case& c, QsaShapes s, const std::vector<float>& q,
                           const std::vector<int8_t>& k_q, const std::vector<uint16_t>& k_scale,
                           const std::vector<block_q4_0>& v_q4, const std::vector<int32_t>& page_table,
                           const std::vector<int32_t>& ids, std::vector<double>& out) {
    const int hd = (int) s.head_dim;
    const int kv = (int) s.n_head_kv;
    const int nh = (int) s.n_head;
    const int groups = hd / strata::kernels::KV_Q8_GROUP;
    const int blocks = hd / strata::kernels::QK4_0;
    const int page_size = c.page_size;
    const double attn_scale = 1.0 / std::sqrt((double) hd);
    out.assign((size_t) c.n_q * nh * hd, 0.0);

    for (int qq = 0; qq < c.n_q; ++qq) {
        const int width = c.width;
        for (int h = 0; h < nh; ++h) {
            const int kvh = h / (nh / kv);
            std::vector<int> rows;
            std::vector<double> scores;
            rows.reserve((size_t) width);
            scores.reserve((size_t) width);
            double max_score = -INFINITY;
            for (int i = 0; i < width; ++i) {
                const int cell = ids[(size_t) qq * c.cap + i];
                const int page = page_table[(size_t) cell / page_size];
                if (page < 0) continue;
                const int row = (page * kv + kvh) * page_size + cell % page_size;
                double dot = 0.0;
                for (int d = 0; d < hd; ++d) {
                    const double k = (double) k_q[(size_t) row * hd + d] *
                                     half_to_double(k_scale[(size_t) row * groups + d / strata::kernels::KV_Q8_GROUP]);
                    dot += (double) q[((size_t) qq * nh + h) * hd + d] * k;
                }
                const double score = dot * attn_scale;
                rows.push_back(row);
                scores.push_back(score);
                max_score = std::max(max_score, score);
            }
            if (rows.empty()) continue;

            double sum_exp = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                sum_exp += score;
            }
            for (int d = 0; d < hd; ++d) {
                double value = 0.0;
                for (size_t i = 0; i < rows.size(); ++i) {
                    const int row = rows[i];
                    const block_q4_0& block = v_q4[(size_t) row * blocks + d / strata::kernels::QK4_0];
                    const int rem = d % strata::kernels::QK4_0;
                    const uint8_t packed = block.qs[rem < 16 ? rem : rem - 16];
                    const int code = (int) (rem < 16 ? packed & 0x0f : packed >> 4) - 8;
                    const double v = (double) code * half_to_double(block.d);
                    value += (scores[i] / sum_exp) * v;
                }
                out[((size_t) qq * nh + h) * hd + d] = value;
            }
        }
    }
}

double max_abs_error(const std::vector<float>& actual, const std::vector<double>& reference, double* ref_scale) {
    double error = 0.0, scale = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        error = std::max(error, std::fabs((double) actual[i] - reference[i]));
        scale = std::max(scale, std::fabs(reference[i]));
    }
    *ref_scale = scale;
    return error;
}

bool alloc(void** p, size_t bytes) {
    if (cudaMalloc(p, bytes) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc failed\n"); return false; }
    return true;
}

int run_case(const Case& c, QsaShapes s) {
    const int n_q = c.n_q;
    s.page_size = c.page_size;
    const int hd = (int) s.head_dim;
    const int kv = (int) s.n_head_kv;
    const int nh = (int) s.n_head;
    const int pages = (c.cap + c.page_size - 1) / c.page_size;
    const int rows = pages * kv * c.page_size;
    const int groups = hd / strata::kernels::KV_Q8_GROUP;              // scale groups per row: 256/64 = 4
    const int blocks = hd / strata::kernels::QK4_0;                    // Q4_0 blocks per row: 256/32 = 8

    const bool lowcode = strstr(c.name, "lowcode") != nullptr;   // a gentle pool: scores stay small, so softmax noise is visible
    const bool unitk = strstr(c.name, "unitk") != nullptr;       // only one dimension carries K codes, so the score is exact for both paths
    int unitk_dim = 0;                                           // which dimension, set by the case name (unitk / unitk32 / unitk96 / unitk224)
    if (unitk) {
      const char* n = c.name;
      if (strstr(n, "unitk224") != nullptr) unitk_dim = 224;
      else if (strstr(n, "unitk96") != nullptr) unitk_dim = 96;
      else if (strstr(n, "unitk32") != nullptr) unitk_dim = 32;
      else if (strstr(n, "unitk31") != nullptr) unitk_dim = 31;
      else if (strstr(n, "unitk15") != nullptr) unitk_dim = 15;
      else if (strstr(n, "unitk7") != nullptr) unitk_dim = 7;
      else if (strstr(n, "unitk1") != nullptr) unitk_dim = 1;
    }
    const bool tinyk = strstr(c.name, "tinyk") != nullptr;       // q ±1, codes ±1, scales 1: quantization cannot explain any gap
    const bool constk = strstr(c.name, "constk") != nullptr;     // codes all +127, scales 1: every score is an exact integer sum
    const bool onq = strstr(c.name, "onq") != nullptr;           // q all 1 with varying ±1 codes: pairing test without quantization
    const bool cellbasis = strstr(c.name, "cellbasis") != nullptr; // one-hot q sweep; cell 0 and 1 have distinct known K/V
    // K: int8 codes plus one fp16 scale per 64 dims.  V: Q4_0 blocks (uint16 d, 16 nibble bytes).
    std::vector<int8_t> k_q((size_t) rows * hd);
    std::vector<uint16_t> k_scale((size_t) rows * groups);
    std::vector<block_q4_0> v_q4((size_t) rows * blocks);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < hd; ++i) k_q[(size_t) r * hd + i] = cellbasis ? (int8_t) (r / kv == 1 ? 1 : 0)
                                                           : lowcode ? (int8_t) ((int) (rnd() % 7) - 3)
                                                           : unitk ? (int8_t) (i == unitk_dim ? 127 : 0)
                                                           : tinyk ? (int8_t) ((int) (rnd() % 3) - 1)
                                                           : onq ? (int8_t) ((int) (rnd() % 3) - 1)
                                                           : constk ? (int8_t) 127
                                                           : (int8_t) ((int) (rnd() % 251) - 125);   // realistic INT8 codes
        for (int g = 0; g < groups; ++g)
            k_scale[(size_t) r * groups + g] = cellbasis || (unitk && unitk_dim / 64 == g) || tinyk || constk || onq ? __half_as_ushort(__float2half(1.f))
                                                               : __half_as_ushort(__float2half(0.004f + 0.02f * (rnd() % 100) / 100.f));
        for (int b = 0; b < blocks; ++b) {
            v_q4[(size_t) r * blocks + b].d = __half_as_ushort(__float2half(cellbasis ? 1.f : 0.01f + 0.03f * (rnd() % 100) / 100.f));
            for (int j = 0; j < strata::kernels::QK4_0 / 2; ++j)
                v_q4[(size_t) r * blocks + b].qs[j] = cellbasis ? (uint8_t) (r / kv == 1 ? 0x99 : 0x77)
                                                                  : (uint8_t) (rnd() & 0xffu);
        }
    }

    std::vector<int32_t> page_table((size_t) pages, -1);
    for (int p = 0; p < pages; ++p) page_table[p] = p;
    if (c.shuffle)
        for (int p = pages - 1; p > 0; --p)
            std::swap(page_table[p], page_table[rnd() % (uint32_t) (p + 1)]);
    if (c.mask_pages)
        for (int p = 0; p < pages; ++p)
            if ((p % 3) == 1) page_table[p] = -1;   // masked on both sides: score -FLT_MAX, weight 0

    std::vector<float> q((size_t) n_q * nh * hd);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = cellbasis ? ((i % (size_t) hd) == ((i / (size_t) hd) % (size_t) hd) ? 1.f : 0.f)
                         : onq ? 1.f : tinyk ? ((rnd() & 1u) ? 1.f : -1.f) : (float) ((int) (rnd() % 2001) - 1000) / 1000.f;

    // ids: legal cell indices; the chunk pass reads only ids[..width].
    std::vector<int32_t> ids((size_t) n_q * c.cap);
    for (int qq = 0; qq < n_q; ++qq)
        for (int i = 0; i < c.cap; ++i) ids[(size_t) qq * c.cap + i] = cellbasis && i < 2
            ? i : (int32_t) ((qq * 7 + i * 13) % c.cap);

    std::vector<int32_t> steps((size_t) n_q * strata::kernels::kStepCount, 0);
    for (int qq = 0; qq < n_q; ++qq)
        steps[(size_t) qq * strata::kernels::kStepCount + strata::kernels::kStepWidth] = c.width;

    const size_t stride = strata::kernels::qsa_decode_attn_scratch_floats((size_t) c.cap, s);
    const size_t attn_floats = (size_t) n_q * nh * hd;

    QsaAttnPools hp;
    hp.k_pool = nullptr;
    hp.v_pool = nullptr;
    hp.k_q = k_q.data();
    hp.v_q = nullptr;
    hp.k_scale = k_scale.data();
    hp.v_scale = nullptr;
    hp.k_q4 = nullptr;
    hp.v_q4 = reinterpret_cast<const uint8_t*>(v_q4.data());
    hp.page_table = page_table.data();

    int8_t* d_kq = nullptr;
    uint16_t* d_ks = nullptr;
    uint8_t* d_vq = nullptr;
    float* d_q = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* d_pt = nullptr;
    int32_t* d_steps = nullptr;
    float* d_scratch[4] = {nullptr, nullptr, nullptr, nullptr};
    float* d_attn[4] = {nullptr, nullptr, nullptr, nullptr};
    if (!alloc((void**) &d_kq, k_q.size()) || !alloc((void**) &d_ks, k_scale.size() * sizeof(uint16_t)) ||
        !alloc((void**) &d_vq, v_q4.size() * sizeof(block_q4_0)) || !alloc((void**) &d_q, q.size() * sizeof(float)) ||
        !alloc((void**) &d_ids, ids.size() * sizeof(int32_t)) ||
        !alloc((void**) &d_pt, page_table.size() * sizeof(int32_t)) ||
        !alloc((void**) &d_steps, steps.size() * sizeof(int32_t)))
        return 77;
    cudaMemcpy(d_kq, k_q.data(), k_q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks, k_scale.data(), k_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vq, v_q4.data(), v_q4.size() * sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt, page_table.data(), page_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_steps, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    QsaAttnPools dp;
    dp.k_pool = nullptr;
    dp.v_pool = nullptr;
    dp.k_q = d_kq;
    dp.v_q = nullptr;
    dp.k_scale = d_ks;
    dp.v_scale = nullptr;
    dp.k_q4 = nullptr;
    dp.v_q4 = d_vq;
    dp.page_table = d_pt;

    for (int i = 0; i < 4; ++i) {
        if (!alloc((void**) &d_scratch[i], stride * (size_t) n_q * sizeof(float)) ||
            !alloc((void**) &d_attn[i], attn_floats * sizeof(float)))
            return 77;
        cudaMemset(d_scratch[i], 0, stride * (size_t) n_q * sizeof(float));
        cudaMemset(d_attn[i], 0, attn_floats * sizeof(float));
    }

    // A: the default scoring. B: the tensor-core chunk kernel, named. C: the default scoring again. The production
    // batch entry is checked against A or B below according to the exact opt-in environment value.
    strata::kernels::qsa_decode_attn_chunks_only(d_q, dp, d_ids, d_steps, (size_t) c.cap, s, d_scratch[0], n_q, 0, nullptr);
    strata::kernels::qsa_decode_attn_merge_only(d_steps, (size_t) c.cap, s, d_scratch[0], d_attn[0], n_q, 0, nullptr);
    strata::kernels::qsa_decode_attn_chunks_only(d_q, dp, d_ids, d_steps, (size_t) c.cap, s, d_scratch[1], n_q, 2, nullptr);
    strata::kernels::qsa_decode_attn_merge_only(d_steps, (size_t) c.cap, s, d_scratch[1], d_attn[1], n_q, 0, nullptr);
    strata::kernels::qsa_decode_attn_chunks_only(d_q, dp, d_ids, d_steps, (size_t) c.cap, s, d_scratch[2], n_q, 0, nullptr);
    strata::kernels::qsa_decode_attn_merge_only(d_steps, (size_t) c.cap, s, d_scratch[2], d_attn[2], n_q, 0, nullptr);
    strata::kernels::qsa_decode_attn_batch(d_q, dp, d_ids, d_steps, (size_t) c.cap, s, d_scratch[3], d_attn[3], n_q, nullptr);
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "case %s: %s\n", c.name, cudaGetErrorString(cudaGetLastError()));
        ++failures;
        return 1;
    }

    std::vector<float> a(attn_floats), b(attn_floats), again(attn_floats), dispatched(attn_floats);
    cudaMemcpy(a.data(), d_attn[0], attn_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_attn[1], attn_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(again.data(), d_attn[2], attn_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dispatched.data(), d_attn[3], attn_floats * sizeof(float), cudaMemcpyDeviceToHost);

    if (!all_finite(a, c.name) || !all_finite(b, c.name) || !all_finite(dispatched, c.name)) ++failures;
    if (memcmp(a.data(), again.data(), attn_floats * sizeof(float)) != 0) {
        std::fprintf(stderr, "case %s: the default dispatch is not bitwise stable across two runs\n", c.name);
        ++failures;
    }
    const char* tc_env = std::getenv("STRATA_ATTN_DECODE_TC");
    const bool expect_tc_dispatch = tc_env && tc_env[0] == '1' && tc_env[1] == '\0';
    const float* expected_dispatch = expect_tc_dispatch ? b.data() : a.data();
    if (std::memcmp(dispatched.data(), expected_dispatch, attn_floats * sizeof(float)) != 0) {
        std::fprintf(stderr, "case %s: production batch dispatch did not select the expected %s path\n", c.name,
                     expect_tc_dispatch ? "TC" : "default");
        ++failures;
    }
    std::vector<double> reference;
    host_double_reference(c, s, q, k_q, k_scale, v_q4, page_table, ids, reference);
    double ref_scale = 0.0;
    const double default_error = max_abs_error(a, reference, &ref_scale);
    double tc_ref_scale = 0.0;
    const double tc_error = max_abs_error(b, reference, &tc_ref_scale);
    ref_scale = std::max(ref_scale, tc_ref_scale);
    double tc_default = 0.0, default_scale = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        tc_default = std::max(tc_default, (double) std::fabs(a[i] - b[i]));
        default_scale = std::max(default_scale, (double) std::fabs(a[i]));
    }
    size_t nz_a = 0, nz_b = 0;
    for (size_t i = 0; i < a.size(); ++i) { if (a[i] != 0.0f) ++nz_a; if (b[i] != 0.0f) ++nz_b; }
    std::printf("case %s: nonzero default=%zu TC=%zu; default-vs-math %.3e, TC-vs-math %.3e, TC-vs-default %.3e, ref_scale %.3e\n",
                c.name, nz_a, nz_b, default_error, tc_error, tc_default, ref_scale);
    // Match prompt attention's i8v3 parity policy: accept only if the candidate's error against the FP64
    // mathematical reference is within four times the measured default-path error, with a scale-relative floor.
    report(c.name, "TC-vs-math", tc_error, std::max(4.0 * default_error, 1e-6 * ref_scale));
    report(c.name, "TC-vs-default", tc_default, 1e-4 * default_scale);

    if (c.width == 0) {
        // An empty split must produce exact zeros: the chunk pass writes only masked partials (-FLT_MAX, 0).
        bool zeros = true;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != 0.0f || b[i] != 0.0f) { zeros = false; break; }
        if (!zeros) { std::fprintf(stderr, "case %s: empty split must produce exact zeros\n", c.name); ++failures; }
    }

    cudaFree(d_kq); cudaFree(d_ks); cudaFree(d_vq); cudaFree(d_q); cudaFree(d_ids); cudaFree(d_pt); cudaFree(d_steps);
    for (int i = 0; i < 4; ++i) { cudaFree(d_scratch[i]); cudaFree(d_attn[i]); }
    return 0;   // failures accumulate in the global; every case runs so the diagnostics stay readable
}

}  // namespace

int main(void) {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    std::printf("skip (no CUDA device)\n");
    return 77;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 77;
  if (prop.major < 8) {   // the tensor-core score phase traps below sm_80; a non-Ampere run proves nothing
    std::printf("skip (compute capability %d.%d is not Ampere or newer)\n", prop.major, prop.minor);
    return 77;
  }

  QsaShapes s;
  s.n_head = strata::core::kCompiledGeometry.n_head;
  s.n_head_kv = strata::core::kCompiledGeometry.n_head_kv;
  s.head_dim = strata::core::kCompiledGeometry.head_dim;
  s.n_rot = strata::core::kCompiledGeometry.n_rot;
  s.idx_n_head = strata::core::kCompiledGeometry.idx_q_heads;
  s.idx_dim = strata::core::kCompiledGeometry.idx_key_dim;
  s.idx_block = strata::core::kCompiledGeometry.idx_block;
  s.idx_top_k = strata::core::kCompiledGeometry.idx_top_k;
  s.page_size = 1;
  const int basis_queries = (int) ((s.head_dim + s.n_head - 1) / s.n_head);

  // Short and long attention, the tail block (a second chunk with one cell), an empty split, a paged gather
  // through a shuffled table, streamed slots whose pages were never written, a multi-query batch window, and a
  // one-hot sweep over all 256 dimensions with distinguishable K and V rows for cells 0 and 1.
  const Case cases[] = {
      {"short-1", 64, 1, 1, false, false, 1},
      {"w2", 64, 2, 1, false, false, 1},
      {"w8", 64, 8, 1, false, false, 1},
      {"w9", 64, 9, 1, false, false, 1},
      {"full-64", 256, 64, 1, false, false, 1},
      {"long-200", 256, 200, 1, false, false, 1},
      {"tail-65", 128, 65, 1, false, false, 1},
      {"empty-split", 64, 0, 1, false, false, 1},
      {"paged-shuffle", 256, 120, 16, true, false, 1},
      {"streamed-mask", 256, 150, 16, false, true, 1},
      {"multi-query", 128, 40, 1, false, false, 2},
      {"long-200-lowcode", 256, 200, 1, false, false, 1},
      {"short-2-unitk", 64, 2, 1, false, false, 1},
      {"full-64-unitk", 256, 64, 1, false, false, 1},
      {"short-2-unitk32", 64, 2, 1, false, false, 1},
      {"short-2-unitk96", 64, 2, 1, false, false, 1},
      {"short-2-unitk224", 64, 2, 1, false, false, 1},
      {"short-2-unitk1", 64, 2, 1, false, false, 1},
      {"short-2-unitk7", 64, 2, 1, false, false, 1},
      {"short-2-unitk15", 64, 2, 1, false, false, 1},
      {"short-2-unitk31", 64, 2, 1, false, false, 1},
      {"short-2-cellbasis", 64, 2, 1, false, false, basis_queries},
      {"short-2-onq", 64, 2, 1, false, false, 1},
      {"long-200-onq", 256, 200, 1, false, false, 1},
      {"short-2-constk", 64, 2, 1, false, false, 1},
      {"long-200-constk", 256, 200, 1, false, false, 1},
      {"short-2-tinyk", 64, 2, 1, false, false, 1},
      {"long-200-tinyk", 256, 200, 1, false, false, 1},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const int rc = run_case(cases[i], s);
    if (rc == 77) return 77;
  }
  if (failures != 0) {
    std::fprintf(stderr, "qsa_decode_tc_parity: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("qsa_decode_tc_parity: all cases passed\n");
  return 0;
}
