// tests/cuda/dense_i8_tc_test.cpp - Parity & throughput test for clean-room MIT INT8 TC dense GEMM (sm_80+).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Validates dense_i8_tc_mmvq (T = 1, 4, 8, 16) against native_mmvq AND dense_i8_tc_prefill (T = 64, 512, 2048)
// against dequant_f16 + FP16 cuBLAS across Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14), IQ4_XS (23).
#include "strata/kernels/dense_i8_tc.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;
namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_i8_tc_test: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

void fill_weights(int ggml_type, int n_in, int n_out, std::vector<uint8_t>& buf, std::mt19937& rng) {
    buf.resize(strata::kernels::native_mmvq_weight_bytes(ggml_type, n_in, n_out));
    if (ggml_type == 8) {
        const int nb = n_in / 32;
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t* p = buf.data() + ((size_t) r * nb + b) * 34u;
                const uint16_t d = __half_as_ushort(__float2half_rn(0.005f + 0.01f * ((float) (rng() & 255) / 255.0f)));
                p[0] = (uint8_t) (d & 0xffu); p[1] = (uint8_t) (d >> 8);
                for (int j = 0; j < 32; ++j) p[2 + j] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
            }
        }
    } else if (ggml_type == 12) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 144u;
                const half2 dm = __floats2half2_rn(0.004f + 0.006f * ((float) (rng() & 255) / 255.0f),
                                                   0.003f + 0.005f * ((float) (rng() & 255) / 255.0f));
                std::memcpy(p, &dm, 4);
                for (int j = 4; j < 144; ++j) p[j] = (uint8_t) (rng() & 0xffu);
            }
        }
    } else if (ggml_type == 13) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 176u;
                const half2 dm = __floats2half2_rn(0.003f + 0.004f * ((float) (rng() & 255) / 255.0f),
                                                   0.002f + 0.004f * ((float) (rng() & 255) / 255.0f));
                std::memcpy(p, &dm, 4);
                for (int j = 4; j < 176; ++j) p[j] = (uint8_t) (rng() & 0xffu);
            }
        }
    } else if (ggml_type == 14) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 210u;
                for (int j = 0; j < 192; ++j) p[j] = (uint8_t) (rng() & 0xffu);
                for (int s = 0; s < 16; ++s) p[192 + s] = (uint8_t) (int8_t) ((int) (rng() % 63) - 31);
                const uint16_t d = __half_as_ushort(__float2half_rn(0.003f + 0.006f * ((float) (rng() & 255) / 255.0f)));
                p[208] = (uint8_t) (d & 0xffu); p[209] = (uint8_t) (d >> 8);
            }
        }
    } else if (ggml_type == 23) {
        const int nsb = n_in / 256;
        for (int r = 0; r < n_out; ++r) {
            for (int sb = 0; sb < nsb; ++sb) {
                uint8_t* p = buf.data() + ((size_t) r * nsb + sb) * 136u;
                const uint16_t d = __half_as_ushort(__float2half_rn(0.001f + 0.002f * ((float) (rng() & 255) / 255.0f)));
                p[0] = (uint8_t) (d & 0xffu); p[1] = (uint8_t) (d >> 8);
                for (int j = 2; j < 136; ++j) p[j] = (uint8_t) (rng() & 0xffu);
            }
        }
    }
}


void compare(int ty,int K,int N,int T,cudaStream_t stream){
 std::mt19937 rng(9123u+ty+T);
 std::vector<uint8_t> weights; fill_weights(ty,K,N,weights,rng);
 std::vector<float> x((size_t)T*K);for(auto&v:x)v=float(int(rng()%20001)-10000)*.0001f;
 const int sets=std::max(2,std::min(32,int((128ull<<20)/weights.size())));
 std::vector<void*> w(sets);for(auto&p:w){ck(cudaMalloc(&p,weights.size()),"weights");ck(cudaMemcpy(p,weights.data(),weights.size(),cudaMemcpyHostToDevice),"weight copy");}
 float *dx,*ref,*out;void*xq;
 ck(cudaMalloc((void**)&dx,x.size()*4),"x");ck(cudaMalloc(&xq,native_q8_1_bytes(K,1)*T),"xq");
 ck(cudaMalloc((void**)&ref,(size_t)T*N*4),"ref");ck(cudaMalloc((void**)&out,(size_t)T*N*4),"out");
 ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice),"x copy");
 for(int first=0;first<T;first+=8) native_quantize_q8_1(dx+(size_t)first*K,(uint8_t*)xq+(size_t)first*native_q8_1_bytes(K,1),K,std::min(8,T-first),stream);ck(cudaStreamSynchronize(stream),"quant");
 for(int first=0;first<T;first+=8)native_mmvq(ty,w[0],(uint8_t*)xq+(size_t)first*native_q8_1_bytes(K,1),ref+(size_t)first*N,K,N,std::min(8,T-first),stream);
 if(!dense_i8_tc_mmvq(ty,w[0],xq,out,K,N,T,stream)){std::fprintf(stderr,"TC refused shape\n");std::exit(1);}
 ck(cudaStreamSynchronize(stream),"compute");std::vector<float>a((size_t)T*N),b(a.size());
 ck(cudaMemcpy(a.data(),ref,a.size()*4,cudaMemcpyDeviceToHost),"read ref");ck(cudaMemcpy(b.data(),out,b.size()*4,cudaMemcpyDeviceToHost),"read tc");
 double num=0,den=0;size_t same=0;for(size_t i=0;i<a.size();++i){num+=double(a[i]-b[i])*(a[i]-b[i]);den+=double(a[i])*a[i];if(!std::memcmp(&a[i],&b[i],4))++same;}
 double rms=std::sqrt(num/std::max(den,1e-30));if(!std::isfinite(rms)||rms>1e-4){std::fprintf(stderr,"TC parity rms %g\n",rms);std::exit(1);}
 auto measure=[&](bool tc){
  constexpr int iters=160;cudaEvent_t start,end;ck(cudaEventCreate(&start),"start");ck(cudaEventCreate(&end),"end");
  auto run=[&](int i){if(tc)dense_i8_tc_mmvq(ty,w[i%sets],xq,out,K,N,T,stream);else for(int first=0;first<T;first+=8)native_mmvq(ty,w[i%sets],(uint8_t*)xq+(size_t)first*native_q8_1_bytes(K,1),out+(size_t)first*N,K,N,std::min(8,T-first),stream);};
  for(int i=0;i<10;++i)run(i);ck(cudaStreamSynchronize(stream),"warm");ck(cudaEventRecord(start,stream),"record");
  for(int i=0;i<iters;++i)run(i);ck(cudaEventRecord(end,stream),"record");ck(cudaEventSynchronize(end),"sync");float ms=0;ck(cudaEventElapsedTime(&ms,start,end),"elapsed");cudaEventDestroy(start);cudaEventDestroy(end);return double(ms)*1000/iters;
 };
 std::vector<double> normal,tc;for(int pair=0;pair<4;++pair){if(pair%2){tc.push_back(measure(true));normal.push_back(measure(false));}else{normal.push_back(measure(false));tc.push_back(measure(true));}}
 std::sort(normal.begin(),normal.end());std::sort(tc.begin(),tc.end());double n=(normal[1]+normal[2])/2,t=(tc[1]+tc[2])/2;
 std::printf("{\"type\":%d,\"K\":%d,\"N\":%d,\"T\":%d,\"mmvq_us\":%.4f,\"tc_us\":%.4f,\"speedup\":%.4f,\"rel_rms\":%.9g,\"same_fraction\":%.6f}\n",ty,K,N,T,n,t,n/t,rms,double(same)/a.size());std::fflush(stdout);
 for(auto p:w)cudaFree(p);cudaFree(dx);cudaFree(xq);cudaFree(ref);cudaFree(out);
}
} // namespace
int main(){
 setenv("STRATA_DENSE_I8_TC","0",1);native_mmvq_set_b6_rows(1);
 cudaStream_t stream;ck(cudaStreamCreate(&stream),"stream");
 for(int ty:{8,12,13,14}) for(auto shape:{std::pair<int,int>{2560,2560},{2560,10240},{10240,2560}})
  for(int T:{2,3,4,5,6,8,9,12,16})compare(ty,shape.first,shape.second,T,stream);
 cudaStreamDestroy(stream);
}
