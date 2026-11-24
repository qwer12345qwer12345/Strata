#include "strata/kernels/ple.hpp"
#include "strata/kernels/ngram.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace strata::kernels;
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    constexpr int rows = 411; // 65760 values: every encoding and an odd final block
    const size_t n = rows * PLE_HEAD_DIM;
    std::vector<uint8_t> raw(n*2);
    for (size_t i=0;i<n;++i) { raw[2*i]=uint8_t(i); raw[2*i+1]=uint8_t(i>>8); }
    uint8_t* d = nullptr; float* out = nullptr;
    if (cudaMalloc((void**)&d,raw.size()) != cudaSuccess || cudaMalloc((void**)&out,n*4) != cudaSuccess) return 1;
    cudaStream_t upload, compute; cudaEvent_t copied, used;
    cudaStreamCreateWithFlags(&upload,cudaStreamNonBlocking); cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&copied,cudaEventDisableTiming); cudaEventCreateWithFlags(&used,cudaEventDisableTiming);
    std::vector<uint32_t> got(n);
    for (int pass=0;pass<5;++pass) {
        cudaStreamWaitEvent(upload,used,0);
        cudaMemcpyAsync(d,raw.data(),raw.size(),cudaMemcpyHostToDevice,upload);
        cudaEventRecord(copied,upload); cudaStreamWaitEvent(compute,copied,0);
        ple_bf16_widen(d,rows,out,compute); cudaEventRecord(used,compute);
        if (cudaStreamSynchronize(compute) != cudaSuccess) return 1;
        if (cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        for (size_t i=0;i<n;++i) {
            const uint32_t want=(uint32_t(raw[2*i]) | uint32_t(raw[2*i+1])<<8)<<16;
            if (got[i]!=want) { std::fprintf(stderr,"BF16 bits differ at %zu: %08x != %08x\n",i,got[i],want);return 1; }
        }
    }
    cudaFree(d);cudaFree(out);cudaStreamDestroy(upload);cudaStreamDestroy(compute);cudaEventDestroy(copied);cudaEventDestroy(used);
    std::printf("ple_bf16_widen_test: all 65536 encodings exact, 5 cross-stream passes\n");return 0;
}
