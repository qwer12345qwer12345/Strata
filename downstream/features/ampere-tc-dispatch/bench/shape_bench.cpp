// SM86 cold-weight sweep, CUDA graph replay. JSONL; no host paging or inference.
#include "strata/kernels/dflash2.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using strata::kernels::Df2Mat;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){try{
 int repeats=argc>1?atoi(argv[1]):3, iters=argc>2?atoi(argv[2]):128;
 struct Shape{const char*name;int k,n,m;};
 Shape ss[]={{"gate_up",2560,7680,2},{"down",7680,2560,1},{"qkv",2560,4096,3},{"o",4096,2560,1},{"fc",12800,2560,1},{"kv",2560,1024,2},{"akp",2560,640,1},{"sel",2560,256,1}};
 std::mt19937 rng(451);cudaStream_t st;ck(cudaStreamCreate(&st));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
 for(auto sh:ss){
  if(argc>3 && std::strcmp(argv[3],sh.name)!=0)continue;
  // Each QKV segment has its own row count. Its planar scales immediately
  // follow that segment's nibbles, not the largest segment's allocation.
  std::vector<int> ns(sh.m,sh.n);if(sh.m==3){ns[1]=1024;ns[2]=1024;}
  size_t per_set=0;for(int n:ns)per_set+=size_t(sh.k)*n*18/32;
  size_t sets=std::max<size_t>(2,(128ull<<20)/per_set);
  std::vector<void*>w(sets*sh.m);
  for(size_t i=0;i<w.size();++i){int n=ns[i%sh.m];size_t wb=size_t(sh.k)*n*18/32;
   std::vector<unsigned char>hw(wb);for(auto&v:hw)v=rng()>>24;
   auto*d=(__half*)(hw.data()+size_t(sh.k)*n/2);
   for(int j=0;j<sh.k/32*n;++j)d[j]=__float2half(.002f+(rng()%31)*.001f);
   ck(cudaMalloc(&w[i],wb));ck(cudaMemcpy(w[i],hw.data(),wb,cudaMemcpyHostToDevice));
  }
  int8_t*x;float *d,*y;ck(cudaMalloc(&x,sh.k*8));ck(cudaMalloc(&d,sh.k/32*8*4));ck(cudaMalloc(&y,sh.n*sh.m*8*4));
  std::vector<int8_t>hx(sh.k*8);for(auto&v:hx)v=int(rng()%255)-127;ck(cudaMemcpy(x,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  std::vector<float>hd(sh.k/32*8);for(auto&v:hd)v=.002f+(rng()%31)*.001f;ck(cudaMemcpy(d,hd.data(),hd.size()*4,cudaMemcpyHostToDevice));
  auto call=[&](int i,int cols){Df2Mat ms[3];for(int j=0;j<sh.m;++j)ms[j]={w[(i%sets)*sh.m+j],y+j*sh.n*8,sh.n,true};if(sh.m==3){ms[1].n_out=1024;ms[2].n_out=1024;}strata::kernels::df2_gemm(ms,sh.m,x,d,sh.k,cols,st);};
  strata::kernels::df2_gemm_variant(17);auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
  while(std::chrono::steady_clock::now()<until){for(int i=0;i<128;++i)call(i,8);ck(cudaStreamSynchronize(st));}
  for(int cols=1;cols<=8;++cols){
   size_t ny=sh.n*sh.m*8;std::vector<float>ref(ny),now(ny);
   ck(cudaMemsetAsync(y,0,ny*4,st));strata::kernels::df2_gemm_variant(1);call(0,cols);ck(cudaStreamSynchronize(st));ck(cudaMemcpy(ref.data(),y,ny*4,cudaMemcpyDeviceToHost));
   std::vector<int>vars={1};for(int v=10;v<=51;++v)vars.push_back(v);
   for(int rep=0;rep<repeats;++rep){std::shuffle(vars.begin(),vars.end(),rng);
    for(int v:vars){strata::kernels::df2_gemm_variant(v);ck(cudaMemsetAsync(y,0,ny*4,st));call(0,cols);ck(cudaStreamSynchronize(st));ck(cudaMemcpy(now.data(),y,ny*4,cudaMemcpyDeviceToHost));bool eq=memcmp(now.data(),ref.data(),ny*4)==0;
     cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));for(int i=0;i<iters;++i)call(i,cols);ck(cudaStreamEndCapture(st,&g));ck(cudaGraphInstantiate(&ex,g,0));
     ck(cudaGraphLaunch(ex,st));ck(cudaStreamSynchronize(st));ck(cudaEventRecord(a,st));for(int k=0;k<4;++k)ck(cudaGraphLaunch(ex,st));ck(cudaEventRecord(b,st));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
     printf("{\"shape\":\"%s\",\"cols\":%d,\"variant\":%d,\"repeat\":%d,\"bitwise\":%s,\"us\":%.5f,\"weight_cycle_mib\":%.2f}\n",sh.name,cols,v,rep,eq?"true":"false",ms*1000/(iters*4),per_set*sets/1048576.);fflush(stdout);
     ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));if(!eq)return 3;
    }
   }
  }
  for(auto p:w)ck(cudaFree(p));ck(cudaFree(x));ck(cudaFree(d));ck(cudaFree(y));
 }
 return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
