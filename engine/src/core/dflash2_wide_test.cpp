// Real Q4 drafter: wide bridge versus explicit context + unchanged one-row draft.
#include "strata/core/dflash2.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using strata::core::Dflash2Drafter;
int main(int argc,char**argv) {
 if(argc!=2){std::fprintf(stderr,"usage: dflash2_wide_test PACK\n");return 2;}
 try {
  auto ck=[](bool b,const std::string&s){if(!b)throw std::runtime_error(s);};
  auto cu=[&](cudaError_t e){ck(e==cudaSuccess,cudaGetErrorString(e));};
  constexpr int H=2560,T=12800,V=64;
  float *tap,*last,*emb; uint8_t*head;
  cu(cudaMalloc(&tap,16*T*4));cu(cudaMalloc(&last,8*T*4));cu(cudaMalloc(&emb,8*H*4));
  std::vector<float> e(8*H);for(size_t i=0;i<e.size();++i)e[i]=std::sin(float(i)*.013f)*.25f;
  cu(cudaMemcpy(emb,e.data(),e.size()*4,cudaMemcpyHostToDevice));
  // Packed Q8_0 native head, nonuniform signed values and finite fp16 scales.
  std::vector<uint8_t> hh(V*(H/32)*34);uint32_t rs=349;
  for(size_t b=0;b<hh.size();b+=34){hh[b]=0x1f;hh[b+1]=0x21;for(int j=2;j<34;++j){rs=rs*1664525+1013904223;hh[b+j]=rs>>24;}}
  cu(cudaMalloc(&head,hh.size()));cu(cudaMemcpy(head,hh.data(),hh.size(),cudaMemcpyHostToDevice));
  std::vector<int32_t> vocab(V);for(int i=0;i<V;++i)vocab[i]=i+10;
  auto embed=[&](const int32_t*,int n,float*out,cudaStream_t st){cu(cudaMemcpyAsync(out,emb,n*H*4,cudaMemcpyDeviceToDevice,st));};
  Dflash2Drafter ref,wide;std::string err;
  for(auto*d:{&ref,&wide}){ck(d->load(argv[1],0,err),err);ck(d->bind_raw(embed,head,8,V,vocab,err),err);d->set_max_drafts(5);}
  ck(ref.prepare(last,err),err);ck(wide.prepare(tap,err),err);
  const uint64_t initial=wide.vram_bytes();
  int64_t pos=2040;
  auto equal=[&](){std::vector<float>a,b;std::vector<int64_t>ta,tb;ck(ref.debug_ring(a,ta,err),err);ck(wide.debug_ring(b,tb,err),err);ck(a.size()==b.size()&&memcmp(a.data(),b.data(),a.size()*4)==0&&ta==tb,"ring or epoch tags differ");};
  int32_t a[7],b[7];float pa[7],pb[7];
  // Off and invalid calls must reject before modifying the context ring.
  unsetenv("STRATA_DF2_WIDE_CONTEXT");
  ck(!wide.draft_extended(tap,9,pos,4,b,pb,err),"wide unexpectedly enabled");equal();
  setenv("STRATA_DF2_WIDE_CONTEXT","1",1);
  for(int invalid:{0,17,-1}) { ck(!wide.draft_extended(tap,invalid,pos,4,b,pb,err),"invalid width accepted"); }
  equal();
  for(int rows:{9,1,12,4,16,8,9,12,16}) {
   std::vector<float> ht(rows*T);for(size_t i=0;i<ht.size();++i){rs=rs*1664525+1013904223;ht[i]=(int(rs>>16)-32768)/32768.f;}
   cu(cudaMemcpy(tap,ht.data(),ht.size()*4,cudaMemcpyHostToDevice));
   cu(cudaDeviceSynchronize()); // pageable H2D must complete before either nonblocking stream reads taps
   if(rows>8){for(int from=0;from<rows-1;from+=8)ck(ref.context(tap+from*T,std::min(8,rows-1-from),pos+from,err),err);cu(cudaMemcpyAsync(last,tap+(rows-1)*T,T*4,cudaMemcpyDeviceToDevice,ref.stream()));ck(ref.draft(last,1,pos+rows-1,4,a,pa,err),err);}
   else {cu(cudaMemcpyAsync(last,tap,rows*T*4,cudaMemcpyDeviceToDevice,ref.stream()));ck(ref.draft(last,rows,pos,4,a,pa,err),err);}
   ck(wide.draft_extended(tap,rows,pos,4,b,pb,err),err);
   bool same=memcmp(a,b,sizeof(a))==0&&memcmp(pa,pb,sizeof(pa))==0;
   if(!same)for(int j=0;j<7;++j)std::fprintf(stderr,"rows=%d pick %d ref=%d %.9g wide=%d %.9g\n",rows,j,a[j],pa[j],b[j],pb[j]);
   equal();ck(same,"draft tokens or probabilities differ");
   std::vector<float> fn;ck(wide.debug_rows(fn,err),err);
   uint64_t hash=1469598103934665603ull;
   auto mix=[&](const void* data,size_t n){auto*p=static_cast<const unsigned char*>(data);for(size_t i=0;i<n;++i){hash^=p[i];hash*=1099511628211ull;}};
   mix(b,sizeof(b));mix(pb,sizeof(pb));mix(fn.data(),fn.size()*sizeof(float));
   std::printf("state rows=%d hash=%016llx\n",rows,(unsigned long long)hash);
   std::vector<int32_t> ids(rows);for(int i=0;i<rows;++i)ids[i]=int(pos+i)%V;
   ref.note_rows(ids.data(),rows,pos);wide.note_rows(ids.data(),rows,pos);
   std::printf("rows=%d pos=%lld tokens/probabilities/ring/tags bitwise equal\n",rows,(long long)pos);pos+=rows;
   // Changing spec invalidates both the normal and landing graph caches.
   if(rows==16){ref.set_max_drafts(3);wide.set_max_drafts(3);ck(ref.prepare(last,err),err);ck(wide.prepare(tap,err),err);}
  }
  ck(wide.wide_rounds==6&&wide.wide_rows==74,"wide counters differ");
  ck(initial==wide.vram_bytes(),"bridge allocated model or scratch VRAM");
  std::printf("dflash2_wide_test: PASS (6 wide + 3 narrow, ring wrap, graph invalidation, no extra arena)\n");
  // Drafters die before their bound buffers are released by CUDA process teardown.
  return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
