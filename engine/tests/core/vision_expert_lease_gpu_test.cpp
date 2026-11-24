#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_cache_refuse.hpp"
#include "strata/core/vision_expert_lease.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace strata::core;
static bool refuse_unmap(const char* name,int64_t seg){return std::string(name)=="unmap" && seg==3;}
int main(){
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
 try{
 auto ck=[](bool ok,const std::string&e){if(!ok)throw std::runtime_error(e);};
 auto cu=[&](cudaError_t e){ck(e==cudaSuccess,cudaGetErrorString(e));};
 for(int sized:{0,1}){
  ExpertCache cache;cache.set_segment_bytes(8ll<<20);std::string err;
  std::vector<int64_t> sizes(80,1<<20);if(sized)for(int i=0;i<40;++i)sizes[i]=1<<19;
  ck(sized?cache.open_sized(sizes,2,64,err):cache.open(80,2,64,1<<20,err),err);
  if(!cache.segmented())return 77;
  std::vector<int32_t> res(128,-1);std::vector<int64_t> offsets(128,0);std::FILE* disk=std::tmpfile();ck(disk!=nullptr,"tmpfile");
  std::vector<uint8_t> blob(1<<20);int64_t offset=0;
  for(int slot=0;slot<80;++slot){const int layer=slot/40,ex=slot%40,index=layer*64+ex;
   res[index]=cache.admit(layer,ex);ck(res[index]==slot,"slot admission");offsets[index]=offset;
   for(int64_t b=0;b<sizes[slot];++b)blob[b]=(uint8_t)((slot*19+b*37+(b>>9))%251);
   ck(std::fwrite(blob.data(),1,sizes[slot],disk)==(size_t)sizes[slot],"fixture write");offset+=sizes[slot];
   ck(cache.fill_slot_blocking(slot,blob.data(),err,sizes[slot]),err);
  }
  std::fflush(disk);const auto original=res;auto*base=cache.device_slot(0);const auto original_mapped=cache.mapped_bytes();
  uint8_t* pinned=nullptr;cu(cudaHostAlloc((void**)&pinned,1<<20,cudaHostAllocDefault));
  VisionLeaseOps ops;int fail_read=0;size_t bytes_read=0;
  ops.mapped=[&]{return cache.mapped_bytes();};ops.slots=[&]{return cache.slots();};
  ops.shrink=[&](int64_t n,std::string&e){return cache.shrink(n,e);};ops.grow=[&](int64_t n,std::string&e){return cache.grow(n,e);};
  ops.refill=[&](int32_t idx,int32_t slot,std::string&e){
   if(fail_read){e="injected original-file read failure";return false;}
   if(std::fseek(disk,(long)offsets[idx],SEEK_SET) || std::fread(pinned,1,sizes[slot],disk)!=(size_t)sizes[slot]){e="fixture read";return false;}
   bytes_read+=sizes[slot];return cache.fill_slot_blocking(slot,pinned,e,sizes[slot]);};
  ops.finish=[&](std::string&e){return cache.sync_queued(e);};
  VisionExpertLease lease;ck(lease.begin(20ll<<20,ops,res,err),err);ck(lease.active()&&res==original,"borrow ownership");
  ck(cache.mapped_bytes()<original_mapped,"VRAM not lent");
  // A separate encoder-like allocation really occupies the released physical VRAM.
  void* encoder=nullptr;cu(cudaMalloc(&encoder,16<<20));cu(cudaMemset(encoder,0x47,16<<20));cu(cudaFree(encoder));
  fail_read=1;ck(!lease.restore(ops,res,err)&&lease.active(),"read failure did not retain gate");
  fail_read=0;ck(lease.restore(ops,res,err),err);ck(!lease.active()&&res==original&&base==cache.device_slot(0),"restore identity");
  auto verify=[&]{for(int slot=0;slot<80;++slot){
   const int idx=(slot/40)*64+slot%40;std::fseek(disk,(long)offsets[idx],SEEK_SET);ck(std::fread(pinned,1,sizes[slot],disk)==(size_t)sizes[slot],"verify source");
   ck(cache.verify_slot(slot,pinned,err,sizes[slot]),err);
  }};verify();
  ec_test::refuse_segment_call=refuse_unmap;
  ck(!lease.begin(8ll<<20,ops,res,err),"injected partial unmap was not refused");
  ec_test::refuse_segment_call=nullptr;
  ck(!lease.active()&&cache.mapped_bytes()==original_mapped&&res==original,"partial-shrink rollback");verify();
  cu(cudaFreeHost(pinned));std::fclose(disk);
  std::printf("sized=%d PASS: original bytes/slots/addresses restored, partial unmap rollback, read retry, staging=1MiB, read=%zu bytes\n",sized,bytes_read);
 }
 std::puts("vision_expert_lease_gpu_test: PASS");return 0;
 }catch(const std::exception&e){ec_test::refuse_segment_call=nullptr;std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
