#include "strata/core/vision_expert_lease.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core;
int main(){
 VisionExpertLease lease;std::vector<int32_t> res={-1,3,0,2,1};const auto original=res;
 int64_t bytes=64,slots=4;int fail_grow=0,fail_refill=0,fail_shrink=0;int finish=0;
 std::vector<std::pair<int,int>> written;
 VisionLeaseOps ops;
 ops.mapped=[&]{return bytes;};ops.slots=[&]{return slots;};
 ops.shrink=[&](int64_t keep,std::string&e){bytes=keep;slots=keep/16;if(fail_shrink){e="injected partial shrink";return false;}return true;};
 ops.grow=[&](int64_t want,std::string&e){if(fail_grow){e="injected grow";return false;}bytes=want;slots=want/16;return true;};
 ops.refill=[&](int32_t i,int32_t s,std::string&e){if(fail_refill){e="injected read/copy";return false;}written.emplace_back(i,s);return true;};
 ops.finish=[&](std::string&){++finish;return true;};std::string err;
 assert(lease.begin(32,ops,res,err)&&lease.active()&&res==original&&lease.restore_experts()==2);
 assert(!lease.begin(16,ops,res,err));assert(lease.active());
 fail_grow=1;assert(!lease.restore(ops,res,err)&&lease.active()&&res==original);
 fail_grow=0;fail_refill=1;assert(!lease.restore(ops,res,err)&&lease.active());
 fail_refill=0;assert(lease.restore(ops,res,err)&&!lease.active());
 assert((written==std::vector<std::pair<int,int>>{{1,3},{3,2}}));assert(bytes==64&&finish==1);
 assert(lease.restore(ops,res,err)&&finish==1); // idempotent END
 written.clear();fail_shrink=1;assert(!lease.begin(16,ops,res,err)&&!lease.active());
 assert(bytes==64&&slots==4&&written.size()==3&&res==original);
 fail_grow=1;assert(!lease.begin(16,ops,res,err)&&lease.active());
 fail_grow=0;fail_shrink=0;assert(lease.restore(ops,res,err)&&!lease.active());
 assert(lease.begin(32,ops,res,err));res[1]=-1;assert(!lease.restore(ops,res,err)&&lease.active());
 res=original;assert(lease.restore(ops,res,err));
 assert(!lease.begin(-1,ops,res,err)&&!lease.active());
 std::cout<<"vision_expert_lease_test: PASS (borrow, partial shrink rollback, failed map/read retry, ownership gate, idempotent END)\n";
}
