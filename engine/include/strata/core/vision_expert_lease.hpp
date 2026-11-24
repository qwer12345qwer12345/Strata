#pragma once
// A paused single-GPU vision turn may lend physical cache segments without changing
// expert ownership. The command dispatcher MUST block GPU work while active().
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
struct VisionLeaseOps {
    std::function<int64_t()> mapped, slots;
    std::function<bool(int64_t,std::string&)> shrink, grow;
    std::function<bool(int32_t,int32_t,std::string&)> refill; // residency index, original slot; synchronous
    std::function<bool(std::string&)> finish;
};

class VisionExpertLease {
    bool held_ = false;
    int64_t original_bytes_ = 0, original_slots_ = 0, kept_slots_ = 0;
    std::vector<int32_t> original_res_;
public:
    bool active() const { return held_; }
    int64_t original_bytes() const { return original_bytes_; }
    int64_t original_slots() const { return original_slots_; }
    int64_t kept_slots() const { return kept_slots_; }
    size_t restore_experts() const {
        size_t n=0;for(int32_t slot:original_res_)if(slot>=kept_slots_)++n;return n;
    }
    bool restore(const VisionLeaseOps& ops, const std::vector<int32_t>& res, std::string& err) {
        if (!held_) return true; // retry after an acknowledged END is harmless
        if (res != original_res_) { err="vision lease: expert ownership changed during the lease";return false; }
        if (!ops.grow(original_bytes_,err) || ops.mapped()!=original_bytes_ || ops.slots()!=original_slots_) {
            if(err.empty())err="vision lease: original cache mapping is not fully restored";
            return false; // retain the snapshot and gate; END may be retried
        }
        for(size_t i=0;i<original_res_.size();++i) {
            const int32_t slot=original_res_[i];
            if(slot>=kept_slots_ && !ops.refill((int32_t)i,slot,err))return false;
        }
        if(!ops.finish(err))return false;
        held_=false;original_res_.clear();
        return true;
    }
    bool begin(int64_t keep_bytes, const VisionLeaseOps& ops,
               const std::vector<int32_t>& res, std::string& err) {
        if(held_){err="vision lease: already active; restore it before another borrow";return false;}
        original_bytes_=ops.mapped();original_slots_=ops.slots();
        if(keep_bytes<0 || keep_bytes>original_bytes_){err="vision lease: invalid retained byte count";return false;}
        // Allocate metadata before releasing any VRAM. Never copy the expert complement.
        original_res_=res;kept_slots_=original_slots_;held_=true;
        const bool ok=ops.shrink(keep_bytes,err);
        kept_slots_=ops.slots(); // includes partial driver failure
        if(ok)return true;
        const std::string cause=err;std::string rollback;
        if(!restore(ops,res,rollback))err=cause+"; rollback incomplete: "+rollback;
        else err=cause+"; original cache restored";
        return false;
    }
};
} // namespace strata::core
