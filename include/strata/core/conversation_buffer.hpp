#pragma once

#include "strata/core/conversation_segment_pool.hpp"
#include "strata/core/progress.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {

class ConversationBufferBacking {
public:
    static bool requested() {
        static const bool on = init();
        return on;
    }
    static bool disk_on() {
        return requested() && !failed_.load(std::memory_order_relaxed);
    }
    static const char* dir_label() { return dir_path(); }

    // Returns a pointer the caller sees as `n` usable bytes; a 64-byte header in front carries the
    // allocation's kind for drop_pages()/deallocate().
    static uint8_t* allocate(size_t n) {
        const size_t total = n + kHeader;
        if (disk_on()) {
            if (uint8_t* p = map_file(total, true)) return p + kHeader;
            say_failed();
        }
        uint8_t* p = map_file(total, false);   // anonymous MAP_PRIVATE
        return p ? p + kHeader : nullptr;
    }
    static void deallocate(uint8_t* p, size_t n) noexcept {
        if (p == nullptr) return;
#if defined(__linux__)
        ::munmap(p - kHeader, n + kHeader);
#endif
    }
    // Write the resident pages of [p, p+n) back and drop them from the process.  Anonymous memory
    // is left alone (DONTNEED there would destroy data).
    static void drop_pages(uint8_t* p, size_t n) noexcept {
        if (p == nullptr) return;
        uint8_t* base = p - kHeader;
        if (std::memcmp(base, kMagicFile, sizeof kMagicFile) != 0) return;
#if defined(__linux__)
        ::msync(base, n + kHeader, MS_ASYNC);
        ::madvise(base, n + kHeader, MADV_DONTNEED);
#endif
    }

private:
    static constexpr size_t kHeader = 64;
    static constexpr char kMagicFile[8] = {'S', 'B', 'S', 'E', 'G', 'F', '1', '\0'};
    static constexpr char kMagicAnon[8] = {'S', 'B', 'S', 'E', 'G', 'A', '1', '\0'};
    static inline std::atomic<bool> failed_{false};

    static bool init() {
#if !defined(__linux__)
        return false;
#endif
        const char* dir = std::getenv("STRATA_CONV_DISK_DIR");
        if (dir == nullptr || dir[0] == '\0') return false;
        if (const char* off = std::getenv("STRATA_CONV_DISK"); off != nullptr && std::atoi(off) == 0) return false;
        return true;   // the directory itself is validated by the first allocation
    }
    static const char* dir_path() {
        static const char* dir = std::getenv("STRATA_CONV_DISK_DIR");
        return dir;
    }
    static void say_failed() {
        bool expect = false;
        if (failed_.compare_exchange_strong(expect, true))
            std::fprintf(stderr, "strata: STRATA_CONV_DISK_DIR=%s is not usable (%s); park snapshots stay in "
                                 "anonymous memory for this process\n", dir_path(), std::strerror(errno));
    }
    static uint8_t* map_file(size_t total, bool file) {
#if !defined(__linux__)
        (void) total; (void) file;
        return nullptr;
#else
        int fd = -1;
        char path_tmp[512] = "";
        if (file) {
            fd = ::open(dir_path(), O_TMPFILE | O_RDWR, 0600);
            if (fd < 0) {
                std::snprintf(path_tmp, sizeof path_tmp, "%s/seg-pid%d-XXXXXX", dir_path(), (int) ::getpid());
                fd = ::mkstemp(path_tmp);
                if (fd >= 0) ::unlink(path_tmp);   // unnamed from here on; gone when the mapping closes
            }
            if (fd < 0) return nullptr;
            if (::ftruncate(fd, (off_t) total) != 0) {   // a hole: fresh pages read as zeros, no disk traffic
                ::close(fd);
                return nullptr;
            }
        }
        void* base = ::mmap(nullptr, total, PROT_READ | PROT_WRITE,
                            file ? MAP_SHARED : (MAP_PRIVATE | MAP_ANONYMOUS), file ? fd : -1, 0);
        if (fd >= 0) ::close(fd);   // the mapping keeps the inode alive either way
        if (base == MAP_FAILED) return nullptr;
        std::memcpy(base, file ? static_cast<const void*>(kMagicFile) : static_cast<const void*>(kMagicAnon), 8);
        ::madvise(base, total, MADV_SEQUENTIAL);   // park writes and restore reads are linear
        return (uint8_t*) base;
#endif
    }
};


namespace conversation_buffer_detail {
// Where a segment's bytes come from: the process-wide segment pool (STRATA_PARK_SEGPOOL_MIB, conversation_segment_pool.hpp)
// when one is installed and has a free run for it, else ::operator new as always.  The pool decides by pointer on the way
// back, so allocate and deallocate always agree.
inline void* segment_alloc(size_t bytes) {
    if (ConversationBufferBacking::requested()) {
        if (void* p = ConversationBufferBacking::allocate(bytes)) return p;
        throw std::bad_alloc();
    }
    if (SegmentPool* pool = segment_pool())
        if (void* p = pool->allocate(bytes)) return p;
    return ::operator new(bytes);
}
inline void segment_free(void* p, size_t bytes) noexcept {
    if (ConversationBufferBacking::requested()) {
        ConversationBufferBacking::deallocate(static_cast<uint8_t*>(p), bytes);
        return;
    }
    if (SegmentPool* pool = segment_pool())
        if (pool->deallocate(p, bytes)) return;
    ::operator delete(p);
}
}   // namespace conversation_buffer_detail

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    // STRATA_CONV_DISK (local patch): whether a park snapshot's bytes are backed by a file.
    static bool disk_backed() { return ConversationBufferBacking::disk_on(); }
    // Hand every segment's resident pages back to the kernel (written back first).  Safe next to a
    // writer: pages being written stay until they are clean.  No effect without the env.
    void drop_resident() {
        if (!ConversationBufferBacking::requested()) return;
        for (auto& segment : segments_)
            ConversationBufferBacking::drop_pages(segment.data(), segment.capacity());
    }
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) { resize_impl(n, &value); }
    void resize_impl(size_t n, const uint8_t* value) {
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                if (value) segments_.back().resize(segments_.back().size() + extend, *value);
                else segments_.back().resize(segments_.back().size() + extend);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing, held));
                if (value) segment.resize(count, *value);
                else segment.resize(count);
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }
    // resize() without the zero fill of what it adds (the fresh pages are not touched here): only for a caller that
    // writes every added byte before anything reads it (STRATA_PARK_ASYNC's background copy)
    void resize_for_overwrite(size_t n) { resize_impl(n, nullptr); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    // value-initializes only when asked (resize's explicit fill value): resize_for_overwrite leaves the bytes alone
    template<class T> struct DefaultInit : std::allocator<T> {
        template<class U> struct rebind { using other = DefaultInit<U>; };
        DefaultInit() = default;
        template<class U> DefaultInit(const DefaultInit<U>&) noexcept {}
        T* allocate(size_t n) { return static_cast<T*>(conversation_buffer_detail::segment_alloc(n * sizeof(T))); }
        void deallocate(T* p, size_t n) noexcept { conversation_buffer_detail::segment_free(p, n * sizeof(T)); }
        template<class U> void construct(U* p) { ::new ((void*) p) U; }
        template<class U, class... A> void construct(U* p, A&&... a) { ::new ((void*) p) U(std::forward<A>(a)...); }
    };
    using Segment = std::vector<uint8_t, DefaultInit<uint8_t>>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

} // namespace strata::core
