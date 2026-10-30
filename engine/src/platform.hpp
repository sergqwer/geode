// platform.hpp - what the engine needs from the OS: positional file reads from any thread, aligned memory, a clock.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace geode {

// A file read at explicit offsets, safely from several threads at once (pread; ReadFile with an offset on Windows).
class RandomFile {
public:
    RandomFile() = default;
    ~RandomFile();
    RandomFile(const RandomFile &) = delete;
    RandomFile &operator=(const RandomFile &) = delete;

    // direct: O_DIRECT where the OS has it, for reads whose offset, length and buffer are 4096-aligned (they go
    // straight into the caller's buffer, no page cache copy); other reads, or a file system that refuses O_DIRECT,
    // use a buffered descriptor.
    bool open(const std::string &path, std::string &err, bool direct = false);
    bool direct() const { return direct_; }
    // Exactly n bytes at offset; false on an error or a short read.
    bool read_at(uint64_t offset, void *dst, size_t n) const;
    uint64_t size() const { return size_; }
    bool is_open() const;

private:
#if defined(_WIN32)
    void *handle_ = nullptr;
#else
    int fd_ = -1, dfd_ = -1;
#endif
    uint64_t size_ = 0;
    bool direct_ = false;
};

struct AlignedFree {
    void operator()(uint8_t *p) const;
};
using AlignedBuffer = std::unique_ptr<uint8_t[], AlignedFree>;

// 4096-aligned: fine for SIMD loads and for O_DIRECT reads into it.
AlignedBuffer aligned_bytes(size_t n);

double now_seconds();

// The system's CPU performance hints (Android's APerformanceHint, API 33+, looked up at run time): the threads that
// compute say how long each piece of work took against a short target, so the governor keeps their cores' clocks
// up while a prompt or an answer runs (else they sat at 1-1.4 GHz on the 8 Elite between the GPU's kernels). No-ops
// where there is none (the PC, older Android) or with GEODE_ADPF=0.
void perf_thread();               // the calling thread computes: into the session
void perf_work(double seconds);   // a piece of work just took this long
bool perf_active();               // a session is open (the profile says so)

// Gives the physical pages of [p, p + n) back to the system (they read as zeros if touched again); the range stays
// allocated. Rounds inward to whole pages.
void release_pages(void *p, size_t n);

}  // namespace geode
