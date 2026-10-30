// platform.cpp - RandomFile, aligned memory and the clock for Windows (the PC parity runs) and Linux/Android.
#include "platform.hpp"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif
#if defined(__ANDROID__)
#include <dlfcn.h>
#include <algorithm>
#endif

namespace geode {

#if defined(_WIN32)

RandomFile::~RandomFile() {
    if (handle_) CloseHandle(static_cast<HANDLE>(handle_));
}

bool RandomFile::is_open() const { return handle_ != nullptr; }

bool RandomFile::open(const std::string &path, std::string &err, bool) {
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen > 0 ? wlen : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = "cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz;
    GetFileSizeEx(h, &sz);
    handle_ = h;
    size_ = static_cast<uint64_t>(sz.QuadPart);
    return true;
}

bool RandomFile::read_at(uint64_t offset, void *dst, size_t n) const {
    auto *p = static_cast<uint8_t *>(dst);
    while (n > 0) {
        const DWORD chunk = static_cast<DWORD>(n > (1u << 30) ? (1u << 30) : n);
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(offset);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD got = 0;
        if (!ReadFile(static_cast<HANDLE>(handle_), p, chunk, &got, &ov) || got == 0) return false;
        p += got;
        offset += got;
        n -= got;
    }
    return true;
}

void AlignedFree::operator()(uint8_t *p) const { _aligned_free(p); }

AlignedBuffer aligned_bytes(size_t n) {
    return AlignedBuffer(static_cast<uint8_t *>(_aligned_malloc(n ? n : 1, 4096)));
}

#else

RandomFile::~RandomFile() {
    if (fd_ >= 0) ::close(fd_);
    if (dfd_ >= 0) ::close(dfd_);
}

bool RandomFile::is_open() const { return fd_ >= 0; }

bool RandomFile::open(const std::string &path, std::string &err, bool direct) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        err = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
#if defined(O_DIRECT)
    // a second descriptor for the aligned reads; the buffered one stays for anything unaligned
    if (direct) dfd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
    direct_ = dfd_ >= 0;
    struct stat st {};
    fstat(fd_, &st);
    size_ = static_cast<uint64_t>(st.st_size);
    return true;
}

bool RandomFile::read_at(uint64_t offset, void *dst, size_t n) const {
    auto *p = static_cast<uint8_t *>(dst);
    const bool aligned = dfd_ >= 0 && offset % 4096 == 0 && n % 4096 == 0 && reinterpret_cast<uintptr_t>(dst) % 4096 == 0;
    const int fd = aligned ? dfd_ : fd_;
    while (n > 0) {
        const ssize_t got = ::pread(fd, p, n, static_cast<off_t>(offset));
        if (got < 0 && errno == EINTR) continue;
        if (got < 0 && errno == EFAULT && fd == dfd_) {
            // memory O_DIRECT cannot pin (a dma-buf's mapping: the NPU's share of the expert cache): through an aligned
            // buffer of this thread's, still past the page cache
            thread_local AlignedBuffer bounce;
            constexpr size_t kBounce = size_t{2} << 20;
            if (!bounce) bounce = aligned_bytes(kBounce);
            if (!bounce) return false;
            while (n > 0) {
                const size_t chunk = n < kBounce ? n : kBounce;
                const ssize_t b = ::pread(dfd_, bounce.get(), chunk, static_cast<off_t>(offset));
                if (b < 0 && errno == EINTR) continue;
                if (b <= 0) return false;
                std::memcpy(p, bounce.get(), static_cast<size_t>(b));
                p += b;
                offset += static_cast<uint64_t>(b);
                n -= static_cast<size_t>(b);
            }
            return true;
        }
        if (got <= 0) return false;
        p += got;
        offset += static_cast<uint64_t>(got);
        n -= static_cast<size_t>(got);
    }
    return true;
}

void AlignedFree::operator()(uint8_t *p) const { std::free(p); }

AlignedBuffer aligned_bytes(size_t n) {
    void *p = nullptr;
    if (posix_memalign(&p, 4096, n ? n : 1) != 0) p = nullptr;
    return AlignedBuffer(static_cast<uint8_t *>(p));
}

#endif

void release_pages(void *p, size_t n) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    const uintptr_t first = (a + 4095) / 4096 * 4096, last = (a + n) / 4096 * 4096;
    if (last <= first) return;
#if defined(_WIN32)
    VirtualAlloc(reinterpret_cast<void *>(first), last - first, MEM_RESET, PAGE_READWRITE);
    // and out of the working set now (unlocking pages that are not locked takes them out of it)
    VirtualUnlock(reinterpret_cast<void *>(first), last - first);
#else
    madvise(reinterpret_cast<void *>(first), last - first, MADV_DONTNEED);
#endif
}

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

#if defined(__ANDROID__)
namespace {
// libandroid's APerformanceHint_* (the NDK's performance_hint.h), dlsym'd: the engine runs on Android versions
// without them too
struct Hint {
    void *(*get_manager)() = nullptr;
    void *(*create)(void *, const int32_t *, size_t, int64_t) = nullptr;
    int (*report)(void *, int64_t) = nullptr;
    int (*set_threads)(void *, const int32_t *, size_t) = nullptr;            // API 34
    int (*prefer_efficiency)(void *, bool) = nullptr;                          // API 35
    std::mutex mu;
    std::vector<int32_t> tids;
    void *session = nullptr;
    bool tried = false, ok = false, dirty = false;
    // the target a piece of work is held to: far below any real piece, so every report asks for more speed
    static constexpr int64_t kTargetNs = 1000000;
    bool load() {
        if (tried) return ok;
        tried = true;
        if (const char *e = std::getenv("GEODE_ADPF"); e && std::atoi(e) == 0) return false;
        void *lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) return false;
        get_manager = reinterpret_cast<decltype(get_manager)>(dlsym(lib, "APerformanceHint_getManager"));
        create = reinterpret_cast<decltype(create)>(dlsym(lib, "APerformanceHint_createSession"));
        report = reinterpret_cast<decltype(report)>(dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
        set_threads = reinterpret_cast<decltype(set_threads)>(dlsym(lib, "APerformanceHint_setThreads"));
        prefer_efficiency =
            reinterpret_cast<decltype(prefer_efficiency)>(dlsym(lib, "APerformanceHint_setPreferPowerEfficiency"));
        ok = get_manager && create && report;
        return ok;
    }
};
Hint &hint() {
    static Hint h;
    return h;
}
}  // namespace

void perf_thread() {
    Hint &h = hint();
    std::lock_guard<std::mutex> lock(h.mu);
    if (!h.load()) return;
    const auto tid = static_cast<int32_t>(gettid());
    for (int32_t t : h.tids)
        if (t == tid) return;
    h.tids.push_back(tid);
    h.dirty = true;
}

void perf_work(double seconds) {
    Hint &h = hint();
    std::lock_guard<std::mutex> lock(h.mu);
    if (!h.ok || h.tids.empty()) return;
    if (!h.session) {
        void *m = h.get_manager();
        h.session = m ? h.create(m, h.tids.data(), h.tids.size(), Hint::kTargetNs) : nullptr;
        if (!h.session) {
            h.ok = false;   // not for this process: no more tries
            return;
        }
        if (h.prefer_efficiency) h.prefer_efficiency(h.session, false);
        h.dirty = false;
    } else if (h.dirty && h.set_threads) {
        h.set_threads(h.session, h.tids.data(), h.tids.size());
        h.dirty = false;
    }
    h.report(h.session, std::max<int64_t>(1, static_cast<int64_t>(seconds * 1e9)));
}

bool perf_active() {
    Hint &h = hint();
    std::lock_guard<std::mutex> lock(h.mu);
    return h.session != nullptr;
}
#else
void perf_thread() {}
void perf_work(double) {}
bool perf_active() { return false; }
#endif

}  // namespace geode
