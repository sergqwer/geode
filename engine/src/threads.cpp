// threads.cpp - ThreadPool: workers spin, then sleep on a condition variable, between products; a dynamic product's
// chunks are taken with an atomic counter, a weighted product's pieces are dealt by speed (threads.hpp).
#include "threads.hpp"

#include "platform.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

namespace geode {

namespace {

// how long a thread keeps spinning for the next product before it sleeps
constexpr auto kSpin = std::chrono::milliseconds(3);
// a timed weighted round every this many (the threads read the clock around their shares then)
constexpr uint64_t kTimeEvery = 4;
// the weight follows the timings once this many timed rounds had both kinds of thread in them
constexpr int kLearnRounds = 16;

thread_local bool t_helper = false;

}  // namespace

ThreadPool::ThreadPool(int n, std::vector<int> cpus, int helpers, std::vector<int> helper_cpus) {
    // more threads than cores to pin them to would spin against each other on those cores (a product's threads wait
    // for the slowest): then the scheduler places them
    if (static_cast<int>(cpus.size()) < n) cpus.clear();
    if (static_cast<int>(helper_cpus.size()) < helpers) helper_cpus.clear();
    const int nb = std::max(1, n), nh = std::max(0, helpers);
    shares_ = std::make_unique<Share[]>(static_cast<size_t>(nb + nh));
    parts_.assign(static_cast<size_t>(nb + nh), 0);
    for (int i = 1; i < nb; ++i)
        workers_.emplace_back([this, cpus, i] {
            pin_current_thread(cpus);
            perf_thread();
            worker_loop(false, i);
        });
    for (int i = 0; i < nh; ++i)
        helpers_.emplace_back([this, helper_cpus, i] {
            pin_current_thread(helper_cpus);
            perf_thread();
            t_helper = true;
            worker_loop(true, i);
        });
}

void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

bool ThreadPool::on_helper() { return t_helper; }

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        quit_.store(true);
    }
    wake_.notify_all();
    for (auto &t : workers_) t.join();
    for (auto &t : helpers_) t.join();
}

ThreadPool::Stats ThreadPool::stats() const {
    Stats s;
    s.weighted = weighted_count_.load(std::memory_order_relaxed);
    s.stolen = stolen_.load(std::memory_order_relaxed);
    s.learned = learned_.load(std::memory_order_relaxed);
    return s;
}

void ThreadPool::add_done(int64_t items) {
    if (items <= 0) return;
    if (done_items_.fetch_add(items) + items == n_ && caller_sleeps_.load()) {
        std::lock_guard<std::mutex> lock(mu_);
        done_.notify_one();
    }
}

void ThreadPool::run_chunks() {
    for (;;) {
        const int64_t b = next_.fetch_add(chunk_);
        if (b >= n_) break;
        const int64_t e = std::min(n_, b + chunk_);
        (*fn_)(b, e);
        add_done(e - b);
    }
}

int64_t ThreadPool::run_pieces(int64_t b, int64_t e) {
    const int64_t lo = b * piece_, hi = std::min(n_, e * piece_);
    if (lo >= hi) return 0;
    (*fn_)(lo, hi);
    return hi - lo;
}

int64_t ThreadPool::run_share(int me, bool caller) {
    Share &s = shares_[static_cast<size_t>(me)];
    int64_t items = 0, stolen = 0;
    if (s.next.load(std::memory_order_relaxed) < s.end) {
        const auto t0 = timed_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        int64_t own = 0;
        for (;;) {
            const int64_t b = s.next.fetch_add(s.step, std::memory_order_relaxed);
            if (b >= s.end) break;
            const int64_t e = std::min(s.end, b + s.step);
            items += run_pieces(b, e);
            own += e - b;
        }
        if (timed_ && own > 0) {
            s.own = own;
            s.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            s.round.store(round_, std::memory_order_release);
        }
    }
    // the others' pieces: from the share that would finish last, while it would finish later than one piece here
    // (a helper leaves a big thread's last pieces to it); the caller takes whatever is left, so a thread that never
    // came holds nothing up
    const double mine = s.speed;
    for (;;) {
        int victim = -1;
        double worst = 0.0;
        for (int i = 0; i < nparts_; ++i) {
            const int p = parts_[static_cast<size_t>(i)];
            if (p == me) continue;
            const Share &o = shares_[static_cast<size_t>(p)];
            const int64_t rem = o.end - o.next.load(std::memory_order_relaxed);
            if (rem <= 0) continue;
            const double t = static_cast<double>(rem) / o.speed;
            if (t > worst) {
                worst = t;
                victim = p;
            }
        }
        if (victim < 0 || (!caller && worst * mine < 1.0 - 1e-9)) break;
        Share &o = shares_[static_cast<size_t>(victim)];
        const int64_t b = o.next.fetch_add(1, std::memory_order_relaxed);
        if (b >= o.end) continue;
        items += run_pieces(b, b + 1);
        ++stolen;
    }
    if (stolen) stolen_.fetch_add(static_cast<uint64_t>(stolen), std::memory_order_relaxed);
    return items;
}

void ThreadPool::deal(int64_t p, const int *w, int t, int64_t *count) {
    int64_t W = 0;
    for (int i = 0; i < t; ++i) W += std::max(0, w[i]);
    for (int i = 0; i < t; ++i) count[i] = 0;
    if (W <= 0 || p <= 0) return;
    int64_t left = p;
    for (int i = 0; i < t; ++i) {
        count[i] = w[i] > 0 ? p * w[i] / W : 0;
        left -= count[i];
    }
    // identical pieces on threads of different speeds: each next one to whoever would finish it first (the shortest
    // schedule; the proportional shares are where that rule stands at p / W)
    for (; left > 0; --left) {
        int best = -1;
        for (int i = 0; i < t; ++i)
            if (w[i] > 0 && (best < 0 || (count[i] + 1) * w[best] < (count[best] + 1) * w[i])) best = i;
        ++count[best];
    }
}

void ThreadPool::set_width(int k) {
    k = std::max(1, k);
    if (width_.exchange(k) == k) return;
    std::lock_guard<std::mutex> lock(mu_);
    wake_.notify_all();
}

void ThreadPool::set_little(int k) {
    k = std::clamp(k, 0, helpers());
    if (little_.exchange(k) == k) return;
    std::lock_guard<std::mutex> lock(mu_);
    wake_.notify_all();
}

void ThreadPool::set_wide_helpers(int k) {
    k = std::max(0, k);
    if (wide_helpers_.exchange(k) == k) return;
    std::lock_guard<std::mutex> lock(mu_);
    wake_.notify_all();
}

void ThreadPool::set_little_weight(float w, bool learn) {
    little_w_.store(std::clamp(w, 0.05f, 1.5f));
    learn_ = learn;
    acc_pb_ = acc_sb_ = acc_pl_ = acc_sl_ = 0.0;
    acc_n_ = 0;
}

void ThreadPool::worker_loop(bool helper, int index) {
    // workers wake for every round (within the width), helpers for the rounds they are in
    std::atomic<uint64_t> &gen = helper ? helper_generation_ : state_;
    std::atomic<int> &sleepers = helper ? helper_sleepers_ : big_sleepers_;
    const int me = helper ? size() + index : index;
    uint64_t seen = gen.load();
    const auto benched = [&] {
        return helper ? index >= std::max(little_.load(), wide_helpers_.load()) : index >= width_.load();
    };
    const auto ready = [&] {
        const uint64_t g = gen.load();
        return benched() || (g != seen && (helper || (g & 1) != 0));
    };
    for (;;) {
        if (benched()) {   // outside the width: sleep, not spin, until it grows
            std::unique_lock<std::mutex> lock(mu_);
            wake_.wait(lock, [&] { return quit_.load() || !benched(); });
            if (quit_.load()) return;
            seen = gen.load();
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (!ready() && !quit_.load(std::memory_order_relaxed)) {
            cpu_relax();
            ++spins;
            // a helper shares its little core with the flash readers: let a woken one run
            if (helper && spins % 2048 == 0) std::this_thread::yield();
            if (spins % 256 == 0 && std::chrono::steady_clock::now() - t0 > kSpin) {
                std::unique_lock<std::mutex> lock(mu_);
                sleepers.fetch_add(1);
                wake_.wait(lock, [&] { return quit_.load() || ready(); });
                sleepers.fetch_sub(1);
                break;
            }
        }
        if (quit_.load()) return;
        if (benched()) continue;
        seen = gen.load(std::memory_order_acquire);
        const uint64_t s = state_.load();
        if ((s & 1) == 0) continue;   // closed already
        inside_.fetch_add(1);
        if (state_.load() == s) {
            if (!helper) note_cpu(s);
            if (weighted_) {
                if (helper ? index < nlit_ : index < nbig_) add_done(run_share(me, false));
            } else if (!helper || (wide_round_ && index < wide_n_)) {
                run_chunks();
            }
        }
        inside_.fetch_sub(1);
    }
}

void ThreadPool::note_cpu(uint64_t state) {
#if defined(__linux__)
    // every sixteenth round: getcpu is a system call on the phone (~300 ns), against a token's products of tens of us
    // and the tests' of one
    if (((state >> 1) & 15) != 0) return;
    const int c = sched_getcpu();
    if (c >= 0 && c < 64) cpus_.fetch_or(uint64_t{1} << c, std::memory_order_relaxed);
#else
    (void)state;
#endif
}

void ThreadPool::wait_done(int64_t n) {
    const auto t0 = std::chrono::steady_clock::now();
    int spins = 0;
    while (done_items_.load(std::memory_order_acquire) < n) {
        cpu_relax();
        if (++spins % 256 == 0 && std::chrono::steady_clock::now() - t0 > kSpin) {
            std::unique_lock<std::mutex> lock(mu_);
            caller_sleeps_.store(true);
            done_.wait(lock, [&] { return done_items_.load() >= n; });
            caller_sleeps_.store(false);
            break;
        }
    }
}

void ThreadPool::learn_from_round() {
    double pb = 0, sb = 0, pl = 0, sl = 0;
    for (int i = 0; i < nparts_; ++i) {
        const int p = parts_[static_cast<size_t>(i)];
        const Share &o = shares_[static_cast<size_t>(p)];
        if (o.round.load(std::memory_order_acquire) != round_) continue;
        (p < size() ? pb : pl) += static_cast<double>(o.own);
        (p < size() ? sb : sl) += o.secs;
    }
    if (pb <= 0 || pl <= 0 || sb <= 0 || sl <= 0) return;
    acc_pb_ += pb;
    acc_sb_ += sb;
    acc_pl_ += pl;
    acc_sl_ += sl;
    if (++acc_n_ < kLearnRounds) return;
    const double r = (acc_pl_ / acc_sl_) / (acc_pb_ / acc_sb_);
    const double w = 0.5 * little_w_.load() + 0.5 * r;
    little_w_.store(static_cast<float>(std::clamp(w, 0.05, 1.5)));
    learned_.fetch_add(1, std::memory_order_relaxed);
    acc_pb_ = acc_sb_ = acc_pl_ = acc_sl_ = 0.0;
    acc_n_ = 0;
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)> &fn, int64_t min_chunk,
                              bool wide, const std::function<void()> *first, bool learn) {
    if (n <= 0) {
        if (first) (*first)();
        return;
    }
    min_chunk = std::max<int64_t>(1, min_chunk);
    const int nlit = wide ? 0 : little();
    const int nwide = wide ? wide_helpers() : 0;
    const int nbig = width();
    const bool weighted = nlit > 0;
    const int threads = nbig + (weighted ? nlit : nwide);
    int64_t chunk = min_chunk, pieces = 0;
    if (weighted) {
        pieces = (n + min_chunk - 1) / min_chunk;
    } else {
        // a few chunks per thread so uneven cores balance out (more of them with the slow helpers), never below
        // min_chunk
        const int per = nwide > 0 ? 8 : 4;
        chunk = std::max<int64_t>(min_chunk, (n + threads * per - 1) / (threads * per));
        // a whole number of min_chunk (a product's 16 rows: a chunk of 86 split a quad, both threads computing it)
        chunk = (chunk + min_chunk - 1) / min_chunk * min_chunk;
    }
    if (threads == 1 || (weighted ? pieces <= 1 : n <= chunk)) {
        if (first) (*first)();
        fn(0, n);
        return;
    }
    fn_ = &fn;
    n_ = n;
    done_items_.store(0);
    weighted_ = weighted;
    if (weighted) {
        // the shares: the big threads 0 .. nbig - 1 (the caller 0, nothing of its own when it runs `first`), then the
        // helpers, each a contiguous run of pieces by its speed
        piece_ = min_chunk;
        nbig_ = nbig;
        nlit_ = nlit;
        nparts_ = 0;
        for (int i = 0; i < nbig; ++i) parts_[static_cast<size_t>(nparts_++)] = i;
        for (int j = 0; j < nlit; ++j) parts_[static_cast<size_t>(nparts_++)] = size() + j;
        const float lw = little_w_.load();
        int w[256];
        int64_t count[256];
        const int t = std::min(nparts_, 256);
        nparts_ = t;
        for (int i = 0; i < t; ++i) {
            const bool big = parts_[static_cast<size_t>(i)] < size();
            w[i] = i == 0 && first ? 0 : big ? 1024 : std::max(1, static_cast<int>(std::lround(lw * 1024.0f)));
        }
        deal(pieces, w, t, count);
        int64_t at = 0;
        for (int i = 0; i < t; ++i) {
            Share &s = shares_[static_cast<size_t>(parts_[static_cast<size_t>(i)])];
            s.next.store(at, std::memory_order_relaxed);
            s.end = at + count[i];
            s.step = std::max<int64_t>(1, count[i] / 8);   // a share in ~8 takes: one taken and held up is not much
            s.speed = parts_[static_cast<size_t>(i)] < size() ? 1.0f : lw;
            at += count[i];
        }
        ++round_;
        timed_ = learn && learn_ && round_ % kTimeEvery == 0;
        weighted_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
        chunk_ = chunk;
        wide_round_ = nwide > 0;
        wide_n_ = nwide;
        next_.store(0);
    }
    const bool helpers_in = weighted || nwide > 0;
    const uint64_t opened = state_.fetch_add(1) + 1;   // open
    note_cpu(opened);
    if (helpers_in) helper_generation_.fetch_add(1);
    // wake only who sleeps (threads.hpp): the counts are read after the round opened
    if (big_sleepers_.load() > 0 || (helpers_in && helper_sleepers_.load() > 0)) {
        std::lock_guard<std::mutex> lock(mu_);
        wake_.notify_all();
    }
    if (first) (*first)();
    if (weighted) {
        add_done(run_share(0, true));
    } else {
        run_chunks();
    }
    // every item done (whoever did it); threads that never came to this round are not waited for
    wait_done(n);
    if (weighted && timed_) learn_from_round();
    state_.fetch_add(1);   // closed
    // a thread that entered checks the round and leaves (no chunk is left): only then may the fields change
    while (inside_.load() != 0) cpu_relax();
    fn_ = nullptr;
}

namespace {

// cpu -> max frequency (kHz), for the cpus that report one
std::vector<std::pair<int, int>> core_freqs() {
    std::vector<std::pair<int, int>> out;
#if defined(__ANDROID__)
    for (int cpu = 0; cpu < 16; ++cpu) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq");
        int khz = 0;
        if (f >> khz) out.emplace_back(cpu, khz);
    }
#endif
    return out;
}

// Each core's EAS capacity (/sys/devices/system/cpu/cpuN/cpu_capacity, 1024 = the fastest), else its highest
// clock: the kernel's own measure of what a core does, which a clock alone is not across core types.
std::vector<std::pair<int, int>> core_weights() {
    std::vector<std::pair<int, int>> out;
#if defined(__ANDROID__)
    for (int cpu = 0; cpu < 16; ++cpu) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpu_capacity");
        int cap = 0;
        if (f >> cap && cap > 0) out.emplace_back(cpu, cap);
    }
#endif
    return out;
}

// big: at least 60% of the fastest core's capacity. The Snapdragon 720G: 2 x A76 (its A55s are ~40%); the 8 Elite:
// all 8 Oryon cores (792 and 1024). Without capacities, the cores of the highest clock.
bool is_big(int weight, int best, bool capacities) { return capacities ? weight * 10 >= best * 6 : weight == best; }

std::vector<int> cores(bool big) {
    auto w = core_weights();
    const bool caps = !w.empty();
    if (!caps) w = core_freqs();
    int best = 0;
    for (const auto &[cpu, v] : w) best = std::max(best, v);
    std::vector<int> out;
    for (const auto &[cpu, v] : w)
        if (is_big(v, best, caps) == big) out.push_back(cpu);
    return out;
}

}  // namespace

std::vector<int> big_cores() { return cores(true); }

std::vector<int> little_cores() { return cores(false); }

std::string core_capacities() {
    auto w = core_weights();
    const char *unit = "";
    if (w.empty()) {
        w = core_freqs();
        unit = " kHz";
    }
    std::string s;
    for (const auto &[cpu, v] : w) s += (s.empty() ? "" : " ") + std::to_string(cpu) + ":" + std::to_string(v) + unit;
    return s;
}

void pin_current_thread(const std::vector<int> &cpus) {
#if defined(__linux__)
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof set, &set);
#else
    (void)cpus;
#endif
}

int default_compute_threads() {
#if defined(__ANDROID__)
    // the big cores (2 x A76 on the Snapdragon 720G, all 8 Oryon cores on the 8 Elite); the rest serve flash reads
    return std::max(1, static_cast<int>(big_cores().size()));
#else
    return std::max(1u, std::thread::hardware_concurrency());
#endif
}

int default_io_threads() {
#if defined(__ANDROID__)
    return 4;   // a block's read in 4 parts (ExpertStore::set_read_parts): UFS 4.0 serves them in parallel
#else
    return 4;
#endif
}

}  // namespace geode
