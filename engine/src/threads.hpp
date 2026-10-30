// threads.hpp - a fixed pool of compute threads for splitting one matrix product (or one batch of rows) across cores.
//
// Between products the threads spin for a little while before they sleep. A token is a few thousand short products;
// threads that sleep between them look half idle to the phone's frequency governor, which then keeps the big cores
// near their lowest clock (measured on the Redmi: 652 MHz of 2.3 GHz, 2.7 instead of 3.4 tokens/s).
//
// Two kinds of product. Dynamic: the big threads (and, for a prompt's wide products, helpers) take chunks from one
// counter, a slower core simply takes fewer of them. Weighted (set_little > 0, generating): the big threads and the
// little cores' helpers each get a contiguous share of the pieces in proportion to their speed (a little core does
// about a third of a big one's arithmetic), so all finish together and none waits for an A55's last chunk; a thread
// done with its share takes single pieces from the shares that would finish later than it can do one.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace geode {

class ThreadPool {
public:
    // n threads on the big cores, the caller of parallel_for included (n - 1 workers are started, pinned to cpus when
    // it is not empty); helpers more threads, pinned to helper_cpus: the phone's little cores. Helpers join the
    // weighted products (set_little) and, the first wide_helpers() of them (all, unless set), the wide ones.
    explicit ThreadPool(int n, std::vector<int> cpus = {}, int helpers = 0, std::vector<int> helper_cpus = {});
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }
    int helpers() const { return static_cast<int>(helpers_.size()); }

    // fn(begin, end) over [0, n), on every thread of the product; returns when all items are done. Not reentrant: one
    // parallel_for at a time. A wide product, or any while little() is 0: dynamic, in contiguous chunks of at least
    // min_chunk (wide: the wide helpers too). Otherwise weighted: [0, n) in pieces of min_chunk items dealt to the big
    // threads within the width and the first little() helpers by their speed.
    // first: run on the calling thread once the product is open, before it computes; its own share is then none (it
    // only takes the others' late pieces) - the GPU's launch while the CPU's threads start on their rows.
    // learn: the threads' timings of their own shares teach little_weight() (products of equal pieces only, not ones
    // that wait for flash reads).
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)> &fn, int64_t min_chunk = 1,
                      bool wide = false, const std::function<void()> *first = nullptr, bool learn = true);

    // From now on the products run on the first k big threads (the caller and k - 1 workers); the others sleep until
    // the width grows again. Generating a token wants fewer threads than reading a prompt on a phone that lowers
    // its clocks as more cores work (the 8 Elite: a prompt 67 tokens/s on 8, a token faster on 4).
    void set_width(int k);
    int width() const { return std::min(size(), width_.load()); }
    // The helpers in every product that is not wide (weighted): 0 .. helpers(); 0 = none, the products as before.
    // Helpers outside both this and wide_helpers() sleep.
    void set_little(int k);
    int little() const { return std::min(helpers(), little_.load()); }
    void set_wide_helpers(int k);
    int wide_helpers() const { return std::min(helpers(), wide_helpers_.load()); }
    // A helper's speed against a big thread's: kept (learn false) or learned from the weighted products' timings,
    // starting at w. Clamped to [0.05, 1.5].
    void set_little_weight(float w, bool learn);
    float little_weight() const { return little_w_.load(); }
    bool learning() const { return learn_; }

    // The cores the big threads ran their products on since the last call (a bit a cpu; every sixteenth product looked
    // at; 0 where unknown): fewer of them than threads means the system gave fewer cores than the pool has - the
    // 8 Elite outside its performance mode parks its two prime cores (core_ctl) and 8 threads share the other 6.
    uint64_t take_cpus() { return cpus_.exchange(0, std::memory_order_relaxed); }

    struct Stats {
        uint64_t weighted = 0;   // weighted products
        uint64_t stolen = 0;     // pieces a thread took from another's share
        uint64_t learned = 0;    // updates of the weight from timings
    };
    Stats stats() const;

    static bool on_helper();   // the calling thread is one of a pool's helpers (tests)

    // The weighted products' deal: p equal pieces over t threads of integer speeds w (0: none for it) - each one's
    // count, the shortest finish (whole pieces, each thread its count / its speed): proportional shares rounded down,
    // the rest one at a time to whoever finishes it first. (Public for the tests.)
    static void deal(int64_t p, const int *w, int t, int64_t *count);

private:
    void worker_loop(bool helper, int index);
    void run_chunks();
    int64_t run_share(int me, bool caller);
    int64_t run_pieces(int64_t b, int64_t e);
    void add_done(int64_t items);
    void wait_done(int64_t n);
    void learn_from_round();
    void note_cpu(uint64_t state);   // the calling big thread's cpu into cpus_ (every sixteenth round)

    std::vector<std::thread> workers_, helpers_;
    std::mutex mu_;
    std::condition_variable wake_, done_;
    // A product is a round: state_ odd while it is open (its fields below are set and it takes threads), even once
    // it is closed. The caller waits for the round's work, not for every thread: a thread the scheduler has not run
    // yet does not hold up each of a token's ~400 products (on the 8 Elite the scheduler kept 7 of 8 threads queued
    // on one core and decode stopped). inside_ counts the threads in the open round's chunks; a thread enters, then
    // checks the round is still open, and the caller closes it, then waits for inside_ to drain (both seq_cst: either
    // the caller sees the thread inside, or the thread sees the round closed), so no thread reads fields being
    // rewritten for the next round.
    // (each written often on a cache line of its own: the spinning threads' line is not the finishing threads')
    alignas(64) std::atomic<uint64_t> state_{0};
    alignas(64) std::atomic<uint64_t> helper_generation_{0};   // helpers follow the rounds they are in only
    alignas(64) std::atomic<int> inside_{0};
    alignas(64) std::atomic<int64_t> done_items_{0};
    alignas(64) std::atomic<int> width_{1 << 30}, little_{0}, wide_helpers_{1 << 30};
    alignas(64) std::atomic<uint64_t> cpus_{0};
    std::atomic<bool> quit_{false};
    // Waking only who sleeps: a thread about to sleep counts itself, then checks for a round; the caller opens the
    // round, then looks at the counts (seq_cst both: one of them sees the other) - no futex call a product while all
    // spin (Android's condition variable makes one on every notify). The same for the caller waiting for the work.
    std::atomic<int> big_sleepers_{0}, helper_sleepers_{0};
    std::atomic<bool> caller_sleeps_{false};

    const std::function<void(int64_t, int64_t)> *fn_ = nullptr;
    int64_t n_ = 0, chunk_ = 1;
    bool wide_round_ = false;
    int wide_n_ = 0;   // the wide helpers in this round
    alignas(64) std::atomic<int64_t> next_{0};

    // the weighted round: its pieces, who is in it (big threads 0 .. size() - 1, the caller 0, then the helpers), each
    // one's share
    struct alignas(64) Share {
        std::atomic<int64_t> next{0};   // the share's next piece not taken (by its owner or by others)
        int64_t end = 0, step = 1;      // one past its last piece; the pieces its owner takes at a time
        float speed = 1.0f;             // against a big thread (whose pieces are worth stealing)
        // the owner's timing of its own pieces in the last timed round it had any
        std::atomic<uint64_t> round{0};
        int64_t own = 0;
        double secs = 0.0;
    };
    std::unique_ptr<Share[]> shares_;
    std::vector<int> parts_;
    int nparts_ = 0, nbig_ = 0, nlit_ = 0;
    bool weighted_ = false, timed_ = false;
    int64_t piece_ = 1;
    uint64_t round_ = 0;
    std::atomic<float> little_w_{1.0f / 3.0f};
    bool learn_ = true;
    // the timed rounds' sums since the weight's last update: big and little pieces and seconds
    double acc_pb_ = 0, acc_sb_ = 0, acc_pl_ = 0, acc_sl_ = 0;
    int acc_n_ = 0;
    std::atomic<uint64_t> weighted_count_{0}, stolen_{0}, learned_{0};
};

// The number of compute threads to use when the caller says 0: the big cores on a phone, the cores on a PC.
int default_compute_threads();

// The phone's big cores (the highest max frequency) and the rest; both empty where that is not known (the PC).
std::vector<int> big_cores();
std::vector<int> little_cores();
// The cores' EAS capacities (cpuN/cpu_capacity, 1024 = the fastest) as "cpu:capacity ..." - empty where unknown.
std::string core_capacities();

// Pins the calling thread to these cpus (Linux/Android; nothing elsewhere or when cpus is empty): the arithmetic on
// the big cores, the flash readers on the little ones.
void pin_current_thread(const std::vector<int> &cpus);

// A spin-wait hint to the core (yield / pause).
void cpu_relax();

// Expert readers: flash wants a few requests in flight (UFS 2.1 saturates at 2-3), and they mostly sleep in the kernel.
int default_io_threads();

}  // namespace geode
