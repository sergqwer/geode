// experts.hpp - the routed experts between flash and the kernels: a fixed set of cache slots, I/O threads that read
// expert blocks into them, and the eviction policy (PLAN.md, P5).
//
// The forward pass asks for the experts the router picked (demand) and for those it predicts the next layer will pick
// (prefetch), then waits only for the blocks it is about to multiply. Reads of one layer's misses run in parallel,
// and they overlap the shared expert's and the hits' arithmetic. A slot handed out stays pinned until released.
//
// Eviction: decayed frequency (an exponential moving average of picks per expert, decayed per token) - in the
// simulator (tools/cachesim.cpp, ema_hits) 2-3 points of hit rate above LRU. The victim is the unpinned slot whose
// expert has the lowest score.
//
// Size classes: a pack whose experts mix formats (a q2 block is about half a q4 one) gets a pool of slots per block
// size, each slot as big as its class's blocks, so the same memory holds more of the small ones. An expert goes into
// a slot of its own class; with every one of those pinned, into a larger class's (never a smaller one). One format:
// one class, the cache as it always was.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "platform.hpp"

namespace geode {

struct ExpertRead {
    const RandomFile *file = nullptr;
    uint64_t offset = 0;
    size_t bytes = 0;
};

class ExpertStore {
public:
    ~ExpertStore();
    // the readers stopped (each finishes the part it is reading): before the memory of outside slabs goes away
    void stop();

    // keys: 0 .. n_keys-1 (layer * n_experts + expert); slot_bytes: the largest expert block.
    // io_cpus: where the readers run (the little cores on a phone; they sleep in the kernel most of the time)
    // slabs: the cache's memory from outside (the GPU's buffers, which the CPU reads as well), each (its memory, the
    // slots in it) in slot order, slot_bytes apart (rounded up to 4 KiB, round_slot); none: its own memory.
    // Retired slots of outside memory keep their pages (taken ones lose them with the piece the caller frees).
    struct Slab {
        uint8_t *mem = nullptr;
        size_t slots = 0;
    };
    bool init(size_t slots, size_t slot_bytes, uint32_t n_keys, int io_threads, std::vector<int> io_cpus = {},
              std::vector<Slab> slabs = {});
    // Size classes: class c has slots[c] slots of bytes[c] each (ascending sizes), key_class[key] is the class of each
    // key's block. Its own memory only (outside slabs hold one size).
    bool init_classes(const std::vector<size_t> &slots, const std::vector<size_t> &bytes, std::vector<uint8_t> key_class,
                      uint32_t n_keys, int io_threads, std::vector<int> io_cpus = {});
    // A block is read in this many parts at once by as many I/O threads (each a 4 KiB multiple): a miss on UFS 4.0
    // takes ~1 ms as one 1.7 MB read, a third of that as four (the device serves parallel requests). 1 = whole.
    void set_read_parts(int parts) { read_parts_ = std::max(1, parts); }
    static size_t round_slot(size_t bytes) { return (bytes + 4095) / 4096 * 4096; }
    uint8_t *slot_memory(int slot) const;   // where a slot's block lies (whether it is read yet or not)

    // Starts getting a block (a cache hit costs nothing) and pins it. Returns the slot. prefetch: a guess, read after
    // the demand reads; it does not count as a pick.
    int acquire(uint32_t key, const ExpertRead &read, bool prefetch);
    // The block of a pinned slot once it is read; nullptr if the read failed.
    const uint8_t *wait(int slot);
    bool ready(int slot) const { return slots_[static_cast<size_t>(slot)].state.load(std::memory_order_acquire) != Loading; }
    // the block in a slot, read or being read (an acquire would not have to start a read)
    bool resident(uint32_t key);
    // resident() of keys first .. first + n - 1 into out (1 / 0), under one lock (the route prior, once a layer)
    void residency(uint32_t first, uint32_t n, uint8_t *out);
    void release(int slot);

    // Once per token: the scores decay.
    void next_token() { ++now_; }
    // A prompt's pass through the layers (a cyclic scan of nearly every expert): while it is at `layer` (keys are
    // layer * per_layer + expert), the victims are experts of the layers it has passed, then those of the layers
    // farthest ahead - never the ones it needs next, which the decayed scores would pick (they were last used a whole
    // batch ago). -1: the scores alone again.
    void set_scan(int layer, uint32_t per_layer);

    // What the cache learned, across runs (Strata: a cache that starts from a profile hits more from the first
    // token): save_profile writes every picked key with its score now (u32 count, then u32 key + f32 score each);
    // load_profile takes the scores back and reads the hottest blocks into the empty slots, after every demand read
    // and guess (a third queue), at most `max` of them; read_of(key) says where a block is. Returns how many it queued.
    bool save_profile(const std::string &path);
    size_t load_profile(const std::string &path, const std::function<ExpertRead(uint32_t)> &read_of, size_t max);
    size_t preloads_left();   // blocks of load_profile not read yet
    void clear();

    // The context grows into the expert cache's memory (Model's dynamic context): retire_bytes() gives the coldest
    // slots back to the system (their pages released) until at least `bytes` went, and returns how many bytes it gave;
    // restore() takes them all back (a new conversation).
    uint64_t retire_bytes(uint64_t bytes);
    uint64_t unretire_bytes(uint64_t bytes);   // retired slots back, at most `bytes` of them; returns how many
    // retire() never leaves the cache fewer slots than this (what a layer and its guesses pin at once: fewer, and a
    // token would wait for a slot no one releases) - the reserve may have taken many
    void set_floor(size_t slots) { floor_ = std::max<size_t>(1, slots); }
    void restore();

    // The reserve (Model::reserve_memory): slots taken out of the cache whatever retire / unretire / restore do,
    // their memory given back - its own memory's pages released here, outside memory's by the caller (it frees the
    // piece they lie in). take() takes the given slots, or the coldest `coldest` usable ones (keeping at least `keep`
    // usable), once the reads into them are over; a pinned one is not taken. Returns the slots taken.
    std::vector<int> take(const std::vector<int> &slots, size_t coldest, size_t keep);
    // the coldest usable slots until at least `bytes` of them are taken (keeping at least `keep` usable)
    std::vector<int> take_bytes(uint64_t bytes, size_t keep);
    // back into the cache, empty; mem: where each now lies (outside memory allocated again), or empty (own memory)
    void give_back(const std::vector<int> &slots, const std::vector<uint8_t *> &mem = {});
    size_t taken() const { return gone_; }

    // the slots the cache can use: neither retired nor taken
    size_t slots() const { return n_ - retired_ - gone_ + gone_retired_; }
    size_t slot_bytes() const { return slot_bytes_; }   // the largest class's
    size_t slot_size(int slot) const { return class_bytes_[slot_class_[static_cast<size_t>(slot)]]; }
    uint64_t bytes() const { return usable_bytes_; }
    // how many blocks may be pinned at once whatever their classes: the largest class's usable slots (any block fits
    // one of those; a smaller one takes it only with its own class's all pinned)
    size_t pin_slots() const { return class_usable_.size() == 1 ? slots() : class_usable_.back(); }
    // the classes: their slot size, usable slots, demand picks and the misses among them
    size_t classes() const { return class_bytes_.size(); }
    size_t class_bytes(size_t c) const { return class_bytes_[c]; }
    size_t class_slots(size_t c) const { return class_usable_[c]; }
    uint64_t class_picks(size_t c) const { return class_picks_[c]; }
    uint64_t class_misses(size_t c) const { return class_misses_[c]; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t prefetches() const { return prefetches_; }
    uint64_t prefetch_hits() const { return prefetch_hits_; }   // demand picks that a prefetch had already started
    uint64_t prefetch_dropped() const { return prefetch_dropped_; }   // guesses let go before their read began
    uint64_t bytes_read() const { return bytes_read_.load(); }

private:
    enum State : int { Empty = 0, Loading = 1, Ready = 2, Failed = 3 };
    struct Slot {
        uint32_t key = kNoKey;
        std::atomic<int> state{Empty};
        int pins = 0;
        bool prefetched = false;   // loaded on a guess and not demanded yet
        bool retired = false;      // given back to the system for the context (its pages are released)
        bool gone = false;         // taken for the reserve (take): its memory is given back, retired or not
        ExpertRead read;
        // the read in parts, taken by the I/O threads one at a time (under mu_); the slot is read when none is left
        int parts = 1, next_part = 0, parts_left = 0;
        bool part_failed = false;
    };
    void start_read(Slot &s, const ExpertRead &read);   // (under mu_) the read's parts, none taken yet
    static constexpr uint32_t kNoKey = 0xffffffffu;
    static constexpr float kDecay = 0.95f;

    void bump(uint32_t key);
    // the slot to (re)use for a block of class cls: its own class's coldest, else a larger class's; cls < 0 (retiring,
    // taking): any class's coldest, the largest class kept at its floor
    int victim(int cls);
    int coldest(size_t from, size_t to, bool keep_largest);
    // a slot that becomes usable (add) or stops being (not retired and not taken)
    void usable(size_t slot, bool add);
    void io_loop();

    AlignedBuffer mem_;
    std::vector<uint8_t *> slot_mem_;   // each slot's place
    bool own_mem_ = true;               // the memory is the store's (its pages released here) or outside slabs
    std::vector<size_t> class_bytes_, class_first_, class_usable_;   // per class (class_first_: + n_ at the end)
    std::vector<uint8_t> slot_class_, key_class_;                     // key_class_ empty: one class
    std::vector<uint64_t> class_picks_, class_misses_;
    uint64_t usable_bytes_ = 0;
    size_t n_ = 0, slot_bytes_ = 0, retired_ = 0;
    size_t gone_ = 0, gone_retired_ = 0;   // taken slots, and of them the retired ones (slots() counts each once)
    size_t floor_ = 1;
    std::unique_ptr<Slot[]> slots_;
    std::unordered_map<uint32_t, int> where_;
    std::vector<float> ema_;          // per key, as of ema_t_
    std::vector<uint32_t> ema_t_;
    std::vector<double> rank_;        // per key: log(ema) - t log(decay), comparable across keys
    uint32_t now_ = 0;
    int scan_ = -1;                   // set_scan (under mu_)
    uint32_t scan_per_ = 1;

    std::mutex mu_;
    std::condition_variable io_cv_, done_cv_;
    std::deque<int> demand_q_, prefetch_q_, preload_q_;
    // (under mu_) a slot still queued among the guesses or the preloads moved to the end of q (a demand needs it now:
    // front of the demand queue; a guess: the guesses')
    void promote(int slot, std::deque<int> &q, bool front);
    int read_parts_ = 1;
    // GEODE_STORE_CHECK=1 (diagnosis): every read checked against a second read, a copy of every slot kept, and the
    // slot compared with it whenever it is handed out or let go
    bool check_ = false;
    std::vector<AlignedBuffer> shadow_;
    void check_slot(int slot, const char *when);   // (under mu_)
    std::vector<std::thread> io_;
    bool quit_ = false;

    uint64_t hits_ = 0, misses_ = 0, prefetches_ = 0, prefetch_hits_ = 0, prefetch_dropped_ = 0;
    std::atomic<uint64_t> bytes_read_{0};
};

}  // namespace geode
