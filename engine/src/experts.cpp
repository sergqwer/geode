// experts.cpp - ExpertStore: slot table under one mutex (the forward pass touches it a few hundred times a token),
// block reads outside it on the I/O threads.
#include "experts.hpp"

#include "threads.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <limits>

namespace geode {

ExpertStore::~ExpertStore() { stop(); }

void ExpertStore::stop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        quit_ = true;
    }
    io_cv_.notify_all();
    for (auto &t : io_) t.join();
    io_.clear();
}

bool ExpertStore::init(size_t slots, size_t slot_bytes, uint32_t n_keys, int io_threads, std::vector<int> io_cpus,
                       std::vector<Slab> slabs) {
    n_ = slots;
    slot_bytes_ = round_slot(slot_bytes);
    class_bytes_ = {slot_bytes_};
    class_first_ = {0, n_};
    slot_class_.assign(n_, 0);
    if (slabs.empty()) {
        mem_ = aligned_bytes(n_ * slot_bytes_);
        if (!mem_) return false;
        for (size_t i = 0; i < n_; ++i) slot_mem_.push_back(mem_.get() + i * slot_bytes_);
    } else {
        own_mem_ = false;
        for (const Slab &b : slabs)
            for (size_t i = 0; i < b.slots && slot_mem_.size() < n_; ++i) slot_mem_.push_back(b.mem + i * slot_bytes_);
        if (slot_mem_.size() < n_) return false;
    }
    class_usable_ = {n_};
    usable_bytes_ = static_cast<uint64_t>(n_) * slot_bytes_;
    class_picks_.assign(1, 0);
    class_misses_.assign(1, 0);
    slots_ = std::make_unique<Slot[]>(n_);
    ema_.assign(n_keys, 0.0f);
    ema_t_.assign(n_keys, 0);
    rank_.assign(n_keys, -std::numeric_limits<double>::infinity());
    check_ = std::getenv("GEODE_STORE_CHECK") != nullptr;
    if (check_) shadow_.resize(n_);
    for (int i = 0; i < std::max(1, io_threads); ++i)
        io_.emplace_back([this, io_cpus] {
            pin_current_thread(io_cpus);
            io_loop();
        });
    return true;
}

bool ExpertStore::init_classes(const std::vector<size_t> &slots, const std::vector<size_t> &bytes,
                               std::vector<uint8_t> key_class, uint32_t n_keys, int io_threads, std::vector<int> io_cpus) {
    if (slots.empty() || slots.size() != bytes.size() || (!key_class.empty() && key_class.size() != n_keys)) return false;
    n_ = 0;
    uint64_t total = 0;
    class_bytes_.clear();
    class_first_.clear();
    for (size_t c = 0; c < slots.size(); ++c) {
        class_bytes_.push_back(round_slot(bytes[c]));
        if (c && class_bytes_[c] <= class_bytes_[c - 1]) return false;   // ascending sizes
        class_first_.push_back(n_);
        n_ += slots[c];
        total += static_cast<uint64_t>(slots[c]) * class_bytes_[c];
    }
    class_first_.push_back(n_);
    slot_bytes_ = class_bytes_.back();
    // one buffer, each class's slots after the smaller ones'; every size a 4 KiB multiple, so every slot page-aligned
    mem_ = aligned_bytes(static_cast<size_t>(total));
    if (!mem_) return false;
    slot_mem_.clear();
    slot_class_.clear();
    uint8_t *at = mem_.get();
    for (size_t c = 0; c < slots.size(); ++c)
        for (size_t i = 0; i < slots[c]; ++i) {
            slot_mem_.push_back(at);
            slot_class_.push_back(static_cast<uint8_t>(c));
            at += class_bytes_[c];
        }
    class_usable_ = slots;
    usable_bytes_ = total;
    class_picks_.assign(slots.size(), 0);
    class_misses_.assign(slots.size(), 0);
    key_class_ = std::move(key_class);
    slots_ = std::make_unique<Slot[]>(n_);
    ema_.assign(n_keys, 0.0f);
    ema_t_.assign(n_keys, 0);
    rank_.assign(n_keys, -std::numeric_limits<double>::infinity());
    check_ = std::getenv("GEODE_STORE_CHECK") != nullptr;
    if (check_) shadow_.resize(n_);
    for (int i = 0; i < std::max(1, io_threads); ++i)
        io_.emplace_back([this, io_cpus] {
            pin_current_thread(io_cpus);
            io_loop();
        });
    return true;
}

void ExpertStore::usable(size_t slot, bool add) {
    const size_t c = slot_class_[slot];
    if (add) {
        ++class_usable_[c];
        usable_bytes_ += class_bytes_[c];
    } else {
        --class_usable_[c];
        usable_bytes_ -= class_bytes_[c];
    }
}

void ExpertStore::bump(uint32_t key) {
    // the score now: ema * decay^(tokens since its last pick), plus this pick
    ema_[key] = ema_[key] * std::pow(kDecay, static_cast<float>(now_ - ema_t_[key])) + 1.0f;
    ema_t_[key] = now_;
    // decay^now is common to every key, so log(ema) - t log(decay) orders them without a pow per comparison
    rank_[key] = std::log(static_cast<double>(ema_[key])) - static_cast<double>(now_) * std::log(static_cast<double>(kDecay));
}

void ExpertStore::set_scan(int layer, uint32_t per_layer) {
    std::lock_guard<std::mutex> lock(mu_);
    scan_ = layer;
    scan_per_ = std::max<uint32_t>(1, per_layer);
}

int ExpertStore::victim(int cls) {
    if (cls < 0) {
        // retiring or taking: any class, the largest kept at the floor (what a layer and its guesses pin fits there)
        const bool keep = class_bytes_.size() > 1 && class_usable_.back() <= floor_;
        return coldest(0, n_, keep);
    }
    const size_t c = static_cast<size_t>(cls);
    const int v = coldest(class_first_[c], class_first_[c + 1], false);
    return v >= 0 || c + 1 >= class_bytes_.size() ? v : coldest(class_first_[c + 1], n_, false);
}

int ExpertStore::coldest(size_t from, size_t to, bool keep_largest) {
    int best = -1;
    double best_rank = std::numeric_limits<double>::infinity();
    // a prompt's pass: the layers behind it first (their scores among them), then the farthest ahead
    const double layers = static_cast<double>(ema_.size() / scan_per_);
    const size_t largest = class_first_[class_first_.size() - 2];
    for (size_t i = from; i < to; ++i) {
        if (keep_largest && i >= largest) break;
        Slot &s = slots_[i];
        // pinned, retired, taken, or an I/O thread is writing into it
        if (s.pins > 0 || s.retired || s.gone || s.state.load(std::memory_order_acquire) == Loading) continue;
        if (s.key == kNoKey) return static_cast<int>(i);
        // an unused guess goes first; then the coldest expert
        double r = s.prefetched ? -std::numeric_limits<double>::infinity() : rank_[s.key];
        if (scan_ >= 0 && !s.prefetched) {
            const auto layer = static_cast<int>(s.key / scan_per_);
            r = layer < scan_ ? r - 1e12 : 1e6 * (layers - layer) + r;
        }
        if (r < best_rank || best < 0) {
            best_rank = r;
            best = static_cast<int>(i);
        }
    }
    return best;
}

bool ExpertStore::resident(uint32_t key) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = where_.find(key);
    return it != where_.end() && slots_[static_cast<size_t>(it->second)].state.load() != Failed;
}

void ExpertStore::residency(uint32_t first, uint32_t n, uint8_t *out) {
    std::lock_guard<std::mutex> lock(mu_);
    for (uint32_t i = 0; i < n; ++i) {
        const auto it = where_.find(first + i);
        out[i] = it != where_.end() && slots_[static_cast<size_t>(it->second)].state.load() != Failed ? 1 : 0;
    }
}

void ExpertStore::check_slot(int slot, const char *when) {
    const Slot &s = slots_[static_cast<size_t>(slot)];
    const AlignedBuffer &sh = shadow_[static_cast<size_t>(slot)];
    if (!sh || s.state.load() != Ready) return;
    const uint8_t *m = slot_memory(slot);
    const size_t n = s.read.bytes;
    if (std::memcmp(m, sh.get(), n) == 0) return;
    size_t first = n, count = 0;
    for (size_t i = 0; i < n; ++i)
        if (m[i] != sh[i]) {
            if (first == n) first = i;
            ++count;
        }
    std::fprintf(stderr, "store check (%s): slot %d key %u changed since its read: %zu bytes, the first at %zu of %zu (was %02x, now %02x)\n",
                 when, slot, s.key, count, first, n, sh[first], m[first]);
    std::memcpy(sh.get(), m, n);   // reported once
}

void ExpertStore::start_read(Slot &s, const ExpertRead &read) {
    s.read = read;
    // parts of whole 4 KiB pages (O_DIRECT), at least 256 KiB each
    const size_t pages = (read.bytes + 4095) / 4096;
    s.parts = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(read_parts_), pages / 64)));
    s.next_part = 0;
    s.parts_left = s.parts;
    s.part_failed = false;
}

int ExpertStore::acquire(uint32_t key, const ExpertRead &read, bool prefetch) {
    std::unique_lock<std::mutex> lock(mu_);
    const int cls = key_class_.empty() ? 0 : key_class_[key];
    if (!prefetch) {
        bump(key);
        ++class_picks_[static_cast<size_t>(cls)];
    }
    const auto it = where_.find(key);
    if (it != where_.end()) {
        Slot &s = slots_[static_cast<size_t>(it->second)];
        if (s.state.load() == Failed) {
            // a guess whose read failed after it was released: read it again
            s.state.store(Loading);
            start_read(s, read);
            (prefetch ? prefetch_q_ : demand_q_).push_back(it->second);
            io_cv_.notify_one();
        }
        ++s.pins;
        if (!prefetch) {
            if (s.prefetched) ++prefetch_hits_;
            else ++hits_;
            // a guess (or a preload) still queued behind others: the forward pass needs it now, ahead of them
            if (s.state.load() == Loading) promote(it->second, demand_q_, false);
            s.prefetched = false;
        } else if (s.state.load() == Loading) {
            promote(it->second, prefetch_q_, false);   // a preload guessed: read with the guesses
        }
        return it->second;
    }
    int v = victim(cls);
    while (v < 0) {
        // every slot it fits pinned: wait for one to be released (the budget is checked to hold top_k + the
        // prefetches, in the largest class: pin_slots)
        done_cv_.wait(lock);
        v = victim(cls);
    }
    Slot &s = slots_[static_cast<size_t>(v)];
    if (s.key != kNoKey) where_.erase(s.key);
    s.key = key;
    s.state.store(Loading);
    s.pins = 1;
    s.prefetched = prefetch;
    start_read(s, read);
    where_[key] = v;
    if (prefetch) {
        ++prefetches_;
        prefetch_q_.push_back(v);
    } else {
        ++misses_;
        ++class_misses_[static_cast<size_t>(cls)];
        demand_q_.push_back(v);
    }
    lock.unlock();
    io_cv_.notify_one();
    return v;
}

const uint8_t *ExpertStore::wait(int slot) {
    Slot &s = slots_[static_cast<size_t>(slot)];
    if (s.state.load(std::memory_order_acquire) == Loading) {
        {
            // a prefetch (or a preload) still queued behind others: promote it, the forward pass needs it now
            std::lock_guard<std::mutex> lock(mu_);
            promote(slot, demand_q_, true);
        }
        // spin first (a read takes a few ms): a big core that sleeps here is clocked down by the governor
        const auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (s.state.load(std::memory_order_acquire) == Loading) {
            cpu_relax();
            if (++spins % 256 == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(10)) {
                std::unique_lock<std::mutex> lock(mu_);
                done_cv_.wait(lock, [&] { return s.state.load(std::memory_order_acquire) != Loading; });
                break;
            }
        }
    }
    if (check_) {
        std::lock_guard<std::mutex> lock(mu_);
        check_slot(slot, "handed out");
    }
    return s.state.load(std::memory_order_acquire) == Ready ? slot_memory(slot) : nullptr;
}

uint8_t *ExpertStore::slot_memory(int slot) const { return slot_mem_[static_cast<size_t>(slot)]; }

void ExpertStore::release(int slot) {
    std::lock_guard<std::mutex> lock(mu_);
    Slot &s = slots_[static_cast<size_t>(slot)];
    if (check_) check_slot(slot, "let go");
    if (s.pins > 0) --s.pins;
    if (s.pins == 0 && s.prefetched && s.state.load() == Loading && s.next_part == 0) {
        // a guess nobody picked whose read has not started: dropped, the I/O threads go to the reads that matter
        auto q = std::find(prefetch_q_.begin(), prefetch_q_.end(), slot);
        if (q != prefetch_q_.end()) {
            prefetch_q_.erase(q);
            where_.erase(s.key);
            s.key = kNoKey;
            s.state.store(Empty);
            s.prefetched = false;
            ++prefetch_dropped_;
        }
    }
    if (s.state.load() == Failed && s.pins == 0) {
        // a failed read caches nothing
        where_.erase(s.key);
        s.key = kNoKey;
        s.state.store(Empty);
    }
    if (s.pins == 0) done_cv_.notify_all();
}

uint64_t ExpertStore::retire_bytes(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t done = 0;
    while (done < bytes && slots() > floor_) {
        const int v = victim(-1);
        if (v < 0) break;
        Slot &s = slots_[static_cast<size_t>(v)];
        if (s.key != kNoKey) where_.erase(s.key);
        s.key = kNoKey;
        s.state.store(Empty);
        s.prefetched = false;
        s.retired = true;
        usable(static_cast<size_t>(v), false);
        if (own_mem_) release_pages(slot_memory(v), slot_size(v));
        ++retired_;
        done += slot_size(v);
    }
    return done;
}

uint64_t ExpertStore::unretire_bytes(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    // the ones the cache can use at once first; a taken one comes back with the reserve's end. Never more than
    // `bytes` (the budget lent them), so with mixed sizes perhaps a little less
    uint64_t done = 0;
    for (const bool taken : {false, true})
        for (size_t i = 0; i < n_ && done < bytes; ++i)
            if (slots_[i].retired && slots_[i].gone == taken && done + slot_size(static_cast<int>(i)) <= bytes) {
                slots_[i].retired = false;
                --retired_;
                if (taken) --gone_retired_;
                else usable(i, true);
                done += slot_size(static_cast<int>(i));
            }
    return done;
}

void ExpertStore::restore() {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t i = 0; i < n_; ++i) {
        if (slots_[i].retired && !slots_[i].gone) usable(i, true);
        slots_[i].retired = false;
    }
    retired_ = 0;
    gone_retired_ = 0;
}

std::vector<int> ExpertStore::take(const std::vector<int> &want, size_t coldest, size_t keep) {
    std::unique_lock<std::mutex> lock(mu_);
    // a slot taken once no read goes into it any more; false: pinned (or taken already)
    const auto pull = [&](int i) {
        Slot &s = slots_[static_cast<size_t>(i)];
        if (s.gone || s.pins > 0) return false;
        if (s.state.load() == Loading && s.next_part == 0) {
            // queued and not begun: out of its queue
            for (std::deque<int> *q : {&demand_q_, &prefetch_q_, &preload_q_}) {
                auto it = std::find(q->begin(), q->end(), i);
                if (it == q->end()) continue;
                q->erase(it);
                s.state.store(Empty);
                break;
            }
        }
        // begun: its parts are read to the end first (the I/O threads write into the slot)
        done_cv_.wait(lock, [&] { return s.state.load(std::memory_order_acquire) != Loading; });
        if (s.gone || s.pins > 0) return false;
        if (s.key != kNoKey) where_.erase(s.key);
        s.key = kNoKey;
        s.state.store(Empty);
        s.prefetched = false;
        s.gone = true;
        ++gone_;
        if (s.retired) ++gone_retired_;
        else usable(static_cast<size_t>(i), false);
        if (own_mem_) release_pages(slot_memory(i), slot_size(i));
        return true;
    };
    std::vector<int> out;
    if (!want.empty()) {
        for (int i : want)
            if (pull(i)) out.push_back(i);
    } else {
        while (out.size() < coldest && slots() > keep) {
            const int v = victim(-1);
            if (v < 0 || !pull(v)) break;
            out.push_back(v);
        }
    }
    return out;
}

std::vector<int> ExpertStore::take_bytes(uint64_t bytes, size_t keep) {
    std::vector<int> out;
    uint64_t got = 0;
    while (got < bytes) {
        int v;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (slots() <= keep) break;
            v = victim(-1);
        }
        if (v < 0) break;
        const std::vector<int> one = take({v}, 0, 0);
        if (one.empty()) break;   // pinned meanwhile
        out.push_back(v);
        got += slot_size(v);
    }
    return out;
}

void ExpertStore::give_back(const std::vector<int> &slots, const std::vector<uint8_t *> &mem) {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t k = 0; k < slots.size(); ++k) {
        Slot &s = slots_[static_cast<size_t>(slots[k])];
        if (!s.gone) continue;
        if (k < mem.size() && !own_mem_) slot_mem_[static_cast<size_t>(slots[k])] = mem[k];
        s.gone = false;
        --gone_;
        if (s.retired) --gone_retired_;
        else usable(static_cast<size_t>(slots[k]), true);
        s.key = kNoKey;
        s.state.store(Empty);
    }
    done_cv_.notify_all();
}

void ExpertStore::promote(int slot, std::deque<int> &q, bool front) {
    for (std::deque<int> *from : {&prefetch_q_, &preload_q_}) {
        if (from == &q) continue;
        // only a slot no part of which an I/O thread took (it is at the front then, and goes on as it is)
        auto it = std::find(from->begin(), from->end(), slot);
        if (it == from->end() || slots_[static_cast<size_t>(slot)].next_part > 0) continue;
        from->erase(it);
        if (front) q.push_front(slot);
        else q.push_back(slot);
        io_cv_.notify_one();
        return;
    }
}

bool ExpertStore::save_profile(const std::string &path) {
    std::vector<std::pair<uint32_t, float>> v;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint32_t k = 0; k < ema_.size(); ++k)
            if (ema_[k] > 0.0f)
                v.emplace_back(k, ema_[k] * std::pow(kDecay, static_cast<float>(now_ - ema_t_[k])));
    }
    if (v.empty()) return false;
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    // written aside, then moved over the old one: a crash leaves the old profile, never half of one
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const uint32_t n = static_cast<uint32_t>(v.size());
        f.write(reinterpret_cast<const char *>(&n), 4);
        for (const auto &[k, sc] : v) {
            f.write(reinterpret_cast<const char *>(&k), 4);
            f.write(reinterpret_cast<const char *>(&sc), 4);
        }
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

size_t ExpertStore::load_profile(const std::string &path, const std::function<ExpertRead(uint32_t)> &read_of,
                                 size_t max) {
    std::ifstream f(path, std::ios::binary);
    uint32_t n = 0;
    if (!f || !f.read(reinterpret_cast<char *>(&n), 4) || n > ema_.size()) return 0;
    std::vector<std::pair<uint32_t, float>> v(n);
    for (auto &[k, sc] : v)
        if (!f.read(reinterpret_cast<char *>(&k), 4) || !f.read(reinterpret_cast<char *>(&sc), 4) ||
            k >= ema_.size() || !(sc > 0.0f) || !std::isfinite(sc))
            return 0;
    std::lock_guard<std::mutex> lock(mu_);
    // the scores as they were (the hottest are the slowest to be evicted), then the blocks into the empty slots
    for (const auto &[k, sc] : v) {
        ema_[k] = sc;
        ema_t_[k] = now_;
        rank_[k] = std::log(static_cast<double>(sc)) - static_cast<double>(now_) * std::log(static_cast<double>(kDecay));
    }
    // each block into an empty slot of its own class (a larger one stays for its own blocks)
    size_t queued = 0;
    std::vector<size_t> nexts(class_first_.begin(), class_first_.end() - 1);
    for (const auto &[k, sc] : v) {
        if (queued >= max) break;
        if (where_.count(k)) continue;
        const size_t c = key_class_.empty() ? 0 : key_class_[k];
        size_t &next = nexts[c];
        const size_t end = class_first_[c + 1];
        while (next < end && (slots_[next].key != kNoKey || slots_[next].retired || slots_[next].gone)) ++next;
        if (next >= end) continue;
        Slot &s = slots_[next];
        s.key = k;
        s.state.store(Loading);
        s.pins = 0;
        s.prefetched = false;
        start_read(s, read_of(k));
        where_[k] = static_cast<int>(next);
        preload_q_.push_back(static_cast<int>(next));
        ++queued;
    }
    io_cv_.notify_all();
    return queued;
}

size_t ExpertStore::preloads_left() {
    std::lock_guard<std::mutex> lock(mu_);
    return preload_q_.size();
}

void ExpertStore::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    hits_ = misses_ = prefetches_ = prefetch_hits_ = 0;
    bytes_read_.store(0);
}

void ExpertStore::io_loop() {
    for (;;) {
        int slot;
        ExpertRead r;
        size_t at = 0, len = 0;
        {
            std::unique_lock<std::mutex> lock(mu_);
            io_cv_.wait(lock, [&] { return quit_ || !demand_q_.empty() || !prefetch_q_.empty() || !preload_q_.empty(); });
            if (quit_) return;
            auto &q = !demand_q_.empty() ? demand_q_ : !prefetch_q_.empty() ? prefetch_q_ : preload_q_;
            slot = q.front();
            Slot &s = slots_[static_cast<size_t>(slot)];
            r = s.read;
            // part p of the read: whole pages, the last one takes the rest
            const size_t pages = (r.bytes + 4095) / 4096, per = (pages + static_cast<size_t>(s.parts) - 1) / static_cast<size_t>(s.parts);
            const int p = s.next_part++;
            at = std::min(r.bytes, static_cast<size_t>(p) * per * 4096);
            len = std::min(r.bytes - at, per * 4096);
            if (s.next_part == s.parts) q.pop_front();
            else io_cv_.notify_one();   // the next part for another thread
        }
        const bool ok = len == 0 || r.file->read_at(r.offset + at, slot_memory(slot) + at, len);
        bytes_read_ += len;
        {
            std::lock_guard<std::mutex> lock(mu_);
            Slot &s = slots_[static_cast<size_t>(slot)];
            s.part_failed |= !ok;
            if (--s.parts_left > 0) continue;
            if (check_ && !s.part_failed) {
                // the block read again, whole, and kept
                auto &sh = shadow_[static_cast<size_t>(slot)];
                if (!sh) sh = aligned_bytes(slot_size(slot));
                const bool again = r.file->read_at(r.offset, sh.get(), r.bytes);
                if (!again || std::memcmp(sh.get(), slot_memory(slot), r.bytes) != 0)
                    std::fprintf(stderr, "store check: slot %d key %u read differently twice (%s)\n", slot, s.key,
                                 again ? "bytes differ" : "the second read failed");
                std::memcpy(sh.get(), slot_memory(slot), r.bytes);
            }
            s.state.store(s.part_failed ? Failed : Ready, std::memory_order_release);
        }
        done_cv_.notify_all();
    }
}

}  // namespace geode
