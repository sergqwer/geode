// state.cpp - a conversation's state in a file and back (Model::save_state / load_state; geode.h, geode_state_save /
// geode_state_load): instead of reading a long conversation again (minutes on a phone), the states it left are read
// back from flash (a fraction of a second).
//
// The file (little-endian, the machine's own layout):
//   [0, 4096)      the header (StateHeader), a checksum of its bytes last
//   fixed_offset   the fixed-size part: gen, the images' (kv0, s, gh, gw) x image_cap, the last hidden state (the MTP
//                  layer drafts from it), the GDN layers' conv states, their recurrent states, gen again
//   kv_offset      the KV cache in blocks of kStateBlock positions: [K, V][attention layer][kStateBlock][kv_row] elements, then
//                  (int8) their scales [K, V][attention layer][kStateBlock][kv_heads] floats
// The blocks do not follow the model's KV chunks (their size differs between the CPU and the GPU path): a state saved
// by one path loads into the other.
//
// An append (a later state of the same conversation) writes the fixed part in place, the blocks from the one holding
// the last saved position on (the MTP layer rewrites the row before a new prompt), then the header: a save cut short
// leaves a header whose gen the fixed part no longer carries at both ends, which a load refuses (it never reads a
// mixed state). A whole write goes to path.tmp, renamed over path at the end.
#include "model.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace geode {

namespace {

constexpr char kMagic[8] = {'G', 'E', 'O', 'D', 'E', 'S', 'T', 'A'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kStateBlock = 256;           // positions a KV block of the file holds
constexpr uint64_t kHeaderBytes = 4096;
constexpr size_t kIoBatch = 16u << 20;     // the blocks read or written at once

struct StateHeader {
    char magic[8];
    uint32_t version, header_size;
    uint64_t fingerprint;      // the pack (geode.json) and the state's shapes (Model::fingerprint)
    uint64_t conv_id;          // the conversation: an append goes only to its own file
    uint64_t gen;              // bumped by every save; the fixed part carries it at both ends
    int32_t position;          // the tokens computed (the KV rows)
    int32_t extra;             // the caller's (geode.cpp: the token passed on but not computed yet)
    int32_t rope_delta;
    uint32_t n_images, image_cap;
    uint32_t kv_elem, n_attn, kv_heads, kv_row, hidden, has_hidden, block;
    uint64_t conv_n, gdn_n;
    uint64_t fixed_offset, fixed_bytes, kv_offset, block_bytes;
    uint64_t checksum;         // FNV-1a of every byte before it
};
static_assert(sizeof(StateHeader) <= kHeaderBytes, "the header fits its page");

uint64_t fnv(const void *data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

uint64_t checksum(const StateHeader &h) { return fnv(&h, offsetof(StateHeader, checksum)); }

uint64_t align4k(uint64_t x) { return (x + 4095) / 4096 * 4096; }

// a file read and written at offsets, in big pieces
class StateFile {
public:
    ~StateFile() { close(); }
    bool open(const std::string &path, bool write, bool create) {
#if defined(_WIN32)
        int flags = _O_BINARY | (write ? _O_RDWR : _O_RDONLY);
        if (create) flags |= _O_CREAT | _O_TRUNC;
        fd_ = _open(path.c_str(), flags, _S_IREAD | _S_IWRITE);
#else
        int flags = O_CLOEXEC | (write ? O_RDWR : O_RDONLY);
        if (create) flags |= O_CREAT | O_TRUNC;
        fd_ = ::open(path.c_str(), flags, 0600);
#endif
        return fd_ >= 0;
    }
    void close() {
        if (fd_ < 0) return;
#if defined(_WIN32)
        _close(fd_);
#else
        // its pages out of the page cache: a state is read or written once, and on a phone the memory is the model's
        // (a save's dirty pages are on the flash after its fsyncs; a few hundred MB kept beside a 3 GB model got the
        // engine killed under pressure)
        ::fdatasync(fd_);
        ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd_);
#endif
        fd_ = -1;
    }
    uint64_t size() const {
#if defined(_WIN32)
        const long long n = _lseeki64(fd_, 0, SEEK_END);
#else
        struct stat st {};
        const long long n = fstat(fd_, &st) == 0 ? static_cast<long long>(st.st_size) : -1;
#endif
        return n < 0 ? 0 : static_cast<uint64_t>(n);
    }
    bool write(uint64_t off, const void *data, size_t n) {
        const auto *p = static_cast<const uint8_t *>(data);
        while (n > 0) {
            const size_t k = std::min<size_t>(n, 1u << 30);
#if defined(_WIN32)
            if (_lseeki64(fd_, static_cast<long long>(off), SEEK_SET) < 0) return false;
            const int w = _write(fd_, p, static_cast<unsigned>(k));
#else
            const ssize_t w = ::pwrite(fd_, p, k, static_cast<off_t>(off));
#endif
            if (w <= 0) return false;
            p += w;
            off += static_cast<uint64_t>(w);
            n -= static_cast<size_t>(w);
        }
        return true;
    }
    bool read(uint64_t off, void *data, size_t n) {
        auto *p = static_cast<uint8_t *>(data);
        while (n > 0) {
            const size_t k = std::min<size_t>(n, 1u << 30);
#if defined(_WIN32)
            if (_lseeki64(fd_, static_cast<long long>(off), SEEK_SET) < 0) return false;
            const int r = _read(fd_, p, static_cast<unsigned>(k));
#else
            const ssize_t r = ::pread(fd_, p, k, static_cast<off_t>(off));
#endif
            if (r <= 0) return false;
            p += r;
            off += static_cast<uint64_t>(r);
            n -= static_cast<size_t>(r);
        }
        return true;
    }
    // what was written reaches the flash (the order of a save's parts: a header never ahead of what it names)
    bool sync() {
#if defined(_WIN32)
        return _commit(fd_) == 0;
#else
        return ::fsync(fd_) == 0;
#endif
    }

private:
    int fd_ = -1;
};

bool replace_file(const std::string &from, const std::string &to) {
#if defined(_WIN32)
    return MoveFileExA(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

bool read_header(StateFile &f, StateHeader &h, std::string &err) {
    if (!f.read(0, &h, sizeof h)) {
        err = "not a saved conversation (too short)";
        return false;
    }
    if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0) {
        err = "not a saved conversation";
        return false;
    }
    if (h.version != kVersion || h.header_size != sizeof(StateHeader)) {
        err = "a saved conversation of another version (" + std::to_string(h.version) + ")";
        return false;
    }
    if (h.checksum != checksum(h)) {
        err = "the saved conversation's header is damaged";
        return false;
    }
    return true;
}

// where the fixed part's pieces are, for image_cap images
struct Layout {
    uint64_t images, hidden, conv, gdn, trail, fixed_bytes, kv_offset;
};

Layout layout(const StateHeader &h) {
    Layout L{};
    L.images = h.fixed_offset + sizeof(uint64_t);
    L.hidden = L.images + static_cast<uint64_t>(h.image_cap) * 4 * sizeof(int32_t);
    L.conv = L.hidden + static_cast<uint64_t>(h.hidden) * sizeof(float);
    L.gdn = L.conv + h.conv_n * sizeof(float);
    L.trail = L.gdn + h.gdn_n * sizeof(float);
    L.fixed_bytes = L.trail + sizeof(uint64_t) - h.fixed_offset;
    L.kv_offset = align4k(h.fixed_offset + L.fixed_bytes);
    return L;
}

}  // namespace

void Model::new_conversation() {
    // unique enough between this process's conversations and others': the clock, a counter, where the model is
    static uint64_t counter = 0;
    uint64_t x = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
                 (static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()) << 1) ^
                 (++counter * 0x9E3779B97F4A7C15ull) ^ reinterpret_cast<uintptr_t>(this);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    conv_id_ = x;
}

uint64_t Model::fingerprint() {
    if (!pack_print_) {
        // the pack: its manifest names every file, tensor and format (another pack, or a pack made again, differs)
        std::FILE *f = std::fopen((pack_.dir() + "/geode.json").c_str(), "rb");
        uint64_t h = fnv(nullptr, 0);
        if (f) {
            std::vector<char> buf(1 << 16);
            size_t n;
            while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) h = fnv(buf.data(), n, h);
            std::fclose(f);
        }
        pack_print_ = h ? h : 1;
    }
    // and the shapes of what a state holds
    const uint64_t shapes[] = {kVersion, kv_elem(), static_cast<uint64_t>(n_attn_), static_cast<uint64_t>(kv_heads_),
                               kv_row_, conv_n_, gdn_n_, static_cast<uint64_t>(cfg_.hidden),
                               static_cast<uint64_t>(cfg_.n_layers), mtp_ ? 1u : 0u, has_vision() ? 1u : 0u};
    return fnv(shapes, sizeof shapes, pack_print_);
}

uint8_t *Model::kv_row_bytes(int pos, int layer, bool value) {
    return kv_ptrs_[static_cast<size_t>(pos / kv_chunk_)] +
           ((static_cast<size_t>(value) * n_attn_ + static_cast<size_t>(layer)) * kv_chunk_ +
            static_cast<size_t>(pos % kv_chunk_)) * kv_row_ * kv_elem();
}

bool Model::save_state(const std::string &path, bool append, int32_t extra, std::string &err) {
    if (image_next_ != images_.size() || image_fed_ != 0) {
        err = "images were added that the conversation has not taken yet";
        return false;
    }
    const size_t elem = kv_elem(), row_bytes = kv_row_ * elem;
    const size_t scale_bytes = kv_i8_ ? static_cast<size_t>(kv_heads_) * sizeof(float) : 0;   // a position's
    StateHeader h{};
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kVersion;
    h.header_size = sizeof(StateHeader);
    h.fingerprint = fingerprint();
    h.conv_id = conv_id_;
    h.gen = 1;
    h.position = pos_;
    h.extra = extra;
    h.rope_delta = rope_delta_;
    h.n_images = static_cast<uint32_t>(images_.size());
    h.image_cap = std::max<uint32_t>(16, (h.n_images + 15) / 16 * 16 + 16);
    h.kv_elem = static_cast<uint32_t>(elem);
    h.n_attn = static_cast<uint32_t>(n_attn_);
    h.kv_heads = static_cast<uint32_t>(kv_heads_);
    h.kv_row = static_cast<uint32_t>(kv_row_);
    h.hidden = static_cast<uint32_t>(cfg_.hidden);
    h.has_hidden = keeps_hidden() && pos_ > 0 && last_hid_.size() == static_cast<size_t>(cfg_.hidden) ? 1 : 0;
    h.block = kStateBlock;
    h.conv_n = conv_n_;
    h.gdn_n = gdn_n_;
    h.fixed_offset = kHeaderBytes;
    h.block_bytes = 2 * static_cast<uint64_t>(n_attn_) * kStateBlock * (row_bytes + scale_bytes);

    // an append: the file holds an earlier state of this conversation, in this shape
    int first_block = 0;
    bool in_place = false;
    if (append) {
        StateFile f;
        StateHeader old{};
        std::string e;
        if (f.open(path, false, false) && read_header(f, old, e) && old.fingerprint == h.fingerprint &&
            old.conv_id == h.conv_id && old.position <= h.position && h.n_images <= old.image_cap &&
            old.block == h.block && old.block_bytes == h.block_bytes && old.fixed_offset == h.fixed_offset) {
            h.image_cap = old.image_cap;
            h.gen = old.gen + 1;
            // the MTP layer writes the row before a new prompt again: from the block that holds it
            first_block = std::max(0, old.position - 1) / static_cast<int>(kStateBlock);
            in_place = true;
        }
    }
    const Layout L = layout(h);
    h.fixed_bytes = L.fixed_bytes;
    h.kv_offset = L.kv_offset;
    h.checksum = checksum(h);

    const std::string target = in_place ? path : path + ".tmp";
    StateFile f;
    if (!f.open(target, true, !in_place)) {
        err = "cannot write " + target;
        return false;
    }
    const auto fail = [&](const char *what) {
        f.close();
        if (!in_place) std::remove(target.c_str());
        err = std::string("cannot write the conversation's ") + what + " to " + target;
        return false;
    };
    // the fixed part: gen first, gen last once the rest is on the flash
    {
        std::vector<uint8_t> head(static_cast<size_t>(L.conv - h.fixed_offset), 0);
        std::memcpy(head.data(), &h.gen, sizeof h.gen);
        auto *im = reinterpret_cast<int32_t *>(head.data() + (L.images - h.fixed_offset));
        for (size_t i = 0; i < images_.size(); ++i) {
            im[4 * i] = images_[i].kv0;
            im[4 * i + 1] = images_[i].s;
            im[4 * i + 2] = images_[i].gh;
            im[4 * i + 3] = images_[i].gw;
        }
        if (h.has_hidden) std::memcpy(head.data() + (L.hidden - h.fixed_offset), last_hid_.data(), h.hidden * sizeof(float));
        if (!f.write(h.fixed_offset, head.data(), head.size()) || !f.write(L.conv, conv_st_, conv_n_ * sizeof(float)) ||
            !f.write(L.gdn, gdn_st_, gdn_n_ * sizeof(float)) || !f.sync() || !f.write(L.trail, &h.gen, sizeof h.gen))
            return fail("states");
    }
    // the KV cache, block by block from first_block, a batch of blocks a write
    const int blocks = (pos_ + static_cast<int>(kStateBlock) - 1) / static_cast<int>(kStateBlock);
    if (blocks > first_block) {
        const size_t bb = static_cast<size_t>(h.block_bytes);
        const int per = static_cast<int>(std::max<size_t>(1, kIoBatch / bb));
        std::vector<uint8_t> buf(static_cast<size_t>(std::min(per, blocks - first_block)) * bb);
        for (int b0 = first_block; b0 < blocks; b0 += per) {
            const int nb = std::min(per, blocks - b0);
            std::fill(buf.begin(), buf.begin() + static_cast<ptrdiff_t>(nb * bb), uint8_t{0});
            for (int b = b0; b < b0 + nb; ++b) {
                uint8_t *blk = buf.data() + static_cast<size_t>(b - b0) * bb;
                const int p0 = b * static_cast<int>(kStateBlock), p1 = std::min(pos_, p0 + static_cast<int>(kStateBlock));
                uint8_t *scales = blk + 2 * static_cast<size_t>(n_attn_) * kStateBlock * row_bytes;
                for (int v = 0; v < 2; ++v)
                    for (int l = 0; l < n_attn_; ++l) {
                        const size_t seg = static_cast<size_t>(v * n_attn_ + l) * kStateBlock;
                        for (int p = p0; p < p1;) {
                            const int run = std::min(p1, (p / kv_chunk_ + 1) * kv_chunk_) - p;   // within a chunk
                            std::memcpy(blk + (seg + static_cast<size_t>(p - p0)) * row_bytes, kv_row_bytes(p, l, v != 0),
                                        static_cast<size_t>(run) * row_bytes);
                            if (scale_bytes)
                                std::memcpy(scales + (seg + static_cast<size_t>(p - p0)) * scale_bytes,
                                            kv_scale(p, l, v != 0), static_cast<size_t>(run) * scale_bytes);
                            p += run;
                        }
                    }
            }
            if (!f.write(h.kv_offset + static_cast<uint64_t>(b0) * h.block_bytes, buf.data(), static_cast<size_t>(nb) * bb))
                return fail("KV cache");
        }
    }
    // the header last, once what it names is on the flash
    if (!f.sync() || !f.write(0, &h, sizeof h) || !f.sync()) return fail("header");
    f.close();
    if (!in_place && !replace_file(target, path)) {
        std::remove(target.c_str());
        err = "cannot move " + target + " to " + path;
        return false;
    }
    return true;
}

bool Model::load_state(const std::string &path, int32_t &extra, std::string &err) {
    // whatever the failure, the conversation is empty after it (geode.h): never this one with another's extra
    const auto refuse = [&](const std::string &e) {
        reset();
        err = e;
        return false;
    };
    StateFile f;
    if (!f.open(path, false, false)) return refuse("cannot open " + path);
    StateHeader h{};
    std::string e;
    if (!read_header(f, h, e)) return refuse(e);
    if (h.fingerprint != fingerprint() || h.kv_elem != kv_elem() || h.n_attn != static_cast<uint32_t>(n_attn_) ||
        h.kv_heads != static_cast<uint32_t>(kv_heads_) || h.kv_row != kv_row_ || h.hidden != static_cast<uint32_t>(cfg_.hidden) ||
        h.conv_n != conv_n_ || h.gdn_n != gdn_n_ || h.block == 0)
        return refuse("the conversation was saved by another model or another KV cache format");
    if (h.position < 0 || h.position >= ctx_)
        return refuse("the saved conversation (" + std::to_string(h.position) + " tokens) is longer than this context (" +
                      std::to_string(ctx_) + ")");
    const Layout L = layout(h);
    const size_t elem = kv_elem(), row_bytes = kv_row_ * elem;
    const size_t scale_bytes = kv_i8_ ? static_cast<size_t>(kv_heads_) * sizeof(float) : 0;
    const int B = static_cast<int>(h.block);
    const int blocks = (h.position + B - 1) / B;
    if (L.fixed_bytes != h.fixed_bytes || L.kv_offset != h.kv_offset || h.n_images > h.image_cap ||
        h.block_bytes != 2 * static_cast<uint64_t>(n_attn_) * static_cast<uint64_t>(B) * (row_bytes + scale_bytes) ||
        f.size() < h.kv_offset + static_cast<uint64_t>(blocks) * h.block_bytes)
        return refuse("the saved conversation is cut short");
    uint64_t lead = 0, trail = 0;
    if (!f.read(h.fixed_offset, &lead, sizeof lead) || !f.read(L.trail, &trail, sizeof trail) || lead != h.gen ||
        trail != h.gen)
        return refuse("the saved conversation is incomplete (its save was interrupted)");
    // from here the model's conversation is replaced
    reset();
    const auto &fail = refuse;
    if (!reserve_context(std::max(1, h.position), e)) return fail(e);
    std::vector<int32_t> im(static_cast<size_t>(h.n_images) * 4);
    if (!im.empty() && !f.read(L.images, im.data(), im.size() * sizeof(int32_t))) return fail("cannot read the images' positions");
    images_.resize(h.n_images);
    for (size_t i = 0; i < images_.size(); ++i) {
        images_[i].kv0 = im[4 * i];
        images_[i].s = im[4 * i + 1];
        images_[i].gh = im[4 * i + 2];
        images_[i].gw = im[4 * i + 3];
        images_[i].feats.clear();   // its tokens are in the KV cache already
    }
    image_next_ = images_.size();
    image_fed_ = 0;
    if (h.has_hidden && keeps_hidden()) {
        last_hid_.resize(static_cast<size_t>(cfg_.hidden));
        if (!f.read(L.hidden, last_hid_.data(), last_hid_.size() * sizeof(float))) return fail("cannot read the last state");
    } else {
        last_hid_.clear();
    }
    if (!f.read(L.conv, conv_st_, conv_n_ * sizeof(float)) || !f.read(L.gdn, gdn_st_, gdn_n_ * sizeof(float)))
        return fail("cannot read the states");
    if (blocks > 0) {
        const size_t bb = static_cast<size_t>(h.block_bytes);
        const int per = static_cast<int>(std::max<size_t>(1, kIoBatch / bb));
        std::vector<uint8_t> buf(static_cast<size_t>(std::min(per, blocks)) * bb);
        for (int b0 = 0; b0 < blocks; b0 += per) {
            const int nb = std::min(per, blocks - b0);
            if (!f.read(h.kv_offset + static_cast<uint64_t>(b0) * h.block_bytes, buf.data(), static_cast<size_t>(nb) * bb))
                return fail("cannot read the KV cache");
            for (int b = b0; b < b0 + nb; ++b) {
                const uint8_t *blk = buf.data() + static_cast<size_t>(b - b0) * bb;
                const int p0 = b * B, p1 = std::min(h.position, p0 + B);
                const uint8_t *scales = blk + 2 * static_cast<size_t>(n_attn_) * static_cast<size_t>(B) * row_bytes;
                for (int v = 0; v < 2; ++v)
                    for (int l = 0; l < n_attn_; ++l) {
                        const size_t seg = static_cast<size_t>(v * n_attn_ + l) * static_cast<size_t>(B);
                        for (int p = p0; p < p1;) {
                            const int run = std::min(p1, (p / kv_chunk_ + 1) * kv_chunk_) - p;
                            std::memcpy(kv_row_bytes(p, l, v != 0), blk + (seg + static_cast<size_t>(p - p0)) * row_bytes,
                                        static_cast<size_t>(run) * row_bytes);
                            if (scale_bytes)
                                std::memcpy(kv_scale(p, l, v != 0), scales + (seg + static_cast<size_t>(p - p0)) * scale_bytes,
                                            static_cast<size_t>(run) * scale_bytes);
                            p += run;
                        }
                    }
            }
        }
    }
    set_rope_delta(h.rope_delta);
    pos_ = h.position;
    conv_id_ = h.conv_id;
    extra = h.extra;
    return true;
}

}  // namespace geode
