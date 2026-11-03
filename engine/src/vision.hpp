// vision.hpp - Qwen3.6's vision encoder (PLAN.md, "Images"): an image into the LLM's embeddings at its image_pad
// tokens. A ViT of 27 blocks over 16x16 patches, then a merger of 2x2 patches into one token, on the CPU's int8
// kernels. Its weights stay in the pack's vision.bin (tools/pack_vision.py): each block is read while the one before
// computes and let go after it, so between images the encoder holds no memory and the generating speed is untouched.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu8.hpp"
#include "kernels.hpp"
#include "pack.hpp"
#include "threads.hpp"

namespace geode {

class Vision {
public:
    struct Shape {
        int depth = 0, hidden = 0, heads = 0, ff = 0, patch = 0, merge = 0, out_hidden = 0, pos_grid = 0;
        float theta = 10000.0f, eps = 1e-6f;
        float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
        int64_t min_pixels = 0, max_pixels = 0;
        int start_token = -1, end_token = -1, pad_token = -1;   // <|vision_start|>, <|vision_end|>, <|image_pad|>
        // Gemma 4's encoder (geode.json's vision.arch "gemma4", tools/pack_gemma4_vision.py): merge is its 3 x 3
        // pooling, the tokens <|image>, <|image|>, <image|>; soft_tokens the default budget an image fits in (fit)
        bool gemma = false;
        int head_dim = 0, pos_size = 0, soft_tokens = 0;
        // a video's frames (Gemma 4): the same encoder, their tokens <|video|>, each at most video_soft_tokens (70)
        int video_pad = -1, video_soft_tokens = 0;
    };

    // The pack's "vision" section checked; false with err empty when the pack has none, with err when it is broken.
    // direct: vision.bin read with O_DIRECT (the phone: no page cache filled with what is read once).
    bool open(const Pack &pack, bool direct, std::string &err);
    bool available() const { return ok_; }
    void disable() { ok_ = false; }
    const Shape &shape() const { return s_; }
    // The size an image of w x h goes in at (Qwen's smart_resize): each side a multiple of patch * merge, near the
    // aspect, at least min_pixels and at most max_pixels (<= 0: the pack's) pixels. Gemma 4: the largest with the
    // aspect kept, both sides multiples of 48 and at most max_pixels / 48^2 tokens (<= 0: soft_tokens, 280).
    void fit(int w, int h, int64_t max_pixels, int &out_w, int &out_h) const;
    // the image_pad tokens of an image of w x h (fitted)
    int tokens(int w, int h) const;
    // rgb: h rows of w pixels (R, G, B bytes), w and h multiples of patch * merge. out: tokens(w, h) rows of
    // out_hidden floats, the merger's output, in the order of the image's pad tokens (rows of merged tokens).
    // gpu: the Adreno 7xx+ path, when there is one: the products run there (Gpu8::vision_gemm), the rest on pool.
    bool encode(const uint8_t *rgb, int w, int h, ThreadPool &pool, std::vector<float> &out, std::string &err,
                Gpu8 *gpu = nullptr);
    // where the last encode's time went
    std::string profile() const;
    // An image's features kept in a file (the app's cache): written after an encode, read instead of one when the
    // file is this image's at this size from this vision.bin (a header says so). False when it is not.
    bool load_features(const std::string &path, int w, int h, std::vector<float> &out) const;
    bool has_features(const std::string &path, int w, int h) const;   // the header only
    bool save_features(const std::string &path, int w, int h, const std::vector<float> &feats) const;

private:
    struct Range {
        uint64_t offset = 0, bytes = 0;
    };
    // Gemma 4's encoder (vision.cpp): open's and encode's
    bool open_gemma(const Pack &pack, bool direct, std::string &err);
    bool encode_gemma(const uint8_t *rgb, int w, int h, ThreadPool &pool, std::vector<float> &out, std::string &err);
    // a tensor of vision.bin in a buffer holding [base, ...) of the file
    const Tensor *tensor(const std::string &name, std::string &err) const;
    bool read(const Range &r, AlignedBuffer &buf, size_t &cap, uint64_t &base, std::string &err) const;

    const Pack *pack_ = nullptr;
    RandomFile file_;
    bool ok_ = false;
    Shape s_;
    std::string fingerprint_;   // vision.bin's sha256 as geode.json lists it (the feature files carry it)
    Range head_, merger_;   // the patch embedding and the position table; the merger (Gemma 4: the embedder)
    std::vector<Range> blocks_;
    struct Prof {
        double total = 0, wait = 0, embed = 0, matmul = 0, attention = 0, other = 0;
        int patches = 0;
        const char *where = "CPU";   // the products
    } prof_;
};

}  // namespace geode
