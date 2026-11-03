// audio.hpp - Gemma 4's audio encoder (tools/gemma4_audio.py is the spec, tools/pack_gemma4_audio.py the pack): a clip
// of 16 kHz PCM into the LLM's embeddings at its <|audio|> tokens. Log-mel features, two stride-2 convolutions, 12
// Conformer layers (attention over the token and the 11 before it with relative positions, a light causal
// convolution, two half-step MLPs), on the CPU's int8 kernels (SMMLA where the CPU has it). Its weights stay in the
// pack's audio.bin: a layer is read while the one before computes and let go after it (as the vision encoder's), so
// between clips the encoder holds no memory.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pack.hpp"
#include "platform.hpp"
#include "threads.hpp"

namespace geode {

class Audio {
public:
    struct Shape {
        int layers = 0, hidden = 0, heads = 0, head_dim = 0, ff = 0, window = 0, n_rel = 0, kernel = 0;
        int ch0 = 0, ch1 = 0, out_hidden = 0, text_hidden = 0, mel_bins = 0, frame = 0, hop = 0, fft = 0;
        int64_t max_samples = 0;
        float cap = 0, residual = 0, eps = 1e-6f, k_scale = 0, mel_floor = 1e-3f;
        int start_token = -1, end_token = -1, pad_token = -1;   // <|audio>, <audio|>, <|audio|>
    };

    // The pack's "audio" section checked; false with err empty when the pack has none, with err when it is broken.
    // direct: audio.bin read with O_DIRECT.
    bool open(const Pack &pack, bool direct, std::string &err);
    bool available() const { return ok_; }
    void disable() { ok_ = false; }
    const Shape &shape() const { return s_; }
    // the <|audio|> tokens of a clip of n samples (the first max_samples of a longer one): one per 40 ms
    int tokens(int64_t n) const;
    // pcm: n samples at 16 kHz in [-1, 1]. out: tokens(n) rows of text_hidden floats (the text model's embeddings
    // there), in order.
    bool encode(const float *pcm, int64_t n, ThreadPool &pool, std::vector<float> &out, std::string &err);
    // where the last encode's time went
    std::string profile() const;

private:
    struct Range {
        uint64_t offset = 0, bytes = 0;
    };
    bool read(const Range &r, AlignedBuffer &buf, size_t &cap, uint64_t &base, std::string &err) const;
    // the log-mel frames of the clip padded as the feature extractor pads it; valid: the frames of real samples
    void features(const float *pcm, int64_t n, const float *win, const float *filters, ThreadPool &pool,
                  std::vector<float> &mel, std::vector<uint8_t> &valid) const;

    const Pack *pack_ = nullptr;
    RandomFile file_;
    bool ok_ = false;
    Shape s_;
    Range head_, tail_;
    std::vector<Range> layers_;
    struct Prof {
        double total = 0, wait = 0, features = 0, sub = 0, matmul = 0, attention = 0, other = 0;
        int frames = 0, tokens = 0;
    } prof_;
};

}  // namespace geode
