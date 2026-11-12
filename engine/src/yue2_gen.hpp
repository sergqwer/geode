// yue2_gen.hpp - YuE2's token generation (the reference's sampling.py generate_tokens) on FastAr: the ABC score
// (one branch), then the semantic codes (CFG: the positive and the negative branch, the same sampled token fed to
// both). The sampler works on the allowed rows only and keeps the survivors of top-k as candidates - the same
// probabilities as yue2.hpp's distribution() (the oracle) without a vocabulary-wide sort a token.
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "yue2_ar.hpp"

namespace geode::yue2 {

// The rows a phase may sample, as FastAr computes them: [lo, hi) and, when extra >= 0, the one id `extra` after
// them (ABC: the text ids [0, EOD) and ABC_END; semantic: [MUSIC_END, CODEC_OFFSET + CODEC_SIZE), no extra).
struct Allowed {
    int lo = 0, hi = 0, extra = -1, end = 0;   // end: the phase's end token (inside the rows)
    int size() const { return hi - lo + (extra >= 0); }
    int id(int i) const { return i < hi - lo ? lo + i : extra; }
    int index(int id) const;                   // -1 if not allowed
};
Allowed allowed_rows(const Config &c, bool abc);

// distribution() over the allowed rows' logits (vals [allowed.size()], fp32): the tokens with a non-zero weight
// and their probabilities (in descending order of probability, ties by id).
struct Candidates {
    std::vector<int> id;
    std::vector<float> p;
};
void distribution_rows(const float *vals, const Allowed &a, const Sampling &s, int step, const std::vector<int> &history,
                       Candidates &out);

// A deterministic draw from the candidates: u in [0, 1) from the 53 high bits of the engine's next number, the
// first candidate whose running sum (double) passes u. Not torch's multinomial stream: a seed is reproducible
// here, not against the reference.
int draw(const Candidates &c, std::mt19937_64 &rng);

// A branch and the tokens in its cache (what extend() compares the next prefix against).
struct Stream {
    ArBranch b;
    std::vector<int32_t> fed;
};

struct GenRequest {
    bool abc = false;            // the phase: ABC (one branch) or semantic
    Sampling s;
    float cfg = 1.0f;            // semantic: the guidance; != 1 needs the negative stream
    uint64_t seed = 0;
    // the prefixes the phase continues from: the streams' fed tokens must be a prefix of them (the rest is fed first)
    std::vector<int32_t> prefix, negative;
    // the repetition penalty's history before the first token (an edit sung on from a song's codes: the ones before
    // it, as the reference's _remembering); empty: none
    std::vector<int32_t> history;
    std::function<bool()> cancelled;              // polled every token
    std::function<void(int step, int token)> on_token;
    // ABC: prompt-lookup drafts verified several at a time (the score repeats its bars and sections). Every
    // position still draws its own number from the phase's generator and keeps a draft only when the draw is that
    // token, so the tokens are the one-at-a-time loop's (bit for bit while the batched products are - kernels.hpp).
    // With an int8 KV cache (the songs'); the others take the loop.
    bool lookup = true;
};
struct GenResult {
    std::vector<int32_t> tokens;   // without the end token
    bool eos = false;               // false: max_tokens reached (truncated)
    double prefill_seconds = 0, seconds = 0;
    int prefill_tokens = 0;
    int calls = 0, drafted = 0, accepted = 0;   // the lookup's verifications, their drafts, the drafts kept
    double slope = 0;                          // the lookup's learnt cost of a draft (a token's step = 1)
};

// One phase: the prefixes' missing tokens fed (one run, both streams), then up to s.max_tokens sampled. pos / neg
// grow by what is fed (the sampled tokens but the last / the end token, as the reference feeds them). neg: nullptr
// unless cfg != 1. The streams' capacities must hold prefix + max_tokens (checked).
bool generate(FastAr &ar, Stream &pos, Stream *neg, const GenRequest &req, GenResult &out, ThreadPool &pool,
              std::string &err);

// Several phases at once - takes of one song, each its own seed: every step's tokens of every stream through one
// run, so the weights are read once for all of them. Each stream gets the tokens generate() gives it alone (the
// products are per row - kernels.hpp - and the attention per branch); a stream whose cache and prefix equal an
// earlier one's takes that one's prefill by copy. One phase for all (reqs[i].abc the same); neg empty or one a
// stream (nullptr where cfg == 1).
bool generate_many(FastAr &ar, const std::vector<Stream *> &pos, const std::vector<Stream *> &neg,
                   const std::vector<GenRequest> &reqs, std::vector<GenResult> &outs, ThreadPool &pool,
                   std::string &err);

}  // namespace geode::yue2
