// A finished song's editable state - what YuE2-ComfyUI's Edit Track keeps of a song so that a later edit (a retake of
// some bars, a cut, the song carried on) sings from it instead of making the whole song again: the NAR's prompt, the
// score, the codes, the acoustic latents and the noise they were solved from, and what the mastering did. One file
// beside the song's audio (song.state), a few MB for a long song: 25 frames a second, 64 latent values each.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace geode::yue2 {

struct SongState {
    static constexpr int kVersion = 2;
    std::string lora_sha256;   // empty: unadapted (including version 1 states)
    float lora_scale = 1;
    uint64_t seed = 0;            // the song's seed: its noise is noise_of(seed) until an edit splices other runs in
    int ode_steps = 0;            // the ODE steps its latents were solved with
    int latent = 0;               // values a frame (the config's latent: 64)
    int rate = 48000;             // the audio's sample rate
    std::string ar_weights, nar_weights, nar_precision;   // what made it ("q8", "f16", "balanced", ...)
    bool mastered = false;
    double master_gain_db = 0;    // the mastering's gain: an edit's new audio gets the same
    std::vector<int32_t> prompt;  // the NAR's prompt: the positive head, the score between its delimiters, music start
                                  // (the codes' positive branch began with the same)
    std::vector<int32_t> negative;  // the codes' negative branch's (CFG): the negative head, the score, music start
    float cfg = 1.0f;             // the guidance the codes were sampled with (1: no negative branch)
    std::vector<int32_t> abc;     // the score's text ids (empty: a song without one)
    bool abc_eos = true;
    std::vector<int32_t> codes;   // codec indices 0 .. 32767, 25 a second
    bool codes_eos = true;
    std::vector<float> latents;   // [codes.size() * latent]: the acoustic stage's solution, the VAE's input
    std::vector<float> noise;     // [codes.size() * latent]: where the ODE started each frame
};

// N(0, 1)[n] from [seed]: a song's noise - Box-Muller over mt19937_64's 53-bit doubles, the same numbers on every
// platform (not torch's stream: a seed is reproducible here, not against the reference)
std::vector<float> noise_of(uint64_t seed, size_t n);

// [s] into [path], whole (written as .part, then renamed); false with [err] when [s] does not hold together (latents
// and noise not codes x latent) or the file could not be written
bool write_state(const std::string &path, const SongState &s, std::string &err);

// [path] into [s]; false with [err] for a missing, truncated, foreign or newer file
bool read_state(const std::string &path, SongState &s, std::string &err);

}  // namespace geode::yue2
