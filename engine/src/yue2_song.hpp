// yue2_song.hpp - a whole YuE2 song on the phone in one call (docs/music-plan.md M1): the score (ABC, FastAr), the
// codes (semantic, CFG's two branches), the NAR per original chunk (nar.py song_chunks: prefix + the chunk's codes +
// MUSIC_END, the noise drawn once for the song) on the DSP when the phone has HMX, else on the CPU, the VAE (tiled),
// the mastering (yue2_master.hpp) and the file. The caller tokenizes (the app's tokenizer): the heads are
// [EOD] + encode(text) without the ABC delimiters, which are the engine's (yue2/protocol.py token_prefixes).
//
// Memory, for a phone with less free memory than the weights (docs/music-plan.md): the CPU's weights are mapped
// (Pack::map), the generation's KV is int8, the NAR's prefix is recomputed per chunk into an fp16 branch (or on the
// DSP in fp16 from ar_f16.bin), every stage's memory is given back before the next one starts.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "yue2.hpp"
#include "yue2_lora.hpp"
#include "yue2_master.hpp"

namespace geode::yue2 {

enum class SongStage { Load, Score, Codes, Prefix, Acoustic, Decode, Master, Write, Join };   // (Join: a retake's; last, so
// the app's numbers of the others stay)
const char *song_stage_name(SongStage s);
enum class NarUnit { Auto, Dsp, Gpu, Cpu };   // Auto: the DSP where it has HMX, else an Adreno, else the CPU

struct SongRequest {
    std::string expected_lora_sha256; // internal: parent identity across an edit; never parsed from JSON
    LoraSpec lora;   // empty path: no adapter; CPU Q8 decode, optional merged fp16 DSP prompts/prefix (NAR unchanged)
    std::string pack_dir;
    std::string npu_dir;       // libgeode_npu.so and its skels; empty: no NPU
    std::string output_path;   // .flac or .wav (written as .part, renamed when complete)
    std::vector<int32_t> positive_head;   // [EOD] + encode(INSTRUCTIONS[cot] + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics + "\n")
    std::vector<int32_t> negative_head;   // [EOD] + encode(INSTRUCTIONS[cot]) (CFG's negative branch)
    enum class Score { Generate, Given, Off } score = Score::Generate;   // cot full / melody: Generate or Given; off: Off
    std::vector<int32_t> abc;  // Given: the score's text ids (no delimiters)
    // Generate: the score goes on from these ids (a song going on: its score's first bars) - the ABC phase's prefix
    // after abc_start, its repetition penalty remembering them; the result's abc begins with them
    std::vector<int32_t> abc_head;
    bool default_sampling = true;   // the pack's config.generation sampling; false: the two below
    Sampling abc_sampling, semantic_sampling;
    float cfg = 1.0f;          // guidance (1: no negative branch)
    uint64_t seed = 0;
    int max_frames = 0;        // the codes' ceiling (25 a second); 0: the sampling's max_tokens
    int ode_steps = 0;         // 0: the config's (32)
    int threads = 0;           // 0: every core
    NarUnit nar = NarUnit::Auto;
    // the score's and the codes' weights in Q4 (ar_q4.bin, when the pack has it): faster, at Q4's rounding; the NAR's
    // prefix stays the default's (the DSP's fp16, or Q8 on the CPU)
    bool ar_q4 = false;
    // The sound stage's precision. Best: as the model is (the DSP's fp16, the CPU's fp32 arithmetic). Balanced: only
    // what costs at most about a third of the bf16 run's own error - the CPU backend's attention scores in int8 (the
    // DSP: nothing yet). Fast: also the 8-bit weights (nar_q8.bin, made from nar_f16.bin the first time: yue2_q4.hpp
    // make_nar_q8) - the DSP's layers, the CPU's products by SDOT with int8 activations; noticeably less precise.
    enum class NarPrecision { Best, Balanced, Fast };
    NarPrecision nar_precision = NarPrecision::Best;
    bool nar_precision_given = false;   // (the request named it: an edit keeps it, else takes its song's)
    // (the former switch: nar_weights "q8" is Fast)
    // the takes queued after this one (the same request but the seed): their score and codes made with this song's,
    // the weights read once for all of them (yue2_gen.hpp generate_many: each its own tokens, as alone), and each
    // written to its ar_out as JSON (seed, abc, abc_eos, codes, codes_eos, ar) - as many as the free memory holds
    struct Companion {
        uint64_t seed = 0;
        std::string ar_out;
    };
    std::vector<Companion> companions;
    // a take whose score and codes a companion's run made: given here (score Given or Off), the AR left out
    bool codes_given = false;
    std::vector<int32_t> codes;    // codec indices 0 .. 32767
    bool abc_eos = true, codes_eos = true;
    bool master = true;
    MasterOptions mastering;
    int bits = 24;
    // the score alone (a plan: YuE2-ComfyUI's Plan / Plan Batch): written, given back in the result's abc, nothing
    // sung - the codes, the NAR and the decode neither opened nor run, no audio written; the companions' scores into
    // their ar_out (codes empty). Needs score Generate.
    bool score_only = false;
    // where the song's editable state goes (yue2_state.hpp: the NAR's prompt, the codes, the latents and their noise,
    // the mastering's gain), written after the audio; empty: none. Not for a plan (nothing sung).
    std::string state_out;
    // where its waveform and pitch classes go (yue2_view.hpp: what the track editor shows); empty: none
    std::string view_out;
    // An edit of a sung song (yue2_edit.hpp): with op set, make_song is edit_song - the heads, the score and the codes
    // are the song's (its state's), the output and state_out the edited song's
    struct Edit {
        enum class Op { None, Cut, Retake, GoOn, Move, Insert } op = Op::None;
        std::string state_in, audio_in;   // the song's state and its audio file
        int first = 0, stop = 0;          // its frames [first, stop) (25 a second)
        // a retake: how far its length may move from the stretch's either way, at most (a third of the stretch, and
        // what the context leaves, are the other bounds; 0: exactly the stretch's), and how many of the old codes after
        // it judge where it ends (inpaint/ops.py WIDTH, JOIN_FRAMES)
        int width = 50, join_frames = 75;
        // a song going on (from `first` to its end, `stop` ignored): the frames the new score asks for from `first` (the
        // app's measure of what the model wrote); its new prompts are built from the request's heads and its abc.
        // An insert (a break, track.py _break; `stop` ignored): about `length` new frames put in at `first`, nothing
        // taken out - the join chooses the count as a retake's does, judged on the old frames from `first` on
        int length = 0;
        int margin = 12;                  // frames drawn again either side of a seam (0.48 s)
        int window = 1000;                // the frames a drawn window reaches past them either way (40 s)
        bool redraw_edges = false;        // a cut at the start or the end drawn again too (else trimmed and faded)
        double fade_head = 0.2, fade_tail = 1.5;   // seconds: the fades a cut at the start / the end leaves
        // the edited song's NAR prompt and score (the app's: the lyrics and the score without the cut's bars);
        // empty: the song's own
        std::vector<int32_t> prompt, abc;
        // a move (track.py _move): the song's frames [f0, f1) in their new order (every frame once), then the stretches
        // before the joins that are sung again, in the assembled song's frames before any is sung, in order; `seeds`
        // takes of each (the best join kept: the request's seed and the next ones)
        std::vector<std::pair<int, int>> pieces, sung;
        int seeds = 2;
    } edit;
    // set by edit_song for the edited song's acoustic stage and decode (not from JSON): the NAR's prompt, the noise,
    // the known latents and the frames held on them; the frames the decode covers (given the decoder's halo) and how
    // the decoded sound goes into the old
    struct Repaint {
        std::vector<int32_t> prompt, negative;   // (negative and cfg: the song's, kept in the edited song's state)
        float cfg = 1.0f;
        std::vector<float> noise, known;   // [frames * latent]
        std::vector<uint8_t> held;         // [frames]: 1 = kept on known
        int window = 1000;
        bool mastered = false;             // the song's mastering (the edit's sound gets its gain in splice)
        double gain_db = 0;
        // the windows of latents to decode, [low, high) each, in order and apart
        std::function<void(int halo, std::vector<std::pair<int, int>> &windows)> decode_ranges;
        // each window's sound (interleaved, its natural length) -> the whole edited song's; false: err
        std::function<bool(std::vector<std::vector<float>> &decoded, const std::vector<std::pair<int, int>> &windows, int halo,
                           std::vector<float> &song, std::string &err)> splice;
    };
    const Repaint *repaint = nullptr;
};

struct SongProgress {
    SongStage stage = SongStage::Load;
    int chunk = 0, chunks = 1;
    int64_t done = 0, total = 0;   // the stage's units (tokens, ODE evaluations, VAE tiles); total may be a ceiling
    double elapsed = 0;            // seconds since the call began
};

struct SongCallbacks {
    std::function<void(const SongProgress &)> progress;   // on the calling thread; must not call back in
    std::function<bool()> cancelled;                       // polled between tokens, pieces, evaluations, tiles
};

struct SongResult {
    std::string lora_sha256;
    float lora_scale = 1;
    bool ok = false, cancelled = false;
    std::string error, output_path, nar_backend, vae_backend;   // vae: "cpu" or the GPU's name
    std::string ar_weights;        // "q8" or "q4": the score's and the codes' weights
    std::string nar_weights;       // "f16" or "q8": the acoustic stage's (its DSP layers, or the CPU's products' twins)
    std::string nar_precision;     // "best", "balanced" or "fast": what the sound stage ran at
    std::vector<int32_t> abc;      // the score's text ids (generated or given)
    std::vector<int32_t> codes;    // codec indices 0 .. 32767 (not vocabulary ids)
    bool abc_eos = true, codes_eos = true;   // false: stopped at the ceiling (truncated)
    int chunks = 0;
    int64_t samples = 0;           // audio frames at 48 kHz
    MasterResult mastering;
    bool mastered = false;
    double t_load = 0, t_score = 0, t_codes = 0, t_prefix = 0, t_acoustic = 0, t_decode = 0, t_master = 0,
           t_write = 0, t_all = 0;
    int prompt_dsp_tokens = 0;     // the generation prompts' tokens the DSP computed (the rest: the CPU's)
    int companions = 0;            // the companions whose score and codes were written
    bool state = false;            // the state_out written
    bool view = false;             // the view_out written
    double t_prompt_dsp = 0;       // its calls (within t_score / t_codes)
    // a retake: the new frames sung in place of the stretch, the join's score of that count (NaN: not chosen by the
    // join - the model ended the song, or the retake reaches the end), whether the model ended the song itself, and
    // the join's time (beside t_codes, the new part's)
    int edit_count = -1, edit_length = 0;
    double edit_join = std::nan(""), t_join = 0;
    bool edit_ended = false;
    bool edit_words = false;       // a retake sung to new words (the request's heads): "words": true
    std::vector<std::array<int, 3>> edit_sung;   // a move's stretches sung again: [a, b) -> count ("move": {"sung"})
    std::string edit_kind;         // "retake", "go_on" (no join; edit_length what its score asked) or "insert"
};

SongResult make_song(const SongRequest &req, const SongCallbacks &cb);

// A song's codes' prompts (protocol.py token_prefixes / negative_prefix): the heads, then the score between its
// delimiters and music start - off: no score, its negative without the score's delimiters
void song_prompts(const Tokens &T, bool off, const std::vector<int32_t> &positive_head, const std::vector<int32_t> &negative_head,
                  const std::vector<int32_t> &abc, std::vector<int32_t> &sp, std::vector<int32_t> &np);

// The result as JSON (the JNI layer's answer; the CLI prints it)
std::string song_result_json(const SongResult &r);

// A request from JSON (the app's and tools/yue2_request.py's): positive_head, negative_head (token ids), score
// ("generate" | "given" | "off"), abc (ids), cfg, seed, max_frames, ode_steps, master, target_lufs, max_gain_db,
// bits, threads, score_only, state_out, view_out, edit ({op "cut" | "retake" | "go_on", state_in, audio_in, first, stop, length, margin, window, width, join_frames,
// redraw_edges, fade_head, fade_tail, prompt, abc}; go_on: the new lyrics' heads and the whole new abc at the top level,
// score "given"; a cut by bars: its lyrics' heads and the score without them, score "given" (or "off" for a song with
// none); new words for a retake: its lyrics' heads alone, the score the song's), abc_head (a score the model writes going on from these ids), nar ("auto" | "dsp" | "cpu"), ar ("q8" | "q4"), nar_precision ("best" | "balanced" | "fast"; nar_weights "q8": fast), companions ([{seed, ar_out}]), codes +
// codes_eos + abc_eos (given), pack_dir / npu_dir / output (when given: else the caller's), and
// abc_sampling / semantic_sampling objects (temperature, top_p, top_k, repetition_penalty, penalty_window,
// min_tokens, max_tokens: overrides on the pack's defaults, which need the pack - read here when they are given)
bool song_request_from_json(const std::string &text, SongRequest &q, std::string &err);

}  // namespace geode::yue2
