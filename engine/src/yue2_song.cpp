// yue2_song.cpp - a whole song (yue2_song.hpp)
#include "yue2_song.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <thread>

#include "json.hpp"
#include "platform.hpp"
#include "threads.hpp"
#include "yue2_ar.hpp"
#include "yue2_gen.hpp"
#include "yue2_nar.hpp"
#include "yue2_pack.hpp"
#include "yue2_q4.hpp"
#include "yue2_edit.hpp"
#include "yue2_q8gemm.hpp"
#include "yue2_state.hpp"
#include "yue2_view.hpp"
#include "yue2_vae.hpp"

namespace geode::yue2 {

const char *song_stage_name(SongStage s) {
    switch (s) {
        case SongStage::Load: return "load";
        case SongStage::Score: return "score";
        case SongStage::Codes: return "codes";
        case SongStage::Prefix: return "prefix";
        case SongStage::Acoustic: return "acoustic";
        case SongStage::Decode: return "decode";
        case SongStage::Master: return "master";
        case SongStage::Write: return "write";
        case SongStage::Join: return "join";
    }
    return "?";
}

namespace {

constexpr int kRate = 48000;

// protocol.py chunk_ranges
std::vector<std::pair<int, int>> chunk_ranges(int frames, int prefix, int context) {
    const int size = std::min((context - prefix - 3) / 2, context);
    std::vector<std::pair<int, int>> r;
    if (frames < 1 || size < 1) return r;
    for (int a = 0; a < frames; a += size) r.emplace_back(a, std::min(a + size, frames));
    return r;
}

std::string json_escape(const std::string &s) {
    std::string o;
    for (const char ch : s) {
        if (ch == '"' || ch == '\\') o += '\\', o += ch;
        else if (static_cast<unsigned char>(ch) < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", ch);
            o += b;
        } else o += ch;
    }
    return o;
}

}  // namespace

namespace {
// MemAvailable in bytes (0: unknown - no companions then)
int64_t mem_free() {
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    long long kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    std::fclose(f);
    return static_cast<int64_t>(kb) * 1024;
}
}  // namespace

// A companion's score and codes into its ar_out as JSON (seed, ar, abc, abc_eos, codes, codes_eos; a plan's: made, the
// seconds its batch took to write it), written whole: .part, then renamed. False: it could not be (the take then makes
// its own).
static bool write_take(const std::string &out, uint64_t seed, const std::string &ar, const std::vector<int32_t> &abc,
                       bool abc_eos, const std::vector<int32_t> &codes, bool codes_eos, const SongResult &identity, double made = -1.0) {
    std::ostringstream o;
    const auto ints = [&](const std::vector<int32_t> &v) {
        o << '[';
        for (size_t k = 0; k < v.size(); ++k) o << (k ? "," : "") << v[k];
        o << ']';
    };
    o << "{\"seed\":" << seed << ",\"ar\":\"" << ar << "\",\"abc\":";
    ints(abc);
    o << ",\"abc_eos\":" << (abc_eos ? "true" : "false") << ",\"codes\":";
    ints(codes);
    o << ",\"codes_eos\":" << (codes_eos ? "true" : "false");
    if (!identity.lora_sha256.empty()) o << ",\"lora\":{\"sha256\":\"" << identity.lora_sha256 << "\",\"scale\":" << lora_scale_text(identity.lora_scale) << "}";
    if (made >= 0.0) o << ",\"made\":" << made;
    o << "}";
    const std::string text = o.str(), part = out + ".part";
    std::FILE *f = std::fopen(part.c_str(), "wb");
    bool ok = f && std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f && std::fclose(f) != 0) ok = false;
    if (ok && std::rename(part.c_str(), out.c_str()) == 0) return true;
    std::remove(part.c_str());
    return false;
}

SongResult make_song(const SongRequest &asked, const SongCallbacks &cb) {
    SongRequest req = asked;
    if (!req.lora.path.empty() && req.ar_q4) {
        std::fprintf(stderr, "lora: Q4 ignored; AR uses merged CPU Q8 weights\n");
        req.ar_q4 = false;
    }
    if (req.edit.op != SongRequest::Edit::Op::None && !req.repaint) return edit_song(req, cb);
    const SongRequest::Repaint *rp = req.repaint;
    SongResult r;
    const double t0 = now_seconds();
    double mark = t0;
    const auto lap = [&](double &into) {
        const double t = now_seconds();
        into += t - mark;
        mark = t;
    };
    const auto cancelled = [&]() { return cb.cancelled && cb.cancelled(); };
    SongProgress pg;
    const auto report = [&](SongStage st, int64_t done, int64_t total) {
        if (!cb.progress) return;
        pg.stage = st, pg.done = done, pg.total = total, pg.elapsed = now_seconds() - t0;
        cb.progress(pg);
    };
    const auto fail = [&](const std::string &e) {
        r.ok = false;
        r.cancelled = e == "cancelled" || cancelled();
        r.error = r.cancelled ? "cancelled" : e;
        r.t_all = now_seconds() - t0;
        return r;
    };
    if (!rp && (req.positive_head.empty() || req.negative_head.empty())) return fail("song: empty prompt");
    if (req.score_only && (req.score != SongRequest::Score::Generate || req.codes_given))
        return fail("song: a score alone is a score the model writes");
    if (req.output_path.empty()) return fail("song: no output path");
    report(SongStage::Load, 0, 1);

    const int threads = req.threads > 0 ? req.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    ThreadPool pool(threads);
    std::string err;
    clean_ar_q4_parts(req.pack_dir);   // (a killed conversion's leftovers)
    clean_nar_q8_parts(req.pack_dir);
    // the Q4 weights asked for and the pack without them (never made, or a manifest that did not merge): made now
    // from ar_q8.bin (yue2_q4.hpp; ~12 s on an 8 Elite), the load stage counting its matrices. A pack that has them
    // from the PC (pack_yue2.py --ar-q4, in geode.json) is left as it is.
    if (req.ar_q4) {
        bool have = false;
        {
            Pack probe;
            if (!probe.open(req.pack_dir, err)) return fail(err);
            have = FastAr::has_q4(probe);
        }
        if (!have && !make_ar_q4(req.pack_dir, pool, [&](int d, int n) { report(SongStage::Load, d, n); }, cb.cancelled, err))
            return fail(err == "cancelled" ? err : "song: the Q4 weights: " + err);
    }
    LoraWeights selected;
    if (!req.lora.path.empty()) {
        if (!prepare_lora(req.pack_dir, req.lora, pool, [&](int d, int n) { report(SongStage::Load, d, n); }, cb.cancelled, selected, err)) return fail(err);
        if (!req.expected_lora_sha256.empty() && selected.sha256 != req.expected_lora_sha256)
            return fail("edit: adapter changed while preparing the song");
        r.lora_sha256 = selected.sha256; r.lora_scale = selected.scale;
    }
    Pack pack;
    if (!pack.open_selected(req.pack_dir, selected.directory, err)) return fail(err);
    if (req.ar_q4 && !FastAr::has_q4(pack)) return fail("song: the Q4 weights were made but do not open (see the log)");
    Config c;
    if (!read_config(pack, c, err)) return fail(err);
    const Tokens &T = c.tok;
    Sampling sa = req.default_sampling ? c.abc : req.abc_sampling;
    Sampling ss = req.default_sampling ? c.semantic : req.semantic_sampling;
    if (req.max_frames > 0) ss.max_tokens = std::min(ss.max_tokens, req.max_frames);
    ss.min_tokens = std::min(ss.min_tokens, ss.max_tokens);
    const int steps = req.ode_steps > 0 ? req.ode_steps : c.ode_steps;

    // the NAR's unit: the DSP when the phone has it (its driver and HMX), else the CPU
    auto nar = std::make_unique<NarDsp>();
    bool dsp = false;
    std::string why_not;
    if (!req.score_only && (req.nar == NarUnit::Auto || req.nar == NarUnit::Dsp) && !req.npu_dir.empty()) {
        if (nar->open(pack, req.npu_dir, pool, why_not) && nar->backend() == "dsp" && nar->driver()) dsp = true;
        else nar = std::make_unique<NarDsp>();
    }
    if (!dsp && !req.score_only) {
        if (req.nar == NarUnit::Dsp) return fail("song: the NPU cannot run the NAR" + (why_not.empty() ? "" : ": " + why_not));
        // the GPU's products (yue2_nar_gpu.hpp): asked for, or on its own on an Adreno (measured there: 1.4x the CPU's)
        bool gpu = false;
        if (req.nar == NarUnit::Gpu || req.nar == NarUnit::Auto) {
            std::string why_gpu;
            gpu = nar->open(pack, "gpu", pool, why_gpu) && (req.nar == NarUnit::Gpu || nar->device().find("Adreno") != std::string::npos);
            if (!gpu && req.nar == NarUnit::Gpu) return fail("song: the GPU cannot run the NAR: " + why_gpu);
            if (!gpu) nar = std::make_unique<NarDsp>();
        }
        if (!gpu && !nar->open(pack, "cpu", pool, err)) return fail(err);
    }
    const bool dsp_prefix = req.lora.path.empty() && dsp && NarDsp::has_ar_f16(pack);
    r.nar_backend = req.score_only ? "" : nar->backend() + (dsp && !dsp_prefix ? " (CPU prefix)" : "");
    // 8-bit sound: on the DSP the layers' 8-bit weights; on the CPU / GPU backends the CPU's products from them (the
    // GPU's stay fp16) and the attention's scores in int8. The weights made now when the pack has none (from
    // nar_f16.bin; ~2 s on an 8 Elite, its file in the page cache) and merged - before anything reads the pack on a
    // thread
    using P = SongRequest::NarPrecision;
    const bool attn8 = !req.score_only && !dsp && have_q8_gemm() && req.nar_precision != P::Best;
    const bool eight = !req.score_only && req.nar_precision == P::Fast && (dsp || have_q8_gemm());
    if (attn8) nar->set_attn8(true);
    r.nar_precision = eight ? "fast" : attn8 ? "balanced" : "best";
    if (eight) {   // (the DSP's h8.*; the CPU's turned c8.*: yue2_q4.hpp)
        if (!pack.find(dsp ? "h8.model.layers.0.nar_qkv" : "c8.model.layers.0.nar_qkv")) {
            if (!make_nar_q8(req.pack_dir, pool, [&](int d, int n) { report(SongStage::Load, d, n); }, cb.cancelled, err, !dsp))
                return fail(err == "cancelled" ? err : "song: the 8-bit sound weights: " + err);
            if (!pack.merge_extra(dsp ? kNarQ8Manifest : kNarC8Manifest, err))
                return fail("song: the 8-bit sound weights do not open: " + err);
        }
        nar->set_weights_q8(true);
        // the products also on a Hexagon v66 .. v72 DSP when the phone has one (npu/hvx66: no HMX, integer HVX)
        if (!dsp && !req.npu_dir.empty() && nar->backend() == "gpu") {
            std::string why;
            if (nar->set_hvx(req.npu_dir, why)) r.nar_backend += " + DSP (HVX)";
        }
    }

    // ---- the AR: the score, then the codes
    FastAr ar;
    {
        ArOptions o;
        o.kv = KvType::I8;
        o.map = true;
        o.q4 = req.ar_q4;   // (has_q4 checked above)
        if (!ar.open(pack, o, err)) return fail(err);
        ar.cancelled = cb.cancelled;
        r.ar_weights = o.q4 ? "q4" : "q8";
    }
    lap(r.t_load);
    // the generation's prompts on the DSP as well when it has the AR's fp16 layers: read from now, beside the CPU's
    // work (the Pack is done changing: FastAr mapped its file), kept for the chunks' prefixes
    const bool dsp_prompt = dsp_prefix && nar->load_ar_async();
    // a stream's prompt but its last token (generate() feeds that one: its logits start the phase); a short one, or
    // one already begun, stays the CPU's - so does one the DSP fails (the stream left empty)
    const auto dsp_fill = [&](Stream &s, const std::vector<int32_t> &want) {
        const int n = static_cast<int>(want.size()) - 1;
        if (!dsp_prompt || n < 256 || !s.fed.empty() || s.b.len || n > s.b.cap) return;
        std::string e;
        double sec = 0;
        if (!nar->ar_prompt(want.data(), n, [&](int l, float *k, float *v) { return ar.put_kv(s.b, l, 0, n, k, v, pool); },
                            e, &sec)) {
            std::fprintf(stderr, "song: the prompt on the DSP failed (%s): the CPU computes it\n", e.c_str());
            return;
        }
        s.b.len = n;
        s.fed.assign(want.begin(), want.begin() + n);
        r.prompt_dsp_tokens += n;
        r.t_prompt_dsp += sec;
    };
    // ---- the takes: this song, then each companion its caches fit beside (the free memory less 1.5 GB)
    struct Take {
        uint64_t seed = 0;
        std::string out;
        Stream pos, neg;
        std::vector<int32_t> abc, codes, sp, np;
        bool abc_eos = true, codes_eos = true;
        Sampling s;
    };
    const bool cfg = req.cfg != 1.0f;
    const int sem_room = ss.max_tokens;
    const auto pos_cap = [&](int head) { return std::min<int>(T.context, head + sem_room + 8); };
    std::vector<int32_t> prompt(req.positive_head);   // the score's prompt (and the head it goes on from)
    prompt.push_back(T.abc_start);
    if (req.score == SongRequest::Score::Generate) prompt.insert(prompt.end(), req.abc_head.begin(), req.abc_head.end());
    if (!req.abc_head.empty() && (req.score != SongRequest::Score::Generate || !req.companions.empty()))
        return fail("song: a score's head goes with a score the model writes, alone");
    if (req.score == SongRequest::Score::Generate && static_cast<int>(prompt.size()) + sa.max_tokens > T.context)
        return fail("song: the lyrics and the score so far leave no room in the context for the rest of the score");
    // (a plan's cache holds its score alone; a song's goes on into the codes)
    const int score_cap = std::min<int>(T.context, static_cast<int>(prompt.size()) + sa.max_tokens + 2 +
                                                       (req.score_only ? 0 : sem_room + 8));
    std::vector<Take> tk(1);
    tk[0].seed = req.seed;
    if (!req.codes_given && !req.companions.empty()) {
        const int64_t per_pos = static_cast<int64_t>(c.layers) * c.kv_heads * (2 * c.head_dim + 8);   // (int8 + scales)
        // a take's caches at their largest: the score's (or a given one's tokens) + the codes, the negative's too
        const int abc_len = req.score == SongRequest::Score::Given ? static_cast<int>(req.abc.size())
                            : req.score == SongRequest::Score::Generate ? sa.max_tokens : 0;
        const int pos_len = std::min<int>(T.context, static_cast<int>(req.positive_head.size()) + abc_len + 4 + sem_room + 8);
        const int neg_len = std::min<int>(T.context, static_cast<int>(req.negative_head.size()) + abc_len + 4 + sem_room + 8);
        const int64_t take_bytes = req.score_only ? per_pos * score_cap
                                                  : per_pos * (std::max(score_cap, pos_len) + (cfg ? neg_len : 0));
        int64_t room = mem_free() - (int64_t{3} << 29) - take_bytes;   // (this song's own first)
        for (const auto &cp : req.companions) {
            if (room < take_bytes) break;
            room -= take_bytes;
            tk.emplace_back();
            tk.back().seed = cp.seed, tk.back().out = cp.ar_out;
        }
        if (tk.size() < req.companions.size() + 1)
            std::fprintf(stderr, "song: %zu of %zu companions fit the free memory\n", tk.size() - 1, req.companions.size());
    }
    const size_t K = tk.size();
    if (req.codes_given) {
        if (req.score == SongRequest::Score::Generate || req.codes.empty()) return fail("song: given codes need their score");
        r.abc = req.abc, r.abc_eos = req.abc_eos;
        r.codes = req.codes, r.codes_eos = req.codes_eos;
    } else {
        // ---- the score: every take's at once, their prompt computed once (the DSP's for this song, copied)
        if (req.score == SongRequest::Score::Generate) {
            for (Take &t : tk)
                if (!ar.branch_init(t.pos.b, score_cap, err)) return fail(err);
            dsp_fill(tk[0].pos, prompt);
            for (size_t i = 1; i < K; ++i) {
                if (!ar.branch_copy(tk[0].pos.b, tk[i].pos.b, tk[0].pos.b.len)) return fail("song: a take's prompt cache");
                tk[i].pos.fed = tk[0].pos.fed;
            }
            std::vector<GenRequest> g(K);
            std::vector<Stream *> ps;
            for (size_t i = 0; i < K; ++i) {
                g[i].abc = true, g[i].s = sa, g[i].seed = tk[i].seed, g[i].prefix = prompt, g[i].cancelled = cb.cancelled;
                g[i].history = req.abc_head;   // (core.py score_on: _remembering(written))
                ps.push_back(&tk[i].pos);
            }
            g[0].on_token = [&](int step, int) {
                if ((step & 15) == 0) report(SongStage::Score, step + 1, sa.max_tokens);
            };
            std::vector<GenResult> out;
            if (!generate_many(ar, ps, {}, g, out, pool, err)) return fail(err);
            for (size_t i = 0; i < K; ++i) {
                tk[i].abc = req.abc_head;   // (the head, then what the model wrote after it)
                tk[i].abc.insert(tk[i].abc.end(), out[i].tokens.begin(), out[i].tokens.end());
                tk[i].abc_eos = out[i].eos;
            }
            r.abc = tk[0].abc, r.abc_eos = tk[0].abc_eos;
            lap(r.t_score);
            if (req.score_only) {
                // the plans: the companions' scores into their files (codes none), this one's in the result
                r.t_all = now_seconds() - t0;
                for (size_t i = 1; i < K; ++i) {
                    if (write_take(tk[i].out, tk[i].seed, r.ar_weights, tk[i].abc, tk[i].abc_eos, {}, true, r, r.t_all))
                        ++r.companions;
                }
                r.ok = true;
                return r;
            }
        } else if (req.score == SongRequest::Score::Given) {
            if (req.abc.empty()) return fail("song: a given score with no tokens");
            for (Take &t : tk) t.abc = req.abc;
            r.abc = req.abc;
        }
        // ---- the codes: every take's at once (its prefixes from its score)
        for (Take &t : tk) song_prompts(T, req.score == SongRequest::Score::Off, req.positive_head, req.negative_head, t.abc, t.sp, t.np);
        // (a companion whose score leaves no room is left out; this song's own is the error)
        if (static_cast<int>(tk[0].sp.size()) + 16 >= T.context) return fail("song: the lyrics and the score leave no room for music");
        for (size_t i = K; i-- > 1;)
            if (static_cast<int>(tk[i].sp.size()) + 16 >= T.context) tk.erase(tk.begin() + static_cast<std::ptrdiff_t>(i));
        const size_t K2 = tk.size();
        for (size_t i = 0; i < K2; ++i) {
            Take &t = tk[i];
            t.s = ss;
            t.s.max_tokens = std::min(t.s.max_tokens, T.context - static_cast<int>(t.sp.size()) - 2);
            t.s.min_tokens = std::min(t.s.min_tokens, t.s.max_tokens);
            // a fresh stream (a given score's or none): the prefix computed - by copy where it is this song's
            if (!t.pos.b.cap && !ar.branch_init(t.pos.b, pos_cap(static_cast<int>(t.sp.size())), err)) return fail(err);
            if (cfg && !ar.branch_init(t.neg.b, std::min<int>(T.context, static_cast<int>(t.np.size()) + t.s.max_tokens + 8), err))
                return fail(err);
            const auto fill = [&](Stream &st, Stream &mine, const std::vector<int32_t> &want, const std::vector<int32_t> &main_want) {
                if (i > 0 && want == main_want && st.fed.empty() && !st.b.len && mine.b.len && mine.b.len <= st.b.cap) {
                    if (!ar.branch_copy(mine.b, st.b, mine.b.len)) return false;
                    st.fed = mine.fed;
                    return true;
                }
                dsp_fill(st, want);
                return true;
            };
            if (!fill(t.pos, tk[0].pos, t.sp, tk[0].sp) || (cfg && !fill(t.neg, tk[0].neg, t.np, tk[0].np)))
                return fail("song: a take's prefix cache");
        }
        // the NAR's layers read into the DSP's memory beside the codes' generation (unmapped: the prefix maps the
        // AR's after it), so the sound starts without reading them
        if (dsp_prefix) nar->load_nar_async();
        std::vector<GenRequest> g(K2);
        std::vector<Stream *> ps, ns;
        for (size_t i = 0; i < K2; ++i) {
            Take &t = tk[i];
            g[i].s = t.s, g[i].cfg = req.cfg, g[i].seed = t.seed, g[i].prefix = t.sp, g[i].negative = t.np;
            g[i].cancelled = cb.cancelled;
            ps.push_back(&t.pos);
            if (cfg) ns.push_back(&t.neg);
        }
        const int sem_total = tk[0].s.max_tokens;
        g[0].on_token = [&](int step, int) {
            if ((step & 15) == 0) report(SongStage::Codes, step + 1, sem_total);
        };
        std::vector<GenResult> out;
        if (!generate_many(ar, ps, ns, g, out, pool, err)) return fail(err);
        for (size_t i = 0; i < K2; ++i) {
            tk[i].codes.reserve(out[i].tokens.size());
            for (const int t : out[i].tokens) tk[i].codes.push_back(t - T.codec_offset);
            tk[i].codes_eos = out[i].eos;
        }
        r.codes = tk[0].codes, r.codes_eos = tk[0].codes_eos;
        // the companions' score and codes into their files (written whole: .part, then renamed)
        for (size_t i = 1; i < K2; ++i) {
            const Take &t = tk[i];
            if (write_take(t.out, t.seed, r.ar_weights, t.abc, t.abc_eos, t.codes, t.codes_eos, r)) ++r.companions;
        }
    }
    tk.clear();   // (the generation's caches given back)
    lap(r.t_codes);
    // this song's codes' prefix: the NAR's AR tokens begin with it
    std::vector<int32_t> sp, np;
    if (rp) sp = rp->prompt, np = rp->negative;
    else song_prompts(T, req.score == SongRequest::Score::Off, req.positive_head, req.negative_head, r.abc, sp, np);
    if (static_cast<int>(sp.size()) + 16 >= T.context) return fail("song: the lyrics and the score leave no room for music");
    const int frames = static_cast<int>(r.codes.size());
    if (frames < 1) return fail("song: the model wrote no music");
    if (dsp_prefix) pack.unload(ar.file());   // (the DSP computes every chunk's prefix: the CPU's AR is done)
    // the CPU's prefix from Q8 when the generation ran Q4: the NAR's keys and values as the default's (the NAR reads
    // them in every evaluation; Q8 is what the gates measured)
    FastAr ar8;
    FastAr *pre_ar = &ar;
    if (!dsp_prefix && ar.options().q4) {
        pack.unload(ar.file());
        ArOptions o8;
        o8.kv = KvType::I8;
        o8.map = true;
        if (!ar8.open(pack, o8, err)) return fail(err);
        ar8.cancelled = cb.cancelled;
        pre_ar = &ar8;
    }

    // ---- the NAR, per original chunk (an edit's: per window around what it draws again, the rest held)
    const int L = c.latent;
    std::vector<std::pair<int, int>> ranges;
    if (rp) {
        const size_t want = static_cast<size_t>(frames) * L;
        if (rp->noise.size() != want || rp->known.size() != want || rp->held.size() != static_cast<size_t>(frames))
            return fail("edit: the song's state is not this model's (its latents)");
        const int size = std::min((T.context - static_cast<int>(sp.size()) - 3) / 2, T.context);
        if (!windows_of(rp->held, size, rp->window, ranges, err)) return fail(err);
    } else {
        ranges = chunk_ranges(frames, static_cast<int>(sp.size()), T.context);
    }
    if (ranges.empty()) return fail("song: no acoustic context left");
    r.chunks = static_cast<int>(ranges.size());
    pg.chunks = r.chunks;
    const std::vector<float> noise = rp ? rp->noise : noise_of(req.seed, static_cast<size_t>(frames) * L);
    std::vector<float> latents = rp ? rp->known : std::vector<float>(noise.size());
    nar->cancelled = cb.cancelled;
    const int kd = c.kv_heads * c.head_dim;
    // the VAE opened (its weights read, the GPU's program built and its weights copied in) on a thread of its own
    // while the DSP computes the last chunk's velocities - the CPU and the GPU wait for the DSP then
    // (GEODE_YUE2_VAE_EARLY=0: after the NAR, as before). Joined on every way out of make_song.
    struct VaePrep {
        std::unique_ptr<Vae> vae;
        std::thread th;
        std::string err;
        bool ok = false;
        void join() {
            if (th.joinable()) th.join();
        }
        ~VaePrep() { join(); }
    } prep;
    const bool vae_early = dsp && [] {
        const char *e = std::getenv("GEODE_YUE2_VAE_EARLY");
        return !e || *e != '0';
    }();
    for (int ci = 0; ci < r.chunks; ++ci) {
        const auto [a, b] = ranges[static_cast<size_t>(ci)];
        pg.chunk = ci;
        std::vector<int32_t> ids(sp);
        for (int f = a; f < b; ++f) ids.push_back(r.codes[static_cast<size_t>(f)] + T.codec_offset);
        ids.push_back(T.music_end);
        const int A = static_cast<int>(ids.size()), nf = b - a;
        report(SongStage::Prefix, 0, 1);   // (no steps to count: the bar runs on its own)
        if (dsp_prefix) {
            if (!nar->begin_tokens(ids.data(), A, nf, err)) return fail(err);
            const auto &pt = nar->prefix_times();
            std::fprintf(stderr, "song: prefix %d tokens: AR weights %.2f s, DSP %.2f s, NAR weights %.2f s, all %.2f s\n", A,
                         pt.weights, pt.dsp, pt.nar_weights, pt.all);
        } else {   // the chunk's own sequence from position 0 through the CPU's AR into an fp16 branch
            ArBranch br;
            if (!pre_ar->branch_init(br, A, err, KvType::F16)) return fail(err);
            std::vector<FastAr::Seg> segs(1);
            segs[0].b = &br, segs[0].tokens = ids.data(), segs[0].n = A;
            if (!pre_ar->run(segs, pool, err)) return fail(err);
            if (cancelled()) return fail("cancelled");
            const auto row = [&](const AlignedBuffer &buf, int l, int p, float *dst) {
                const _Float16 *h = reinterpret_cast<const _Float16 *>(buf.get()) +
                                    (static_cast<size_t>(l) * static_cast<size_t>(br.cap) + static_cast<size_t>(p)) * static_cast<size_t>(kd);
                for (int i = 0; i < kd; ++i) dst[i] = static_cast<float>(h[i]);
            };
            if (!nar->begin(A, nf, [&](int l, float *k, float *v) {
                    for (int p = 0; p < A; ++p)
                        row(br.k, l, p, k + static_cast<size_t>(p) * kd), row(br.v, l, p, v + static_cast<size_t>(p) * kd);
                    return true;
                }, err))
                return fail(err);
        }
        lap(r.t_prefix);
        if (vae_early && ci + 1 == r.chunks && pack.map("vae.bin", err)) {   // (mapped here: the thread only reads the Pack)
            prep.vae = std::make_unique<Vae>();
            prep.th = std::thread([&prep, &pack] {
                try {
                    prep.ok = prep.vae->open(pack, prep.err, true);
                } catch (const std::exception &) {   // (not ok: opened again after the NAR, as without this)
                    prep.ok = false;
                    return;
                }
                std::string why;
                try {
                    if (prep.ok && !prep.vae->open_gpu(why)) std::fprintf(stderr, "song: the VAE on the CPU (%s)\n", why.c_str());
                } catch (const std::exception &e) {   // (the GPU's failure: the opened VAE decodes on the CPU)
                    std::fprintf(stderr, "song: the VAE on the CPU (%s)\n", e.what());
                }
            });
        }
        const int64_t evals = 2ll * steps;
        report(SongStage::Acoustic, static_cast<int64_t>(ci) * evals, evals * r.chunks);
        if (!nar->solve(noise.data() + static_cast<size_t>(a) * L, steps, latents.data() + static_cast<size_t>(a) * L, err,
                        [&](int s, int) { report(SongStage::Acoustic, static_cast<int64_t>(ci) * evals + 2ll * s, evals * r.chunks); },
                        rp ? rp->known.data() + static_cast<size_t>(a) * L : nullptr, rp ? rp->held.data() + a : nullptr))
            return fail(err);
        lap(r.t_acoustic);
    }
    prep.join();   // (before the Pack changes: the thread reads it)
    r.nar_weights = nar->weights_q8() ? "q8" : "f16";
    nar.reset();   // (the NAR's weights, banks and the DSP's session given back)
    pack.unload("nar_f16.bin");
    pack.unload("nar_q8.bin");
    pack.unload("nar_c8.bin");
    pack.unload("ar_q8.bin");
    pack.unload("ar_q4.bin");

    // ---- the VAE, the mastering, the file
    std::fprintf(stderr, "song: the NAR closed in %.2f s\n", now_seconds() - mark);
    std::vector<float> audio;
    {
        std::unique_ptr<Vae> own;
        if (prep.ok) {
            own = std::move(prep.vae);
            // its GPU tried again when the early open could not have it (the NAR's buffers held then): the decode's
            // arithmetic must not depend on what the NAR held at the time (open_gpu keeps a GPU it has)
            std::string why;
            if (!own->on_gpu() && !own->open_gpu(why))
                std::fprintf(stderr, "song: the VAE on the CPU (%s)\n", why.c_str());
        } else {
            own = std::make_unique<Vae>();
            if (!own->open(pack, err, true)) return fail(err);
            // the GPU's when there is one (OpenCL; the CPU's arithmetic: the Adreno 830 3.2x the CPU)
            std::string why;
            if (!own->open_gpu(why)) std::fprintf(stderr, "song: the VAE on the CPU (%s)\n", why.c_str());
        }
        Vae &vae = *own;
        std::fprintf(stderr, "song: the VAE open (and its GPU) after %.2f s\n", now_seconds() - mark);
        std::vector<std::pair<int, int>> wins{{0, frames}};
        if (rp) {
            wins.clear();
            rp->decode_ranges(vae.halo(), wins);
        }
        int64_t tiles = 0;
        for (size_t i = 0; i < wins.size(); ++i) {
            const auto [low, high] = wins[i];
            if (low < 0 || high > frames || low >= high || (i && low < wins[i - 1].second)) return fail("edit: nothing to decode");
            tiles += (high - low + 1023) / 1024;
        }
        if (wins.empty()) return fail("edit: nothing to decode");
        report(SongStage::Decode, 0, tiles);
        std::vector<std::vector<float>> decoded(wins.size());
        try {
            int64_t before = 0;
            for (size_t i = 0; i < wins.size(); ++i) {
                const auto [low, high] = wins[i];
                if (!vae.decode_tiled(latents.data() + static_cast<size_t>(low) * L, high - low, decoded[i], pool,
                                      [&](int d, int) { report(SongStage::Decode, before + d, tiles); }, cb.cancelled))
                    return fail("cancelled");
                before += (high - low + 1023) / 1024;
            }
            r.vae_backend = vae.on_gpu() ? vae.gpu_name() : "cpu";   // (a tile it could not take: the CPU's after)
        } catch (const std::exception &e) {
            return fail(std::string("VAE: ") + e.what());
        }
        if (rp) {   // (the old sound around them: read now, the NAR's and the VAE's memory given back first)
            std::vector<float> song;
            if (!rp->splice(decoded, wins, vae.halo(), song, err)) return fail(err);
            audio = std::move(song);
        } else {
            audio = std::move(decoded[0]);
        }
    }
    pack.unload("vae.bin");
    lap(r.t_decode);
    if (cancelled()) return fail("cancelled");   // (one that came with the last tile: no file then)
    r.samples = static_cast<int64_t>(audio.size() / 2);
    if (rp) {   // (splice gave the new sound the song's gain and limiter: no loudness of its own to report)
    } else if (req.master) {
        report(SongStage::Master, 0, 1);
        r.mastering = master(audio, 2, kRate, req.mastering);
        r.mastered = true;
        lap(r.t_master);
    }
    report(SongStage::Write, 0, 1);
    const std::string part = req.output_path + ".part";
    const bool wav = req.output_path.size() >= 4 &&
                     (req.output_path.compare(req.output_path.size() - 4, 4, ".wav") == 0 ||
                      req.output_path.compare(req.output_path.size() - 4, 4, ".WAV") == 0);
    const bool wrote = wav ? write_wav(part, audio.data(), r.samples, 2, kRate, req.bits)
                           : write_flac(part, audio.data(), r.samples, 2, kRate, req.bits);
    if (!wrote) {
        std::remove(part.c_str());
        return fail("song: could not write " + req.output_path);
    }
#if defined(_WIN32)
    std::remove(req.output_path.c_str());   // (Windows' rename does not replace; POSIX's does, atomically)
#endif
    if (std::rename(part.c_str(), req.output_path.c_str()) != 0) {
        std::remove(part.c_str());
        return fail("song: could not move the song to " + req.output_path);
    }
    lap(r.t_write);
    r.output_path = req.output_path;
    if (!req.state_out.empty()) {   // what an edit needs of the song (yue2_state.hpp); a song without it is still a song
        SongState st;
        st.seed = req.seed, st.ode_steps = steps, st.latent = L, st.rate = kRate;
        st.lora_sha256 = r.lora_sha256; st.lora_scale = r.lora_scale;
        st.ar_weights = r.ar_weights, st.nar_weights = r.nar_weights, st.nar_precision = r.nar_precision;
        st.mastered = rp ? rp->mastered : r.mastered;   // (an edit: the song's)
        st.master_gain_db = rp ? rp->gain_db : r.mastered ? r.mastering.gain_db : 0.0;
        st.prompt = sp, st.abc = r.abc, st.abc_eos = r.abc_eos, st.codes = r.codes, st.codes_eos = r.codes_eos;
        st.negative = np, st.cfg = rp ? rp->cfg : req.cfg;   // (the codes' negative branch's prompt)
        st.latents = std::move(latents), st.noise = noise;
        std::string why;
        r.state = write_state(req.state_out, st, why);
        if (!r.state) std::fprintf(stderr, "song: no state (%s)\n", why.c_str());
    }
    if (!req.view_out.empty()) {   // what the track editor shows (yue2_view.hpp)
        std::string why;
        r.view = write_view(req.view_out, view_of(audio.data(), r.samples, 2, kRate), why);
        if (!r.view) std::fprintf(stderr, "song: no view (%s)\n", why.c_str());
    }
    r.ok = true;
    r.t_all = now_seconds() - t0;
    return r;
}

void song_prompts(const Tokens &T, bool off, const std::vector<int32_t> &positive_head, const std::vector<int32_t> &negative_head,
                  const std::vector<int32_t> &abc, std::vector<int32_t> &sp, std::vector<int32_t> &np) {
    sp = positive_head, np = negative_head;
    if (off) {   // (off's negative without the score's delimiters, as the reference's)
        sp.insert(sp.end(), {T.abc_start, T.abc_end, T.music_start});
        np.push_back(T.music_start);
        return;
    }
    for (std::vector<int32_t> *p : {&sp, &np}) {
        p->push_back(T.abc_start);
        p->insert(p->end(), abc.begin(), abc.end());
        p->push_back(T.abc_end);
        p->push_back(T.music_start);
    }
}

namespace {
std::vector<int32_t> ints_of(const json::Value &v) {
    std::vector<int32_t> out;
    for (size_t i = 0; i < v.size(); ++i) out.push_back(static_cast<int32_t>(v.at(i).i64()));
    return out;
}
// the overrides on [s]; false (with why) for values the sampler cannot take - protocol.py Sampling's own checks (a top_k
// of 0 would read before its sorted rows)
bool sampling_of(const json::Value &v, Sampling &s, const char *what, std::string &err) {
    if (!v.is_object()) return true;
    s.temperature = static_cast<float>(v["temperature"].f64(s.temperature));
    s.top_p = static_cast<float>(v["top_p"].f64(s.top_p));
    s.top_k = static_cast<int>(v["top_k"].i64(s.top_k));
    s.repetition_penalty = static_cast<float>(v["repetition_penalty"].f64(s.repetition_penalty));
    s.penalty_window = static_cast<int>(v["penalty_window"].i64(s.penalty_window));
    s.min_tokens = static_cast<int>(v["min_tokens"].i64(s.min_tokens));
    s.max_tokens = static_cast<int>(v["max_tokens"].i64(s.max_tokens));
    const bool ok = std::isfinite(s.temperature) && s.temperature >= 0 && s.temperature <= 5 && std::isfinite(s.top_p) &&
                    s.top_p > 0 && s.top_p <= 1 && s.top_k >= 1 && std::isfinite(s.repetition_penalty) &&
                    s.repetition_penalty > 0 && s.penalty_window >= 1 && s.penalty_window <= 100 && s.min_tokens >= 0 &&
                    s.max_tokens >= 1 && s.min_tokens <= s.max_tokens;
    if (!ok) err = std::string("song request: ") + what + ": a value out of range";
    return ok;
}
}  // namespace

bool song_request_from_json(const std::string &text, SongRequest &q, std::string &err) {
    json::Value j;
    if (!json::parse(text, j, err)) return false;
    if (!j.is_object()) {
        err = "song request: not an object";
        return false;
    }
    q.lora = {};
    if (j.object.count("lora")) {
        const auto &l = j["lora"];
        if (!l.is_object() || !l["path"].is_string() || l["path"].str().empty() ||
            !l["cache"].is_string() || l["cache"].str().empty() ||
            (l.object.count("scale") && !l["scale"].is_number())) {
            err = "song request: lora needs nonempty path/cache strings and a finite numeric scale"; return false;
        }
        q.lora.path = l["path"].str(); q.lora.cache = l["cache"].str();
        q.lora.scale = static_cast<float>(l["scale"].f64(1));
        if (!std::isfinite(q.lora.scale) || q.lora.path.find('\0') != std::string::npos || q.lora.cache.find('\0') != std::string::npos) {
            err = "song request: invalid lora scale/path/cache"; return false;
        }
    }
    if (j["pack_dir"].is_string()) q.pack_dir = j["pack_dir"].str();
    if (j["npu_dir"].is_string()) q.npu_dir = j["npu_dir"].str();
    if (j["output"].is_string()) q.output_path = j["output"].str();
    q.positive_head = ints_of(j["positive_head"]);
    q.negative_head = ints_of(j["negative_head"]);
    const std::string score = j["score"].str("generate");
    q.score = score == "off" ? SongRequest::Score::Off : score == "given" ? SongRequest::Score::Given : SongRequest::Score::Generate;
    q.abc = ints_of(j["abc"]);
    q.abc_head = ints_of(j["abc_head"]);
    q.cfg = static_cast<float>(j["cfg"].f64(1.0));
    q.seed = static_cast<uint64_t>(j["seed"].i64(0));
    q.max_frames = static_cast<int>(j["max_frames"].i64(0));
    q.ode_steps = static_cast<int>(j["ode_steps"].i64(0));
    if (j["threads"].is_number()) q.threads = static_cast<int>(j["threads"].i64());
    q.ar_q4 = j["ar"].str("q8") == "q4";
    {
        const std::string pr = j["nar_precision"].str(j["nar_weights"].str("f16") == "q8" ? "fast" : "best");
        if (pr != "best" && pr != "balanced" && pr != "fast") {
            err = "nar_precision: best, balanced or fast, not '" + pr + "'";
            return false;
        }
        using P = SongRequest::NarPrecision;
        q.nar_precision = pr == "fast" ? P::Fast : pr == "balanced" ? P::Balanced : P::Best;
        q.nar_precision_given = j["nar_precision"].is_string() || j["nar_weights"].is_string();
    }
    for (const json::Value &cpn : j["companions"].array) {
        if (!cpn.is_object() || !cpn["ar_out"].is_string()) continue;
        SongRequest::Companion cp;
        cp.seed = static_cast<uint64_t>(cpn["seed"].i64(0));
        cp.ar_out = cpn["ar_out"].str();
        q.companions.push_back(cp);
    }
    if (j["codes"].kind == json::Value::Kind::Array) {
        q.codes_given = true;
        q.codes = ints_of(j["codes"]);
    }
    q.abc_eos = j["abc_eos"].kind != json::Value::Kind::Bool || j["abc_eos"].boolean;
    q.codes_eos = j["codes_eos"].kind != json::Value::Kind::Bool || j["codes_eos"].boolean;
    const std::string unit = j["nar"].str("auto");
    q.nar = unit == "dsp" ? NarUnit::Dsp : unit == "gpu" ? NarUnit::Gpu : unit == "cpu" ? NarUnit::Cpu : NarUnit::Auto;
    q.master = j["master"].kind != json::Value::Kind::Bool || j["master"].boolean;
    q.mastering.target_lufs = j["target_lufs"].f64(q.mastering.target_lufs);
    q.mastering.max_gain_db = j["max_gain_db"].f64(q.mastering.max_gain_db);
    q.bits = static_cast<int>(j["bits"].i64(24));
    q.score_only = j["score_only"].kind == json::Value::Kind::Bool && j["score_only"].boolean;
    if (j["state_out"].is_string()) q.state_out = j["state_out"].str();
    if (j["view_out"].is_string()) q.view_out = j["view_out"].str();
    if (j["edit"].is_object()) {
        const json::Value &e = j["edit"];
        const std::string op = e["op"].str();
        if (op != "cut" && op != "retake" && op != "go_on" && op != "move" && op != "insert") {
            err = "edit: op cut, retake, go_on, move or insert, not '" + op + "'";
            return false;
        }
        using O = SongRequest::Edit::Op;
        q.edit.op = op == "cut" ? O::Cut : op == "retake" ? O::Retake : op == "go_on" ? O::GoOn : op == "move" ? O::Move : O::Insert;
        const auto spans = [&](const json::Value &v, std::vector<std::pair<int, int>> &out) {
            for (size_t i = 0; i < v.size(); ++i)
                out.emplace_back(static_cast<int>(v.at(i).at(0).i64(-1)), static_cast<int>(v.at(i).at(1).i64(-1)));
        };
        spans(e["pieces"], q.edit.pieces);
        spans(e["sung"], q.edit.sung);
        q.edit.seeds = static_cast<int>(e["seeds"].i64(q.edit.seeds));
        q.edit.length = static_cast<int>(e["length"].i64(0));
        q.edit.state_in = e["state_in"].str(), q.edit.audio_in = e["audio_in"].str();
        q.edit.first = static_cast<int>(e["first"].i64(0)), q.edit.stop = static_cast<int>(e["stop"].i64(0));
        q.edit.margin = static_cast<int>(e["margin"].i64(q.edit.margin));
        q.edit.window = static_cast<int>(e["window"].i64(q.edit.window));
        q.edit.width = static_cast<int>(e["width"].i64(q.edit.width));
        q.edit.join_frames = static_cast<int>(e["join_frames"].i64(q.edit.join_frames));
        q.edit.redraw_edges = e["redraw_edges"].kind == json::Value::Kind::Bool && e["redraw_edges"].boolean;
        q.edit.fade_head = e["fade_head"].f64(q.edit.fade_head), q.edit.fade_tail = e["fade_tail"].f64(q.edit.fade_tail);
        q.edit.prompt = ints_of(e["prompt"]), q.edit.abc = ints_of(e["abc"]);
    }
    if (j["abc_sampling"].is_object() || j["semantic_sampling"].is_object()) {
        Pack pack;
        Config c;
        if (!pack.open(q.pack_dir, err) || !read_config(pack, c, err)) return false;
        q.default_sampling = false;
        q.abc_sampling = c.abc, q.semantic_sampling = c.semantic;
        if (!sampling_of(j["abc_sampling"], q.abc_sampling, "abc_sampling", err) ||
            !sampling_of(j["semantic_sampling"], q.semantic_sampling, "semantic_sampling", err))
            return false;
    }
    return true;
}

std::string song_result_json(const SongResult &r) {
    std::ostringstream o;
    const auto ints = [&](const std::vector<int32_t> &v) {
        o << '[';
        for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
        o << ']';
    };
    const auto num = [](double x) { return std::isfinite(x) ? x : -999.0; };
    o << "{\"ok\":" << (r.ok ? "true" : "false") << ",\"cancelled\":" << (r.cancelled ? "true" : "false")
      << ",\"error\":\"" << json_escape(r.error) << "\",\"output\":\"" << json_escape(r.output_path)
      << "\",\"nar\":\"" << json_escape(r.nar_backend) << "\",\"vae\":\"" << json_escape(r.vae_backend)
      << "\",\"ar\":\"" << json_escape(r.ar_weights) << "\",\"nar_weights\":\"" << json_escape(r.nar_weights) << "\",\"nar_precision\":\"" << json_escape(r.nar_precision)
      << "\",\"abc\":";
    ints(r.abc);
    o << ",\"abc_eos\":" << (r.abc_eos ? "true" : "false") << ",\"codes\":" << r.codes.size()
      << ",\"codes_eos\":" << (r.codes_eos ? "true" : "false") << ",\"chunks\":" << r.chunks << ",\"samples\":" << r.samples
      << ",\"seconds\":" << static_cast<double>(r.samples) / kRate;
    if (r.mastered)
        o << ",\"master\":{\"lufs_before\":" << num(r.mastering.lufs_before) << ",\"lufs_after\":" << num(r.mastering.lufs_after)
          << ",\"gain_db\":" << num(r.mastering.gain_db) << ",\"wanted_db\":" << num(r.mastering.wanted_db)
          << ",\"capped\":" << (r.mastering.capped ? "true" : "false") << ",\"peak_before_db\":" << num(r.mastering.peak_before_db)
          << ",\"peak_after_db\":" << num(r.mastering.peak_after_db) << "}";
    if (!r.lora_sha256.empty()) o << ",\"lora\":{\"sha256\":\"" << r.lora_sha256 << "\",\"scale\":" << lora_scale_text(r.lora_scale) << "}";
    o << ",\"times\":{\"load\":" << r.t_load << ",\"score\":" << r.t_score << ",\"codes\":" << r.t_codes
      << ",\"prefix\":" << r.t_prefix << ",\"acoustic\":" << r.t_acoustic << ",\"decode\":" << r.t_decode
      << ",\"master\":" << r.t_master << ",\"write\":" << r.t_write << ",\"all\":" << r.t_all
      << ",\"prompt_dsp\":" << r.t_prompt_dsp << ",\"join\":" << r.t_join << "},\"prompt_dsp_tokens\":" << r.prompt_dsp_tokens
      << ",\"companions\":" << r.companions << ",\"state\":" << (r.state ? "true" : "false")
      << ",\"view\":" << (r.view ? "true" : "false");
    if (r.edit_count >= 0) {
        o << ",\"" << (r.edit_kind.empty() ? "retake" : r.edit_kind) << "\":{\"count\":" << r.edit_count << ",\"ended\":"
          << (r.edit_ended ? "true" : "false");
        if (std::isfinite(r.edit_join)) o << ",\"join\":" << r.edit_join;
        if (r.edit_length > 0) o << ",\"length\":" << r.edit_length;
        if (r.edit_words) o << ",\"words\":true";
        o << "}";
    }
    if (r.edit_kind == "move") {
        o << ",\"move\":{\"sung\":[";
        for (size_t i = 0; i < r.edit_sung.size(); ++i)
            o << (i ? "," : "") << '[' << r.edit_sung[i][0] << ',' << r.edit_sung[i][1] << ',' << r.edit_sung[i][2] << ']';
        o << "]";
        o << "}";
    }
    o << "}";
    return o.str();
}

}  // namespace geode::yue2
