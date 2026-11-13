// yue2_edit.cpp - an edit of a sung song (yue2_edit.hpp)
#include "yue2_edit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include <limits>
#include <thread>

#include "platform.hpp"
#include "threads.hpp"
#include "yue2_ar.hpp"
#include "yue2_gen.hpp"
#include "yue2_master.hpp"
#include "yue2_pack.hpp"
#include "yue2_q4.hpp"
#include "yue2_state.hpp"
#include "yue2_view.hpp"

namespace geode::yue2 {

namespace {

constexpr int kHop = 1920;          // samples a frame at 48 kHz (25 frames a second)
constexpr int kFade = 4800;         // the crossfades at the decoded window's ends: 0.1 s (inpaint/ops.py FADE_SAMPLES)
constexpr double kPi = 3.14159265358979323846;

// the rise of a crossfade over n samples: a raised cosine (inpaint/core.py _ramp, edit_track.py _faded)
float ramp(int64_t i, int64_t n) {
    return static_cast<float>(0.5 - 0.5 * std::cos(kPi * (static_cast<double>(i) + 0.5) / static_cast<double>(n)));
}

// the edge a cut at the start (head) or the end left bare, faded over `seconds` (edit_track.py _faded: its count
// rounded half to even, its ramp in float32)
void fade_edge(std::vector<float> &audio, int channels, int rate, double seconds, bool head) {
    const int64_t frames = static_cast<int64_t>(audio.size()) / channels;
    const int64_t n = std::min<int64_t>(static_cast<int64_t>(std::nearbyint(seconds * rate)), frames);
    if (n < 2) return;
    for (int64_t i = 0; i < n; ++i) {
        const float g = 0.5f - 0.5f * std::cos(static_cast<float>(kPi) * (static_cast<float>(i) + 0.5f) / static_cast<float>(n));
        const int64_t at = head ? i : frames - 1 - i;
        for (int c = 0; c < channels; ++c) audio[static_cast<size_t>(at * channels + c)] *= g;
    }
}

template <typename T>   // v with [first, stop) replaced by `in`
std::vector<T> replaced(const std::vector<T> &v, size_t first, size_t stop, const std::vector<T> &in = {}) {
    std::vector<T> o(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(first));
    o.insert(o.end(), in.begin(), in.end());
    o.insert(o.end(), v.begin() + static_cast<std::ptrdiff_t>(stop), v.end());
    return o;
}

// A retake's new part (inpaint/core.py retakes, one take): sung on from the song's prompt and its codes before the
// stretch, with the song's guidance and its sampling (the request's), the repetition penalty remembering the codes
// before it (_remembering); `longest` tokens, the end barred inside the song (perform). Then the join (joins): for every
// count from `shortest` to `longest`, the mean log-probability among the codes of the old codes after the stretch
// following the first `count` new ones - the best count is the new part's length. The context and the new tokens are
// the generation's own cache, rewound and grown a token a count (the same keys and values: int8's per position).
//
// A song going on (core.py extended, one take): sung on from frame `first` under the new prompt - the new lyrics and
// the whole new score the app had the model write (the request's heads and abc) - the end barred until `EARLIER`
// frames before the length the new score asks for, stopped `LATER` after it; every code it sang is the new part.
constexpr int kEarlier = 50, kLater = 250;   // (ops.py EARLIER, LATER: 2 s before, 10 s after)

struct Sung {
    std::vector<int32_t> codes;   // the chosen count of new frames (codes 0 .. 32767)
    bool ended = false;           // the model ended the song itself (a retake reaching the end, a song going on)
    double join = std::nan("");   // the chosen count's score (NaN: the count not chosen by the join)
    double t_load = 0, t_codes = 0, t_join = 0;
};

// The prompts an edit sings and draws under, and the score the edited song keeps (track.py plan): the song's own; a
// cut's by bars, its lyrics and its score without them (ops.cut_words, notation.without); new words, the lyrics with
// lines rewritten and the song's score (ops.change_words); a song going on, its new lyrics and its whole new score.
// New lyrics come as the request's heads and the prompts are built from them as a song's are (song_prompts); an older
// app's cut may still send its prompt and score in the edit.
struct Prompts {
    std::vector<int32_t> prompt, negative, abc;
    bool abc_eos = true;
    bool words = false;   // built from new heads
};

bool resolve_prompts(const SongRequest &req, const SongState &st, Prompts &out, std::string &err) {
    using Op = SongRequest::Edit::Op;
    const SongRequest::Edit &e = req.edit;
    out.prompt = st.prompt, out.negative = st.negative, out.abc = st.abc, out.abc_eos = st.abc_eos;
    const bool pos = !req.positive_head.empty(), neg = !req.negative_head.empty();
    if (pos != neg) return (err = "edit: new lyrics come as both heads, the positive and the negative"), false;
    if (!pos) {
        if (e.op == Op::GoOn) return (err = "edit: a song going on needs its new score and its new lyrics' prompts"), false;
        if (e.op == Op::Move) return (err = "edit: a move needs its new score and its lyrics' prompts"), false;
        if (e.op == Op::Insert) return (err = "edit: a break needs its new score and its lyrics' prompts"), false;
        if (!e.prompt.empty()) {   // (a new prompt comes with the score it holds: a song with a score gets one)
            if (e.abc.empty() != st.abc.empty()) return (err = "edit: the new prompt's score does not match the song's (with or without one)"), false;
            if (e.op != Op::Cut) return (err = "edit: a prompt in the edit is a cut's"), false;
            // (it replaces the positive branch alone: with guidance the negative would keep the old score)
            if (st.cfg != 1.0f) return (err = "edit: a song sung with CFG takes its new prompt as the new lyrics' heads"), false;
            out.prompt = e.prompt, out.abc = e.abc;
        }
        return true;
    }
    if (!e.prompt.empty() || !e.abc.empty()) return (err = "edit: new lyrics' heads and a prompt in the edit both given"), false;
    const bool given = req.score == SongRequest::Score::Given;
    bool off = st.abc.empty();
    if (e.op == Op::Retake) {   // (new words: the tune stays)
        if (given || !req.abc.empty()) return (err = "edit: new words are sung to the song's own score"), false;
    } else if (e.op == Op::Cut || e.op == Op::Move || e.op == Op::Insert) {
        if (st.abc.empty() ? (req.score != SongRequest::Score::Off || !req.abc.empty()) : (!given || req.abc.empty()))
            return (err = "edit: the edit's score does not match the song's (with or without one)"), false;
        // (its end as the song's: bars taken out, moved or put in do not finish or cut short what the model wrote)
        out.abc = req.abc;
    } else {   // (a song going on: checked in edit_song)
        out.abc = req.abc, out.abc_eos = req.abc_eos, off = false;
    }
    Pack pack;
    Config c;
    if (!pack.open(req.pack_dir, err) || !read_config(pack, c, err)) return false;
    song_prompts(c.tok, off, req.positive_head, req.negative_head, out.abc, out.prompt, out.negative);
    out.words = true;
    return true;
}

bool sing_again(const SongRequest &asked, const SongState &st, const Prompts &pr, const SongCallbacks &cb, double t0, Sung &out,
                std::string &err) {
    SongRequest req = asked;
    if (!req.lora.path.empty() && req.ar_q4) {
        std::fprintf(stderr, "lora: Q4 ignored; edit AR uses merged CPU Q8 weights\n"); req.ar_q4 = false;
    }
    const SongRequest::Edit &e = req.edit;
    const auto report = [&](SongStage s, int64_t done, int64_t total) {
        if (!cb.progress) return;
        SongProgress pg;
        pg.stage = s, pg.done = done, pg.total = total, pg.elapsed = now_seconds() - t0;
        cb.progress(pg);
    };
    double mark = now_seconds();
    const auto lap = [&](double &into) {
        const double t = now_seconds();
        into += t - mark;
        mark = t;
    };
    report(SongStage::Load, 0, 1);
    const int threads = req.threads > 0 ? req.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    ThreadPool pool(threads);
    // the Q4 weights asked for and not made yet: made now, as a song's run makes them (make_song)
    if (req.ar_q4) {
        clean_ar_q4_parts(req.pack_dir);
        bool have = false;
        {
            Pack probe;
            if (!probe.open(req.pack_dir, err)) return false;
            have = FastAr::has_q4(probe);
        }
        if (!have && !make_ar_q4(req.pack_dir, pool, [&](int d, int n) { report(SongStage::Load, d, n); }, cb.cancelled, err)) {
            if (err != "cancelled") err = "edit: the Q4 weights: " + err;
            return false;
        }
    }
    Pack pack;
    Config c;
    req.lora.f16 = !req.score_only && !req.npu_dir.empty() && (req.nar == NarUnit::Auto || req.nar == NarUnit::Dsp);
    LoraWeights selected;
    if (!req.lora.path.empty() && !prepare_lora(req.pack_dir, req.lora, pool,
        [&](int d, int n) { report(SongStage::Load, d, n); }, cb.cancelled, selected, err)) return false;
    if (selected.sha256 != st.lora_sha256 || (!selected.sha256.empty() && selected.scale != st.lora_scale)) {
        err = "edit: adapter changed while preparing the song"; return false;
    }
    if (!pack.open_selected(req.pack_dir, selected.directory, err, selected.f16) || !read_config(pack, c, err)) return false;
    if (req.ar_q4 && !FastAr::has_q4(pack)) return (err = "edit: the Q4 weights were made but do not open (see the log)"), false;
    const Tokens &T = c.tok;
    const bool going = e.op == SongRequest::Edit::Op::GoOn;
    // (a break: `length` new frames at `first`, its stop the same frame - nothing taken out, the join judged from there)
    const bool inserting = e.op == SongRequest::Edit::Op::Insert;
    const int frames = static_cast<int>(st.codes.size()), length = going || inserting ? e.length : e.stop - e.first;
    // the prompts it sings under (resolve_prompts): the song's, or its new lyrics' (new words, a song going on)
    const std::vector<int32_t> &prompt = pr.prompt, &neg_prompt = pr.negative;
    const bool cfg = st.cfg != 1.0f;   // (the song's guidance, whatever it sings under)
    if (cfg && neg_prompt.empty()) return (err = "edit: the song was sung with CFG and its state has no negative prompt"), false;
    // (ops.py retake: the context holds the prompt, the frames before the stretch and the new part at its longest - the
    // negative branch's too, with guidance)
    const int widest = std::max(static_cast<int>(prompt.size()), cfg ? static_cast<int>(neg_prompt.size()) : 0);
    // (a break, ops.py insert: the old codes after it judge its join, so they are kept room for too - the join is
    // never judged on a code or two at the context's end)
    const int room = T.context - widest - e.first - length - (inserting ? e.join_frames : 0);
    if (room < 0) return (err = going ? "edit: the song and the part it goes on with do not both fit in the model's context"
                                      : "edit: the song's prompt and the part before the retake leave no room for it in the context"), false;
    const int width = going ? 0 : std::max(0, std::min({e.width, length / 3, room}));
    const int shortest = going ? std::max(1, length - kEarlier) : std::max(1, length - width);
    const int longest = going ? std::min(length + kLater, length + room) : length + width;
    const bool at_end = e.stop >= frames;
    Sampling s = req.default_sampling ? c.semantic : req.semantic_sampling;
    s.min_tokens = at_end ? shortest : longest;   // (inside the song the end is barred: `longest` tokens, always)
    s.max_tokens = longest;
    std::vector<int32_t> ids(st.codes.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (st.codes[i] < 0 || st.codes[i] >= T.codec_size) return (err = "edit: the song's state holds a code out of range"), false;
        ids[i] = st.codes[i] + T.codec_offset;
    }
    std::vector<int32_t> context(prompt), negative;
    context.insert(context.end(), ids.begin(), ids.begin() + e.first);
    if (cfg) negative = neg_prompt, negative.insert(negative.end(), ids.begin(), ids.begin() + e.first);
    // the old codes that judge the join: `join_frames` after the stretch, as many as the context holds after the new
    // part at its longest (the join's cache reaches context + longest + suffix - 1; only a song near the context's end
    // has fewer)
    const int fit = T.context - static_cast<int>(context.size()) - longest + 1;
    const std::vector<int32_t> suffix(ids.begin() + e.stop, ids.begin() + std::min({frames, e.stop + e.join_frames, e.stop + std::max(0, fit)}));
    const bool joining = !going && !at_end && !suffix.empty();   // (one count too: its score reported, as the reference's)
    const int pos_cap = static_cast<int>(context.size()) + longest + (joining ? static_cast<int>(suffix.size()) - 1 : 0);

    FastAr ar;
    {
        ArOptions o;
        o.kv = KvType::I8;
        o.map = true;
        o.q4 = req.ar_q4;   // (made above)
        if (!ar.open(pack, o, err)) return false;
        ar.cancelled = cb.cancelled;
    }
    Stream pos, neg;
    if (!ar.branch_init(pos.b, pos_cap, err)) return false;
    if (cfg && !ar.branch_init(neg.b, std::min<int>(T.context, static_cast<int>(negative.size()) + longest), err)) return false;
    lap(out.t_load);
    GenRequest g;
    g.abc = false, g.s = s, g.cfg = st.cfg, g.seed = req.seed, g.prefix = context, g.negative = negative;
    g.history.assign(ids.begin() + std::max(0, e.first - s.penalty_window), ids.begin() + e.first);
    g.cancelled = cb.cancelled;
    g.on_token = [&](int step, int) {
        if ((step & 15) == 0) report(SongStage::Codes, step + 1, longest);
    };
    GenResult gr;
    if (!generate(ar, pos, cfg ? &neg : nullptr, g, gr, pool, err)) return false;
    neg = Stream{};   // (its cache given back)
    lap(out.t_codes);
    const std::vector<int32_t> &tok = gr.tokens;
    int count = length;
    if (gr.eos || going) {   // (a song going on: what it sang, ended or stopped)
        count = static_cast<int>(tok.size()), out.ended = gr.eos;
        if (going)
            std::fprintf(stderr, "edit: went on from frame %d for %d frames (the score asks %d), %s\n", e.first, count, length,
                         gr.eos ? "ending by itself" : "stopped at the limit");
    } else if (joining) {
        if (static_cast<int>(tok.size()) != longest) return (err = "edit: the retake sang fewer tokens than it was asked for"), false;
        const int S = static_cast<int>(suffix.size()), H = c.hidden, V = T.codec_size;
        // the cache: the context and the new tokens but the last of the count - the generation's own, rewound to the
        // first `shortest` - 1 of them and grown a token a count (a causal cache's first positions are the same whatever
        // follows; each count then runs the suffix alone, not again the new tokens before it: (2w + 1) * 75 tokens
        // rather than (2w + 1) * (76 + w))
        const int base = static_cast<int>(context.size()) + shortest - 1;
        if (pos.b.len < base) return (err = "edit: the retake's cache is short of its join"), false;
        pos.b.len = base;
        std::vector<float> hidden(static_cast<size_t>(S) * H), lg(static_cast<size_t>(S) * V);
        double best = -std::numeric_limits<double>::infinity();
        for (int n = shortest; n <= longest; ++n) {
            if (cb.cancelled && cb.cancelled()) return (err = "cancelled"), false;
            // the count's last new token in: its hidden state predicts the suffix's first code
            std::vector<FastAr::Seg> one(1);
            one[0].b = &pos.b, one[0].tokens = &tok[static_cast<size_t>(n - 1)], one[0].n = 1;
            one[0].rows = {0}, one[0].hidden = hidden.data();
            if (!ar.run(one, pool, err)) return false;
            const int keep = pos.b.len;
            // the suffix but its last code after it, each predicting the next (rows 1 .. S - 1); then taken out again
            std::vector<FastAr::Seg> segs(1);
            segs[0].b = &pos.b, segs[0].tokens = suffix.data(), segs[0].n = S - 1;
            for (int j = 0; j + 1 < S; ++j) segs[0].rows.push_back(j);
            segs[0].hidden = hidden.data() + H;
            if (S > 1 && !ar.run(segs, pool, err)) return false;
            pos.b.len = keep;
            if (!ar.logits(hidden.data(), S, T.codec_offset, T.codec_offset + V, lg.data(), pool))
                return (err = "edit: the join's logits"), false;
            double sum = 0;
            for (int j = 0; j < S; ++j) {   // log-softmax among the codes, the old code's
                const float *row = lg.data() + static_cast<size_t>(j) * V;
                const float top = *std::max_element(row, row + V);
                double z = 0;
                for (int v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v]) - top);
                sum += static_cast<double>(row[suffix[static_cast<size_t>(j)] - T.codec_offset]) - top - std::log(z);
            }
            const double score = sum / S;
            if (score > best) best = score, count = n;   // (ties: the shorter, as max over the counts in order)
            report(SongStage::Join, n - shortest + 1, longest - shortest + 1);
        }
        out.join = best;
        std::fprintf(stderr, "edit: retake of %d frames sung as %d (of %d .. %d), join %.6f\n", length, count, shortest, longest, best);
        lap(out.t_join);
    }
    out.codes.resize(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const int code = tok[static_cast<size_t>(i)] - T.codec_offset;
        if (code < 0 || code >= T.codec_size) return (err = "edit: the model sang a token that is not a code"), false;
        out.codes[static_cast<size_t>(i)] = code;
    }
    return true;
}

}  // namespace

bool windows_of(const std::vector<uint8_t> &held, int size, int reach, std::vector<std::pair<int, int>> &out, std::string &err) {
    out.clear();
    const int frames = static_cast<int>(held.size());
    std::vector<std::pair<int, int>> loose, runs;
    for (int f = 0; f < frames; ++f) {
        if (held[static_cast<size_t>(f)]) continue;
        if (!loose.empty() && loose.back().second == f) loose.back().second = f + 1;
        else loose.emplace_back(f, f + 1);
    }
    for (const auto &[low, high] : loose) {
        if (high - low > size) {
            char b[200];
            std::snprintf(b, sizeof b, "edit: this edit draws %.0f s of new sound at once, and the acoustic stage holds at most "
                                       "%.0f s - edit a shorter stretch at a time", (high - low) * 0.04, size * 0.04);
            err = b;
            return false;
        }
        if (!runs.empty() && low - runs.back().second <= 2 * reach && high - runs.back().first <= size) runs.back().second = high;
        else runs.emplace_back(low, high);
    }
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto [low, high] = runs[i];
        const int floor = i ? runs[i - 1].second : 0;
        const int ceiling = i + 1 < runs.size() ? runs[i + 1].first : frames;
        int left = std::min(reach, low - floor), right = std::min(reach, ceiling - high);
        const int room = size - (high - low);
        if (left + right > room) {
            left = std::min(left, std::max(room / 2, room - right));
            right = std::min(right, room - left);
        }
        out.emplace_back(low - left, high + right);
    }
    return true;
}

SongResult edit_song(const SongRequest &asked, const SongCallbacks &cb) {
    SongRequest req = asked;   // (a song going on: its stretch's end set to the song's once the state is read)
    SongResult r;
    const double t0 = now_seconds();
    const auto fail = [&](const std::string &e) {
        r.ok = false;
        r.cancelled = e == "cancelled" || (cb.cancelled && cb.cancelled());
        r.error = r.cancelled ? "cancelled" : e;
        r.t_all = now_seconds() - t0;
        return r;
    };
    const SongRequest::Edit &e = req.edit;
    if (req.output_path.empty()) return fail("edit: no output path");
    using Op = SongRequest::Edit::Op;
    if (e.op != Op::Cut && e.op != Op::Retake && e.op != Op::GoOn && e.op != Op::Move && e.op != Op::Insert)
        return fail("edit: an edit this build does not make");
    const bool going = e.op == Op::GoOn, inserting = e.op == Op::Insert;
    const bool retake = e.op == Op::Retake || going || inserting;   // (retake: the model sings new frames)
    SongState st;
    std::string err;
    if (!read_state(e.state_in, st, err)) return fail(err);
    if (going) {   // (it replaces everything from `first` on; its score and its lyrics' prompts are the request's)
        req.edit.stop = static_cast<int>(st.codes.size());
        if (e.length < 1 || e.length > 24576) return fail("edit: a song going on sings at least a frame");
        if (req.abc.empty() || req.positive_head.empty() || req.negative_head.empty())
            return fail("edit: a song going on needs its new score and its new lyrics' prompts");
        if (st.abc.empty()) return fail("edit: a song without a score cannot go on");
    }
    if (inserting) {   // (a break, ops.py insert: nothing taken out; its score and its lyrics' prompts are the request's)
        req.edit.stop = req.edit.first;
        if (e.length < 1 || e.length > 24576) return fail("edit: a break plays at least a frame");
        if (req.abc.empty() || req.positive_head.empty() || req.negative_head.empty())
            return fail("edit: a break needs its new score and its lyrics' prompts");
        if (st.abc.empty()) return fail("edit: a song without a score cannot have a break put in");
    }
    AudioFile head;
    if (!read_audio(e.audio_in, head, err, true)) return fail(err);
    const int frames = static_cast<int>(st.codes.size());
    const int64_t samples = static_cast<int64_t>(frames) * kHop - 64;   // (the decoder's natural length)
    if (head.channels != 2 || head.rate != st.rate || head.frames != samples)
        return fail("edit: the song's audio is not its state's (" + std::to_string(head.frames) + " samples, the state's " +
                    std::to_string(samples) + ")");
    // A pure trim never opens a model/adapter. Every edit that redraws audio must retain its parent's identity.
    if (!(e.op == Op::Cut && (e.first == 0 || e.stop == frames) && !e.redraw_edges)) {
        std::string sha;
        if (!req.lora.path.empty() && !lora_sha256_remembered(req.lora.path, req.lora.cache, sha, err, cb.cancelled)) return fail(err);
        if (sha != st.lora_sha256 || (!sha.empty() && req.lora.scale != st.lora_scale))
            return fail(st.lora_sha256.empty() ? "edit: this song was made without an adapter" :
                "edit: this song was made with the adapter " + st.lora_sha256 + " at scale " + lora_scale_text(st.lora_scale));
    }
    req.expected_lora_sha256 = st.lora_sha256;
    r.lora_sha256 = st.lora_sha256; r.lora_scale = st.lora_scale;
    const bool moving = e.op == Op::Move;
    if (inserting && (e.first <= 0 || e.first >= frames)) return fail("edit: a break goes between two frames of the song");
    if (!moving && !inserting && (e.first < 0 || e.stop > frames || e.first >= e.stop)) return fail("edit: not a stretch of the song");
    const auto seconds_ok = [](double s) { return std::isfinite(s) && s >= 0 && s <= 60; };
    if (e.margin < 0 || e.margin > 1000 || e.window < 0 || e.window > 100000 || !seconds_ok(e.fade_head) || !seconds_ok(e.fade_tail) ||
        e.width < 0 || e.width > 1000 || e.join_frames < 0 || e.join_frames > 1000 || e.seeds < 1 || e.seeds > 4)
        return fail("edit: a margin, window, fade, width, join or seeds out of range");
    if (moving) {   // (its pieces every frame once; the stretches sung again in order, apart, within the song)
        if (e.pieces.empty() || e.pieces.size() > 64 || e.sung.size() > 64) return fail("edit: a move has no pieces, or too many");
        std::vector<std::pair<int, int>> sorted = e.pieces;
        std::sort(sorted.begin(), sorted.end());
        int at = 0;
        for (const auto &[f0, f1] : sorted) {
            if (f0 != at || f1 <= f0) return fail("edit: a move's pieces are not the song's frames, each once");
            at = f1;
        }
        if (at != frames) return fail("edit: a move's pieces are not the song's frames, each once");
        int last = 0;
        for (const auto &[a, b] : e.sung) {
            if (a < last || b <= a || b > frames) return fail("edit: a move's stretches to sing are not in order within the song");
            last = b;
        }
        if (st.abc.empty()) return fail("edit: a song without a score cannot be moved by bars");
    }
    const int removed = moving ? 0 : e.stop - e.first;
    if (!retake && !moving && frames - removed < 25) return fail("edit: a cut leaves less than a second of the song");
    // the prompts it sings and draws under, and the score the edited song keeps (a trim's too: its state and result)
    Prompts pr;
    if (!resolve_prompts(req, st, pr, err)) return fail(err);
    const bool head_cut = !retake && !moving && e.first == 0, tail_cut = !retake && !moving && e.stop == frames;
    const size_t L = static_cast<size_t>(st.latent);

    // the edited song's state, and what is drawn again: a retake's new part first (its own AR, closed before the
    // sound's), `count` new frames in place of the stretch - the codes, the latents and the noise with the stretch
    // replaced (every old frame keeps its own; the new frames get the first rows of the take's seed, their latents
    // drawn: zeros, not held); a move's pieces put in their new order, then each stretch before a join sung again so
    // (the best of `seeds` takes by its join), the later stretches moved by what the earlier ones gained or lost
    Sung sung;
    SongState ed = st;
    std::vector<std::pair<int, int>> drawn;          // the frames drawn again, [c0, c1), in the edited song
    std::vector<std::array<int, 3>> seams;           // a move's stretches sung again: [a, b) -> count frames
    double t_load = 0, t_codes = 0, t_join = 0;
    int count = 0, nf = frames;
    if (moving) {
        ed.codes.clear(), ed.latents.clear(), ed.noise.clear();
        for (const auto &[f0, f1] : e.pieces) {
            ed.codes.insert(ed.codes.end(), st.codes.begin() + f0, st.codes.begin() + f1);
            ed.latents.insert(ed.latents.end(), st.latents.begin() + f0 * L, st.latents.begin() + f1 * L);
            ed.noise.insert(ed.noise.end(), st.noise.begin() + f0 * L, st.noise.begin() + f1 * L);
        }
        ed.codes_eos = e.pieces.back().second == frames && st.codes_eos;   // (the song's end: still the model's?)
        int delta = 0;
        for (size_t k = 0; k < e.sung.size(); ++k) {
            SongRequest sub = req;
            sub.edit.op = Op::Retake;
            sub.edit.first = e.sung[k].first + delta, sub.edit.stop = e.sung[k].second + delta;
            Sung best;
            bool have = false;
            uint64_t used = 0;
            for (int s = 0; s < e.seeds; ++s) {
                sub.seed = req.seed + static_cast<uint64_t>(k) * static_cast<uint64_t>(e.seeds) + static_cast<uint64_t>(s);
                Sung one;
                if (!sing_again(sub, ed, pr, cb, t0, one, err)) return fail(err);
                t_load += one.t_load, t_codes += one.t_codes, t_join += one.t_join;
                if (!have || (std::isfinite(one.join) && (!std::isfinite(best.join) || one.join > best.join)))
                    best = std::move(one), have = true, used = sub.seed;
            }
            const int a = sub.edit.first, b = sub.edit.stop, n = static_cast<int>(best.codes.size());
            ed.codes = replaced(ed.codes, static_cast<size_t>(a), static_cast<size_t>(b), best.codes);
            ed.latents = replaced(ed.latents, a * L, b * L, std::vector<float>(n * L, 0.0f));
            ed.noise = replaced(ed.noise, a * L, b * L, noise_of(used, n * L));
            seams.push_back({a, b, n});
            std::fprintf(stderr, "edit: move: the join at %d sung as %d frames (of %d), seed %llu, join %.6f\n", b, n, b - a,
                         static_cast<unsigned long long>(used), best.join);
            delta += n - (b - a);
        }
        nf = static_cast<int>(ed.codes.size());
        for (const auto &[a, b, n] : seams) drawn.emplace_back(std::max(0, a - e.margin), std::min(nf, a + n + e.margin));
    } else {
        if (retake && !sing_again(req, st, pr, cb, t0, sung, err)) return fail(err);
        count = static_cast<int>(sung.codes.size()), nf = frames - removed + count;
        t_load = sung.t_load, t_codes = sung.t_codes, t_join = sung.t_join;
        ed.codes = replaced(st.codes, static_cast<size_t>(e.first), static_cast<size_t>(e.stop), sung.codes);
        ed.latents = replaced(st.latents, e.first * L, e.stop * L, std::vector<float>(count * L, 0.0f));
        ed.noise = replaced(st.noise, e.first * L, e.stop * L, noise_of(req.seed, count * L));
        if (retake && e.stop == frames) ed.codes_eos = sung.ended;   // (the song's end is the retake's)
        drawn.emplace_back(std::max(0, e.first - e.margin), std::min(nf, e.first + count + e.margin));
    }
    const double t_sung = now_seconds() - t0;
    ed.prompt = pr.prompt, ed.negative = pr.negative, ed.abc = pr.abc, ed.abc_eos = pr.abc_eos;
    r.ar_weights = st.ar_weights, r.nar_weights = st.nar_weights, r.nar_precision = st.nar_precision;
    r.abc = ed.abc, r.abc_eos = ed.abc_eos, r.codes = ed.codes, r.codes_eos = ed.codes_eos;

    if ((head_cut || tail_cut) && !e.redraw_edges) {
        // ---- a trim: the old samples, faded at the new edge - no model
        if (cb.progress) {
            SongProgress pg;
            pg.stage = SongStage::Write, pg.total = 1, pg.elapsed = now_seconds() - t0;
            cb.progress(pg);
        }
        AudioFile old;
        if (!read_audio(e.audio_in, old, err)) return fail(err);
        if (old.channels != 2 || old.rate != st.rate || old.frames != samples || old.audio.size() != static_cast<size_t>(samples) * 2)
            return fail("edit: the song's audio changed while it was edited");
        const int64_t keep = static_cast<int64_t>(nf) * kHop - 64;
        const int64_t from = head_cut ? samples - keep : 0;   // (a head cut: the old's last `keep` samples)
        std::vector<float> audio(old.audio.begin() + from * 2, old.audio.begin() + (from + keep) * 2);
        old.audio = {};
        fade_edge(audio, 2, st.rate, head_cut ? e.fade_head : e.fade_tail, head_cut);
        const std::string part = req.output_path + ".part";
        const bool wav = req.output_path.size() >= 4 && (req.output_path.compare(req.output_path.size() - 4, 4, ".wav") == 0 ||
                                                        req.output_path.compare(req.output_path.size() - 4, 4, ".WAV") == 0);
        if (!(wav ? write_wav(part, audio.data(), keep, 2, st.rate, req.bits) : write_flac(part, audio.data(), keep, 2, st.rate, req.bits))) {
            std::remove(part.c_str());
            return fail("edit: could not write " + req.output_path);
        }
#if defined(_WIN32)
        std::remove(req.output_path.c_str());
#endif
        if (std::rename(part.c_str(), req.output_path.c_str()) != 0) {
            std::remove(part.c_str());
            return fail("edit: could not move the song to " + req.output_path);
        }
        if (!req.state_out.empty()) {
            r.state = write_state(req.state_out, ed, err);
            if (!r.state) std::fprintf(stderr, "edit: no state (%s)\n", err.c_str());
        }
        if (!req.view_out.empty()) {
            r.view = write_view(req.view_out, view_of(audio.data(), keep, 2, st.rate), err);
            if (!r.view) std::fprintf(stderr, "edit: no view (%s)\n", err.c_str());
        }
        r.samples = keep;
        r.output_path = req.output_path;
        r.ok = true;
        r.t_all = r.t_write = now_seconds() - t0;
        return r;
    }

    // ---- the frames in `drawn` (the new frames and `margin` frames either side of them; a cut's: the seam where it was)
    // drawn again, the rest held (core.py _held)
    SongRequest::Repaint rp;
    rp.prompt = ed.prompt, rp.negative = ed.negative, rp.cfg = ed.cfg;
    rp.noise = ed.noise, rp.known = ed.latents;
    rp.held.assign(static_cast<size_t>(nf), 1);
    for (const auto &[c0, c1] : drawn)
        for (int f = c0; f < c1; ++f) rp.held[static_cast<size_t>(f)] = 0;
    rp.window = e.window;
    rp.mastered = st.mastered, rp.gain_db = st.master_gain_db;
    const int64_t out_samples = static_cast<int64_t>(nf) * kHop - 64;   // (the decoder's natural length)
    // where the new sound is laid in (inpaint/core.py _window): the decoder reaches `halo` frames either way of the
    // drawn frames, the fades lie just outside that, the window covers both fades; windows whose fades would meet are one
    struct Lay {
        bool left = false, right = false;
        int64_t la = 0, lb = 0, ra = 0, rb = 0;   // the fades [la, lb) and [ra, rb), in the edited song's samples
        int first = 0, end = 0;                   // the frames the window covers
    };
    const auto lays = [=](int halo) {
        std::vector<Lay> out;
        for (const auto &[c0, c1] : drawn) {
            Lay w;
            const int64_t left = static_cast<int64_t>(c0 - halo) * kHop, right = static_cast<int64_t>(c1 + halo) * kHop;
            w.left = left - kFade >= 0, w.la = left - kFade, w.lb = left;
            w.right = right + kFade <= out_samples, w.ra = right, w.rb = right + kFade;
            w.first = w.left ? static_cast<int>(w.la / kHop) : 0;
            w.end = w.right ? std::min<int>(nf, static_cast<int>((w.rb + kHop - 1) / kHop)) : nf;
            if (!out.empty()) {
                Lay &p = out.back();
                const int64_t p_end = p.right ? p.rb : out_samples, w_begin = w.left ? w.la : 0;
                if (w_begin <= p_end) {   // (one window from the first's left to this one's right)
                    p.right = w.right, p.ra = w.ra, p.rb = w.rb, p.end = w.end;
                    continue;
                }
            }
            out.push_back(w);
        }
        return out;
    };
    rp.decode_ranges = [=](int halo, std::vector<std::pair<int, int>> &windows) {
        windows.clear();
        for (const Lay &w : lays(halo)) {
            const int low = std::max(0, w.first - halo), high = std::min(nf, w.end + halo);
            if (!windows.empty() && low < windows.back().second) windows.back().second = high;   // (decoded once)
            else windows.emplace_back(low, high);
        }
    };
    // the old sound in the edited song's place, [out_samples x 2]: a cut's or a retake's before the stretch as it was
    // and after it moved by the change of length; a move's pieces in their new order, then each stretch sung again
    // replaced by its count of frames (the new frames' samples only ever inside a decoded window); zeros where only new
    // sound goes
    const std::string audio_in = e.audio_in;
    const int rate = st.rate;
    const int first = e.first, stop = e.stop;
    const std::vector<std::pair<int, int>> pieces = e.pieces;
    const auto old_in_place = [=](const std::vector<float> &old, std::vector<float> &base) {
        base.assign(static_cast<size_t>(out_samples) * 2, 0.0f);
        if (!moving) {
            const int64_t head_end = std::min<int64_t>(static_cast<int64_t>(first) * kHop, out_samples);
            std::copy(old.begin(), old.begin() + head_end * 2, base.begin());
            const int64_t tail_at = static_cast<int64_t>(first + count) * kHop, from = static_cast<int64_t>(stop) * kHop;
            for (int64_t i = tail_at, j = from; i < out_samples && j < samples; ++i, ++j)
                base[static_cast<size_t>(i * 2)] = old[static_cast<size_t>(j * 2)], base[static_cast<size_t>(i * 2 + 1)] = old[static_cast<size_t>(j * 2 + 1)];
            return;
        }
        std::vector<float> laid;   // the pieces in order, each its frames' samples (the song's last 64 short: zeros)
        laid.reserve(static_cast<size_t>(frames) * kHop * 2);
        for (const auto &[f0, f1] : pieces) {
            const int64_t a = static_cast<int64_t>(f0) * kHop, b = static_cast<int64_t>(f1) * kHop;
            laid.insert(laid.end(), old.begin() + a * 2, old.begin() + std::min(b, samples) * 2);
            if (b > samples) laid.insert(laid.end(), static_cast<size_t>(b - samples) * 2, 0.0f);
        }
        for (const auto &[a, b, n] : seams) {   // (each in the song as it was when it was sung)
            std::vector<float> next(laid.begin(), laid.begin() + static_cast<int64_t>(a) * kHop * 2);
            next.insert(next.end(), static_cast<size_t>(n) * kHop * 2, 0.0f);
            next.insert(next.end(), laid.begin() + static_cast<int64_t>(b) * kHop * 2, laid.end());
            laid = std::move(next);
        }
        std::copy(laid.begin(), laid.begin() + std::min<int64_t>(static_cast<int64_t>(laid.size()), out_samples * 2), base.begin());
    };
    // a move that changed the song's first or last piece: those edges faded as a trim's (head: not the song's own
    // start any more; tail: not its own ending)
    const bool fade_head = head_cut || (moving && e.pieces.front().first != 0);
    const bool fade_tail = tail_cut || (moving && e.pieces.back().second != frames);
    const double fh = e.fade_head, ft = e.fade_tail;
    rp.splice = [=](std::vector<std::vector<float>> &decoded, const std::vector<std::pair<int, int>> &windows, int halo,
                         std::vector<float> &song, std::string &why) {
        const std::vector<Lay> ws = lays(halo);
        AudioFile old;
        if (!read_audio(audio_in, old, why)) return false;
        if (old.channels != 2 || old.rate != rate || old.frames != samples || old.audio.size() != static_cast<size_t>(samples) * 2) {
            why = "edit: the song's audio changed while it was edited";
            return false;
        }
        old_in_place(old.audio, song);
        old.audio = {};
        for (const Lay &w : ws) {
            // the decoded window holding this one's frames
            size_t k = 0;
            while (k < windows.size() && !(windows[k].first <= w.first && w.end <= windows[k].second)) ++k;
            if (k == windows.size()) {
                why = "edit: no decoded window holds a seam";
                return false;
            }
            std::vector<float> &dec = decoded[k];
            const int low = windows[k].first;
            const int64_t offset = static_cast<int64_t>(w.first - low) * kHop, begin = static_cast<int64_t>(w.first) * kHop;
            const int64_t end = w.end == nf ? out_samples : static_cast<int64_t>(w.end) * kHop;
            if (offset + (end - begin) > static_cast<int64_t>(dec.size() / 2)) {
                why = "edit: the decoded window is short";
                return false;
            }
            const auto win = [&](int64_t at, int c) { return dec[static_cast<size_t>((offset + at - begin) * 2 + c)]; };
            const auto was = [&](int64_t at, int c) { return song[static_cast<size_t>(at * 2 + c)]; };
            const int64_t inner0 = w.left ? w.lb : 0, inner1 = w.right ? w.ra : out_samples;
            if (w.left)
                for (int64_t i = w.la; i < w.lb; ++i) {
                    const float g = ramp(i - w.la, kFade);
                    for (int c = 0; c < 2; ++c) song[static_cast<size_t>(i * 2 + c)] = was(i, c) * (1 - g) + win(i, c) * g;
                }
            for (int64_t i = inner0; i < inner1; ++i)
                for (int c = 0; c < 2; ++c) song[static_cast<size_t>(i * 2 + c)] = win(i, c);
            if (w.right)
                for (int64_t i = w.ra; i < w.rb; ++i) {
                    const float g = ramp(i - w.ra, kFade);
                    for (int c = 0; c < 2; ++c) song[static_cast<size_t>(i * 2 + c)] = win(i, c) * (1 - g) + was(i, c) * g;
                }
        }
        if (fade_head) fade_edge(song, 2, rate, fh, true);
        if (fade_tail) fade_edge(song, 2, rate, ft, false);
        return true;
    };
    // (the decoded windows get the song's gain and limiter before they go in, as the old sound had: its own loudness is
    // not the song's)
    const auto inner_splice = rp.splice;
    rp.splice = [inner_splice, &rp, rate](std::vector<std::vector<float>> &decoded, const std::vector<std::pair<int, int>> &windows,
                                          int halo, std::vector<float> &song, std::string &why) {
        if (rp.mastered)
            for (std::vector<float> &d : decoded) {
                MasterOptions mo;
                mo.fixed_gain_db = rp.gain_db;
                master(d, 2, rate, mo);
            }
        return inner_splice(decoded, windows, halo, song, why);
    };
    SongRequest q = req;
    q.edit = SongRequest::Edit{};
    q.repaint = &rp;
    q.codes_given = true;
    q.codes = ed.codes, q.codes_eos = ed.codes_eos;
    q.abc = ed.abc, q.abc_eos = ed.abc_eos;
    q.score = ed.abc.empty() ? SongRequest::Score::Off : SongRequest::Score::Given;
    q.companions.clear();
    q.score_only = false;
    q.master = false;
    q.seed = st.seed;
    if (q.ode_steps <= 0) q.ode_steps = st.ode_steps;
    if (!req.nar_precision_given) {   // (the seam drawn as the song was: its precision, unless the request names one)
        using P = SongRequest::NarPrecision;
        q.nar_precision = st.nar_precision == "fast" ? P::Fast : st.nar_precision == "balanced" ? P::Balanced : P::Best;
    }
    SongCallbacks cb2 = cb;   // (the sound's progress counted from this edit's start: a retake sang before it)
    if (cb.progress) cb2.progress = [&](const SongProgress &p) {
        SongProgress at = p;
        at.elapsed += t_sung;
        cb.progress(at);
    };
    SongResult out = make_song(q, cb2);
    out.t_load += t_load, out.t_codes += t_codes, out.t_join = t_join;
    if (moving) {
        out.edit_kind = "move";
        out.edit_sung = seams;
    } else if (retake) {
        out.edit_count = count, out.edit_join = sung.join, out.edit_ended = sung.ended;
        out.edit_kind = going ? "go_on" : inserting ? "insert" : "retake", out.edit_length = going || inserting ? e.length : 0;
        out.edit_words = !going && !inserting && pr.words;
    }
    out.t_all = now_seconds() - t0;
    return out;
}

}  // namespace geode::yue2
