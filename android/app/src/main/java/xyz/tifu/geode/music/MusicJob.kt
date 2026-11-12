package xyz.tifu.geode.music

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import xyz.tifu.geode.tokenizer.Tokenizer

/**
 * A song's job: what the app asks MusicService for (song.json's "job", written in the app's process from the music
 * settings), and the engine's request MusicService makes of it in the :music process - the style and the lyrics
 * tokenized there with the pack's own tokenizer.
 */
object MusicJob {
    /** The job for a new song: the model's folder, the settings, the seed (drawn now when the settings want a new
     *  one; a fixed one plus [take] for the takes after the first), the score to sing ([abc]: another song's, for the
     *  same melody; empty: the model writes one). */
    fun create(pack: File, npuDir: String, mode: YuE2Prompt.Mode, lyrics: String, p: MusicPrefs,
               abc: IntArray = IntArray(0), take: Int = 0, scoreOnly: Boolean = false, lora: JSONObject? = null): JSONObject {
        // (a number short enough to read off a song and type back in the settings)
        // (the takes' seeds kept in 0 until Int.MAX_VALUE: no wrap to a negative one, and exact as the engine's JSON
        // double reads them - two takes never collapse onto one seed)
        val seed = if (p.randomSeed) java.security.SecureRandom().nextInt(Int.MAX_VALUE).toLong()
                   else if (take == 0) p.seed else Math.floorMod(p.seed + take, Int.MAX_VALUE.toLong())
        val seconds = if (p.maxSeconds > 0) p.maxSeconds else YuE2Prompt.autoSeconds(lyrics)
        val score = when {
            mode == YuE2Prompt.Mode.OFF -> "off"
            abc.isNotEmpty() -> "given"
            else -> "generate"
        }
        return JSONObject()
            .put("pack_dir", pack.path).put("npu_dir", npuDir)
            .put("mode", mode.key).put("score", score).put("abc", SongStore.ints(abc))
            .put("cfg", (if (p.cfg > 0f) p.cfg else YuE2Prompt.defaultCfg(mode)).toDouble())
            .put("seed", seed).put("seconds", seconds).put("max_frames", seconds * YuE2Prompt.FRAMES_PER_SECOND)
            .put("ode_steps", p.odeSteps).put("ar", if (p.arQ4 && lora == null) "q4" else "q8")   // (the adapter: Q8 only)
            .put("nar_precision", p.soundPrecision.key)
            .put("default_sampling", p.defaultSampling).put("abc_sampling", p.abc.json())
            .put("semantic_sampling", p.semantic.json())
            .put("nar", p.narUnit.key).put("threads", p.threads)
            .put("master", p.master).put("target_lufs", p.targetLufs.toDouble()).put("max_gain_db", p.maxGainDb.toDouble())
            .put("bits", p.bits).put("format", p.format)
            // a plan: the score alone (the model writes it: not with a given one, nor in the mode without one)
            .also { if (scoreOnly && score == "generate") it.put("score_only", true) }
            // the instrumental adapter (MusicAdapter): every run of this song and of its edits sings with it
            .also { if (lora != null) it.put("lora", lora) }
    }

    /**
     * [plan] sung: its score given (the tune heard), its seed, its settings - the plan's request as a song. The model's
     * folder, the app's libraries, the sound's unit and threads as they are now (as [again]).
     */
    fun fromPlan(plan: Song, pack: File, npuDir: String, p: MusicPrefs): JSONObject {
        val j = JSONObject(plan.job.toString())
        j.remove("output")
        j.remove("score_only")
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir).put("score", "given").put("abc", SongStore.ints(plan.abc))
            .put("nar", p.narUnit.key).put("threads", p.threads)
    }

    /**
     * [song]'s frames [first, stop) (25 a second) cut out - the engine's edit (yue2_edit.hpp): a new song in the library
     * from the song's state and audio, the song itself left as it is. At the start or the end a trim with a fade (no
     * model, a second); inside, the seam sung again around the cut. Its settings are the song's; the model's folder,
     * the app's libraries, the sound's unit and threads as they are now (as [again]).
     */
    fun cut(song: Song, first: Int, stop: Int, pack: File, npuDir: String, p: MusicPrefs, grid: BarGrid? = null,
            plan: TrackEdit.CutPlan? = null): JSONObject {
        val audio = song.audio ?: throw IllegalStateException("the song has no audio")
        val j = JSONObject(song.job.toString())
        j.remove("output")
        j.remove("score_only")
        val e = JSONObject().put("op", "cut").put("parent", song.id).put("first", first).put("stop", stop)
            .put("state_in", File(song.dir, STATE_FILE).path).put("audio_in", audio.path)
        if (plan != null) {
            // a cut by bars (track.py plan): the score and the lyrics without them, the seam drawn under those
            // ([editRequest]: the new lyrics' heads, the score's ids); its bars the song's without the cut ones
            e.put("score_cut", true).put("bars", JSONArray().put(plan.firstBar).put(plan.stopBar))
                .put("score_text", plan.score).put("grid", plan.grid.json())
        } else {
            // (an older kind of cut, or one by time: its bars the song's without the cut ones - its score still holds
            // them, so it is not fitted again)
            grid?.let { e.put("grid", it.cut(first, stop).json()) }
        }
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir).put("nar", p.narUnit.key).put("threads", p.threads)
            .put("edit", e)
    }

    /**
     * [song]'s frames [first, stop) sung again with the same words (the engine's retake, yue2_edit.hpp): a new song in
     * the library, sung on from the song's codes before the stretch with a new seed, up to 2 s longer or shorter so it
     * joins the song where the song goes on best, the frames around it drawn again. Its settings - the guidance, the
     * sampling, the steps - are the song's; the AR's weights, the model's folder, the app's libraries, the sound's unit
     * and threads as they are now. [grid]: the song's bars, kept with the edit (the new part's length is known when it
     * is done: [BarGrid.retake]). [words]: sung to the new song's own lyrics - the song's with lines rewritten
     * (Words.changeWords), the tune the song's score - not the song's ([editRequest]: the new lyrics' heads).
     */
    fun retake(song: Song, first: Int, stop: Int, pack: File, npuDir: String, p: MusicPrefs, grid: BarGrid? = null,
               words: Boolean = false): JSONObject {
        val audio = song.audio ?: throw IllegalStateException("the song has no audio")
        val j = JSONObject(song.job.toString())
        j.remove("output")
        j.remove("score_only")
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir).put("nar", p.narUnit.key).put("threads", p.threads)
            .put("ar", if (p.arQ4 && !j.has("lora")) "q4" else "q8")   // (an adapter's song: Q8 only)
            .put("seed", java.security.SecureRandom().nextInt(Int.MAX_VALUE).toLong())
            .put("edit", JSONObject().put("op", "retake").put("parent", song.id).put("first", first).put("stop", stop)
                .put("state_in", File(song.dir, STATE_FILE).path).put("audio_in", audio.path)
                .also { e -> grid?.let { e.put("grid", it.json()) }; if (words) e.put("words", true) })
    }

    /**
     * [song] going on with new lines (YuE2-ComfyUI's extended; the engine's go_on, yue2_edit.hpp): a new song in the
     * library whose lyrics are [plan]'s (the song's, then the new lines). Made in two steps by the service
     * ([goingOn]): the model writes the rest of the score from the bars before [plan]'s start for the new lyrics (up
     * to three tries, a seed apart, until the score reads: [goOnScore]), then sings on from there under it ([goOnSong])
     * for about as long as the written bars last at the song's tempo ([GoOn.extensionFrames]). Kept for those steps: the
     * song's score as it was ([score]), its clock and length, the plan. A new seed; the AR weights as set now.
     */
    fun goOn(song: Song, plan: GoOn.Plan, text: String, score: String, clock: GoOn.Clock, pack: File, npuDir: String,
             p: MusicPrefs): JSONObject {
        val audio = song.audio ?: throw IllegalStateException("the song has no audio")
        val frames = Math.rint(song.seconds * YuE2Prompt.FRAMES_PER_SECOND).toInt()
        val j = JSONObject(song.job.toString())
        j.remove("output")
        j.remove("score_only")
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir).put("nar", p.narUnit.key).put("threads", p.threads)
            .put("ar", if (p.arQ4 && !j.has("lora")) "q4" else "q8")   // (an adapter's song: Q8 only)
            .put("seed", java.security.SecureRandom().nextInt(Int.MAX_VALUE).toLong())
            .put("edit", JSONObject().put("op", "go_on").put("parent", song.id).put("first", plan.start)
                .put("bar", plan.bar).put("bars", plan.bars).put("head", plan.head).put("text", text).put("score", score)
                .put("clock", JSONObject().put("lines", JSONArray().apply { clock.lines.forEach { put(it) } }).put("speed", clock.speed))
                .put("frames", frames)
                .put("state_in", File(song.dir, STATE_FILE).path).put("audio_in", audio.path))
    }

    /**
     * [song]'s sections moved (YuE2-ComfyUI's move; the engine's move, yue2_edit.hpp): a new song whose lyrics and
     * score are [plan]'s, its frames the song's in [plan]'s order, the stretch before each join sung again under the
     * new prompt (two takes, the better join kept) and the joins drawn again. A new seed; the AR weights as set now.
     */
    fun move(song: Song, plan: TrackEdit.MovePlan, pack: File, npuDir: String, p: MusicPrefs): JSONObject {
        val audio = song.audio ?: throw IllegalStateException("the song has no audio")
        val j = JSONObject(song.job.toString())
        j.remove("output")
        j.remove("score_only")
        fun spans(v: List<Pair<Int, Int>>) = JSONArray().apply { v.forEach { (a, b) -> put(JSONArray().put(a).put(b)) } }
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir).put("nar", p.narUnit.key).put("threads", p.threads)
            .put("ar", if (p.arQ4 && !j.has("lora")) "q4" else "q8")   // (an adapter's song: Q8 only)
            .put("seed", java.security.SecureRandom().nextInt(Int.MAX_VALUE).toLong())
            .put("edit", JSONObject().put("op", "move").put("parent", song.id)
                .put("pieces", spans(plan.pieces)).put("sung", spans(plan.sung))
                .put("score_text", plan.score).put("grid", plan.grid.json())
                .put("state_in", File(song.dir, STATE_FILE).path).put("audio_in", audio.path))
    }

    /**
     * [song] with a break of [plan]'s bars put in before its bar (YuE2-ComfyUI's break; the engine's insert,
     * yue2_edit.hpp): a new song whose lyrics are [plan]'s (an empty [Interlude] where the break goes). Made in two
     * steps by the service ([breaking]): the model writes on from the score's head ([plan]'s: the bars before the break,
     * less the notes sung from where it goes in) - up to three tries, a seed apart, until [plan]'s bars of it can be put
     * into the song's score ([breakScore], ScoreEdit.interluded) - then the break is sung in at [plan]'s frame under the
     * new score ([breakSongRequest]) for about as long as its bars last at the song's tempo (TrackEdit.breakFrames).
     * Kept for those steps: the song's score ([score]) and bars ([grid]), the plan. A new seed; the AR weights as set now.
     */
    fun breakIn(song: Song, plan: TrackEdit.BreakPlan, score: String, grid: BarGrid, pack: File, npuDir: String,
                p: MusicPrefs): JSONObject {
        val audio = song.audio ?: throw IllegalStateException("the song has no audio")
        val j = JSONObject(song.job.toString())
        j.remove("output")
        j.remove("score_only")
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir).put("nar", p.narUnit.key).put("threads", p.threads)
            .put("ar", if (p.arQ4 && !j.has("lora")) "q4" else "q8")   // (an adapter's song: Q8 only)
            .put("seed", java.security.SecureRandom().nextInt(Int.MAX_VALUE).toLong())
            .put("edit", JSONObject().put("op", "insert").put("parent", song.id).put("first", plan.start)
                .put("bar", plan.bar).put("bars", plan.bars)
                .put("seam", JSONArray().put(plan.seam.numerator).put(plan.seam.denominator))
                .put("head", plan.head).put("score", score).put("grid", grid.json())
                .put("state_in", File(song.dir, STATE_FILE).path).put("audio_in", audio.path))
    }

    /** Whether [song] is a song with a break put in (two engine runs: [breakScore], then [breakSongRequest]). */
    fun breaking(song: Song): Boolean = song.job.optJSONObject("edit")?.optString("op") == "insert"

    /** A break's plan, the song's score as it was and its bars back from the song's job. */
    fun breakParts(song: Song): Triple<TrackEdit.BreakPlan, String, BarGrid> {
        val e = song.job.getJSONObject("edit")
        val seam = e.getJSONArray("seam")
        val plan = TrackEdit.BreakPlan(e.getInt("bar"), e.getInt("bars"), e.getInt("first"),
            AbcScore.Fraction(seam.getLong(0), seam.getLong(1)), e.getString("head"), song.lyrics, emptyList())
        val grid = BarGrid.of(e.optJSONObject("grid")) ?: throw IllegalStateException("the song's bars were not kept")
        return Triple(plan, e.getString("score"), grid)
    }

    /** A break's first step: the model writes on from the score's head for the new lyrics - try [attempt] (0..2) from
     *  the seed [attempt] x 7919 on, as a song going on's score is. The score alone: nothing written. */
    fun breakScore(song: Song, tok: Tokenizer, attempt: Int): String = goOnScore(song, tok, attempt)

    /** A break's second step: about [length] new frames sung in at its frame under the new lyrics and [abc] (the song's
     *  score with the break's bars in: the engine's insert), its seed the song's own whatever try wrote the score. */
    fun breakSongRequest(song: Song, tok: Tokenizer, abc: IntArray, length: Int): String {
        val (positive, negative) = heads(song, tok)
        val q = JSONObject(editRequest(song, song.job.getJSONObject("edit"), tok))
        q.getJSONObject("edit").put("length", length)
        q.put("positive_head", SongStore.ints(positive)).put("negative_head", SongStore.ints(negative))
            .put("score", "given").put("abc", SongStore.ints(abc))
        return q.toString()
    }

    /** Whether [song] is a song going on (two engine runs: [goOnScore], then [goOnSong]). */
    fun goingOn(song: Song): Boolean = song.job.optJSONObject("edit")?.optString("op") == "go_on"

    /** A song going on's plan, clock and old score back from its job. */
    fun goOnParts(song: Song): Triple<GoOn.Plan, GoOn.Clock, String> {
        val e = song.job.getJSONObject("edit")
        val lines = e.getJSONObject("clock").getJSONArray("lines")
        val clock = GoOn.Clock(List(lines.length()) { lines.getDouble(it) }, e.getJSONObject("clock").getDouble("speed"))
        val plan = GoOn.Plan(e.getInt("bar"), e.getInt("bars"), e.getInt("first"), song.lyrics, e.getString("head"), false, emptyList())
        return Triple(plan, clock, e.getString("score"))
    }

    private fun heads(song: Song, tok: Tokenizer): Pair<IntArray, IntArray> {
        val mode = YuE2Prompt.Mode.of(song.job.optString("mode"))
        return (intArrayOf(YuE2Prompt.EOD) + tok.encode(YuE2Prompt.text(mode, song.style, song.lyrics))) to
            (intArrayOf(YuE2Prompt.EOD) + tok.encode(mode.instruction))
    }

    /** A song going on's first step: the score written on from its head for the new lyrics - try [attempt] (0..2) from
     *  the seed [attempt] x 7919 on (core.py score_on, SCORE_SEED_STEP). The score alone: nothing written. */
    fun goOnScore(song: Song, tok: Tokenizer, attempt: Int): String {
        val j = song.job
        val e = j.getJSONObject("edit")
        val (positive, negative) = heads(song, tok)
        val q = JSONObject()
            .put("pack_dir", j.optString("pack_dir")).put("npu_dir", j.optString("npu_dir"))
            .put("output", File(song.dir, "score.part").path)   // (a score alone writes no file)
            .put("positive_head", SongStore.ints(positive)).put("negative_head", SongStore.ints(negative))
            .put("score", "generate").put("score_only", true)
            .put("abc_head", SongStore.ints(tok.encode(e.getString("head"))))
            .put("seed", j.optLong("seed") + attempt * GoOn.SCORE_SEED_STEP).put("ar", j.optString("ar", "q8"))
            .put("threads", j.optInt("threads"))
        j.optJSONObject("lora")?.let { q.put("lora", it) }
        if (!j.optBoolean("default_sampling", true)) q.put("abc_sampling", j.optJSONObject("abc_sampling") ?: JSONObject())
        return q.toString()
    }

    /** A song going on's second step: sung on from its start under the new lyrics and [abc] (the whole score written)
     *  for about [length] frames (the engine's go_on), its seed the song's own whatever try wrote the score. */
    fun goOnSong(song: Song, tok: Tokenizer, abc: IntArray, abcEos: Boolean, length: Int): String {
        val j = song.job
        val e = j.getJSONObject("edit")
        val (positive, negative) = heads(song, tok)
        val q = JSONObject(editRequest(song, e, tok))
        q.getJSONObject("edit").put("length", length)
        q.put("positive_head", SongStore.ints(positive)).put("negative_head", SongStore.ints(negative))
            .put("score", "given").put("abc", SongStore.ints(abc)).put("abc_eos", abcEos)
        return q.toString()
    }

    /** [song]'s request when it is an edit of another (its job's "edit"): the engine reads what it needs from that
     *  song's state and audio. A retake also sings: its seed, the AR's weights and the song's sampling (when it had
     *  its own). A cut by bars and new words come with the new song's lyrics as heads (the engine builds the prompts it
     *  sings and draws under from them), a cut by bars with its score without the bars too. */
    private fun editRequest(song: Song, e: JSONObject, tok: Tokenizer): String {
        val j = song.job
        val o = JSONObject()
            .put("pack_dir", j.optString("pack_dir")).put("npu_dir", j.optString("npu_dir"))
            .put("output", j.optString("output"))
            .put("state_out", File(song.dir, STATE_FILE).path).put("view_out", File(song.dir, VIEW_FILE).path)
            .put("ode_steps", j.optInt("ode_steps"))
            .put("nar_precision", j.optString("nar_precision", if (j.optString("nar_weights") == "q8") "fast" else "best"))
            .put("nar", j.optString("nar", "auto")).put("threads", j.optInt("threads")).put("bits", j.optInt("bits", 24))
            .put("edit", JSONObject().put("op", e.optString("op")).put("first", e.optInt("first")).put("stop", e.optInt("stop"))
                .put("state_in", e.optString("state_in")).put("audio_in", e.optString("audio_in")))
        j.optJSONObject("lora")?.let { o.put("lora", it) }   // (an adapter's song: edited with the same adapter)
        val op = e.optString("op")
        if (op == "retake" || op == "go_on" || op == "move" || op == "insert") {
            o.put("seed", j.optLong("seed")).put("ar", j.optString("ar", "q8"))
            if (!j.optBoolean("default_sampling", true)) j.optJSONObject("semantic_sampling")?.let { o.put("semantic_sampling", it) }
        }
        if (op == "move") o.getJSONObject("edit").put("pieces", e.getJSONArray("pieces")).put("sung", e.getJSONArray("sung"))
        val newScore = (op == "cut" && e.optBoolean("score_cut")) || op == "move"
        if (newScore || (op == "retake" && e.optBoolean("words"))) {
            val (positive, negative) = heads(song, tok)
            o.put("positive_head", SongStore.ints(positive)).put("negative_head", SongStore.ints(negative))
            if (newScore) o.put("score", "given").put("abc", SongStore.ints(tok.encode(e.getString("score_text"))))
        }
        return o.toString()
    }

    /**
     * A plan's result when a companion's run wrote its score already (the engine's score_only with companions: each
     * plan's score in its ar.json, no codes): the plan is done without a run of its own. Null: not written (or not
     * this plan's: another seed).
     */
    fun planOf(song: Song): JSONObject? = runCatching {
        if (!song.plan) return null
        val f = File(song.dir, AR_FILE)
        if (!f.isFile) return null
        val o = JSONObject(f.readText())
        if (o.optLong("seed", -1) != song.job.optLong("seed")) return null
        val abc = o.optJSONArray("abc")?.takeIf { it.length() > 0 } ?: return null
        JSONObject().put("ok", true).put("abc", abc).put("abc_eos", o.optBoolean("abc_eos", true))
            .put("ar", o.optString("ar")).put("seconds", 0.0).put("companion", true)
            // (the batch's writing time: the plans were written together)
            .put("times", JSONObject().put("all", o.optDouble("made", 0.0)))
    }.getOrNull()

    /**
     * [song] again, a new take of its own request: its settings as it was made with them (the mode, the length, the
     * guidance, the steps, the sampling, the precision, the mastering, the format), a new seed (the same one would
     * sing the same song), the model's folder and the app's libraries as they are now (a pack replaced, the app
     * updated), the sound's unit and threads as the settings ([p]) say now (this phone's, not the song's). [melody]:
     * on the score the song was sung from (the same tune), else the model writes a new one.
     */
    fun again(song: Song, pack: File, npuDir: String, p: MusicPrefs, melody: Boolean): JSONObject {
        val j = JSONObject(song.job.toString())
        j.remove("output")
        j.remove("edit")   // (an edited song again: a new take of its request, not the edit once more)
        val abc = if (melody) song.abc else IntArray(0)
        return j.put("pack_dir", pack.path).put("npu_dir", npuDir)
            .put("score", when {
                song.mode == YuE2Prompt.Mode.OFF -> "off"
                abc.isNotEmpty() -> "given"
                else -> "generate"
            })
            .put("abc", SongStore.ints(abc))
            .put("seed", java.security.SecureRandom().nextInt(Int.MAX_VALUE).toLong())
            .put("nar", p.narUnit.key).put("threads", p.threads)
            // a plan again: another plan; on its own score: sung (a score alone is one the model writes)
            .also { if (abc.isNotEmpty()) it.remove("score_only") }
    }

    /** A take's score and codes, made with an earlier take's (the engine's companions, yue2_song.hpp). */
    const val AR_FILE = "ar.json"

    /** A sung song's editable state (the engine's state_out, yue2_state.hpp: its codes, latents and noise, the
     *  mastering's gain): what a later edit of some bars sings from instead of the whole song again. */
    const val STATE_FILE = "song.state"

    /** A sung song's waveform and pitch classes every 10 ms (the engine's view_out, yue2_view.hpp): what the track
     *  editor draws and places the score's bars with. */
    const val VIEW_FILE = "song.view"

    /** [b] can be made along with [a] (its score and codes in [a]'s run): the same request but the seed - the takes of
     *  one draft. */
    fun companionOf(a: Song, b: Song): Boolean {
        if (a.id == b.id || a.style != b.style || a.lyrics != b.lyrics) return false
        val x = a.job
        val y = b.job
        if (x.optLong("seed") == y.optLong("seed")) return false
        // (the sound's precision: not the score's or the codes' - nor in a job from before it was a key)
        val keys = (x.keys().asSequence() + y.keys().asSequence()).toSet() - setOf("seed", "output", "nar_weights", "nar_precision")
        return keys.all { k -> x.opt(k)?.toString() == y.opt(k)?.toString() }
    }

    /** The score and codes a companion's run left for [song]; null: none (or not this one's: another seed or weights). */
    fun arOf(song: Song): JSONObject? = runCatching {
        val f = File(song.dir, AR_FILE)
        if (!f.isFile) return null
        JSONObject(f.readText()).takeIf {
            it.optLong("seed", -1) == song.job.optLong("seed") && it.optString("ar") == song.job.optString("ar", "q8") &&
                (it.optJSONArray("codes")?.length() ?: 0) > 0
        }
    }.getOrNull()

    /** The engine's request (yue2::song_request_from_json) for [song]: its job, the heads tokenized; [companions]: the
     *  takes whose score and codes this run makes too; a score and codes made so already given (the AR left out). */
    fun request(song: Song, tok: Tokenizer, companions: List<Song> = emptyList()): String {
        val j = song.job
        j.optJSONObject("edit")?.let { return editRequest(song, it, tok) }
        val mode = YuE2Prompt.Mode.of(j.optString("mode"))
        val positive = intArrayOf(YuE2Prompt.EOD) + tok.encode(YuE2Prompt.text(mode, song.style, song.lyrics))
        val negative = intArrayOf(YuE2Prompt.EOD) + tok.encode(mode.instruction)
        val q = JSONObject()
            .put("pack_dir", j.optString("pack_dir")).put("npu_dir", j.optString("npu_dir"))
            .put("output", j.optString("output"))
            .put("positive_head", SongStore.ints(positive)).put("negative_head", SongStore.ints(negative))
            .put("score", j.optString("score", "generate")).put("abc", j.optJSONArray("abc") ?: JSONArray())
            .put("cfg", j.optDouble("cfg", 1.0)).put("seed", j.optLong("seed"))
            .put("max_frames", j.optInt("max_frames")).put("ode_steps", j.optInt("ode_steps"))
            .put("ar", j.optString("ar", "q8"))
            .put("nar_precision", j.optString("nar_precision", if (j.optString("nar_weights") == "q8") "fast" else "best"))
            .put("nar", j.optString("nar", "auto")).put("threads", j.optInt("threads"))
            .put("score_only", j.optBoolean("score_only"))
            .also {
                if (!j.optBoolean("score_only"))
                    it.put("state_out", File(song.dir, STATE_FILE).path).put("view_out", File(song.dir, VIEW_FILE).path)
            }
            .put("master", j.optBoolean("master", true)).put("target_lufs", j.optDouble("target_lufs", -12.5))
            .put("max_gain_db", j.optDouble("max_gain_db", 4.0)).put("bits", j.optInt("bits", 24))
        j.optJSONObject("lora")?.let { q.put("lora", it) }
        if (!j.optBoolean("default_sampling", true)) {
            q.put("abc_sampling", j.optJSONObject("abc_sampling") ?: JSONObject())
            q.put("semantic_sampling", j.optJSONObject("semantic_sampling") ?: JSONObject())
        }
        val made = arOf(song)
        if (made != null) {
            if (j.optString("score") != "off") q.put("score", "given").put("abc", made.optJSONArray("abc") ?: JSONArray())
            q.put("abc_eos", made.optBoolean("abc_eos", true)).put("codes", made.getJSONArray("codes"))
                .put("codes_eos", made.optBoolean("codes_eos", true))
        } else if (companions.isNotEmpty()) {
            q.put("companions", JSONArray().apply {
                for (c in companions) put(JSONObject().put("seed", c.job.optLong("seed")).put("ar_out", File(c.dir, AR_FILE).path))
            })
        }
        return q.toString()
    }
}
