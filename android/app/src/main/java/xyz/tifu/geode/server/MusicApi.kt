package xyz.tifu.geode.server

import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.music.LyricsLength
import xyz.tifu.geode.music.LyricsWriter
import xyz.tifu.geode.music.MusicAdapter
import xyz.tifu.geode.music.MusicJob
import xyz.tifu.geode.music.MusicModels
import xyz.tifu.geode.music.MusicPrefs
import xyz.tifu.geode.music.MusicPresets
import xyz.tifu.geode.music.MusicProgress
import xyz.tifu.geode.music.ScoreMidi
import xyz.tifu.geode.music.Song
import xyz.tifu.geode.music.SongStatus
import xyz.tifu.geode.music.SongStore
import xyz.tifu.geode.music.SoundPrecision
import xyz.tifu.geode.music.SungScore
import xyz.tifu.geode.music.midi.MidiImport
import xyz.tifu.geode.music.midi.Underlay
import xyz.tifu.geode.ui.music.MidiText
import xyz.tifu.geode.music.YuE2Prompt
import xyz.tifu.geode.tokenizer.Tokenizer
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.bool
import xyz.tifu.geode.util.num
import xyz.tifu.geode.util.str

/**
 * Songs over the local server, for the API (/v1/music/..., its key) and the web chat's Music page (/web/music/..., its
 * sign-in) alike. A song is made the Music screen's way - its folder in SongStore, its job from the music settings with
 * the request's own fields over them, MusicService in the :music process, one song at a time - so a song asked for here
 * is in the phone's library, and one made on the phone is listed here. The lyrics writer is the Music screen's too: a
 * chat model of the phone writes the style and the lyrics from an idea.
 */
class MusicApi(private val app: GeodeApp) {
    private val songs = SongStore(app)
    private val main = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    /** Main thread: MusicClient attached by us (MusicService's queue and progress followed). */
    private var attached = false
    private var idle: Job? = null
    /** Songs asked for whose start waits for the chat's model to be let go (an answer running ends first). */
    private val starting = MutableStateFlow<Set<String>>(emptySet())
    /** The song model's tokenizer (its folder, the tokenizer): a score's ids read back as text for its MIDI. */
    @Volatile private var tokenizer: Pair<String, Tokenizer>? = null

    /** [id] taken off [starting]; false: it was not there (started already, or never asked for here). */
    private fun unmark(id: String): Boolean {
        while (true) {
            val cur = starting.value
            if (id !in cur) return false
            if (starting.compareAndSet(cur, cur - id)) return true
        }
    }

    // ---- MusicService followed ----

    /** MusicService bound while songs wait or are made (its queue, the running song's progress); let go a minute after
     *  it has nothing to do. */
    private suspend fun follow() = withContext(Dispatchers.Main.immediate) {
        if (attached) return@withContext
        attached = true
        app.musicClient.attach()
        idle = main.launch {
            combine(app.musicClient.state, starting) { st, waiting ->
                st.connected && st.running.isEmpty() && st.queued.isEmpty() && waiting.isEmpty()
            }.distinctUntilChanged().collectLatest { done ->
                    if (done) {
                        delay(IDLE_MS)
                        letGo()
                    }
                }
        }
    }

    private fun letGo() {   // (main thread)
        if (!attached) return
        attached = false
        app.musicClient.detach()
        val j = idle
        idle = null
        j?.cancel()
    }

    /** The server stops: MusicService let go (a song it makes goes on; the phone's library has it). */
    fun close() {
        Handler(Looper.getMainLooper()).post { letGo() }
    }

    // ---- the routes: the API's /v1/music and the web chat's /web/music alike ----

    /**
     * [rest]: the path after /music; [base]: the songs' own path as the caller's clients reach them. GET "": the song
     * model, the defaults (the music settings) and what a request may hold; /songs: GET the library, POST a new song
     * (several with "takes": a list) made in the background - GET /songs/<id> follows it, GET /songs/<id>/content is
     * its audio (Range; ?download=1 as a file), GET /songs/<id>/score.mid its score as MIDI, POST /songs/<id>/sing a
     * plan's score sung, POST /songs/<id>/cancel, DELETE /songs/<id>; /midi: POST a MIDI file (base64), what its tune
     * becomes back (key, tempo, bars, its karaoke words, the notices) - a song sings it with "midi"; /lyrics: POST an idea, the
     * writer's style and lyrics back - whole, or with "stream" as server-sent events (delta, restart, done, error).
     * Anything wrong is an ApiError: each caller answers it its own way.
     */
    suspend fun handle(req: HttpRequest, r: HttpResponder, rest: String, base: String) {
        val parts = rest.removePrefix("/").split('/').filter { it.isNotEmpty() }
        val song = parts.size >= 2 && parts[0] == "songs"
        when {
            parts.isEmpty() && req.method == "GET" -> r.json(200, info())
            parts == listOf("songs") && req.method == "GET" -> r.json(200, linkedMapOf("object" to "list", "data" to list(base)))
            parts == listOf("songs") && req.method == "POST" -> {
                val made = create(req.json(), base)
                r.json(200, if (made.size == 1) made[0] else linkedMapOf("object" to "list", "data" to made))
            }
            song && parts.size == 2 && req.method == "GET" -> r.json(200, get(parts[1], base))
            song && parts.size == 2 && req.method == "DELETE" -> {
                delete(parts[1])
                r.json(200, linkedMapOf("id" to parts[1], "object" to "song.deleted", "deleted" to true))
            }
            song && parts.size == 3 && parts[2] == CONTENT && req.method == "GET" -> {
                val (f, type, name) = audio(parts[1])
                val download = req.query["download"].let { it != null && it != "0" }
                withContext(Dispatchers.IO) { r.file(f, type, req.header("range"), name.takeIf { download }) }
            }
            song && parts.size == 3 && parts[2] == SCORE && req.method == "GET" -> {
                val (bytes, name) = scoreMidi(parts[1])
                r.send(200, "audio/midi", bytes, mapOf("Cache-Control" to "private, max-age=3600",
                    "Content-Disposition" to "attachment; filename*=UTF-8''" + java.net.URLEncoder.encode(name, "UTF-8").replace("+", "%20")))
            }
            song && parts.size == 3 && parts[2] == "sing" && req.method == "POST" -> r.json(200, sing(parts[1], base))
            song && parts.size == 3 && parts[2] == "cancel" && req.method == "POST" -> r.json(200, cancel(parts[1], base))
            parts == listOf("midi") && req.method == "POST" -> r.json(200, midiInfo(req.json()))
            parts == listOf("lyrics") && req.method == "POST" -> {
                val b = req.json()
                val w = writerRequest(b)   // (checked before anything is sent: a 400 is still a 400)
                LocalServer.busy {
                    if (b.bool("stream") == true) lyricsStream(r) { onText -> write(w, onText) }
                    else r.json(200, write(w) {})
                }
            }
            else -> throw ApiError(404, "No such endpoint: ${req.method} ${req.path}")
        }
    }

    /** The writer's answer as it is written (delta; restart when it begins again - the first answer missed the
     *  layout), then done with the style and the lyrics, or error. */
    private suspend fun lyricsStream(r: HttpResponder, write: suspend ((String) -> Unit) -> Map<String, Any?>) = coroutineScope {
        r.events()
        var sent = ""
        // (a comment now and then: the model's loading and its reading of the prompt write nothing for a while; the
        // browser gone, the writer stops - not only at its next words)
        val request = coroutineContext[Job]
        val ticker = launch {
            while (true) {
                delay(15_000)
                try {
                    synchronized(r) { r.comment("writing") }
                } catch (e: java.io.IOException) {
                    request?.cancel()
                    break
                }
            }
        }
        try {
            val done = write { text ->
                // (the browser gone: the write throws, and the writer's answer ends with it)
                synchronized(r) {
                    val before = sent
                    sent = text
                    if (text.startsWith(before)) {
                        if (text.length > before.length) r.event("delta", Json.write(mapOf("text" to text.substring(before.length))))
                    } else {
                        r.event("restart", "{}")
                        if (text.isNotEmpty()) r.event("delta", Json.write(mapOf("text" to text)))
                    }
                }
            }
            synchronized(r) { r.event("done", Json.write(done)) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            runCatching { synchronized(r) { r.event("error", Json.write(mapOf("message" to (e.message ?: e.javaClass.simpleName)))) } }
        } finally {
            ticker.cancel()
        }
    }

    // ---- what the client may ask ----

    /** The song model, the settings a song is made with when the request names none, and what a request may hold. */
    private suspend fun info(): Map<String, Any?> {
        val p = app.music.current()
        val model = withContext(Dispatchers.IO) { MusicModels.selected(app, p) }
        return linkedMapOf(
            "model" to model?.let { linkedMapOf("id" to it.id, "title" to it.title) },
            "defaults" to linkedMapOf(
                "mode" to p.mode.key, "seconds" to p.maxSeconds, "steps" to p.odeSteps, "guidance" to p.guidance().toDouble(),
                "precision" to p.soundPrecision.key, "format" to p.format, "seed" to if (p.randomSeed) null else p.seed,
            ),
            "writer" to linkedMapOf(
                "language" to p.writerLanguage, "length" to p.writerLength.name.lowercase(), "stress" to p.writerStress,
                "breaks" to p.writerBreaks, "thinking" to p.writerThinking,
            ),
            "modes" to YuE2Prompt.Mode.entries.map { it.key },
            // the instrumental adapter downloaded: a request may ask for an instrumental ("instrumental": true)
            "instrumental" to withContext(Dispatchers.IO) { MusicAdapter.installed(app) },
            "precisions" to SoundPrecision.entries.map { it.key },
            "formats" to MusicPrefs.FORMATS.map { it.first }.distinct(),
            "steps" to MusicPrefs.STEPS,
            "max_seconds" to YuE2Prompt.MAX_SECONDS,
            "max_takes" to MusicPrefs.MAX_TAKES,
            "tags" to YuE2Prompt.TAGS,
            "breaks" to YuE2Prompt.BREAKS,
            "languages" to LyricsWriter.LANGUAGES.map { (name, tag) -> linkedMapOf("name" to name, "tag" to tag) },
            "lengths" to LyricsLength.entries.map { linkedMapOf("key" to it.name.lowercase(), "lines" to it.lines) },
            "presets" to MusicPresets.EXAMPLES.map { linkedMapOf("name" to it.name, "language" to it.language, "style" to it.style) },
            "genres" to MusicPresets.GENRES,
        )
    }

    // ---- the library ----

    /** Every song, the newest first; [base]: the songs' path (their audio is at <base>/<id>/content). */
    private suspend fun list(base: String): List<Map<String, Any?>> {
        val all = withContext(Dispatchers.IO) { songs.list() }
        if (all.any { it.status == SongStatus.WAITING }) follow()
        return all.map { json(it, base) }
    }

    private suspend fun get(id: String, base: String): Map<String, Any?> {
        val s = find(id)
        if (s.status == SongStatus.WAITING) follow()
        return json(s, base)
    }

    /** [id] cancelled: taken off the queue, or stopped at its next check if it runs. */
    private suspend fun cancel(id: String, base: String): Map<String, Any?> {
        val s = find(id)
        if (s.status != SongStatus.WAITING) return json(s, base)
        follow()
        // not started yet (it waits for the chat's model to go): it never will be
        val unstarted = unmark(id)
        if (!unstarted) app.musicClient.cancel(id)
        // a song no service runs (its process was stopped, or it never got to the queue): marked cancelled here
        val st = app.musicClient.state.value
        if (unstarted || (st.connected && !st.busy(id))) withContext(Dispatchers.IO) { songs.cancelled(id) }
        return json(find(id), base)
    }

    private suspend fun delete(id: String) {
        val s = find(id)
        unmark(id)
        if (s.status == SongStatus.WAITING) app.musicClient.cancel(id)
        // its waiting edits cancelled with it (they read its files)
        for (e in withContext(Dispatchers.IO) { s.waitingEdits(songs.list()) }) {
            val unstarted = unmark(e.id)
            if (!unstarted) app.musicClient.cancel(e.id)
            if (unstarted || !app.musicClient.state.value.busy(e.id)) withContext(Dispatchers.IO) { songs.cancelled(e.id) }
        }
        withContext(Dispatchers.IO) { songs.delete(id) }
    }

    /** [id]'s audio file, its media type and a file name for a download (the title's). */
    private suspend fun audio(id: String): Triple<java.io.File, String, String> {
        val s = find(id)
        val f = s.audio ?: throw ApiError(404, "The song has no audio (it is not made yet, or it failed).")
        val type = if (f.name.endsWith(".wav")) "audio/wav" else "audio/flac"
        return Triple(f, type, fileName(s, f.extension))
    }

    /** A download's file name: [s]'s title, letters, digits, spaces, - and _ only. */
    private fun fileName(s: Song, extension: String) =
        s.title.map { c -> if (c.isLetterOrDigit() || c == ' ' || c == '-' || c == '_') c else '_' }
            .joinToString("").trim().ifEmpty { "song" }.take(80) + "." + extension

    /** The selected song model's tokenizer (kept while that model stays selected); null: no song model. */
    private suspend fun tokenizer(): Tokenizer? = withContext(Dispatchers.IO) {
        val m = MusicModels.selected(app, app.music.current()) ?: return@withContext null
        tokenizer?.takeIf { it.first == m.dir.path }?.second
            ?: runCatching { Tokenizer.load(java.io.File(m.dir, Tokenizer.FILE)) }.getOrNull()?.also { tokenizer = m.dir.path to it }
    }

    /** [id]'s score as a MIDI file (the Music screen's "Share the score as MIDI": ScoreMidi) and its file name. */
    private suspend fun scoreMidi(id: String): Pair<ByteArray, String> {
        val s = find(id)
        val abc = s.abc.takeIf { s.status == SongStatus.DONE && it.isNotEmpty() }
            ?: throw ApiError(404, "The song has no score (made without one, or not made yet).")
        val tok = tokenizer() ?: throw ApiError(503, "No song model is installed: its tokenizer reads the score.")
        val bytes = withContext(Dispatchers.Default) {
            try {
                ScoreMidi.of(tok.decode(abc))
            } catch (e: Exception) {
                throw ApiError(500, "The score could not be written as MIDI: " + (e.message ?: e.javaClass.simpleName))
            }
        }
        return bytes to fileName(s, "mid")
    }

    private suspend fun find(id: String): Song {
        if (!ID.matches(id)) throw ApiError(404, "No such song.")
        return withContext(Dispatchers.IO) { songs.find(id) } ?: throw ApiError(404, "No such song.")
    }

    // ---- a tune from a MIDI file ----

    /** [b]'s "midi" (a MIDI file, base64) read in [mode]: the tune as a score, or an ApiError 400 why it cannot be sung. */
    private suspend fun midi(b: Map<String, Any?>, mode: YuE2Prompt.Mode): MidiImport.Result {
        val text = b.str("midi").orEmpty().substringAfter("base64,").trim()
        val bytes = try {
            java.util.Base64.getMimeDecoder().decode(text)
        } catch (e: IllegalArgumentException) {
            throw ApiError(400, "midi is a MIDI file in base64.")
        }
        if (bytes.isEmpty()) throw ApiError(400, "midi is a MIDI file in base64.")
        if (bytes.size > MidiText.MAX_BYTES) throw ApiError(413, app.getString(R.string.music_midi_failed, MidiText.error(app, MidiText.TooBig())))
        return withContext(Dispatchers.Default) {
            try {
                MidiImport.read(bytes, if (mode == YuE2Prompt.Mode.MELODY) "melody" else "full")
            } catch (e: Exception) {
                throw ApiError(400, app.getString(R.string.music_midi_failed, MidiText.error(app, e)))
            }
        }
    }

    /** What [b]'s MIDI file becomes ([b]: midi, mode, lyrics - the music settings' mode when it names none): its key,
     *  tempo, bars and length, its karaoke words, and the notices - the import's, and how the lyrics are laid along it. */
    private suspend fun midiInfo(b: Map<String, Any?>): Map<String, Any?> {
        var mode = b.str("mode")?.let { k -> YuE2Prompt.Mode.entries.firstOrNull { it.key == k } } ?: app.music.current().mode
        if (mode == YuE2Prompt.Mode.OFF) mode = YuE2Prompt.Mode.FULL
        val r = midi(b, mode)
        val lyrics = b.str("lyrics").orEmpty().trim().ifEmpty { r.lyrics }
        val laid = withContext(Dispatchers.Default) { runCatching { Underlay.lay(r.abc, lyrics) }.getOrNull() }
        return linkedMapOf(
            "mode" to mode.key, "key" to r.key, "bpm" to r.bpm, "meter" to r.meter, "bars" to r.bars, "seconds" to r.seconds,
            "lyrics" to r.lyrics, "max_seconds" to (laid?.maxSeconds ?: 0),
            "notices" to (r.notices + laid?.notices.orEmpty()).map { n ->
                linkedMapOf("code" to n.code.name.lowercase(), "args" to n.args, "message" to MidiText.notice(app, n))
            },
        )
    }

    // ---- making songs ----

    /**
     * New songs ([b]: style, lyrics, title, mode, seconds, seed, steps, guidance, precision, format, takes, melody_of,
     * score_only, midi: a MIDI file's tune as the score, base64; instrumental: with the instrumental adapter - mode
     * full, the lyrics its plan of bare tags or none; by default as the melody_of song was made) - several takes when asked, each its own seed; with score_only plans: their scores alone, written in a
     * fraction of a song's time, to be heard as MIDI and the one liked sung (POST /songs/<id>/sing). They start once the
     * chat's model is let go, when the music settings free its memory for a song (an answer it writes ends first; the
     * request does not wait for it).
     */
    private suspend fun create(b: Map<String, Any?>, base: String): List<Map<String, Any?>> {
        val style = b.str("style").orEmpty().trim()
        var lyrics = b.str("lyrics").orEmpty().replace("\r\n", "\n").trim()
        if (style.length > MAX_STYLE) throw ApiError(413, "style is longer than $MAX_STYLE characters.")
        // (a MIDI file's karaoke words count as the lyrics: checked once they are read)
        val fromMidi = !b.str("midi").isNullOrEmpty()
        if (lyrics.length > MAX_LYRICS) throw ApiError(413, "lyrics are longer than $MAX_LYRICS characters.")
        // the instrumental adapter (MusicAdapter): asked for, or as the song whose melody is sung was made
        val instrumental = b.bool("instrumental")
            ?: b.str("melody_of")?.takeIf { it.isNotEmpty() }?.let { MusicAdapter.usedBy(find(it).job) } ?: false
        // (an instrumental's plan may be empty: the model plans its sections itself)
        if (style.isEmpty() && lyrics.isEmpty() && !fromMidi && !instrumental)
            throw ApiError(400, "style or lyrics is required.")
        var p = app.music.current()
        b.str("mode")?.let { k ->
            p = p.copy(mode = YuE2Prompt.Mode.entries.firstOrNull { it.key == k } ?: throw ApiError(400, "mode is full, melody or off."))
        }
        if (instrumental) {
            if (!withContext(Dispatchers.IO) { MusicAdapter.installed(app) })
                throw ApiError(409, "The instrumental adapter is not downloaded: get it in Geode's Music settings.")
            p = p.copy(mode = YuE2Prompt.Mode.FULL)
        }
        b.num("seconds")?.let { v ->
            val s = v.toInt()
            if (s != 0 && s !in MIN_SECONDS..YuE2Prompt.MAX_SECONDS)
                throw ApiError(400, "seconds is 0 (from the lyrics) or $MIN_SECONDS to ${YuE2Prompt.MAX_SECONDS}.")
            p = p.copy(maxSeconds = s)
        }
        b.num("seed")?.let { v ->
            val s = v.toLong()
            if (s < 0 || s > Int.MAX_VALUE) throw ApiError(400, "seed is 0 to ${Int.MAX_VALUE}.")
            p = p.copy(randomSeed = false, seed = s)
        }
        b.num("steps")?.let { v ->
            val s = v.toInt()
            if (s !in MIN_STEPS..MAX_STEPS) throw ApiError(400, "steps is $MIN_STEPS to $MAX_STEPS (32: what the model was made for).")
            p = p.copy(odeSteps = s)
        }
        (b.num("guidance") ?: b.num("cfg"))?.let { v ->
            val g = v.toFloat()
            if (g != 0f && (g < 1f || g > MAX_GUIDANCE)) throw ApiError(400, "guidance is 0 (the mode's own) or 1 to $MAX_GUIDANCE.")
            p = p.copy(cfg = g)
        }
        b.str("precision")?.let { k ->
            p = p.copy(soundPrecision = SoundPrecision.entries.firstOrNull { it.key == k }
                ?: throw ApiError(400, "precision is best, balanced or fast."))
        }
        b.str("format")?.let { f ->
            if (MusicPrefs.FORMATS.none { it.first == f }) throw ApiError(400, "format is flac or wav.")
            p = p.copy(format = f)
        }
        val takes = b.num("takes")?.toInt() ?: 1
        if (takes !in 1..MusicPrefs.MAX_TAKES) throw ApiError(400, "takes is 1 to ${MusicPrefs.MAX_TAKES}.")
        // another song's score: the same melody sung and played anew
        val abc = b.str("melody_of")?.takeIf { it.isNotEmpty() }?.let { from ->
            if (p.mode == YuE2Prompt.Mode.OFF) throw ApiError(400, "melody_of needs a score: mode full or melody.")
            val src = find(from)
            src.abc.takeIf { src.status == SongStatus.DONE && it.isNotEmpty() }
                ?: throw ApiError(400, "That song has no score to sing again.")
        } ?: IntArray(0)
        val scoreOnly = b.bool("score_only") == true
        if (scoreOnly && p.mode == YuE2Prompt.Mode.OFF) throw ApiError(400, "score_only writes a score: mode full or melody.")
        if (scoreOnly && abc.isNotEmpty()) throw ApiError(400, "score_only writes a new score: not with melody_of.")
        if (fromMidi && (abc.isNotEmpty() || scoreOnly)) throw ApiError(400, "midi is the song's score: not with melody_of nor score_only.")
        if (fromMidi && p.mode == YuE2Prompt.Mode.OFF) throw ApiError(400, "midi needs a score: mode full or melody.")
        val model = withContext(Dispatchers.IO) { MusicModels.selected(app, p) }
            ?: throw ApiError(503, "No song model is installed: download it in Geode's Music settings (about 5.5 GB).")
        // a MIDI file's tune: the lyrics laid along it, its length when the request and the settings name none
        var sung = SungScore(abc)
        if (fromMidi) {
            val tune = midi(b, p.mode)
            if (lyrics.isEmpty() && !instrumental) lyrics = tune.lyrics.trim()   // (its karaoke words; an instrumental's are its plan)
            if (lyrics.length > MAX_LYRICS) throw ApiError(413, "lyrics are longer than $MAX_LYRICS characters.")
            if (style.isEmpty() && lyrics.isEmpty() && !instrumental)
                throw ApiError(400, "style or lyrics is required (the MIDI file has no karaoke words).")
            val tok = tokenizer() ?: throw ApiError(503, "No song model is installed: its tokenizer reads the score.")
            sung = withContext(Dispatchers.Default) { SungScore.of(tune.abc, lyrics, tok) }
            p = sung.prefs(p)
        }
        if (instrumental) lyrics = MusicAdapter.lyricsOf(lyrics)   // (none: [instrumental], the model chooses)
        val lora = if (instrumental) MusicAdapter.lora(app) else null
        val title = b.str("title").orEmpty().trim().take(MAX_TITLE).ifEmpty { YuE2Prompt.defaultTitle(style, lyrics) }
        // the takes made before a failure (the storage full) are made all the same, as the Music screen's are
        val (made, failed) = withContext(Dispatchers.IO) {
            songs.createTakes(title, style, lyrics, takes) { i ->
                MusicJob.create(model.dir, app.applicationInfo.nativeLibraryDir, p.mode, lyrics, p, sung.abc, i, scoreOnly, lora)
            }
        }
        if (made.isEmpty()) throw ApiError(500, "The song could not be saved: " + (failed?.message ?: "unknown"))
        start(made.map { it.id }, p.unloadChat)
        return made.map { json(it, base) }
    }

    /** [id], a plan, sung (the Music screen's "Sing this score"): its score given, its seed and its settings. */
    private suspend fun sing(id: String, base: String): Map<String, Any?> {
        val plan = find(id)
        if (!plan.plan || plan.status != SongStatus.DONE || plan.abc.isEmpty())
            throw ApiError(400, "Only a written plan (score_only) is sung: other songs are sung again with melody_of.")
        val p = app.music.current()
        val model = withContext(Dispatchers.IO) { MusicModels.selected(app, p) }
            ?: throw ApiError(503, "No song model is installed: download it in Geode's Music settings (about 5.5 GB).")
        val made = withContext(Dispatchers.IO) {
            runCatching {
                songs.create(plan.title, plan.style, plan.lyrics,
                    MusicJob.fromPlan(plan, model.dir, app.applicationInfo.nativeLibraryDir, p))
            }
        }.getOrElse { throw ApiError(500, "The song could not be saved: " + (it.message ?: it.javaClass.simpleName)) }
        start(listOf(made.id), p.unloadChat)
        return json(made, base)
    }

    /** [ids] queued for MusicService once the chat's model is let go ([unload]: the music settings free its memory). */
    private suspend fun start(ids: List<String>, unload: Boolean) {
        starting.update { it + ids }
        follow()
        main.launch {
            try {
                // the chat's model out of memory first (the song model needs it): once the agent's turn - its tools
                // between its requests as well - has ended, and the answer of anyone else using the model
                // (a turn begun while the unload waited at the gate: its end waited for, the model let go again; no
                // wait at all once every song of this request was cancelled or deleted meanwhile)
                while (unload && ids.any { it in starting.value }) {
                    app.agent.turn.first { it == null }
                    runCatching { app.gate.unload(app.engine) }
                    if (app.agent.turn.value == null) break
                }
            } finally {
                // (one cancelled or deleted meanwhile is not started)
                ids.filter { unmark(it) }.forEach { app.musicClient.make(it) }   // (queued: one runs at a time)
            }
        }
    }

    // ---- the lyrics writer ----

    /** What the writer is asked: the idea, the user's own instructions, the settings it writes with. */
    private class WriterRequest(val idea: String, val extra: String, val prefs: MusicPrefs)

    /** [b]'s idea checked ([b]: idea, extra, language, length, stress, breaks, thinking - the music settings' for what
     *  it does not name). */
    private suspend fun writerRequest(b: Map<String, Any?>): WriterRequest {
        val idea = b.str("idea").orEmpty().trim()
        if (idea.isEmpty()) throw ApiError(400, "idea is required.")
        if (idea.length > MAX_IDEA) throw ApiError(413, "idea is longer than $MAX_IDEA characters.")
        val extra = b.str("extra").orEmpty().trim()
        if (extra.length > MAX_IDEA) throw ApiError(413, "extra is longer than $MAX_IDEA characters.")
        var p = app.music.current()
        b.str("language")?.let { l ->
            val name = when {
                l.isEmpty() || l.equals("auto", true) -> ""
                else -> LyricsWriter.LANGUAGES.firstOrNull { it.first.equals(l, true) || it.second.equals(l, true) }?.first
                    ?: throw ApiError(400, "language is auto or one of " + LyricsWriter.LANGUAGES.joinToString { it.second } + ".")
            }
            p = p.copy(writerLanguage = name)
        }
        b.str("length")?.let { k ->
            p = p.copy(writerLength = LyricsLength.entries.firstOrNull { it.name.equals(k.replace('-', '_'), true) }
                ?: throw ApiError(400, "length is short, normal, long or very_long."))
        }
        b.bool("stress")?.let { v -> p = p.copy(writerStress = v) }
        b.bool("breaks")?.let { v -> p = p.copy(writerBreaks = v) }
        b.bool("thinking")?.let { v -> p = p.copy(writerThinking = v) }
        return WriterRequest(idea, extra, p)
    }

    /** The style and the lyrics [w] asks for, by the writer's chat model; [onText]: the answer as it is written
     *  (written again from the start when the first one missed the layout). */
    private suspend fun write(w: WriterRequest, onText: (String) -> Unit): Map<String, Any?> {
        val r = LyricsWriter(app).write(w.idea, w.extra, w.prefs, onText)
        return linkedMapOf("style" to r.style, "lyrics" to r.lyrics,
            "notes" to r.notes.map { n -> linkedMapOf("code" to n.code, "message" to app.getString(n.text)) })
    }

    // ---- a song as the client sees it ----

    /** [s] as JSON: queued, in_progress (its stage and the stage's share), completed (its audio), failed, cancelled or
     *  interrupted (its process stopped before it was done: cancel it, or ask for it again). */
    private fun json(s: Song, base: String): Map<String, Any?> {
        val st = app.musicClient.state.value
        val status = when (s.status) {
            SongStatus.DONE -> "completed"
            SongStatus.FAILED -> "failed"
            SongStatus.CANCELLED -> "cancelled"
            SongStatus.WAITING -> when {
                st.running == s.id -> "in_progress"
                s.id in starting.value || s.id in st.queued || !st.connected -> "queued"
                else -> "interrupted"
            }
        }
        val o = linkedMapOf<String, Any?>(
            "id" to s.id, "object" to "song", "status" to status, "title" to s.title, "style" to s.style,
            "lyrics" to s.lyrics, "created" to s.created / 1000, "mode" to s.mode.key, "seed" to s.seed,
            "max_seconds" to s.job.optInt("seconds"), "score_only" to s.plan, "instrumental" to MusicAdapter.usedBy(s.job),
        )
        if (status == "in_progress") st.progress?.takeIf { it.song == s.id }?.let { o["progress"] = progress(it) }
        if (status == "completed") {
            o["seconds"] = s.seconds
            o["made_in"] = s.makeSeconds
            s.audio?.let { f ->
                o["audio"] = "$base/${s.id}/$CONTENT"
                o["format"] = f.extension
                o["bytes"] = f.length()
            }
            if (s.abc.isNotEmpty()) o["score"] = "$base/${s.id}/$SCORE"
        }
        if (status == "failed") o["error"] = s.error
        return o
    }

    private fun progress(p: MusicProgress): Map<String, Any?> = linkedMapOf(
        "stage" to p.musicStage.name.lowercase(),
        "label" to MusicProgress.stageText(app, p),
        "part" to p.chunk + 1, "parts" to p.chunks,
        "done" to p.done, "total" to p.total, "fraction" to p.fraction.toDouble(),
        "elapsed" to p.elapsedAt(SystemClock.elapsedRealtime()),
    )

    companion object {
        /** SongStore's ids (its folder names: nothing else reaches a path). */
        private val ID = Regex("^[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?$")
        private const val IDLE_MS = 60_000L
        private const val MAX_STYLE = 1_000
        private const val MAX_LYRICS = 10_000
        private const val MAX_TITLE = 100
        private const val MAX_IDEA = 4_000
        private const val MIN_SECONDS = 10
        private const val MIN_STEPS = 4
        private const val MAX_STEPS = 64
        private const val MAX_GUIDANCE = 3f
        /** A song's audio under <base>/<id>/ (as OpenAI's videos have theirs). */
        private const val CONTENT = "content"
        private const val SCORE = "score.mid"
    }
}
