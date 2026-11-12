package xyz.tifu.geode.ui.music

import xyz.tifu.geode.music.ScoreMidi
import xyz.tifu.geode.music.SongShare
import xyz.tifu.geode.util.writeAtomic
import android.content.Intent
import android.media.MediaPlayer
import androidx.core.content.FileProvider
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.models.CatalogPack
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.models.PartialModel
import xyz.tifu.geode.models.TransferService
import xyz.tifu.geode.models.TransferSource
import xyz.tifu.geode.models.TransferState
import xyz.tifu.geode.models.Transfers
import xyz.tifu.geode.music.SoundPrecision
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.LyricsWriter
import xyz.tifu.geode.music.GoOn
import xyz.tifu.geode.music.MusicAdapter
import xyz.tifu.geode.music.MusicJob
import xyz.tifu.geode.music.MusicModels
import xyz.tifu.geode.music.MusicPrefs
import xyz.tifu.geode.music.MusicState
import xyz.tifu.geode.music.Song
import xyz.tifu.geode.music.SongStatus
import xyz.tifu.geode.music.SongStore
import xyz.tifu.geode.music.WriterNote
import xyz.tifu.geode.music.YuE2Prompt
import xyz.tifu.geode.music.midi.MidiImport
import xyz.tifu.geode.music.midi.Notice
import xyz.tifu.geode.music.SungScore
import xyz.tifu.geode.music.midi.Underlay
import xyz.tifu.geode.music.BarGrid
import xyz.tifu.geode.music.TrackEdit
import xyz.tifu.geode.music.Words
import xyz.tifu.geode.music.SongView
import xyz.tifu.geode.tokenizer.Tokenizer

/** The song being written on the Create tab (kept in a file: it outlives the app's process). */
data class SongDraft(
    val title: String = "",
    val style: String = "",
    val lyrics: String = "",
    val mode: YuE2Prompt.Mode = YuE2Prompt.Mode.FULL,
    /** Another song's score to sing (its melody), with that song's title; empty: the model writes one. */
    val abc: IntArray = IntArray(0),
    val abcFrom: String = "",
    /** That score moved this many semitones before it is sung (YuE2-ComfyUI's transpose: the style line cannot
     *  change the key, the score can). */
    val transpose: Int = 0,
    /** The score as the user edited it (null: as the song had it; "" is an edit too - one that cannot be sung). */
    val abcText: String? = null,
    /** The score is a MIDI file's tune (the file kept beside the draft: read again when the mode changes). */
    val midi: Boolean = false,
    /** Made with the instrumental adapter (MusicAdapter): [lyrics] is then the section plan, the sung words kept in
     *  [vocalLyrics] until it is turned off. */
    val instrumental: Boolean = false,
    val vocalLyrics: String = "",
) {
    fun json(): String = JSONObject().put("title", title).put("style", style).put("lyrics", lyrics).put("mode", mode.key)
        .put("abc", SongStore.ints(abc)).put("abc_from", abcFrom).put("transpose", transpose).put("abc_text", abcText ?: JSONObject.NULL)
        .put("midi", midi).put("instrumental", instrumental).put("vocal_lyrics", vocalLyrics).toString()

    companion object {
        fun parse(text: String?): SongDraft = runCatching {
            val o = JSONObject(text ?: return SongDraft())
            val a = o.optJSONArray("abc")
            SongDraft(o.optString("title"), o.optString("style"), o.optString("lyrics"), YuE2Prompt.Mode.of(o.optString("mode")),
                a?.let { IntArray(it.length()) { i -> it.getInt(i) } } ?: IntArray(0), o.optString("abc_from"),
                o.optInt("transpose").coerceIn(-AbcScore.LIMIT, AbcScore.LIMIT),
                if (o.isNull("abc_text")) null else o.optString("abc_text"), o.optBoolean("midi"),
                o.optBoolean("instrumental"), o.optString("vocal_lyrics"))
        }.getOrDefault(SongDraft())
    }
}

/** The lyrics writer's state: the idea, the answer as it comes, and what came of it. */
data class WriterState(
    val idea: String = "",
    val extra: String = "",
    val running: Boolean = false,
    val text: String = "",
    val error: String? = null,
    /** The last answer, parsed: shown until it is used or thrown away. */
    val style: String = "",
    val lyrics: String = "",
    val notes: List<WriterNote> = emptyList(),
)

/** The draft's borrowed score as the Create tab shows it: the song's own text, the text sung (the edit, else the
 *  song's), its key, the key it moves to, why it cannot. */
data class ScoreView(
    val original: String = "",
    val text: String = "",
    val key: String? = null,
    val moved: String? = null,
    val error: String? = null,
    /** The score names no sections (a MIDI file's tune): how the lyrics are laid along it when sung (Underlay). */
    val laid: List<Notice> = emptyList(),
)

/** The song model's packs on Hugging Face and the one being downloaded. */
data class MusicCatalog(
    val packs: List<CatalogPack>? = null,
    val loading: Boolean = false,
    val error: String? = null,
)

class MusicViewModel(private val app: GeodeApp) : ViewModel() {
    private val songs = SongStore(app)
    private val draftFile = File(app.filesDir, "music/draft.json")
    /** The draft's MIDI file (SongDraft.midi): read again in another mode. */
    private val draftMidi = File(app.filesDir, "music/draft.mid")

    val prefs: StateFlow<MusicPrefs> = app.music.flow.stateIn(viewModelScope, SharingStarted.Eagerly, MusicPrefs())
    val service: StateFlow<MusicState> = app.musicClient.state

    private val _draft = MutableStateFlow(SongDraft())
    val draft: StateFlow<SongDraft> = _draft.asStateFlow()

    /** Counts the draft's replacements from outside the lyrics field (read from its file, a song edited, the writer's
     *  lyrics used): the field takes the draft's text then and only then - while the user types, the field is the
     *  truth (a draft value a frame late would otherwise put back what was just typed over). */
    private val _replaced = MutableStateFlow(0)
    val replaced: StateFlow<Int> = _replaced.asStateFlow()

    private val _library = MutableStateFlow<List<Song>>(emptyList())
    val library: StateFlow<List<Song>> = _library.asStateFlow()

    /** The song models installed, the downloads that stopped part way. */
    private val _models = MutableStateFlow<List<InstalledModel>>(emptyList())
    val models: StateFlow<List<InstalledModel>> = _models.asStateFlow()
    private val _partials = MutableStateFlow<List<PartialModel>>(emptyList())
    val partials: StateFlow<List<PartialModel>> = _partials.asStateFlow()

    /** The chat models the writer can use. */
    private val _chatModels = MutableStateFlow<List<InstalledModel>>(emptyList())
    val chatModels: StateFlow<List<InstalledModel>> = _chatModels.asStateFlow()
    private val _chatSelected = MutableStateFlow("")
    val chatSelected: StateFlow<String> = _chatSelected.asStateFlow()

    private val _writer = MutableStateFlow(WriterState())
    val writer: StateFlow<WriterState> = _writer.asStateFlow()
    private var writerJob: Job? = null

    private val _catalog = MutableStateFlow(MusicCatalog())
    val catalog: StateFlow<MusicCatalog> = _catalog.asStateFlow()
    val transfer: StateFlow<TransferState?> = Transfers.state

    /** The prompt's tokens as the model will read it (null: no tokenizer yet - no model installed). */
    private val _tokens = MutableStateFlow<Int?>(null)
    val tokens: StateFlow<Int?> = _tokens.asStateFlow()
    private var tokenizer: Pair<String, Tokenizer>? = null

    /** The draft's borrowed score (null: none - the model writes one). */
    private val _score = MutableStateFlow<ScoreView?>(null)
    val score: StateFlow<ScoreView?> = _score.asStateFlow()

    /** The song playing ("" none) and whether it plays now (false: paused). */
    private val _playing = MutableStateFlow("" to false)
    val playing: StateFlow<Pair<String, Boolean>> = _playing.asStateFlow()
    private var player: MediaPlayer? = null

    init {
        app.musicClient.attach()
        viewModelScope.launch(Dispatchers.IO) {
            _draft.value = SongDraft.parse(runCatching { draftFile.readText() }.getOrNull())
            _replaced.update { it + 1 }
            refresh()
        }
        // the service idle: the weights a switch turned off while songs were made go now
        viewModelScope.launch {
            service.map { it.connected && it.running.isEmpty() && it.queued.isEmpty() }.distinctUntilChanged()
                .collect { idle -> if (idle) dropUnwanted() }
        }
        // a song's end: the library read again
        viewModelScope.launch { app.musicClient.finished.collect { withContext(Dispatchers.IO) { refreshSongs() } } }
        // a download's end: the installed models read again
        viewModelScope.launch {
            Transfers.state.collect { t -> if (t != null && !t.active && t.music) withContext(Dispatchers.IO) { refreshModels() } }
        }
        // the running song and the queue: the library's statuses follow them (not every progress message: the files
        // are read again only when what runs or waits changes)
        viewModelScope.launch {
            service.map { Triple(it.connected, it.running, it.queued) }.distinctUntilChanged().collect { (c, _, _) ->
                if (c) withContext(Dispatchers.IO) { refreshSongs() }
            }
        }
        countTokens()
        watchScore()
    }

    override fun onCleared() {
        app.musicClient.detach()
        player?.release()
        player = null
        super.onCleared()
    }

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            refreshSongs()
            refreshModels()
            val store = ModelStore(app)
            _chatModels.value = store.installed().filter { it.manifest.runnable }
            _chatSelected.value = app.settings.flow.first().selectedModel
        }
    }

    private fun refreshSongs() {
        _library.value = songs.list()
    }

    private fun refreshModels() {
        _models.value = MusicModels.installed(app)
        _partials.value = MusicModels.store(app).partials()
    }

    // ---- the draft ----

    fun editDraft(change: (SongDraft) -> SongDraft) {
        _draft.update(change)
        saveDraft()
    }

    private var draftSave: Job? = null

    private fun saveDraft() {
        draftSave?.cancel()
        draftSave = viewModelScope.launch(Dispatchers.IO) {
            delay(500)
            runCatching {
                draftFile.parentFile?.mkdirs()
                writeAtomic(draftFile, _draft.value.json())
            }
        }
    }

    /** A song of the library back into the draft (its words, style and mode; [melody]: its score as well; an
     *  instrumental made with the adapter: an instrumental draft again, its plan as the lyrics). */
    fun edit(song: Song, melody: Boolean = false) {
        val instrumental = MusicAdapter.usedBy(song.job)
        editDraft {
            SongDraft(song.title, song.style, if (instrumental && song.lyrics.trim() == MusicAdapter.BARE) "" else song.lyrics,
                if (instrumental) YuE2Prompt.Mode.FULL else song.mode,
                if (melody) song.abc else IntArray(0), if (melody) song.title else "").copy(instrumental = instrumental)
        }
        _midiNotes.value = emptyList()
        _replaced.update { it + 1 }
    }

    /** The borrowed score let go: the model writes a new one. */
    fun dropMelody() {
        editDraft { it.copy(abc = IntArray(0), abcFrom = "", transpose = 0, abcText = null, midi = false) }
        _midiNotes.value = emptyList()
    }

    // ---- a tune from a MIDI file ----

    private val _midiNotes = MutableStateFlow<List<Notice>>(emptyList())
    /** What the import of the draft's MIDI tune changed or guessed (a tempo averaged, a line moved, the chords guessed). */
    val midiNotes: StateFlow<List<Notice>> = _midiNotes.asStateFlow()

    /**
     * A MIDI file's tune as the draft's score (YuE2-ComfyUI's Load MIDI: MidiImport - the melody, in the full mode the
     * accompaniment's line and the chords as well, the tracks chosen as it chooses them), under the file's name; its
     * karaoke words as the lyrics when the draft has none. The score names no sections: the lyrics are laid along it
     * when sung (scoreIds: Underlay). The mode without a score: the full one. A file that cannot be sung: a toast why.
     */
    fun importMidi(uri: android.net.Uri) {
        viewModelScope.launch {
            val read = withContext(Dispatchers.IO) {
                runCatching {
                    val name = runCatching {
                        app.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
                            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
                    }.getOrNull().orEmpty().substringBeforeLast('.').trim().ifEmpty { "MIDI" }
                    val bytes = app.contentResolver.openInputStream(uri)?.use { MidiText.read(it) }
                        ?: throw java.io.IOException("unreadable")
                    val mode = _draft.value.mode.takeIf { it != YuE2Prompt.Mode.OFF } ?: YuE2Prompt.Mode.FULL
                    val (r, ids) = readMidi(bytes, mode)
                    writeAtomic(draftMidi, sync = false) { it.write(bytes) }
                    Triple(name, r, ids)
                }
            }
            read.onSuccess { (name, r, ids) ->
                editDraft { d ->
                    d.copy(mode = if (d.mode == YuE2Prompt.Mode.OFF) YuE2Prompt.Mode.FULL else d.mode, abc = ids, abcFrom = name,
                        transpose = 0, abcText = null, midi = true,
                        // (an instrumental's lyrics are its plan: the file's karaoke words not taken)
                        lyrics = if (!d.instrumental && d.lyrics.isBlank() && r.lyrics.isNotBlank()) r.lyrics.trim() else d.lyrics)
                }
                _midiNotes.value = r.notices
                _replaced.update { it + 1 }
            }.onFailure { e -> midiFailed(e) }
        }
    }

    /** [bytes] read as a score for [mode] (melody: the tune alone; full: its accompaniment and chords too) and its ids. */
    private fun readMidi(bytes: ByteArray, mode: YuE2Prompt.Mode): Pair<MidiImport.Result, IntArray> {
        val tok = tokenizerNow() ?: throw IllegalStateException(app.getString(xyz.tifu.geode.R.string.music_no_model))
        val r = MidiImport.read(bytes, if (mode == YuE2Prompt.Mode.MELODY) "melody" else "full")
        return r to tok.encode(r.abc)
    }

    private fun midiFailed(e: Throwable) {
        android.widget.Toast.makeText(app, app.getString(xyz.tifu.geode.R.string.music_midi_failed, MidiText.error(app, e)),
            android.widget.Toast.LENGTH_LONG).show()
    }

    /**
     * The draft's mode. The mode without a score lets a borrowed one go (as the web page does); a MIDI file's tune is
     * read again for the new mode - the melody alone, or with its accompaniment and chords - so the score sung is the
     * mode's own (its edit, made for the other mode, let go; the key moved stays).
     */
    /** The draft made with the instrumental adapter, or not: the lyrics field holds the section plan while it is (the
     *  words put aside and back), the mode the score-first one the adapter was trained for. */
    fun setInstrumental(on: Boolean) {
        val d = _draft.value
        if (on == d.instrumental) return
        if (on) {
            editDraft { it.copy(instrumental = true, vocalLyrics = it.lyrics, lyrics = "") }
            setMode(YuE2Prompt.Mode.FULL)
        } else editDraft { it.copy(instrumental = false, lyrics = it.vocalLyrics, vocalLyrics = "") }
        _replaced.update { it + 1 }
    }

    fun setMode(mode: YuE2Prompt.Mode) {
        val d = _draft.value
        if (mode == d.mode) return
        editDraft { it.copy(mode = mode) }
        when {
            d.abc.isEmpty() -> {}
            mode == YuE2Prompt.Mode.OFF -> dropMelody()
            d.midi -> viewModelScope.launch {
                withContext(Dispatchers.IO) { runCatching { readMidi(draftMidi.readBytes(), mode) } }
                    .onSuccess { (r, ids) ->
                        // (the draft changed meanwhile - another score taken, the mode moved on: that one stays)
                        if (_draft.value.midi && _draft.value.mode == mode) {
                            editDraft { it.copy(abc = ids, abcText = null) }
                            _midiNotes.value = r.notices
                        }
                    }.onFailure { e -> midiFailed(e) }
            }
        }
    }

    /** The borrowed score's key moved [delta] semitones more (within an octave either way). */
    fun transposeBy(delta: Int) =
        editDraft { it.copy(transpose = (it.transpose + delta).coerceIn(-AbcScore.LIMIT, AbcScore.LIMIT)) }

    /** The borrowed score as the user edits it; the song's own text again: null. */
    fun editScore(text: String?) = editDraft { it.copy(abcText = text) }

    /** The tokenizer of the selected song model (null: none installed). Off the main thread: it reads a file; one
     *  load at a time (the score's watcher, the token count and make() ask for it at once). */
    @Synchronized
    private fun tokenizerNow(): Tokenizer? {
        val m = MusicModels.selected(app, prefs.value) ?: return null
        return tokenizer?.takeIf { it.first == m.dir.path }?.second
            ?: runCatching { Tokenizer.load(File(m.dir, Tokenizer.FILE)) }.getOrNull()?.also { tokenizer = m.dir.path to it }
    }

    /** The score the draft will sing as text: the edit, else the song's own ids read back. */
    private fun scoreText(d: SongDraft, tok: Tokenizer): String = d.abcText ?: tok.decode(d.abc)

    /** The draft's score as the engine takes it (empty: the model writes one, or the mode has none): the ids as they
     *  are when nothing changed, else the text (edited, moved) tokenized again; a score that names no sections (a MIDI
     *  file's tune) with the lyrics laid along it (SungScore). Throws AbcScore.AbcError with the reason when the text
     *  cannot be read. */
    private fun scoreIds(d: SongDraft): SungScore {
        if (d.mode == YuE2Prompt.Mode.OFF || d.abc.isEmpty()) return SungScore(IntArray(0))
        val tok = tokenizerNow() ?: throw AbcScore.AbcError("no tokenizer")
        val text = scoreText(d, tok)
        val sung = if (d.transpose != 0) AbcScore.move(text, d.transpose).text else text.also { if (d.abcText != null) AbcScore.parse(it) }
        return SungScore.of(sung, d.lyrics, tok, d.abc.takeIf { d.transpose == 0 && d.abcText == null })
    }

    @OptIn(FlowPreview::class)
    private fun watchScore() {
        viewModelScope.launch(Dispatchers.IO) {
            // the draft's score, and the song models (a model installed later: its tokenizer reads the score then)
            // (the lyrics as well: a score that names no sections has them laid along it)
            val drafts = _draft.distinctUntilChanged { a, b ->
                a.abc.contentEquals(b.abc) && a.transpose == b.transpose && a.abcText == b.abcText && a.lyrics == b.lyrics
            }
            combine(drafts, _models.map { list -> list.map { it.dir.path } }.distinctUntilChanged()) { d, _ -> d }
                .debounce(300).collectLatest { d ->
                    val abc = d.abc
                    val t = d.transpose
                    val edited = d.abcText
                    if (abc.isEmpty()) return@collectLatest run { _score.value = null }
                    val tok = tokenizerNow() ?: return@collectLatest run { _score.value = null }
                    val original = tok.decode(abc)
                    val text = edited ?: original
                    val key = AbcScore.key(text)
                    _score.value = try {
                        val moved = if (t != 0) AbcScore.move(text, t) else null
                        if (moved == null && edited != null) AbcScore.parse(text)   // (an edit: read it)
                        val laid = runCatching { Underlay.lay(moved?.text ?: text, d.lyrics)?.notices }.getOrNull().orEmpty()
                        ScoreView(original, text, key, moved?.after, laid = laid)
                    } catch (e: IllegalArgumentException) {
                        ScoreView(original, text, key, error = e.message ?: e.javaClass.simpleName)
                    }
                }
        }
    }

    @OptIn(FlowPreview::class)
    private fun countTokens() {
        viewModelScope.launch(Dispatchers.IO) {
            _draft.debounce(400).collectLatest { d ->
                val tok = tokenizerNow() ?: return@collectLatest run { _tokens.value = null }
                _tokens.value = 1 + tok.encode(YuE2Prompt.text(d.mode, d.style, d.lyrics)).size
            }
        }
    }

    // ---- making songs ----

    /** The draft made: [takes] songs, or with [scoresFirst] that many plans (their scores alone, to hear first), queued
     *  in MusicService. False: no song model installed. */
    fun make(takes: Int = 1, scoresFirst: Boolean = false): Boolean {
        val n = takes.coerceIn(1, MusicPrefs.MAX_TAKES)
        val d = _draft.value
        val p = prefs.value
        val model = MusicModels.selected(app, p) ?: return false
        viewModelScope.launch {
            // the chat's model out of memory first: the song model needs it (the settings can keep it)
            if (p.unloadChat) runCatching { app.unloadNow() }
            // several takes: each its own seed (YuE2-ComfyUI's "four scores, one song": the best chosen by ear). The
            // takes made before a failure (the storage full, the folder not writable) are queued all the same.
            val (list, failed) = withContext(Dispatchers.IO) {
                val sung = try {
                    scoreIds(d)
                } catch (e: IllegalArgumentException) {   // (the edit or the move the score format cannot read)
                    return@withContext emptyList<Song>() to IllegalStateException(app.getString(
                        xyz.tifu.geode.R.string.music_score_unreadable, e.message ?: e.javaClass.simpleName))
                }
                // the lyrics laid along a tune: its length, unless the settings name one (YuE2-ComfyUI's max_seconds 0)
                val pp = sung.prefs(p)
                // the instrumental adapter: its section plan as the lyrics ([instrumental]: the model's own), mode full
                val lora = if (d.instrumental) MusicAdapter.lora(app) else null
                val words = if (d.instrumental) MusicAdapter.lyricsOf(d.lyrics) else d.lyrics
                val mode = if (d.instrumental) YuE2Prompt.Mode.FULL else d.mode
                val title = d.title.trim().ifEmpty { YuE2Prompt.defaultTitle(d.style, words) }
                songs.createTakes(title, d.style.trim(), words.trim(), n) { i ->
                    MusicJob.create(model.dir, app.applicationInfo.nativeLibraryDir, mode, words, pp, sung.abc, i, scoresFirst, lora)
                }
            }
            failed?.let { e ->
                android.widget.Toast.makeText(app, app.getString(xyz.tifu.geode.R.string.music_failed, e.message ?: e.javaClass.simpleName),
                    android.widget.Toast.LENGTH_LONG).show()
            }
            list.forEach { app.musicClient.make(it.id) }   // (queued: one runs at a time)
            withContext(Dispatchers.IO) { refreshSongs() }
        }
        return true
    }

    /**
     * The same song again: a new take of its own request (MusicJob.again: its settings, a new seed; [melody]: on the
     * same score - the same tune, sung and played anew). The draft on the Create tab is left as it is.
     */
    fun again(song: Song, melody: Boolean) {
        val p = prefs.value
        val model = MusicModels.selected(app, p) ?: return
        viewModelScope.launch {
            if (p.unloadChat) runCatching { app.unloadNow() }
            val made = withContext(Dispatchers.IO) {
                runCatching {
                    songs.create(song.title, song.style, song.lyrics,
                        MusicJob.again(song, model.dir, app.applicationInfo.nativeLibraryDir, p, melody))
                }
            }
            made.onSuccess { app.musicClient.make(it.id) }.onFailure { e ->
                android.widget.Toast.makeText(app, app.getString(xyz.tifu.geode.R.string.music_failed, e.message ?: e.javaClass.simpleName),
                    android.widget.Toast.LENGTH_LONG).show()
            }
            withContext(Dispatchers.IO) { refreshSongs() }
        }
    }

    /** [plan]'s score sung (MusicJob.fromPlan): a new song beside the plan, which stays to be sung again. */
    fun sing(plan: Song) {
        val p = prefs.value
        val model = MusicModels.selected(app, p) ?: return
        viewModelScope.launch {
            if (p.unloadChat) runCatching { app.unloadNow() }
            val made = withContext(Dispatchers.IO) {
                runCatching {
                    songs.create(plan.title, plan.style, plan.lyrics,
                        MusicJob.fromPlan(plan, model.dir, app.applicationInfo.nativeLibraryDir, p))
                }
            }
            made.onSuccess { app.musicClient.make(it.id) }.onFailure { e ->
                android.widget.Toast.makeText(app, app.getString(xyz.tifu.geode.R.string.music_failed, e.message ?: e.javaClass.simpleName),
                    android.widget.Toast.LENGTH_LONG).show()
            }
            withContext(Dispatchers.IO) { refreshSongs() }
        }
    }

    fun cancel(song: Song) {
        app.musicClient.cancel(song.id)
        // a song no service runs (its process was stopped): marked cancelled here
        if (!service.value.busy(song.id) && song.status == SongStatus.WAITING) viewModelScope.launch(Dispatchers.IO) {
            songs.cancelled(song.id)
            refreshSongs()
        }
    }

    /** A song that never finished (its process was stopped): made again from its own job. */
    fun resume(song: Song) {
        app.musicClient.make(song.id)
    }

    fun delete(song: Song) {
        if (_playing.value.first == song.id) stop()
        if (service.value.busy(song.id)) app.musicClient.cancel(song.id)
        viewModelScope.launch(Dispatchers.IO) {
            for (e in song.waitingEdits(songs.list())) {   // (they read its files: they cannot be made without it)
                app.musicClient.cancel(e.id)
                if (!service.value.busy(e.id)) songs.cancelled(e.id)
            }
            songs.delete(song.id)
            refreshSongs()
        }
    }

    fun rename(song: Song, title: String) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { songs.rename(song.id, title.trim().ifEmpty { song.title }) }
            refreshSongs()
        }
    }

    /** Seconds of making a second of song on this phone, from the last song made (null: none made yet). */
    /** About how long [takes] takes of a song up to [seconds] take here, from the latest song made: its composing
     *  (the score and the codes - once for takes queued together: MusicService.companionsFor, ~1.25x one take's for
     *  each more, measured on the Redmi) and the rest (the sound) apart. A take whose composing a companion run made
     *  tells only the rest. Null: no song made yet. */
    fun estimate(seconds: Int, takes: Int): Double? {
        val done = _library.value.filter { it.status == SongStatus.DONE && it.seconds > 1 }
        val last = done.firstOrNull() ?: return null
        val composing = { s: Song ->
            s.result?.optJSONObject("times")?.let { it.optDouble("score", 0.0) + it.optDouble("codes", 0.0) } ?: 0.0
        }
        // (a take a companion run composed: its ar.json; its own run still times a few microseconds of codes)
        val withAr = done.firstOrNull { !File(it.dir, MusicJob.AR_FILE).isFile && composing(it) > 0.5 }
            ?: return last.makeSeconds / last.seconds * seconds * takes   // (no song with its own composing timed)
        val runs = 1 + (withAr.result?.optInt("companions", 0) ?: 0)
        val ar = composing(withAr) / (1.0 + 0.25 * (runs - 1)) / withAr.seconds   // one take's, a second of song
        val rest = (last.makeSeconds - composing(last)) / last.seconds
        var all = 0.0
        for (first in 0 until takes step 4) {   // (a run makes at most four takes' composing: itself and three more)
            val n = minOf(4, takes - first)
            all += ar * (1.0 + 0.25 * (n - 1)) * seconds
        }
        return all + rest * seconds * takes
    }

    // ---- playing ----

    fun play(song: Song) {
        play(song.id, song.audio ?: return)
    }

    /** What the track editor shows of a song: its waveform and pitch classes, its score's bars on them (estimated
     *  from the score: BarGrid), its length. */
    data class Track(val view: SongView?, val grid: BarGrid?, val seconds: Double, val score: String? = null)

    /** [song]'s track (the view read, the bars fitted: a few hundred ms for a long song). */
    suspend fun track(song: Song): Track = withContext(Dispatchers.IO) {
        val view = song.view?.let { SongView.read(it) }
        // an edited song's own bars (its score still holds the cut ones; a retake's moved by the length it sang); else
        // the score's fitted to the sound
        val e = song.edit
        val kept = when (e?.optString("op")) {
            null, "" -> null
            // (a song going on, a break put in: the bars the service laid at the song's tempo once its score was written)
            "go_on", "insert" -> BarGrid.of(song.result?.optJSONObject("grid"))
            // (a move: the bars as its pieces laid them, then each join's stretch sung again as the engine reports)
            "move" -> BarGrid.of(e.optJSONObject("grid"))?.let { g ->
                val sung = song.result?.optJSONObject("move")?.optJSONArray("sung") ?: return@let null
                (0 until sung.length()).fold(g) { at, i ->
                    val s = sung.getJSONArray(i)
                    at.retake(s.getInt(0), s.getInt(1), s.getInt(2))
                }
            }
            "retake" -> BarGrid.of(e.optJSONObject("grid"))?.let { g ->
                song.result?.optJSONObject("retake")?.optInt("count", -1)?.takeIf { it >= 0 }
                    ?.let { g.retake(e.optInt("first"), e.optInt("stop"), it) }
            }
            else -> BarGrid.of(e.optJSONObject("grid"))
        }
        val score = songScore(song)
        val grid = kept ?: view?.let { v -> score?.let { BarGrid.measure(it, v) } }
        Track(view, grid, song.seconds, score)
    }

    private var rangeStop: Job? = null

    /** [song] from [fromMs] until [toMs] (the track editor's selection), then stopped; playing already: stopped. */
    fun playRange(song: Song, fromMs: Int, toMs: Int) {
        val id = RANGE + song.id
        if (_playing.value.first == id) return stop()
        play(id, song.audio ?: return, fromMs)
        rangeStop?.cancel()
        rangeStop = viewModelScope.launch {
            while (_playing.value.first == id) {
                val at = runCatching { player?.currentPosition }.getOrNull() ?: break
                if (at >= toMs) {
                    stop()
                    break
                }
                delay(25)
            }
        }
    }

    /**
     * [song]'s frames [first, stop) (25 a second) cut out (MusicJob.cut): a new song in the library, titled as the
     * song with a mark; the song itself stays. A trim (the start or the end) is done in a second; a seam inside is
     * sung again, which takes about as long as the song's own sound did for a window of up to 80 s.
     */
    fun cut(song: Song, first: Int, stop: Int, grid: BarGrid? = null, plan: TrackEdit.CutPlan? = null) {
        val trim = first == 0 || stop >= Math.rint(song.seconds * YuE2Prompt.FRAMES_PER_SECOND).toInt()
        // (a cut by bars: the new song's lyrics without the sections it took out)
        edit(song, !trim, xyz.tifu.geode.R.string.music_track_cut_title, plan?.lyrics ?: song.lyrics) { model, p ->
            MusicJob.cut(song, first, stop, model, app.applicationInfo.nativeLibraryDir, p, grid, plan)
        }
    }

    /** [song]'s frames [first, stop) sung again with the same words (MusicJob.retake): a new song in the library, the
     *  song itself stays. */
    fun retake(song: Song, first: Int, stop: Int, grid: BarGrid? = null) =
        edit(song, true, xyz.tifu.geode.R.string.music_track_retake_title) { model, p ->
            MusicJob.retake(song, first, stop, model, app.applicationInfo.nativeLibraryDir, p, grid)
        }

    /** [song]'s frames [first, stop) sung again to new words (YuE2-ComfyUI's change of words): a new song whose lyrics
     *  are [rewrite]'s (Words.changeWords), the tune the song's score; the song itself stays. */
    fun retakeWords(song: Song, first: Int, stop: Int, rewrite: Words.Rewrite, grid: BarGrid? = null) =
        edit(song, true, xyz.tifu.geode.R.string.music_track_words_title, rewrite.text) { model, p ->
            MusicJob.retake(song, first, stop, model, app.applicationInfo.nativeLibraryDir, p, grid, words = true)
        }

    /** [song]'s sections moved as [plan] says (TrackEdit.move, MusicJob.move): a new song with the plan's lyrics, the
     *  song itself stays. */
    fun move(song: Song, plan: TrackEdit.MovePlan) =
        edit(song, true, xyz.tifu.geode.R.string.music_track_move_title, plan.lyrics) { model, p ->
            MusicJob.move(song, plan, model, app.applicationInfo.nativeLibraryDir, p)
        }

    /** [song] with a break put in as [plan] says (TrackEdit.breakPlan, MusicJob.breakIn): a new song with the plan's
     *  lyrics, the song itself stays. [score]: the song's score; [grid]: its bars. */
    fun breakIn(song: Song, plan: TrackEdit.BreakPlan, score: String, grid: BarGrid) =
        edit(song, true, xyz.tifu.geode.R.string.music_track_break_title, plan.lyrics) { model, p ->
            MusicJob.breakIn(song, plan, score, grid, model, app.applicationInfo.nativeLibraryDir, p)
        }

    /** [song] going on from [plan] (GoOn.plan) with the new lines [text] (MusicJob.goOn): a new song with the plan's
     *  lyrics, the song itself stays. [score]: the song's score; [clock]: its bars. */
    fun goOn(song: Song, plan: GoOn.Plan, text: String, score: String, clock: GoOn.Clock) =
        edit(song, true, xyz.tifu.geode.R.string.music_track_go_on_title, plan.lyrics) { model, p ->
            MusicJob.goOn(song, plan, text, score, clock, model, app.applicationInfo.nativeLibraryDir, p)
        }

    /** An edit of [song] made into a new song titled [title] (the song's title in it) with [lyrics] from the [job] built
     *  for the selected model; [heavy]: it runs the model (the chat's unloaded first when the settings say so). */
    private fun edit(song: Song, heavy: Boolean, title: Int, lyrics: String = song.lyrics,
                     job: (java.io.File, MusicPrefs) -> org.json.JSONObject) {
        val p = prefs.value
        val model = MusicModels.selected(app, p) ?: return
        viewModelScope.launch {
            if (p.unloadChat && heavy) runCatching { app.unloadNow() }
            val made = withContext(Dispatchers.IO) {
                runCatching { songs.create(app.getString(title, song.title), song.style, lyrics, job(model.dir, p)) }
            }
            made.onSuccess { app.musicClient.make(it.id) }.onFailure { e ->
                android.widget.Toast.makeText(app, app.getString(xyz.tifu.geode.R.string.music_failed, e.message ?: e.javaClass.simpleName),
                    android.widget.Toast.LENGTH_LONG).show()
            }
            withContext(Dispatchers.IO) { refreshSongs() }
        }
    }

    /** [f] played under [id] (a song's, a score's: [SCORE], a selection's: [RANGE]) from [fromMs]; the one playing
     *  already: paused or on again. */
    private fun play(id: String, f: File, fromMs: Int = 0) {
        val (now, on) = _playing.value
        val p = player
        if (now == id && p != null) {
            if (on) p.pause() else p.start()
            _playing.value = id to !on
            return
        }
        stop()
        val mp = MediaPlayer()
        player = try {
            mp.setDataSource(f.path)
            mp.setOnCompletionListener { stop() }
            mp.prepare()
            if (fromMs > 0) mp.seekTo(fromMs)
            mp.start()
            mp
        } catch (e: Exception) {   // (a damaged file: the player given back, not left to the collector)
            mp.release()
            null
        }
        _playing.value = if (player != null) id to true else "" to false
    }

    /**
     * The score heard before it is sung (ScoreMidi: the phone's own MIDI synthesizer - the tune, the chords, not the
     * song's sound): the draft's as it will be sung (edited, moved), or [song]'s, the one it was sung from. Playing
     * already: stopped. Nothing when there is no score or it cannot be read (the melody card says why).
     */
    fun listenScore(song: Song? = null) {
        val id = SCORE + (song?.id ?: "draft")
        if (_playing.value.first == id) return stop()
        val d = _draft.value
        viewModelScope.launch {
            val f = withContext(Dispatchers.IO) {
                runCatching {
                    val text = (if (song != null) songScore(song) else draftScore(d)) ?: return@runCatching null
                    java.io.File(app.cacheDir, "score").apply { mkdirs() }.resolve("${song?.id ?: "draft"}.mid")
                        .also { writeAtomic(it, sync = false) { o -> o.write(ScoreMidi.of(text)) } }
                }.getOrNull()
            } ?: return@launch
            play(id, f)
        }
    }

    /** [text], a score the piano roll holds, heard as MIDI (again: stopped). */
    fun listenText(text: String) {
        val id = SCORE + "roll"
        if (_playing.value.first == id) return stop()
        viewModelScope.launch {
            val f = withContext(Dispatchers.IO) {
                runCatching {
                    java.io.File(app.cacheDir, "score").apply { mkdirs() }.resolve("roll.mid")
                        .also { writeAtomic(it, sync = false) { o -> o.write(ScoreMidi.of(text)) } }
                }.getOrNull()
            } ?: return@launch
            play(id, f)
        }
    }

    /** The draft's score as it will be sung (edited, moved); null: none (the model writes one, or the mode has none). */
    private fun draftScore(d: SongDraft): String? {
        if (d.mode == YuE2Prompt.Mode.OFF || d.abc.isEmpty()) return null
        val text = scoreText(d, tokenizerNow() ?: return null)
        val sung = if (d.transpose != 0) AbcScore.move(text, d.transpose).text else text
        // (a bare tune as it will be sung: the lyrics laid along it - passages left out or repeated)
        return runCatching { Underlay.lay(sung, d.lyrics)?.abc }.getOrNull() ?: sung
    }

    /** The score [song] was sung from, as text; null: none (the mode without one). */
    private fun songScore(song: Song): String? = song.abc.takeIf { it.isNotEmpty() }?.let { tokenizerNow()?.decode(it) }

    /** [song]'s score as a MIDI file to other apps (a sequencer, a notation app), under its title; null: no score. */
    suspend fun shareScore(song: Song): Intent? = withContext(Dispatchers.IO) {
        runCatching {
            val text = songScore(song) ?: return@runCatching null
            val base = song.shareBase()
            val f = java.io.File(base.path + ".mid")
            writeAtomic(f, sync = false) { it.write(ScoreMidi.of(text)) }
            val uri = FileProvider.getUriForFile(app, app.packageName + ".files", f)
            Intent.createChooser(Intent(Intent.ACTION_SEND).setType("audio/midi").putExtra(Intent.EXTRA_STREAM, uri)
                .putExtra(Intent.EXTRA_TITLE, song.title).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), song.title)
        }.getOrNull()
    }

    fun stop() {
        player?.release()
        player = null
        _playing.value = "" to false
    }

    /** Where the song plays now and its length (ms), for the seek bar. */
    fun position(): Pair<Int, Int> = player?.let { runCatching { it.currentPosition to it.duration }.getOrNull() } ?: (0 to 0)

    fun seek(ms: Int) {
        runCatching { player?.seekTo(ms) }
    }

    /**
     * The song to other apps (the share sheet: Viber, Telegram, WhatsApp...): for a messenger an .m4a (SongShare: AAC,
     * a few megabytes that play in the chat; made once, kept with the song), [lossless] the file itself. An .m4a that
     * could not be made: the file itself. Null: no audio. Reads and encodes: off the main thread.
     */
    suspend fun share(song: Song, lossless: Boolean = false): Intent? {
        if (_sharing.value != null) return null   // (one at a time: a second tap waits for nothing)
        _sharing.value = song.id
        try {
            return shareFile(song, lossless)
        } finally {
            _sharing.value = null
        }
    }

    /** The song a share is being made for now (its card shows it), or null. */
    private val _sharing = MutableStateFlow<String?>(null)
    val sharing: StateFlow<String?> = _sharing.asStateFlow()

    /** [id]'s share (the song-ready notification's Share); null: gone, or no audio. */
    suspend fun share(id: String): Intent? = withContext(Dispatchers.IO) { songs.find(id) }?.let { share(it) }

    private suspend fun shareFile(song: Song, lossless: Boolean): Intent? = withContext(Dispatchers.IO) {
        val audio = song.audio ?: return@withContext null
        val f = (if (lossless) null else runCatching { song.m4a(audio) }
            .onFailure { if (it is kotlinx.coroutines.CancellationException) throw it; android.util.Log.w("GeodeMusic", "m4a", it) }
            .getOrNull()) ?: song.named(audio) ?: return@withContext null
        val uri = runCatching { FileProvider.getUriForFile(app, app.packageName + ".files", f) }.getOrNull()
            ?: return@withContext null
        val type = when (f.extension) {
            "m4a" -> "audio/mp4"
            "wav" -> "audio/wav"
            else -> "audio/flac"
        }
        val send = Intent(Intent.ACTION_SEND).setType(type)
            .putExtra(Intent.EXTRA_STREAM, uri).putExtra(Intent.EXTRA_TITLE, song.title)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        Intent.createChooser(send, song.title)
    }

    /** Where the song's shares sit, under its title (cache/shared/<id>/: what the other app shows and saves it as). */
    private fun Song.shareBase(): java.io.File {
        val dir = java.io.File(app.cacheDir, "shared/$id").apply { mkdirs() }
        return java.io.File(dir, title.replace(Regex("[\\\\/:*?\"<>|\\p{Cntrl}]"), "_").trim().take(80).ifEmpty { "song" })
    }

    /** The song's shares but [keep] gone (an earlier title's). */
    private fun java.io.File.dropOthers(keep: Set<String>) {
        parentFile?.listFiles()?.forEach { if (it.name !in keep) it.delete() }
    }

    /**
     * The song's file under its title: a hard link to it (a copy where linking fails); deleting the song deletes it too.
     */
    private fun Song.named(audio: java.io.File): java.io.File? = runCatching {
        val base = shareBase()
        val out = java.io.File(base.path + "." + audio.extension)
        base.dropOthers(setOf(out.name, base.name + ".m4a", base.name + ".mid"))
        if (out.length() != audio.length()) {
            out.delete()
            try {
                android.system.Os.link(audio.path, out.path)
            } catch (e: android.system.ErrnoException) {
                audio.copyTo(out, overwrite = true)
            }
        }
        out
    }.getOrNull()

    /** The song as an .m4a under its title (SongShare), made now unless one newer than its audio is there. */
    private suspend fun Song.m4a(audio: java.io.File): java.io.File {
        val base = shareBase()
        val out = java.io.File(base.path + ".m4a")
        base.dropOthers(setOf(out.name, base.name + "." + audio.extension, base.name + ".mid"))
        if (out.length() > 0 && out.lastModified() >= audio.lastModified()) return out
        val job = kotlin.coroutines.coroutineContext[kotlinx.coroutines.Job]
        SongShare.toM4a(audio, out) { job?.isActive == false }
        return out
    }

    // ---- the lyrics writer ----

    fun editWriter(change: (WriterState) -> WriterState) = _writer.update(change)

    fun write() {
        val w = _writer.value
        if (w.running || w.idea.isBlank()) return
        _writer.value = w.copy(running = true, text = "", error = null, style = "", lyrics = "", notes = emptyList())
        writerJob = viewModelScope.launch {
            try {
                val r = LyricsWriter(app).write(w.idea, w.extra, prefs.value) { t -> _writer.update { it.copy(text = t) } }
                _writer.update { it.copy(running = false, style = r.style, lyrics = r.lyrics, notes = r.notes) }
            } catch (e: kotlinx.coroutines.CancellationException) {
                _writer.update { it.copy(running = false) }
                throw e
            } catch (e: Exception) {
                _writer.update { it.copy(running = false, error = e.message ?: e.javaClass.simpleName) }
            }
        }
    }

    fun stopWriting() {
        writerJob?.cancel()
        writerJob = null
        _writer.update { it.copy(running = false) }
    }

    /** The writer's song into the draft (its style and lyrics; the title kept). */
    fun useWritten() {
        val w = _writer.value
        if (w.lyrics.isEmpty() && w.style.isEmpty()) return
        editDraft { it.copy(style = w.style.ifEmpty { it.style }, lyrics = w.lyrics.ifEmpty { it.lyrics }) }
        _replaced.update { it + 1 }
        _writer.update { it.copy(style = "", lyrics = "", text = "", notes = emptyList()) }
    }

    // ---- settings ----

    fun setPrefs(change: (MusicPrefs) -> MusicPrefs) {
        viewModelScope.launch { app.music.update(change) }
    }

    /** The faster composing (Q4) / the sound's 8-bit weights on or off; off gives their space back (1.2 / 1.5 GB) -
     * at once when no song is made or waits (a queued one may have asked for them: it would only make them again),
     * else when the service goes idle (init's collector). */
    fun setArQ4(on: Boolean) {
        viewModelScope.launch {
            app.music.update { it.copy(arQ4 = on) }
            if (!on) dropUnwanted()
        }
    }

    fun setSoundPrecision(v: SoundPrecision) {
        viewModelScope.launch {
            app.music.update { it.copy(soundPrecision = v) }
            if (v != SoundPrecision.FAST) dropUnwanted()
        }
    }

    /** The weights made on the phone that the settings no longer ask for, removed - only while nothing is made or
     * queued, the settings read again just before (a switch turned back on meanwhile keeps them). */
    private fun dropUnwanted() {
        val s = service.value
        if (!s.connected || s.running.isNotEmpty() || s.queued.isNotEmpty()) return
        viewModelScope.launch(Dispatchers.IO) {
            val p = app.music.flow.first()
            if (!p.arQ4) MusicModels.dropArQ4(app)
            if (p.soundPrecision != SoundPrecision.FAST) MusicModels.dropNarQ8(app)
        }
    }

    // ---- the song model ----

    fun loadCatalog() {
        if (_catalog.value.loading) return
        _catalog.value = MusicCatalog(loading = true)
        viewModelScope.launch {
            val token = app.settings.flow.first().hfToken
            val r = withContext(Dispatchers.IO) { runCatching { MusicModels.catalog(token) } }
            _catalog.value = r.fold({ MusicCatalog(it) }, { MusicCatalog(error = it.message ?: it.javaClass.simpleName) })
        }
    }

    fun download(p: CatalogPack) {
        TransferService.start(app, TransferSource.Hf(p.pack.repo, p.pack.revision, p.pack.path, music = true),
            p.manifest.name.ifEmpty { p.pack.label })
    }

    fun resumeDownload(p: PartialModel) {
        val src = p.source ?: return
        TransferService.start(app, src, p.id)
    }

    fun cancelDownload() = TransferService.cancel(app)

    fun deletePartial(p: PartialModel) {
        viewModelScope.launch(Dispatchers.IO) {
            MusicModels.store(app).deletePartial(p.id)
            refreshModels()
        }
    }

    fun deleteModel(m: InstalledModel) {
        viewModelScope.launch(Dispatchers.IO) {
            MusicModels.store(app).delete(m.id)
            tokenizer = null
            refreshModels()
        }
    }

    fun freeBytes(): Long = MusicModels.store(app).freeBytes()

    companion object {
        /** The player's key for a score being heard: SCORE + "draft", or + a song's id. */
        const val SCORE = "score:"
        /** The track editor's selection playing: [RANGE] + the song's id. */
        const val RANGE = "range:"
    }
}
