package xyz.tifu.geode.media

import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.PlaybackParams
import android.os.Process
import android.util.Log
import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import xyz.tifu.geode.models.HfApi

/**
 * Geode's own voice: Supertonic 3 (int8 ONNX through sherpa-onnx), 31 languages read on the phone's cores. Its files
 * are downloaded from Geode's Hugging Face repository only when the user picks it, and stay on the storage; the model
 * is opened when something is to be read and closed again after [IDLE_MS] of silence or when the phone runs short of
 * memory (it holds ~300 MB while open). Measured on the Redmi Note 9 Pro: opened in ~1.7 s, a sentence spoken in half
 * its length on two big cores.
 *
 * Two threads: one synthesizes the sentences in order (a sentence's audio as each of its own sentences is ready), the
 * other plays them, so the next is made while one is heard. A sentence of a language the model does not speak goes
 * to [fallback] (the phone's text-to-speech) in its turn.
 */
class NeuralVoice(context: Context) {
    private val app = context.applicationContext
    val dir = File(app.filesDir, "voices/supertonic3")

    data class VoiceFile(val name: String, val size: Long, val sha256: String)

    /** The download: [done] of [total] bytes; [error] once it failed; running while [running]. */
    data class Download(val running: Boolean = false, val done: Long = 0, val total: Long = TOTAL,
                        val error: String? = null)

    private val _installed = MutableStateFlow(checkInstalled())
    /** Every file of the voice is on the storage, checked. */
    val installed: StateFlow<Boolean> = _installed.asStateFlow()

    private val _download = MutableStateFlow(Download())
    val download: StateFlow<Download> = _download.asStateFlow()
    private var downloadJob: Job? = null

    /**
     * What a sentence is read with: its text, the model's language code for it (null: one the model does not speak,
     * [fallback] reads it), and the utterance's id for [onDone].
     */
    private class Line(val gen: Int, val uid: String, val text: String, val lang: String?, val locale: java.util.Locale,
                       val queuedAt: Long = System.nanoTime())

    private sealed interface Out { val gen: Int }
    private class Chunk(override val gen: Int, val samples: FloatArray, val rate: Int) : Out
    private class End(override val gen: Int, val uid: String) : Out
    private class Other(override val gen: Int, val line: Line) : Out

    @Volatile private var gen = 0
    private val toSynth = LinkedBlockingQueue<Any>()
    private val toPlay = LinkedBlockingQueue<Out>()
    @Volatile private var started = false
    @Volatile private var closeWanted = false

    @Volatile var speakerId = 0
    @Volatile var speed = 1f
    @Volatile var pitch = 1f

    /** A sentence read to its end (or dropped by a stop): its utterance's id. Set by the Speaker. */
    var onDone: (String) -> Unit = {}

    /**
     * A sentence the model does not speak, read by the phone's text-to-speech; returns once it was read (or could not
     * be). Called on the playing thread, in the sentence's turn.
     */
    var fallback: (text: String, locale: java.util.Locale, stopped: () -> Boolean) -> Unit = { _, _, _ -> }

    @Volatile private var tts: OfflineTts? = null

    // ---- the files ----

    private fun checkInstalled(): Boolean =
        File(dir, OK).isFile && FILES.all { File(dir, it.name).length() == it.size }

    /** The voice downloaded (what is there already, checked, is kept; a part from before goes on where it ended). */
    fun startDownload(scope: CoroutineScope) {
        if (downloadJob?.isActive == true) return
        _download.value = Download(running = true, done = 0)
        downloadJob = scope.launch(Dispatchers.IO) {
            try {
                dir.mkdirs()
                var before = 0L
                for (f in FILES) {
                    val target = File(dir, f.name)
                    if (target.length() == f.size && sha256(target) == f.sha256) {
                        before += f.size
                        _download.value = _download.value.copy(done = before)
                        continue
                    }
                    fetch(f, target, before)
                    before += f.size
                }
                File(dir, OK).writeText(REVISION)
                _installed.value = true
                _download.value = Download()
            } catch (e: CancellationException) {
                _download.value = Download()
                throw e
            } catch (e: Exception) {
                Log.w(TAG, "download", e)
                _download.value = Download(error = e.message ?: e.javaClass.simpleName)
            }
        }
    }

    fun cancelDownload() {
        downloadJob?.cancel()
    }

    private suspend fun fetch(f: VoiceFile, target: File, before: Long) {
        val part = File(dir, f.name + ".part")
        var have = part.length().takeIf { it in 1 until f.size } ?: 0L
        if (part.length() >= f.size) part.delete()
        val (conn, ranged) = HfApi.download(HfApi.resolveUrl(REPO, REVISION, f.name), "", have)
        try {
            if (!ranged) have = 0
            conn.inputStream.use { input ->
                FileOutputStream(part, have > 0).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var shown = 0L
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        have += n
                        // the progress every half megabyte (not every read)
                        if (have - shown >= (1 shl 19)) {
                            shown = have
                            _download.value = _download.value.copy(done = before + have)
                        }
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
        if (part.length() != f.size || sha256(part) != f.sha256) {
            part.delete()
            throw java.io.IOException("${f.name}: the file differs from the one expected")
        }
        target.delete()
        if (!part.renameTo(target)) throw java.io.IOException("${f.name}: could not be put in place")
    }

    /** The voice gone from the storage (closed first). */
    fun delete() {
        stop()
        closeWanted = true
        toSynth.offer(WAKE)
        // (the synthesizing thread closes the model; the files go once it did, or right away when it never opened)
        synchronized(this) {
            if (tts == null) removeFiles()
            else pendingDelete = true
        }
    }

    private var pendingDelete = false

    private fun removeFiles() {
        dir.listFiles()?.forEach { it.delete() }
        _installed.value = false
    }

    /** The model's licence (Open RAIL-M), for the settings to show; null before the voice is downloaded. */
    fun licence(): String? = runCatching {
        // its lines are cut for a page: a line that goes on with the sentence is joined to the one before
        File(dir, "LICENSE").readText().replace("\r\n", "\n").replace(Regex("\n(?=[a-z“\"]|\\([“\"A-Z])"), " ")
    }.getOrNull()

    // ---- reading ----

    /** [text] queued to be read in [locale]'s language (its turn after what is queued); [uid] reported to [onDone]. */
    fun add(uid: String, text: String, locale: java.util.Locale) {
        startThreads()
        val lang = locale.language.let { if (it == "in") "id" else it }.takeIf { it in LANGS }
        toSynth.offer(Line(gen, uid, text, lang, locale))
    }

    /** The model opened ahead (an answer to be read aloud is coming): its first sentence is not kept waiting for it. */
    fun warm() {
        if (!_installed.value) return
        startThreads()
        toSynth.offer(WARM)
    }

    /** Nothing queued is read any more, what plays stops now. */
    fun stop() {
        gen++
        toSynth.clear()
        toPlay.clear()
        synchronized(trackLock) {
            runCatching { track?.pause(); track?.flush() }
        }
    }

    fun generation(): Int = gen

    private fun startThreads() {
        if (started) return
        synchronized(this) {
            if (started) return
            started = true
            Thread(::synthLoop, "geode-voice").apply { isDaemon = true }.start()
            Thread(::playLoop, "geode-voice-play").apply { isDaemon = true }.start()
            app.registerComponentCallbacks(object : ComponentCallbacks2 {
                override fun onTrimMemory(level: Int) {
                    // short of memory, or the app out of sight: the model goes once nothing is read
                    if (level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND ||
                        level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
                        closeWanted = true
                        toSynth.offer(WAKE)
                    }
                }
                override fun onConfigurationChanged(newConfig: Configuration) {}
                @Deprecated("the older callback")
                override fun onLowMemory() = onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
            })
        }
    }

    private fun open(): OfflineTts {
        tts?.let { return it }
        fun f(name: String) = File(dir, name).path
        val config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                supertonic = OfflineTtsSupertonicModelConfig(
                    durationPredictor = f("duration_predictor.int8.onnx"),
                    textEncoder = f("text_encoder.int8.onnx"),
                    vectorEstimator = f("vector_estimator.int8.onnx"),
                    vocoder = f("vocoder.int8.onnx"),
                    ttsJson = f("tts.json"),
                    unicodeIndexer = f("unicode_indexer.bin"),
                    voiceStyle = f("voice.bin"),
                ),
                numThreads = THREADS,
                provider = "cpu",
            ),
        )
        val t0 = System.nanoTime()
        return OfflineTts(config = config).also {
            tts = it
            Log.i(TAG, "opened in ${(System.nanoTime() - t0) / 1_000_000} ms, ${it.numSpeakers()} voices, ${it.sampleRate()} Hz")
        }
    }

    private fun close() {
        synchronized(this) {
            tts?.release()
            tts = null
            if (pendingDelete) {
                pendingDelete = false
                removeFiles()
            }
        }
        Log.i(TAG, "closed")
    }

    /** Lines to audio, in order; the model closed after [IDLE_MS] with nothing to read (or when asked to). */
    private fun synthLoop() {
        while (true) {
            val item = try {
                toSynth.poll(IDLE_MS, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                null
            }
            if (item === WARM) {
                runCatching { if (_installed.value) open() }.onFailure { Log.w(TAG, "open", it) }
                continue
            }
            if (item == null || item === WAKE) {
                // idle (or asked): the model goes when the player has nothing left either
                if (tts != null && (item == null || closeWanted) && toSynth.isEmpty() && toPlay.isEmpty()) {
                    closeWanted = false
                    close()
                } else if (tts == null) {
                    closeWanted = false
                    synchronized(this) { if (pendingDelete) { pendingDelete = false; removeFiles() } }
                }
                continue
            }
            val line = item as Line
            if (line.gen != gen) continue
            if (line.lang == null || !_installed.value) {
                toPlay.offer(Other(line.gen, line))
                continue
            }
            try {
                val t = open()
                val rate = t.sampleRate()
                val cfg = GenerationConfig(sid = speakerId.coerceIn(0, maxOf(0, t.numSpeakers() - 1)),
                    speed = speed, numSteps = STEPS, extra = mapOf("lang" to line.lang))
                val feed = Feed(line.gen, rate)
                val t0 = System.nanoTime()
                t.generateWithConfigAndCallback(line.text, cfg, feed)
                Log.d(TAG, "${line.uid}: %.1f s of audio in %d ms, the first %d ms after it was queued".format(
                    feed.frames.toDouble() / rate, (System.nanoTime() - t0) / 1_000_000,
                    (feed.firstAt - line.queuedAt) / 1_000_000))
            } catch (e: Throwable) {
                Log.w(TAG, "synthesis", e)
            }
            toPlay.offer(End(line.gen, line.uid))
        }
    }

    /**
     * Each sentence's audio as the model made it, to the player; 0 (stop) once the reading was stopped. A class, not a
     * lambda: sherpa-onnx's JNI calls `invoke(float[]): Integer` by name, which a Kotlin lambda (an invokedynamic one,
     * desugared by D8) does not have - only its generic bridge (proguard-rules.pro keeps the method in release).
     */
    private inner class Feed(private val g: Int, private val rate: Int) : (FloatArray) -> Int {
        var frames = 0L
        var firstAt = 0L
        override fun invoke(samples: FloatArray): Int {
            if (g != gen) return 0
            if (frames == 0L) firstAt = System.nanoTime()
            frames += samples.size
            toPlay.offer(Chunk(g, samples.copyOf(), rate))
            return 1
        }
    }

    private val trackLock = Any()
    private var track: AudioTrack? = null
    private var trackRate = 0
    private var written = 0L
    /** Utterances played into the track: the frame each ends at, reported once the track got there. */
    private val ends = ArrayDeque<Pair<Long, String>>()
    private var playGen = -1

    private fun playLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
        while (true) {
            val item = try {
                toPlay.poll(40, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                null
            }
            if (playGen != gen) reset()
            report()
            if (item == null) {
                // all played and nothing to come for a while: the track let go (a new one starts with the next)
                val t = track
                if (t != null && ends.isEmpty() && (t.playbackHeadPosition.toLong() and 0xffffffffL) >= written) {
                    if (idleSince == 0L) idleSince = System.currentTimeMillis()
                    else if (System.currentTimeMillis() - idleSince > TRACK_IDLE_MS) releaseTrack()
                }
                continue
            }
            idleSince = 0L
            if (item.gen != gen) continue
            when (item) {
                is Chunk -> play(item)
                is End -> {
                    if (track == null) onDone(item.uid) else ends.addLast(written to item.uid)
                }
                is Other -> {
                    // its turn: what plays ends first, then the phone's engine reads it
                    drain(item.gen)
                    if (item.gen == gen) {
                        fallback(item.line.text, item.line.locale) { item.gen != gen }
                        if (item.gen == gen) onDone(item.line.uid)
                    }
                }
            }
        }
    }

    private fun play(c: Chunk) {
        val t = trackFor(c.rate) ?: return
        var off = 0
        while (off < c.samples.size && c.gen == gen) {
            val n = minOf(WRITE_FRAMES, c.samples.size - off)
            val w = t.write(c.samples, off, n, AudioTrack.WRITE_BLOCKING)
            if (w <= 0) break
            off += w
            written += w
            report()
        }
    }

    private fun trackFor(rate: Int): AudioTrack? {
        track?.let { if (trackRate == rate) return it.also { applyPitch(it) } }
        reset()
        val minBuf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val t = runCatching {
            AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setSampleRate(rate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STREAM)
                // half a second: a stop is heard at once, and writing never waits long
                .setBufferSizeInBytes(maxOf(minBuf, rate * 4 / 2))
                .build()
        }.getOrElse {
            Log.w(TAG, "audio track", it)
            return null
        }
        synchronized(trackLock) { track = t }
        trackRate = rate
        written = 0
        applyPitch(t)
        t.play()
        return t
    }

    private var pitchSet = 1f

    private fun applyPitch(t: AudioTrack) {
        if (pitchSet == pitch) return
        pitchSet = pitch
        runCatching { t.playbackParams = PlaybackParams().setPitch(pitch).setSpeed(1f) }
    }

    /** The utterances the track played to their end reported. */
    private fun report() {
        val t = track ?: return
        val head = t.playbackHeadPosition.toLong() and 0xffffffffL
        while (ends.isNotEmpty() && ends.first().first <= head) onDone(ends.removeFirst().second)
    }

    /** Until the track played what it was given (or a stop). */
    private fun drain(g: Int) {
        val t = track ?: return
        // (frames written but not yet played: at most the buffer's half second, a little more on a slow start)
        val deadline = System.currentTimeMillis() + 5_000
        while (g == gen && (t.playbackHeadPosition.toLong() and 0xffffffffL) < written &&
            System.currentTimeMillis() < deadline) {
            Thread.sleep(20)
        }
        report()
    }

    private var idleSince = 0L

    /**
     * After a stop: the track let go, and what it still had to report dropped (the Speaker counted the stopped reading
     * out already; a report now could be taken for one of a new reading with the same message).
     */
    private fun reset() {
        playGen = gen
        ends.clear()
        releaseTrack()
    }

    private fun releaseTrack() {
        val t = track
        synchronized(trackLock) { track = null }
        if (t != null) runCatching {
            t.pause()
            t.flush()
            t.release()
        }
        written = 0
        trackRate = 0
        pitchSet = 1f
        idleSince = 0L
    }

    private fun sha256(f: File): String {
        val d = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = s.read(buf)
                if (n < 0) break
                d.update(buf, 0, n)
            }
        }
        return d.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val TAG = "GeodeVoice"
        /** Geode's copy of the voice (playstore_assets/hf/README-Supertonic3.md), at the revision checked. */
        const val REPO = "Maximilian228/Geode-Supertonic-3"
        const val REVISION = "dd1208ec3eb6963755d671e4e4c0535bb867da69"
        private const val OK = "installed"

        val FILES = listOf(
            VoiceFile("duration_predictor.int8.onnx", 3700147, "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"),
            VoiceFile("text_encoder.int8.onnx", 36416150, "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff"),
            VoiceFile("vector_estimator.int8.onnx", 78400833, "20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd"),
            VoiceFile("vocoder.int8.onnx", 25991073, "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152"),
            VoiceFile("tts.json", 8253, "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09"),
            VoiceFile("unicode_indexer.bin", 262144, "8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30"),
            VoiceFile("voice.bin", 517168, "67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8"),
            VoiceFile("LICENSE", 15007, "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f"),
        )
        val TOTAL = FILES.sumOf { it.size }

        /** The languages Supertonic 3 speaks (its model card). */
        val LANGS = setOf("en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
            "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi")

        /** Its voice styles: 0-4 the female F1-F5, 5-9 the male M1-M5 (voice.bin merges them sorted by name). */
        const val VOICES = 10

        /** The phone's two big cores (more were no faster on the Redmi Note 9 Pro: the rest are little ones). */
        private const val THREADS = 2
        /** The flow-matching steps (the model's default; 3 is ~35% faster and a little rougher). */
        private const val STEPS = 5
        private const val IDLE_MS = 20_000L
        /** The audio track kept this long after the last sound (a reading that goes on keeps its track). */
        private const val TRACK_IDLE_MS = 3_000L
        private const val WRITE_FRAMES = 4096
        private val WAKE = Any()
        private val WARM = Any()
    }
}
