package xyz.tifu.geode.media

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Answers read aloud: with the phone's own text-to-speech engine (what the system has, offline when its voices are
 * downloaded), or with Geode's own voice ([NeuralVoice], Supertonic 3) when the user picked and downloaded it - the
 * languages it does not speak still read by the phone's engine, in their turn. An answer is spoken sentence by
 * sentence - as it is written ([feed]) or all at once ([speak]) - each sentence in the voice of its script's language
 * (a Ukrainian answer on an English phone is not read with English sounds), the markdown's marks and the code blocks
 * left out.
 */
class Speaker(context: Context) {
    private val app = context.applicationContext

    private val _speaking = MutableStateFlow<Long?>(null)
    /** The message being read now; null: none. */
    val speaking: StateFlow<Long?> = _speaking.asStateFlow()

    private val _available = MutableStateFlow<Boolean?>(null)
    /** Whether the phone has a text-to-speech engine that started (null: not asked yet). */
    val available: StateFlow<Boolean?> = _available.asStateFlow()

    private val main = android.os.Handler(android.os.Looper.getMainLooper())
    private var tts: TextToSpeech? = null
    private var ready = false
    /** What waits for the engine to start: (message, its sentences). */
    private val waiting = ArrayList<Pair<Long, List<String>>>()
    /** The message [feed] reads, how much of its text is spoken (or queued), and its sentences queued. */
    private var fedId = -1L
    private var fedUpTo = 0
    private var queued = 0
    private var done = 0
    private var rate = 1f
    private var pitch = 1f
    /** The voice picked for its language ("": none, every language's default). */
    private var voiceName = ""

    /** A voice the engine has: its name (what the setting keeps), its language, whether it needs the internet. */
    data class VoiceInfo(val name: String, val locale: Locale, val network: Boolean)

    /** Geode's own voice: its files, its download, its reading. */
    val neural = NeuralVoice(app).also { n ->
        n.onDone = { uid -> finished(uid) }
        n.fallback = { text, locale, stopped -> sayWithSystem(text, locale, stopped) }
    }
    /** Geode's voice is the engine picked (it reads once it is downloaded). */
    private var useNeural = false
    private val fallbackSeq = java.util.concurrent.atomic.AtomicLong()
    @Volatile private var fallbackWait: Pair<String, java.util.concurrent.CountDownLatch>? = null

    private val _voices = MutableStateFlow<List<VoiceInfo>>(emptyList())
    /** The engine's installed voices once it started (sorted by language), for the settings' choice. */
    val voices: StateFlow<List<VoiceInfo>> = _voices.asStateFlow()

    /**
     * The phone's sound while an answer is read: asked for as a passing need that others may lower themselves for
     * (music and the navigator get quieter, as for a navigator's own prompt), given back when the reading ends. Losing
     * it ends nothing: a text-to-speech engine may ask for the sound itself, and the reading must not stop for that.
     */
    private val focus = android.media.AudioFocusRequest.Builder(android.media.AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        .setAudioAttributes(android.media.AudioAttributes.Builder()
            .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
            .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
            .build())
        .build()

    init {
        val audio = app.getSystemService(android.media.AudioManager::class.java)
        (app as? xyz.tifu.geode.GeodeApp)?.appScope?.launch(kotlinx.coroutines.Dispatchers.Main) {
            _speaking.map { it != null }.distinctUntilChanged().collect { on ->
                runCatching { if (on) audio?.requestAudioFocus(focus) else audio?.abandonAudioFocusRequest(focus) }
            }
        }
    }

    private fun engine(retry: Boolean): TextToSpeech? {
        // one that did not start is asked again when the user asks (an engine may have been installed or turned on
        // since); reading as the answer is written does not ask again sentence by sentence
        if (_available.value == false) {
            if (!retry) return null
            tts?.shutdown()
            tts = null
        }
        tts?.let { return it }
        ready = false
        _available.value = null
        lateinit var t: TextToSpeech
        // (posted: without an engine the answer may come inside the constructor, before t and tts are set)
        t = TextToSpeech(app) { status -> main.post { started(t, status == TextToSpeech.SUCCESS) } }
        // a phone whose only engine is turned off may never answer (seen on a ZTE): no answer is no engine
        main.postDelayed({ if (tts === t && _available.value == null) started(t, false) }, START_MS)
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String) {}
            override fun onDone(utteranceId: String) = finished(utteranceId)
            @Deprecated("the older callback")
            override fun onError(utteranceId: String) = finished(utteranceId)
            override fun onError(utteranceId: String, errorCode: Int) = finished(utteranceId)
        })
        tts = t
        return t
    }

    /** The engine [t] started ([ok]) or not: what waits for it is read, or the user told why it is not. */
    @Synchronized
    private fun started(t: TextToSpeech, ok: Boolean) {
        if (tts !== t) return
        ready = ok
        _available.value = ok
        if (ok) {
            t.setSpeechRate(rate)
            t.setPitch(pitch)
            _voices.value = runCatching {
                t.voices.orEmpty().filter {
                    android.speech.tts.TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features.orEmpty()
                }.map { VoiceInfo(it.name, it.locale, it.isNetworkConnectionRequired) }
                    .sortedWith(compareBy({ it.locale.getDisplayName(Locale.getDefault()) }, { it.network }, { it.name }))
            }.getOrDefault(emptyList())
            val w = ArrayList(waiting)
            waiting.clear()
            w.forEach { (id, s) -> queue(id, s) }
        } else {
            // something was to be read: why nothing is, said where the user is
            if (waiting.isNotEmpty()) {
                android.widget.Toast.makeText(app, xyz.tifu.geode.R.string.voice_none, android.widget.Toast.LENGTH_LONG).show()
            }
            waiting.clear()
            _speaking.value = null
        }
    }

    private fun finished(utteranceId: String) {
        // a sentence Geode's voice gave the phone's engine: its wait is over
        if (utteranceId.startsWith(FALLBACK)) {
            fallbackWait?.takeIf { it.first == utteranceId }?.second?.countDown()
            return
        }
        val id = utteranceId.substringBefore(':').toLongOrNull() ?: return
        synchronized(this) {
            if (id != _speaking.value) return
            done++
            // the last sentence queued is over, and nothing more is to come for it
            if (done >= queued && fedId != id) _speaking.value = null
        }
    }

    fun setRate(r: Float) {
        rate = r
        neural.speed = r
        tts?.setSpeechRate(r)
    }

    fun setPitch(p: Float) {
        pitch = p
        neural.pitch = p
        tts?.setPitch(p)
    }

    /** Geode's own voice ([geode]) or the phone's engine; what is being read stops when it changes. */
    fun setEngine(geode: Boolean) {
        if (geode == useNeural) return
        stop()
        useNeural = geode
    }

    /** Geode's voice's style: 0-4 female, 5-9 male. */
    fun setNeuralVoice(id: Int) {
        neural.speakerId = id
    }

    private fun neuralOn() = useNeural && neural.installed.value

    fun setVoice(name: String) {
        voiceName = name
    }

    /** The engine started (if it has not yet) so its voices are known: the settings' voice choice opens. */
    fun prepare() {
        engine(retry = true)
    }

    /** [text] read as an example of the voice, rate and pitch set (whatever was being read stopped). */
    fun sample(text: String) = speak(SAMPLE_ID, text)

    /** [text] read from its start, whatever was being read stopped. */
    @Synchronized
    fun speak(id: Long, text: String) {
        stop()
        val s = sentences(clean(text))
        if (s.isEmpty()) return
        fedId = -1L
        start(id, s, retry = true)
    }

    /**
     * The answer [id] as it is written: its sentences that are complete in [text] and not read yet go to the queue;
     * [final]: the answer ended, the rest goes too.
     */
    @Synchronized
    fun feed(id: Long, text: String, final: Boolean) {
        if (id != fedId) {
            if (fedId != -1L || _speaking.value != null) stop()
            fedId = id
            fedUpTo = 0
        }
        val end = if (final) text.length else completeUpTo(text, fedUpTo)
        if (end > fedUpTo) {
            val s = sentences(clean(text.substring(fedUpTo, end)))
            fedUpTo = end
            if (s.isNotEmpty()) {
                if (_speaking.value != id) start(id, s) else queue(id, s)
            }
        }
        if (final) {
            fedId = -1L
            if (done >= queued) _speaking.value = null
        }
    }

    private fun start(id: Long, s: List<String>, retry: Boolean = false) {
        if (neuralOn()) {
            _speaking.value = id
            queued = 0
            done = 0
            queueNeural(id, s)
            return
        }
        val t = engine(retry) ?: return
        _speaking.value = id
        queued = 0
        done = 0
        if (!ready) waiting += id to s else queue(id, s, t)
    }

    /**
     * Each sentence to Geode's voice, in the language of its script (one it does not speak goes to the phone's); a
     * long one in its clauses, so its first words are heard before all of it is made.
     */
    private fun queueNeural(id: Long, s: List<String>) {
        for (sentence in s) {
            val lang = voiceFor(sentence)
            for (piece in clauses(sentence, CLAUSE)) {
                val n = queued++
                neural.add("$id:$n", piece, lang)
            }
        }
    }

    /** Geode's voice opened ahead: an answer is being written that will be read aloud. */
    fun warm() {
        if (neuralOn()) neural.warm()
    }

    /**
     * A sentence of Geode's reading in a language its voice does not speak, read by the phone's engine; returns when
     * it was read, could not be (no engine), or the reading stopped. On Geode's voice's playing thread.
     */
    private fun sayWithSystem(text: String, locale: Locale, stopped: () -> Boolean) {
        val got = java.util.concurrent.CountDownLatch(1)
        var t: TextToSpeech? = null
        main.post {
            t = engine(retry = false)
            got.countDown()
        }
        if (!got.await(2, java.util.concurrent.TimeUnit.SECONDS)) return
        val engine = t ?: return
        val until = System.currentTimeMillis() + START_MS + 500
        while (!ready && _available.value != false && System.currentTimeMillis() < until && !stopped()) Thread.sleep(50)
        if (!ready || stopped()) return
        val uid = FALLBACK + fallbackSeq.incrementAndGet()
        val read = java.util.concurrent.CountDownLatch(1)
        fallbackWait = uid to read
        main.post {
            if (engine.setLanguage(locale) < TextToSpeech.LANG_AVAILABLE) engine.setLanguage(Locale.getDefault())
            engine.speak(text, TextToSpeech.QUEUE_ADD, Bundle(), uid)
        }
        // (a sentence takes well under 200 ms a letter; past that the engine is taken for stuck)
        val deadline = System.currentTimeMillis() + maxOf(30_000L, text.length * 200L)
        while (!read.await(50, java.util.concurrent.TimeUnit.MILLISECONDS)) {
            if (stopped()) {
                main.post { engine.stop() }
                break
            }
            if (System.currentTimeMillis() > deadline) break
        }
        fallbackWait = null
    }

    private fun queue(id: Long, s: List<String>, t: TextToSpeech? = tts) {
        if (neuralOn()) {
            queueNeural(id, s)
            return
        }
        t ?: return
        val picked = voiceName.takeIf { it.isNotEmpty() }?.let { n -> runCatching { t.voices?.firstOrNull { it.name == n } }.getOrNull() }
        for (sentence in s) {
            val lang = voiceFor(sentence)
            // the voice picked reads its own language; another language its default voice (and with no voice for
            // that language on the phone, the phone's own)
            if (picked != null && picked.locale.language == lang.language) t.voice = picked
            else if (t.setLanguage(lang) < TextToSpeech.LANG_AVAILABLE) t.setLanguage(Locale.getDefault())
            val n = queued++
            t.speak(sentence, TextToSpeech.QUEUE_ADD, Bundle(), "$id:$n")
        }
    }

    /** Nothing read any more (Stop, another chat, a new question). */
    @Synchronized
    fun stop() {
        waiting.clear()
        fedId = -1L
        fedUpTo = 0
        queued = 0
        done = 0
        _speaking.value = null
        neural.stop()
        fallbackWait?.second?.countDown()
        tts?.stop()
    }

    fun shutdown() {
        stop()
        tts?.shutdown()
        tts = null
        ready = false
    }

    companion object {
        /** How long an engine may take to start before it is taken for none. */
        private const val START_MS = 6000L

        /** The utterances Geode's voice hands the phone's engine. */
        private const val FALLBACK = "fb:"

        /** The settings' example ([sample]): no message's id. */
        const val SAMPLE_ID = -2L

        /** Geode's voice's pieces: a sentence longer than this is read in its clauses. */
        private const val CLAUSE = 90

        /**
         * [s] in pieces of at most about [max] letters, cut only after a clause's mark (a comma, a dash, a semicolon,
         * a colon): never inside a clause, so its intonation is whole. A piece without such a mark stays as it is.
         */
        internal fun clauses(s: String, max: Int): List<String> {
            if (s.length <= max) return listOf(s)
            val out = ArrayList<String>()
            var rest = s
            while (rest.length > max) {
                val head = rest.substring(0, max)
                val mark = CLAUSE_MARK.findAll(head).lastOrNull { it.range.first >= MIN_CLAUSE }
                    ?: CLAUSE_MARK.find(rest, max)?.takeIf { rest.length - it.range.last > MIN_CLAUSE }
                    ?: break
                val at = mark.range.last + 1
                out += rest.substring(0, at).trim()
                rest = rest.substring(at).trim()
            }
            if (rest.isNotEmpty()) out += rest
            return out
        }

        /** A clause shorter than this is kept with the next (a piece of two words sounds clipped). */
        private const val MIN_CLAUSE = 20

        /** Where a clause ends: a comma, semicolon or colon and a space, or a dash between spaces. */
        private val CLAUSE_MARK = Regex("[,;:]\\s|\\s[—–-]\\s")

        /** A sentence's longest piece for the engine (TextToSpeech.getMaxSpeechInputLength is 4000). */
        private const val MAX_PIECE = 3000

        /**
         * Where the complete sentences of [text] end, from [from] on: after the last ". ", "! ", "? ", "…", a line's
         * end (and not inside a code block still open). [from] when none.
         */
        internal fun completeUpTo(text: String, from: Int): Int {
            // an open code block: nothing after its start is read before it closes
            val fences = Regex("```").findAll(text).count()
            val limit = if (fences % 2 == 1) text.lastIndexOf("```") else text.length
            var end = from
            var i = from
            while (i < limit) {
                val c = text[i]
                val next = if (i + 1 < limit) text[i + 1] else null
                // (a mark at the very end may not end a sentence yet: "3." of "3.14" still being written)
                val mark = c == '.' || c == '!' || c == '?' || c == '…'
                if (c == '\n' || c == '。' || c == '！' || c == '？' || (mark && (next == ' ' || next == '\n'))) end = i + 1
                i++
            }
            return end
        }

        /** The text as it is said: code blocks, markdown's marks, links' addresses and tables' rules left out. */
        internal fun clean(text: String): String {
            var s = text.replace(Regex("```[\\s\\S]*?(```|$)"), "\n")
            s = s.replace(Regex("!\\[[^\\]]*]\\([^)]*\\)"), "")
            s = s.replace(Regex("\\[([^\\]]+)]\\([^)]*\\)"), "$1")
            s = s.replace(Regex("(?m)^\\s*\\|?\\s*:?-{3,}.*$"), "")
            s = s.replace(Regex("(?m)^\\s{0,3}(#{1,6}\\s+|>\\s?|[-*+]\\s+|\\d+[.)]\\s+)"), "")
            // a formula as its text ($\text{H}_2\text{O}$ is "H2O"); a $ without one (a price) stays
            s = s.replace(Regex("\\$\\$?([^$\\n]*[\\\\_^][^$\\n]*)\\$\\$?")) { m -> formula(m.groupValues[1]) }
            s = s.replace(Regex("[*_~`|]+"), " ")
            s = s.replace(Regex("<[^>]+>"), "")
            s = s.replace(Regex("https?://\\S+"), "")
            return s.replace(Regex("[ \\t]+"), " ")
        }

        /** A LaTeX formula as it is said: \text{..} and friends unwrapped, \frac{a}{b} "a/b", the rest of its marks gone. */
        internal fun formula(f: String): String {
            var s = f.replace(Regex("\\\\(?:text|mathrm|mathbf|mathit|operatorname)\\{([^}]*)\\}"), "$1")
            s = s.replace(Regex("\\\\frac\\{([^}]*)\\}\\{([^}]*)\\}"), "$1/$2")
            s = s.replace(Regex("\\\\(?:times|cdot)"), " × ")
            s = s.replace(Regex("\\\\[a-zA-Z]+"), " ")
            s = s.replace(Regex("[{}_^\\\\]"), "")
            return s.replace(Regex("\\s+"), " ").trim()
        }

        /** [text] in pieces the engine takes: its lines, each cut at its sentences when too long; empty ones gone. */
        internal fun sentences(text: String): List<String> {
            val out = ArrayList<String>()
            for (line in text.split('\n')) {
                var l = line.trim()
                while (l.length > MAX_PIECE) {
                    val cut = l.lastIndexOfAny(charArrayOf('.', '!', '?', ',', ' '), MAX_PIECE).takeIf { it > 0 } ?: MAX_PIECE
                    out += l.substring(0, cut + 1).trim()
                    l = l.substring(cut + 1).trim()
                }
                if (l.any { it.isLetterOrDigit() }) out += l
            }
            return out
        }

        /**
         * The voice for [s]: by the script most of its letters are in (Cyrillic with і ї є ґ is Ukrainian, without them
         * the phone's language when that is a Cyrillic one, else Russian; kana Japanese; ...), the phone's language for
         * the Latin script when that is a Latin one, else English.
         */
        internal fun voiceFor(s: String): Locale {
            val sys = Locale.getDefault()
            val counts = HashMap<Character.UnicodeScript, Int>()
            var kana = false
            for (ch in s) {
                if (!ch.isLetter()) continue
                val sc = Character.UnicodeScript.of(ch.code)
                if (sc == Character.UnicodeScript.HIRAGANA || sc == Character.UnicodeScript.KATAKANA) kana = true
                counts[sc] = (counts[sc] ?: 0) + 1
            }
            val main = counts.maxByOrNull { it.value }?.key ?: return sys
            fun sysIf(vararg langs: String) = if (sys.language in langs) sys else null
            return when (main) {
                Character.UnicodeScript.CYRILLIC -> when {
                    s.any { it in "іїєґІЇЄҐ" } -> Locale("uk")
                    else -> sysIf("uk", "ru", "bg", "sr", "mk", "be", "kk") ?: Locale("ru")
                }
                Character.UnicodeScript.HAN -> if (kana) Locale.JAPANESE else sysIf("zh", "ja") ?: Locale.CHINESE
                Character.UnicodeScript.HIRAGANA, Character.UnicodeScript.KATAKANA -> Locale.JAPANESE
                Character.UnicodeScript.HANGUL -> Locale.KOREAN
                Character.UnicodeScript.ARABIC -> sysIf("ar", "fa", "ur") ?: Locale("ar")
                Character.UnicodeScript.DEVANAGARI -> sysIf("hi", "mr") ?: Locale("hi")
                Character.UnicodeScript.BENGALI -> Locale("bn")
                Character.UnicodeScript.GUJARATI -> Locale("gu")
                Character.UnicodeScript.GURMUKHI -> Locale("pa")
                Character.UnicodeScript.TAMIL -> Locale("ta")
                Character.UnicodeScript.TELUGU -> Locale("te")
                Character.UnicodeScript.KANNADA -> Locale("kn")
                Character.UnicodeScript.THAI -> Locale("th")
                Character.UnicodeScript.GREEK -> Locale("el")
                Character.UnicodeScript.HEBREW -> Locale("he")
                Character.UnicodeScript.LATIN -> if (sys.language in LATIN_LANGS) sys else Locale.ENGLISH
                else -> sys
            }
        }

        private val LATIN_LANGS = setOf("en", "de", "fr", "es", "it", "pt", "tr", "in", "id", "jv", "vi", "sw", "ha",
            "tl", "fil", "pl", "nl", "cs", "sk", "ro", "hu", "sv", "da", "no", "nb", "fi", "hr", "sl", "lt", "lv", "et")
    }
}
