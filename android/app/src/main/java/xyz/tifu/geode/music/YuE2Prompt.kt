package xyz.tifu.geode.music

/**
 * What YuE2 reads (its package's protocol.py, "yue2-native-v1"): the mode's instruction, then the style under [Tags]
 * and the lyrics under [Lyrics], verbatim. The engine adds the score's and the music's delimiters itself
 * (yue2_song.hpp): the heads here are [EOD] + the text's tokens.
 *
 * The lyrics' conventions and the length ceiling follow pytraveler/YuE2-ComfyUI (Apache-2.0: its constants.py and
 * writer.py), which measured them on this model.
 */
object YuE2Prompt {
    const val EOD = 151643
    /** 25 codec frames a second (1920 samples at 48 kHz each). */
    const val FRAMES_PER_SECOND = 25
    const val SAMPLE_RATE = 48000

    /** The model's modes (cot): a score with chords first, a melody-only score first, or no score. */
    enum class Mode(val key: String, val instruction: String) {
        FULL("full", "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."),
        MELODY("melody", "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."),
        OFF("off", "Generate music with codec tokens from the given conditions.");

        companion object {
            fun of(key: String?): Mode = entries.firstOrNull { it.key == key } ?: FULL
        }
    }

    /** The section tags the model was trained around (the lyrics' convention, not a list it checks). */
    val TAGS = listOf("Intro", "Verse", "Pre-Chorus", "Chorus", "Hook", "Bridge", "Refrain", "Interlude", "Instrumental", "Outro")

    /** The tags whose sections are played, not sung: written alone (no lines under them), a pause for the voice. */
    val BREAKS = listOf("Interlude", "Instrumental")

    fun text(mode: Mode, style: String, lyrics: String): String =
        "${mode.instruction}\n[Tags]\n$style\n[Lyrics]\n$lyrics\n"

    /** CFG's default per mode (protocol.py's guidance): 1 is no negative branch; "off" leans on it slightly. */
    fun defaultCfg(mode: Mode): Float = if (mode == Mode.OFF) 1.01f else 1.0f

    // ---- the length ceiling (constants.py: auto_seconds) ----

    const val MAX_SECONDS = 360
    private const val AUTO_BASE_SECONDS = 12.0
    private const val AUTO_SECONDS_PER_LINE = 12.0
    private const val AUTO_MIN_SECONDS = 40.0
    private const val AUTO_INSTRUMENTAL_SECONDS = 180.0
    private const val AUTO_WORDS_PER_LINE = 8
    private val DIRECTION = Regex("\\[[^\\]]*]")

    /** Lines that are sung: directions in brackets taken out first, a long line counting one per eight words. */
    fun sungLines(lyrics: String): Int = lyrics.lines().sumOf { line ->
        val words = DIRECTION.replace(line, " ").split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.none { w -> w.any { it.isLetterOrDigit() } }) 0
        else maxOf(1, (words.size + AUTO_WORDS_PER_LINE / 2) / AUTO_WORDS_PER_LINE)
    }

    /**
     * The ceiling when the user leaves the length on automatic: 12 s + 12 s a sung line, 40 s at least, 180 s with no
     * words at all, 360 s at most. It only cuts a song short (the model decides when a song ends): on a short lyric it
     * keeps the model from singing on long after the words ran out.
     */
    fun autoSeconds(lyrics: String): Int {
        val lines = sungLines(lyrics)
        if (lines == 0) return AUTO_INSTRUMENTAL_SECONDS.toInt()
        return (AUTO_BASE_SECONDS + AUTO_SECONDS_PER_LINE * lines).coerceIn(AUTO_MIN_SECONDS, MAX_SECONDS.toDouble()).toInt()
    }

    /** No words to sing: an instrumental (the lyrics empty, or only section tags). */
    fun instrumental(lyrics: String): Boolean = sungLines(lyrics) == 0

    /** A line that is a section tag ("[Chorus]", "[Verse 2]", "[Verse - breathy voice]"). */
    /** A song's title when it is given none: its first sung line, else its style's first words. */
    fun defaultTitle(style: String, lyrics: String): String =
        lyrics.lines().map { it.trim() }.firstOrNull { it.isNotEmpty() && !isTag(it) }?.take(40)
            ?: style.substringBefore(',').trim().take(40).ifEmpty { "Song" }

    fun isTag(line: String): Boolean {
        val t = line.trim()
        return t.length > 2 && t.startsWith("[") && t.endsWith("]")
    }
}
