package xyz.tifu.geode.music

import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.server.GenEvent
import xyz.tifu.geode.server.GenRequest
import xyz.tifu.geode.tokenizer.Part
import xyz.tifu.geode.tokenizer.TemplateMessage

/** A song the writer wrote: the style line and the tagged lyrics, with what is worth saying about them. */
data class WrittenSong(val style: String, val lyrics: String, val notes: List<WriterNote>)

/** Something about an answer that came back usable anyway (never an error: the song still sings). */
enum class WriterNote(
    /** The API's name for it. */
    val code: String,
    /** What the user is told. */
    @androidx.annotation.StringRes val text: Int,
) {
    SHORT_OR_LONG("length", xyz.tifu.geode.R.string.music_note_length),
    NO_TEMPO("tempo", xyz.tifu.geode.R.string.music_note_tempo),
    NO_TAGS("tags", xyz.tifu.geode.R.string.music_note_tags),
}

/**
 * Turns an idea into a style line and lyrics YuE2 can sing, with one of the phone's chat models (the music settings'
 * writer model, else the chat's). A port of pytraveler/YuE2-ComfyUI's writer.py (Apache-2.0): its instruction sheet,
 * which was measured to keep a small model to the layout (sections scaled with the line budget), its parsing and its
 * clean-up - with the rest of what YuE2 reads added for this app: every section tag, the instrumental breaks (a tag
 * alone: the voice rests) and the stressed vowel as a capital ("ФраголІно", GUIDE.md: 3 takes of 3 right).
 */
class LyricsWriter(private val app: GeodeApp) {
    /** The song for [idea] ([extra]: the user's own instructions, after the rules, so they can override them);
     *  [onText] gets the answer as it is written. Throws ApiError when the model cannot answer, and an
     *  IOException when it answered twice with no lyrics in it (writer.py's refusal: an empty song is no answer). */
    suspend fun write(idea: String, extra: String, p: MusicPrefs, onText: (String) -> Unit = {}): WrittenSong {
        var answer = ask(idea, extra, p, repair = false, onText)
        var (style, lyrics) = split(answer)
        if (!complete(style, lyrics)) {
            // one more try with the layout said again (writer.py's REPAIR)
            answer = ask(idea, extra, p, repair = true, onText)
            val second = split(answer)
            if (complete(second.first, second.second) || second.second.isNotEmpty()) {
                style = second.first
                lyrics = second.second
            }
            if (lyrics.isBlank()) throw java.io.IOException(app.getString(xyz.tifu.geode.R.string.music_writer_unusable))
        }
        return WrittenSong(style, lyrics, findings(style, lyrics, p.writerLength.lines))
    }

    private suspend fun ask(idea: String, extra: String, p: MusicPrefs, repair: Boolean, onText: (String) -> Unit): String {
        var system = systemPrompt(p)
        if (extra.isNotBlank()) system += "\n\n" + extra.trim()
        if (repair) system += REPAIR
        val req = GenRequest(
            messages = listOf(
                TemplateMessage("system", listOf(Part.Text(system))),
                TemplateMessage("user", listOf(Part.Text(idea.trim()))),
            ),
            tools = emptyList(),
            thinking = p.writerThinking,
            maxTokens = p.writerMaxTokens + if (p.writerThinking) 2048 else 0,
            temperature = p.writerTemperature,
            topP = p.writerTopP,
            topK = p.writerTopK,
            // none: a chorus comes back word for word (a presence penalty would push the model off it)
            presencePenalty = 0f,
            modelId = p.writerModel,
        )
        val text = StringBuilder()
        val r = app.apiEngine.run(req, "music-") { e ->
            if (e is GenEvent.Content) {
                text.append(e.text)
                onText(text.toString())
            }
        }
        return r.content.ifEmpty { text.toString() }
    }

    companion object {
        private const val SECTION_LINES = 4

        private const val EXAMPLE_STYLE = "English, warm piano pop, expressive female voice, acoustic piano, rounded " +
            "bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"

        private val SYSTEM = """You write songs for YuE2, a model that sings exactly what you hand it.

Answer with these two sections and nothing else. No title, no explanation, no markdown, no code fences.

STYLE
One line, comma separated, in this order: the language the song is sung in, the genre, the singing voice, two or three instruments, the character of the melody, the phrasing, and the tempo as a number followed by BPM. Name sounds, not feelings. The whole line looks like this:

{example}

LYRICS
Section tags on their own line in square brackets -- [Intro], [Verse], [Pre-Chorus], [Chorus], [Hook], [Bridge], [Refrain], [Outro] -- and under each tag the lines that are sung, one per line, with a blank line between sections. Nothing on a sung line but the words: no tags, no brackets, no parentheses, no stage directions, no numbering, no quotation marks.

Write {sections} sections of about {per} sung lines each, {lines} sung lines in total, not counting the tags. Bring the same chorus back rather than writing a new one.{breaks}{stress}{language}

Lay the answer out exactly like this:

STYLE
<the one line>

LYRICS
[Verse]
<line>
<line>

[Chorus]
<line>
<line>"""

        private const val BREAKS_LINE = "\n\nWhere the voice should rest and the band play on, write [Interlude] (or " +
            "[Instrumental]) on its own line with no sung lines under it: an instrumental break. One or two in a song " +
            "at most, never two in a row."

        private const val STRESS_LINE = "\n\nThe singer cannot guess where the stress falls in a name, a borrowed or " +
            "rare word, or a word whose meaning hangs on its stress. In such a word write the stressed vowel as a " +
            "capital letter and the rest of the word in lower case, for example ФраголІно, or зАмок (a castle) against " +
            "замОк (a lock). Leave ordinary words as they are, and never write a whole word in capitals."

        private const val LANGUAGE_LINE = "\n\nSing it in {language}: write the lyrics in {language}, and name " +
            "{language} first in the style line."

        private const val REPAIR = "\n\nYour last answer did not have the two sections. Answer again with the word " +
            "STYLE on its own line, then the one style line, then the word LYRICS on its own line, then the tagged " +
            "lyrics. Nothing else."

        /** The languages the writer can be told to sing in (English names: they go into the prompt), with their tags. */
        val LANGUAGES = listOf("English" to "en", "Ukrainian" to "uk", "Russian" to "ru", "Chinese" to "zh",
            "Japanese" to "ja", "Korean" to "ko", "Spanish" to "es", "French" to "fr", "German" to "de",
            "Italian" to "it", "Portuguese" to "pt", "Polish" to "pl")

        /** (sections, lines a section) for a line budget: the sections scale with it (writer.py section_plan). */
        fun sectionPlan(lines: Int): Pair<Int, Int> {
            val wanted = lines.coerceIn(2, 40)
            val sections = maxOf(2, Math.round(wanted / SECTION_LINES.toFloat()))
            return sections to maxOf(2, Math.round(wanted / sections.toFloat()))
        }

        fun systemPrompt(p: MusicPrefs): String {
            val lines = p.writerLength.lines
            val (sections, per) = sectionPlan(lines)
            val language = p.writerLanguage.takeIf { l -> LANGUAGES.any { it.first == l } }
            return SYSTEM.replace("{example}", EXAMPLE_STYLE)
                .replace("{sections}", sections.toString()).replace("{per}", per.toString())
                .replace("{lines}", lines.toString())
                .replace("{breaks}", if (p.writerBreaks) BREAKS_LINE else "")
                .replace("{stress}", if (p.writerStress) STRESS_LINE else "")
                .replace("{language}", language?.let { LANGUAGE_LINE.replace("{language}", it) } ?: "")
        }

        private val TAGS = setOf("intro", "verse", "pre-chorus", "prechorus", "chorus", "hook", "bridge", "refrain",
            "outro", "instrumental", "interlude")
        private val HEADING = Regex("^[\\s>#*_`-]*(style|lyrics)\\s*[:.\\-]*\\s*$", RegexOption.IGNORE_CASE)
        private val TAG_LINE = Regex("^[\\s>*_]*\\[?\\s*([A-Za-z][A-Za-z -]*?)\\s*(\\d+)?\\s*]?[\\s:*_]*$")
        private val FENCE = Regex("^\\s*```")
        private val NUMBERED = Regex("^\\s*\\d+\\s*[.)]\\s+")
        private val PARENTHETICAL = Regex("^\\s*[(\\[][^)\\]]*[)\\]]\\s*$")
        private val THINK_CLOSE = Regex("</think\\s*>", RegexOption.IGNORE_CASE)
        private val THINK_OPEN = Regex("<think\\s*>", RegexOption.IGNORE_CASE)
        private val TEMPO = Regex("\\d+\\s*bpm", RegexOption.IGNORE_CASE)
        // a note about a section instead of its words ("No tags" under an [Intro]): nothing to sing
        private val META = Regex("^(no\\s+(tags?|lyrics|words|text|vocals?|singing)|none|n/?a|empty|silence|" +
            "(instrumental|music)\\s+(only|section|break|part)|без\\s+(слів|тексту|вокалу))[.!]?$", RegexOption.IGNORE_CASE)

        /** What the model wrote after it finished reasoning; an unclosed reasoning is no answer (""). */
        fun stripThinking(text: String): String {
            THINK_CLOSE.findAll(text).lastOrNull()?.let { return text.substring(it.range.last + 1) }
            return if (THINK_OPEN.containsMatchIn(text)) "" else text
        }

        /** The normalised section tag [line] is ("[Pre-Chorus]", "[Verse 2]"), or "" if it is not one. */
        fun tag(line: String): String {
            val m = TAG_LINE.matchEntire(line) ?: return ""
            val word = m.groupValues[1].trim().lowercase()
            if (word !in TAGS) return ""
            val pretty = if (word == "prechorus" || word == "pre-chorus") "Pre-Chorus"
            else word.split("-").joinToString("-") { it.replaceFirstChar(Char::uppercaseChar) }
            val number = m.groupValues[2]
            return "[" + pretty + (if (number.isNotEmpty()) " $number" else "") + "]"
        }

        private fun unquote(line: String): String =
            if (line.length > 1 && (line[0] == '"' || line[0] == '\'') && line.last() == line[0]) line.substring(1, line.length - 1).trim()
            else line

        /** The style line out of whatever the model wrapped it in. */
        fun tidyStyle(block: String): String {
            for (raw in block.lines()) {
                var line = raw.trim().trim('`')
                if (line.isEmpty() || FENCE.containsMatchIn(line) || HEADING.matches(line)) continue
                line = unquote(line.trim('*', '_', ' ').trim())
                if (line.isNotEmpty()) return line.replace(Regex("\\s+"), " ")
            }
            return ""
        }

        /** The lyrics with the tags normalised and the model's debris removed; the blank lines rebuilt. */
        fun tidyLyrics(block: String): String {
            val kept = mutableListOf<String>()
            for (raw in block.lines()) {
                var line = raw.trim()
                if (line.isEmpty() || FENCE.containsMatchIn(line) || HEADING.matches(line)) continue
                val t = tag(line)
                if (t.isNotEmpty()) {
                    if (kept.isNotEmpty()) kept += ""
                    kept += t
                    continue
                }
                if (PARENTHETICAL.matches(line)) continue
                line = unquote(NUMBERED.replaceFirst(line, "").trim().trim('*', '_', '`').trim())
                if (line.isNotEmpty() && !META.matches(line)) kept += line
            }
            while (kept.isNotEmpty() && kept.last().isEmpty()) kept.removeAt(kept.size - 1)
            return kept.joinToString("\n")
        }

        /** (style, lyrics) out of one answer: the headed layout, else the first section tag as the seam. */
        fun split(answer: String): Pair<String, String> {
            val body = stripThinking(answer)
            val style = mutableListOf<String>()
            val lyrics = mutableListOf<String>()
            val head = mutableListOf<String>()
            var current = ""
            var seen = false
            for (line in body.lines()) {
                val h = HEADING.matchEntire(line)
                if (h != null) {
                    current = h.groupValues[1].lowercase()
                    seen = true
                    continue
                }
                when {
                    current == "style" -> style += line
                    current == "lyrics" -> lyrics += line
                    !seen -> head += line
                }
            }
            if (seen && (style.isNotEmpty() || lyrics.isNotEmpty() || head.isNotEmpty())) {
                val s = tidyStyle(style.joinToString("\n")).ifEmpty { tidyStyle(head.joinToString("\n")) }
                return s to tidyLyrics(lyrics.joinToString("\n"))
            }
            val lines = body.lines()
            for ((i, line) in lines.withIndex())
                if (tag(line.trim()).isNotEmpty())
                    return tidyStyle(lines.take(i).joinToString("\n")) to tidyLyrics(lines.drop(i).joinToString("\n"))
            return tidyStyle(body) to ""
        }

        /** Lines that carry words. */
        fun sung(lyrics: String): Int = lyrics.lines().count { val l = it.trim(); l.isNotEmpty() && !(l.startsWith("[") && l.endsWith("]")) }

        fun complete(style: String, lyrics: String): Boolean = style.isNotBlank() && sung(lyrics) > 0

        fun findings(style: String, lyrics: String, wanted: Int): List<WriterNote> {
            val notes = mutableListOf<WriterNote>()
            val written = sung(lyrics)
            if (written > 0 && kotlin.math.abs(written - wanted) > maxOf(2, wanted / 3)) notes += WriterNote.SHORT_OR_LONG
            if (style.isNotEmpty() && !TEMPO.containsMatchIn(style)) notes += WriterNote.NO_TEMPO
            if (lyrics.isNotEmpty() && lyrics.lines().none { tag(it.trim()).isNotEmpty() }) notes += WriterNote.NO_TAGS
            return notes
        }
    }
}
