package xyz.tifu.geode.music

import java.util.Locale
import kotlin.math.abs
import xyz.tifu.geode.music.midi.Phrasing

/** Lyric edits ported from YuE2-ComfyUI's inpaint/ops.py and track.py (Apache-2.0).
 * No model, tokenizer or Android dependencies. Score times are quarter notes. */
object Words {
    data class Breaking(val text: String, val matched: Boolean)

    /** Empty interlude at the paired section boundary; port of ops.break_words (Apache-2.0). */
    fun breakWords(lyrics: String, score: String, bar: Int): Breaking {
        val sections = sections(EditText.read(score), 0, 1)
        val blocks = blocks(lyrics)
        val singing = blocks.indices.filter { blocks[it].sung > 0 }
        val labels = singing.map { if (blocks[it].tag.isEmpty()) "verse" else Phrasing.labelOf(blocks[it].tag) }
        val ahead = mutableListOf<Int>()
        var at = 0
        for (label in labels) {
            while (at < sections.size && sections[at].label != label) at++
            if (at == sections.size) break
            ahead += at++
        }
        var place: Int? = null
        for (pairing in listOf(ahead, paired(labels, sections).orEmpty())) {
            val later = singing.zip(pairing).firstOrNull { (_, s) -> sections[s].bar >= bar }
            if (later != null) { place = later.first; break }
            if (singing.isNotEmpty() && pairing.size == singing.size) { place = singing.last() + 1; break }
        }
        val cut = place ?: return Breaking(lyrics, false)
        val ending = if ("\r\n" in lyrics) "\r\n" else "\n"
        val head = blocks.take(cut).joinToString("") { it.raw.joinToString("") }
        val tail = blocks.drop(cut).joinToString("") { it.raw.joinToString("") }
        if (tail.isNotEmpty()) return Breaking(head + "[Interlude]" + ending.repeat(2) + tail, true)
        val closed = head.trimEnd('\r', '\n')
        return Breaking(closed + ending.repeat(2) + "[Interlude]" + head.substring(closed.length), true)
    }
    data class Cut(val text: String, val dropped: List<String>, val matched: Boolean)
    data class Moving(val text: String, val moved: List<String>, val matched: Boolean, val tagged: Boolean = false)
    data class Rewrite(val text: String, val before: List<String>, val after: List<String>, val tag: String,
                       val oldSyllables: Int, val newSyllables: Int)

    private val tag = Regex("\\[\\s*([^\\]]*?)\\s*\\]")
    private fun tagged(line: String) = tag.matches(EditText.strip(line))
    private fun tagOf(line: String) = tag.matchEntire(EditText.strip(line))?.groupValues?.get(1)?.let(EditText::strip)
    private data class Block(val tag: String, val raw: MutableList<String> = mutableListOf(), var sung: Int = 0)
    private data class Section(val bar: Int, val label: String, val notes: Int, val cut: Int)

    private fun blocks(lyrics: String): List<Block> {
        val out = mutableListOf(Block(""))
        for (raw in EditText.lines(lyrics, true)) {
            val line = EditText.strip(raw)
            val found = tagOf(line)
            if (found != null) out += Block(found, mutableListOf(raw))
            else {
                out.last().raw += raw
                if (Phrasing.syllables(line) > 0) out.last().sung++
            }
        }
        return out
    }

    private fun sections(sheet: AbcScore.Score, first: Int, stop: Int): List<Section> {
        val vocal = sheet.voices.getValue("Vocal")
        val bars = vocal.bars
        require(first >= 0 && first < stop && stop <= bars.size) {
            "Bars ${first + 1} to $stop are not bars of this score, which has ${bars.size}."
        }
        val begin = bars[first].start
        val end = bars[stop - 1].let { it.start + it.duration }
        val named = AbcScore.sections(sheet.text)
        return named.mapIndexed { i, s ->
            val high = named.getOrNull(i + 1)?.start ?: vocal.time
            val inside = vocal.notes.filter { it.start >= s.start && it.start < high }
            val label = EditText.strip(s.name).lowercase(Locale.ROOT).map { if (EditText.space(it)) ' ' else it }.joinToString("").split(' ')
                .filter { it.isNotEmpty() }.joinToString(" ")
            Section(bars.indexOfFirst { it.start == s.start }, label, inside.size, inside.count { it.start >= begin && it.start < end })
        }
    }

    private fun paired(labels: List<String>, sections: List<Section>): List<Int>? {
        val found = mutableListOf<Int>()
        var at = 0
        for (label in labels) {
            while (at < sections.size && sections[at].label != label) at++
            if (at == sections.size) break
            found += at++
        }
        if (found.size == labels.size) return found
        return sections.indices.filter { sections[it].notes > 0 }.takeIf { it.size == labels.size }
    }

    /** Go on uses exactly the same all-or-nothing section pairing as a cut. */
    internal fun lastSung(lyrics: String, sheet: AbcScore.Score): Int? {
        val labels = blocks(lyrics).filter { it.sung > 0 }.map { if (it.tag.isEmpty()) "verse" else Phrasing.labelOf(it.tag) }
        if (labels.isEmpty()) return null
        val sections = sections(sheet, 0, sheet.voices.getValue("Vocal").bars.size)
        return paired(labels, sections)?.lastOrNull()?.let { sections[it].bar }
    }

    /** Delete whole lyric blocks losing strictly more than half their vocal note onsets. */
    fun cutWords(lyrics: String, score: String, first: Int, stop: Int): Cut {
        val sections = sections(EditText.read(score), first, stop)
        val blocks = blocks(lyrics)
        val singing = blocks.indices.filter { blocks[it].sung > 0 }
        val labels = singing.map { if (blocks[it].tag.isEmpty()) "verse" else Phrasing.labelOf(blocks[it].tag) }
        val paired = paired(labels, sections) ?: return Cut(lyrics, emptyList(), false)
        val dropped = singing.zip(paired).filter { (_, s) -> sections[s].notes > 0 && sections[s].cut * 2 > sections[s].notes }
            .map { it.first }.toSet()
        if (dropped.isEmpty()) return Cut(lyrics, emptyList(), true)
        val text = blocks.filterIndexed { i, _ -> i !in dropped }.joinToString("") { it.raw.joinToString("") }
        return Cut(EditText.strip(text), dropped.map { blocks[it].tag }, true)
    }

    /** Lyric reorder ported from YuE2-ComfyUI's ops.move_words (Apache-2.0).
     * Block bodies move, but blank-line gaps stay at their original positions. */
    fun moveWords(lyrics: String, score: String, first: Int, stop: Int, to: Int): Moving {
        val sheet = EditText.read(score)
        ScoreEdit.order(first, stop, to, sheet.voices.getValue("Vocal").bars.size)
        val sections = sections(sheet, 0, 1)
        val blocks = blocks(lyrics)
        val singing = blocks.indices.filter { blocks[it].sung > 0 }
        val labels = singing.map { if (blocks[it].tag.isEmpty()) "verse" else Phrasing.labelOf(blocks[it].tag) }
        val paired = paired(labels, sections) ?: return Moving(lyrics, emptyList(), false)
        val where = singing.zip(paired.map { sections[it].bar }).toMap()
        val going = singing.filter { where.getValue(it) in first until stop }
        if (going.isEmpty()) return Moving(lyrics, emptyList(), true)
        val rest = blocks.indices.filter { it !in going }
        val place = rest.indexOfFirst { (where[it] ?: -1) >= to }.takeIf { it >= 0 }
            ?: (rest.indexOfLast { it in where }.takeIf { it >= 0 }?.plus(1) ?: rest.size)
        val order = rest.take(place) + going + rest.drop(place)
        val ending = if ("\r\n" in lyrics) "\r\n" else "\n"
        val bodies = mutableListOf<String>(); val gaps = mutableListOf<String>()
        for (block in blocks) {
            val raw = block.raw.toMutableList()
            val gap = mutableListOf<String>()
            while (raw.isNotEmpty() && EditText.strip(raw.last()).isEmpty()) gap.add(0, raw.removeAt(raw.lastIndex))
            var body = raw.joinToString("")
            if (body.isNotEmpty() && !body.endsWith("\n") && !body.endsWith("\r")) body += ending
            bodies += body; gaps += gap.joinToString("")
        }
        var tagged = false
        var result = order.mapIndexed { position, index ->
            var body = bodies[index]
            if (blocks[index].tag.isEmpty() && blocks[index].sung > 0 &&
                order.take(position).any { EditText.strip(bodies[it]).isNotEmpty() }) {
                body = "[Verse]" + ending + body
                tagged = true
            }
            body + gaps[position]
        }.joinToString("")
        if (!lyrics.endsWith("\n") && !lyrics.endsWith("\r") && result.endsWith(ending)) result = result.dropLast(ending.length)
        return Moving(result, going.map { blocks[it].tag }, true, tagged)
    }

    /** Replace original lines [first, stop), counting tags and blanks, preserving untouched text and endings. */
    fun changeWords(lyrics: String, first: Int, stop: Int, text: String): Rewrite {
        val plain = EditText.lines(lyrics)
        val raw = EditText.lines(lyrics, true)
        require(first >= 0 && first < stop && stop <= plain.size) {
            "Lines ${first + 1} to $stop are not lines of these words, which have ${plain.size}."
        }
        val chosen = plain.subList(first, stop)
        require(chosen.none(::tagged)) { "A section tag is not a line to rewrite. Choose the lines that are sung in it." }
        require(chosen.any { Phrasing.syllables(it) > 0 }) { "Nothing is sung in the lines chosen, so there are no words to change." }
        val fresh = EditText.lines(text).map { it.trimEnd(EditText::space) }.toMutableList()
        while (fresh.firstOrNull() == "") fresh.removeAt(0)
        while (fresh.lastOrNull() == "") fresh.removeAt(fresh.lastIndex)
        require(fresh.none(::tagged)) { "New words cannot name a section: a tag among them would start one." }
        require(fresh.any { Phrasing.syllables(it) > 0 }) {
            "The new words say nothing. A stretch of a song is taken out with a cut, not with silence."
        }
        val ending = (first until stop).firstNotNullOfOrNull { i -> raw[i].substring(plain[i].length).ifEmpty { null } } ?: "\n"
        val tail = raw[stop - 1].substring(plain[stop - 1].length)
        val written = fresh.mapIndexed { i, line -> line + if (i == fresh.lastIndex) tail else ending }.joinToString("")
        val label = plain.take(first).mapNotNull(::tagOf).lastOrNull().orEmpty()
        return Rewrite(raw.take(first).joinToString("") + written + raw.drop(stop).joinToString(""),
            chosen.toList(), fresh.toList(), label, chosen.sumOf(Phrasing::syllables), fresh.sumOf(Phrasing::syllables))
    }

    fun needsFitNotice(rewrite: Rewrite): Boolean {
        val difference = abs(rewrite.newSyllables.toLong() - rewrite.oldSyllables)
        return difference >= 2 && difference * 4 > rewrite.oldSyllables
    }
}

/** Python's whitespace and splitlines semantics shared by the pure edit helpers. */
internal object EditText {
    fun space(c: Char) = c in '\t'..'\r' || c in '\u001c'..' ' || c == '\u0085' ||
        Character.isWhitespace(c) || Character.isSpaceChar(c)
    fun strip(s: String) = s.trim(::space)
    private val breaks = Regex("\\r\\n|[\\n\\r\\u000b\\u000c\\u001c-\\u001e\\u0085\\u2028\\u2029]")
    fun lines(s: String, keepEnds: Boolean = false): List<String> = buildList {
        var at = 0
        for (m in breaks.findAll(s)) {
            add(s.substring(at, if (keepEnds) m.range.last + 1 else m.range.first))
            at = m.range.last + 1
        }
        if (at < s.length) add(s.substring(at))
    }
    fun read(text: String): AbcScore.Score {
        val source = strip(text)
        require(source.isNotEmpty()) { "The song has no score." }
        require(source.codePointCount(0, source.length) <= 200000) { "That score is far longer than any song." }
        try {
            return AbcScore.parse(source).also {
                require(it.unit.denominator >= 4) { "L:1/${it.unit.denominator} is coarser than a quarter note" }
            }
        } catch (e: IllegalArgumentException) {
            throw IllegalArgumentException("This score cannot be read note by note: ${reason(e)}.", e)
        }
    }
    fun reason(e: IllegalArgumentException) = strip(e.message.orEmpty()).trimEnd('.')
}
