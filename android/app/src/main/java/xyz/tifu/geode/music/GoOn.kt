package xyz.tifu.geode.music

import xyz.tifu.geode.music.AbcScore.Fraction
import xyz.tifu.geode.music.midi.Phrasing

/** Score and timing helpers from YuE2-ComfyUI's inpaint/{ops,grid,track}.py and notation.head
 * (Apache-2.0). No model, tokenizer or audio dependencies. Parsed times below are quarter notes. */
object GoOn {
    const val TAG = "[Verse]"
    const val LONGEST_WORDS = 4000
    const val SOUND_REACHED = 0.5
    const val EARLIER = 50
    const val LATER = 250
    const val SCORE_TRIES = 3
    const val SCORE_SEED_STEP = 7919L

    data class Going(val text: String, val added: List<String>, val sung: List<String>, val tagged: Boolean)
    class Clock(val lines: List<Double>, val speed: Double)
    data class From(val bar: Int, val bars: Int, val start: Int)
    data class Plan(val bar: Int, val bars: Int, val start: Int, val lyrics: String, val head: String,
                    val tagged: Boolean, val sung: List<String>)

    private const val FRAME = 0.04
    private val voices = listOf("Vocal", "Ins")
    // (no Unicode character-class flag: Android's ICU regex refuses it; the tags and the rests are ASCII)
    private val tag = Regex("\\[\\s*([^\\]]*?)\\s*\\]")
    private val fullRest = Regex("\\s*Z([2-4])?\\s*")
    private val breaks = Regex("\\r\\n|[\\n\\r\\u000b\\u000c\\u001c-\\u001e\\u0085\\u2028\\u2029]")
    private fun space(c: Char) = Character.isWhitespace(c) || Character.isSpaceChar(c) || c == '\u0085'
    private fun strip(s: String) = s.trim(::space)
    private fun rstrip(s: String) = s.trimEnd(::space)
    private fun lines(s: String, keepEnds: Boolean = false): List<String> {
        val out = mutableListOf<String>()
        var at = 0
        for (m in breaks.findAll(s)) {
            out += s.substring(at, if (keepEnds) m.range.last + 1 else m.range.first)
            at = m.range.last + 1
        }
        if (at < s.length) out += s.substring(at)
        return out
    }
    private fun tagged(s: String) = tag.matches(strip(s))
    private fun frame(seconds: Double) = Math.rint(seconds / FRAME).toInt()
    private operator fun Fraction.minus(other: Fraction) = this + other * -1

    fun goOn(lyrics: String, text: String): Going {
        val fresh = lines(text).map(::rstrip).toMutableList()
        while (fresh.isNotEmpty() && strip(fresh.first()).isEmpty()) fresh.removeAt(0)
        while (fresh.isNotEmpty() && strip(fresh.last()).isEmpty()) fresh.removeAt(fresh.lastIndex)
        val joined = fresh.joinToString("\n")
        require(joined.codePointCount(0, joined.length) <= LONGEST_WORDS) {
            "The new words run to more than $LONGEST_WORDS characters. A song goes on by a few sections at a time."
        }
        if (fresh.isEmpty()) return Going(lyrics, emptyList(), emptyList(), false)
        val sung = fresh.filter { !tagged(it) && Phrasing.syllables(it) > 0 }.map(::strip)
        val lead = !tagged(fresh.first())
        val added = (if (lead) listOf(TAG) else emptyList()) + fresh.map(::strip)
        val ending = if ("\r\n" in lyrics) "\r\n" else "\n"
        val raw = lines(lyrics, true)
        var cut = raw.size
        while (cut > 0 && (strip(raw[cut - 1]).isEmpty() || tagged(raw[cut - 1]))) cut--
        val before = rstrip(raw.take(cut).joinToString(""))
        val after = strip(raw.drop(cut).joinToString(""))
        val result = before + (if (before.isNotEmpty()) ending.repeat(2) else "") + added.joinToString(ending) +
            (if (after.isNotEmpty()) ending.repeat(2) + after else "")
        return Going(result, added, sung, lead)
    }

    private data class Section(val bar: Int, val name: String)
    private class Sheet(val score: AbcScore.Score) {
        val vocal = score.voices.getValue("Vocal")
        val starts = vocal.bars.map { it.start } + vocal.time
        val sections = AbcScore.sections(score.text).map { e ->
            Section(vocal.bars.indexOfFirst { it.start == e.start }, e.name)
        }
    }
    private fun read(text: String): Sheet {
        val source = strip(text)
        require(source.isNotEmpty()) { "The song has no score." }
        require(source.codePointCount(0, source.length) <= 200000) { "That score is far longer than any song." }
        try {
            val score = AbcScore.parse(source)
            require(score.unit.denominator >= 4) { "L:1/${score.unit.denominator} is coarser than a quarter note" }
            return Sheet(score)
        } catch (e: IllegalArgumentException) {
            throw IllegalArgumentException("This score cannot be read note by note: ${reason(e)}.", e)
        }
    }
    private fun reason(e: IllegalArgumentException) = e.message.orEmpty().trim().trimEnd('.')

    fun lastSung(lyrics: String, score: String): Int? = lastSung(lyrics, read(score))
    private fun lastSung(lyrics: String, sheet: Sheet): Int? = Words.lastSung(lyrics, sheet.score)

    private data class Part(val head: List<String>, val music: String)
    private data class Block(val names: List<String>, val parts: Map<String, Part>)
    private fun blocks(bodies: List<String>): List<Block> {
        var cursor = 8
        return buildList {
            while (cursor < bodies.size) {
                val names = mutableListOf<String>()
                while (bodies[cursor].startsWith("% ")) names += strip(bodies[cursor++].substring(2))
                val parts = voices.associateWith {
                    val headers = mutableListOf(bodies[cursor++])
                    while (bodies[cursor].startsWith("M:") || bodies[cursor].startsWith("K:")) headers += bodies[cursor++]
                    Part(headers, bodies[cursor++])
                }
                add(Block(names, parts))
            }
        }
    }
    private fun restBars(piece: String) = fullRest.matchEntire(piece)?.let { it.groupValues[1].ifEmpty { "1" }.toInt() }
    private fun barCount(body: String) = body.dropLast(1).split('|').sumOf { restBars(it) ?: 1 }
    private fun restText(bars: Int) = "Z" + if (bars > 1) bars.toString() else ""
    private fun cutMusic(body: String, at: Int): Pair<String, String> {
        val head = mutableListOf<String>(); val tail = mutableListOf<String>()
        var seen = 0
        for (piece in body.dropLast(1).split('|')) {
            val bars = restBars(piece) ?: 1
            when {
                seen + bars <= at -> head += piece
                seen >= at -> tail += piece
                else -> { head += restText(at - seen); tail += restText(bars - (at - seen)) }
            }
            seen += bars
        }
        return (head.joinToString("|") + "|") to (tail.joinToString("|") + "|")
    }
    private fun splitBlock(block: Block, at: Int) = Block(block.names, block.parts.mapValues { (_, p) ->
        Part(p.head, cutMusic(p.music, at).first)
    })

    fun head(score: String, stop: Int, merged: Set<Int> = emptySet()): String {
        val sheet = read(score)
        val count = sheet.vocal.bars.size
        require(stop in 1..count) { "The score cannot end after bar $stop: it has $count bars." }
        val source = strip(score)
        val bodies = lines(source)
        val ending = if (source.startsWith(bodies.first() + "\r\n")) "\r\n" else
            breaks.find(source)?.value ?: "\n"
        try {
            var at = 0
            val result = buildString {
                bodies.take(8).forEach { append(it).append(ending) }
                for (original in blocks(bodies)) {
                    if (at >= stop) break
                    val bars = barCount(original.parts.getValue("Vocal").music)
                    val block = if (at + bars > stop) splitBlock(original, stop - at) else original
                    if (at !in merged) block.names.forEach { append("% ").append(it).append(ending) }
                    for (name in voices) {
                        val part = block.parts.getValue(name)
                        var music = part.music
                        if (at + bars >= stop && rstrip(music).endsWith("-|")) music = rstrip(music).dropLast(2) + "|"
                        part.head.forEach { append(it).append(ending) }
                        append(music).append(ending)
                    }
                    at += bars
                }
            }
            val after = AbcScore.parse(result.trim())
            for (name in voices) {
                val old = sheet.score.voices.getValue(name); val new = after.voices.getValue(name)
                val label = if (name == "Vocal") "sung" else "instrumental"
                require(new.bars.size == stop) { "the $label part has ${new.bars.size} bars where $stop should be left" }
                for (i in new.bars.indices) {
                    require(old.bars[i].duration == new.bars[i].duration && old.bars[i].meter == new.bars[i].meter) {
                        "bar ${i + 1} of the $label part changed its length"
                    }
                    fun key(v: AbcScore.Voice, bar: AbcScore.Bar) = v.keys.last { it.start <= bar.start }.name
                    require(key(old, old.bars[i]) == key(new, new.bars[i])) { "bar ${i + 1} of the $label part changed its key" }
                }
            }
            require(after.bpm == sheet.score.bpm && after.unit == sheet.score.unit) { "the tempo or the note length changed" }
            return result
        } catch (e: IllegalArgumentException) {
            throw IllegalArgumentException("The score could not be ended after bar $stop: ${reason(e)}.\n\n" +
                "Nothing was changed. The score is as it was.", e)
        }
    }

    private data class Phrase(val start: Fraction, val end: Fraction, val onsets: List<Fraction>)
    private data class Seam(val low: Double, val high: Double)
    private fun phrases(sheet: Sheet): List<Phrase> = buildList {
        for (n in sheet.vocal.notes.sortedBy { it.start }) {
            val end = n.start + n.duration
            val last = lastOrNull()
            if (last != null && n.start <= last.end)
                set(lastIndex, Phrase(last.start, maxOf(last.end, end), last.onsets + n.start))
            else add(Phrase(n.start, end, listOf(n.start)))
        }
    }
    private fun usualPickup(sheet: Sheet, runs: List<Phrase>, bar: Int): Fraction? {
        val section = sheet.sections.indexOfLast { it.bar <= bar }.coerceAtLeast(0)
        val first = sheet.starts[sheet.sections[section].bar]
        val last = sheet.starts[sheet.sections.getOrNull(section + 1)?.bar ?: sheet.vocal.bars.size]
        val leads = runs.mapNotNull { run ->
            if (run.start < first || run.start >= last) null else {
                val line = sheet.starts.firstOrNull { it >= run.start }
                when {
                    line == run.start -> Fraction(0)
                    line != null && run.end > line -> line - run.start
                    else -> null
                }
            }
        }.groupingBy { it }.eachCount()
        return leads.keys.minWithOrNull(compareByDescending<Fraction> { leads.getValue(it) }.thenBy { it })
    }
    private fun seamAt(sheet: Sheet, runs: List<Phrase>, bar: Int): Seam {
        val starts = sheet.starts
        if (bar <= 0 || bar >= starts.lastIndex) return Seam(0.0, 0.0)
        val down = starts[bar]
        val index = runs.indexOfFirst { it.end > down }.let { if (it < 0) runs.size else it }
        val before = runs.getOrNull(index - 1)?.end ?: starts.first()
        fun seam(low: Fraction, high: Fraction) = Seam((down - low).toDouble(), (down - high).toDouble())
        if (index == runs.size) return seam(starts.last(), before)
        val run = runs[index]
        if (run.start >= starts[bar - 1]) return seam(run.start, before)
        val following = runs.getOrNull(index + 1)
        if (down !in run.onsets && following != null && run.end <= starts[bar + 1]) return seam(following.start, run.end)
        val lead = usualPickup(sheet, runs, bar)
        if (lead == null) return seam(following?.start ?: starts.last(), run.end)
        return if (down - lead in run.onsets) Seam(lead.toDouble(), lead.toDouble()) else Seam(0.0, 0.0)
    }
    private fun opening(seam: Seam, margin: Double): Double {
        val room = seam.high - seam.low
        return if (room >= 2 * margin) minOf(maxOf(0.0, seam.low + minOf(maxOf(margin, 1.0), room / 2)), seam.high - margin)
        else seam.low + margin
    }
    private fun validClock(clock: Clock, bars: Int) {
        require(clock.lines.size == bars + 1 && clock.lines.all { it.isFinite() } &&
            clock.lines.zipWithNext().all { (a, b) -> b >= a } && clock.speed.isFinite() && clock.speed > 0) {
            "The song's clock does not match its score."
        }
    }
    private fun from(sheet: Sheet, clock: Clock, frames: Int): From {
        val count = sheet.vocal.bars.size
        validClock(clock, count)
        require(frames > 0) { "The song has no frames to go on from." }
        var bar = count
        sheet.vocal.notes.maxOfOrNull { it.start + it.duration }?.let { end ->
            bar = (1..count).firstOrNull { sheet.starts[it] >= end } ?: count
        }
        while (bar > 1 && clock.lines[bar] > frames * FRAME - SOUND_REACHED) bar--
        val start = if (bar < count) {
            val unit = clock.speed * 60.0 / sheet.score.bpm
            val before = opening(seamAt(sheet, phrases(sheet), bar), 0.08 / unit)
            val at = frame(clock.lines[bar] - before * unit).coerceIn(0, frames)
            require(at < frames) { "Bars ${bar + 1} to $count of the score are not a stretch of the song as it was sung." }
            at
        } else frame(clock.lines[count]).coerceIn(0, frames - 1)
        return From(bar, count, start)
    }

    fun goingOnFrom(score: String, clock: Clock, frames: Int): From? =
        if (strip(score).isEmpty() || clock.lines.isEmpty()) null else from(read(score), clock, frames)

    fun plan(lyrics: String, score: String, text: String, clock: Clock, frames: Int): Plan? {
        if (strip(score).isEmpty() || clock.lines.isEmpty()) return null
        val sheet = read(score)
        val from = from(sheet, clock, frames)
        val going = goOn(lyrics, text)
        val last = lastSung(lyrics, sheet)
        val merged = if (last == null) emptySet() else sheet.sections.filter { last < it.bar && it.bar < from.bar }.map { it.bar }.toSet()
        return Plan(from.bar, from.bars, from.start, going.text, head(score, from.bar, merged), going.tagged, going.sung)
    }

    fun extensionFrames(score: String, clock: Clock, frames: Int, plan: Plan, written: String): Int {
        val old = read(score)
        validClock(clock, old.vocal.bars.size)
        val next = read(written)
        require(plan.bar in 0..next.vocal.bars.size && plan.bars == old.vocal.bars.size) { "The extension does not match its score." }
        val length = (next.starts.last() - next.starts[plan.bar]).toDouble()
        val tail = maxOf(0, frames - frame(clock.lines[plan.bars]))
        return maxOf(1, frame(clock.lines[plan.bar]) - plan.start +
            frame(clock.speed * 60.0 / next.score.bpm * length) + tail)
    }

    fun linesAfter(clock: Clock, plan: Plan, written: String): Pair<List<Double>, List<Pair<Int, String>>> {
        validClock(clock, plan.bars)
        val sheet = read(written)
        require(plan.bar in 0..minOf(plan.bars, sheet.vocal.bars.size)) { "The extension does not match its score." }
        val line = clock.lines[plan.bar]
        val unit = clock.speed * 60.0 / sheet.score.bpm
        val lines = clock.lines.take(plan.bar + 1) + sheet.starts.drop(plan.bar + 1).map {
            line + (it - sheet.starts[plan.bar]).toDouble() * unit
        }
        return lines to sheet.sections.map { it.bar to it.name }
    }
}
