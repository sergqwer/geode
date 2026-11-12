package xyz.tifu.geode.music

/** Cut planning ported from YuE2-ComfyUI's inpaint/track.py (Apache-2.0).
 * No model, tokenizer or Android dependencies. Bars and frames are zero-based, stops exclusive. */
object TrackEdit {
    sealed interface BreakNotice { data object Unmatched : BreakNotice }
    /** [seam] is an absolute time in quarter notes, rounded to the score's L: tick grid. */
    data class BreakPlan(val bar: Int, val bars: Int, val start: Int, val seam: AbcScore.Fraction,
                         val head: String, val lyrics: String, val notices: List<BreakNotice>)

    /** Phrase-aware insertion before an interior section; port of track._break (Apache-2.0).
     * [start] counts 25 Hz frames; [BreakPlan.seam] counts quarter notes, not reference L: ticks. */
    fun breakPlan(score: String, lyrics: String, grid: BarGrid, bar: Int, bars: Int, frames: Int): BreakPlan {
        require(score.isNotBlank()) { "The song has no score and no sections to put a break between." }
        val sheet = EditText.read(score)
        val timing = ScoreTiming(sheet)
        val count = timing.vocal.bars.size
        require(bars in 1..16) { "A break must have 1 to 16 whole bars." }
        require(bar > 0 && bar < count) {
            "A break goes between two sections: after the first bar of the song and before its end."
        }
        require(timing.sections.any { it.start == timing.starts[bar] }) {
            "Bar line ${bar + 1} is inside a section. A break goes between two sections, before the first bar of one."
        }
        ScoreEdit.breakKeys(score, bar)   // (refused here: every score written for it would be refused too)
        val lines = grid.bars
        require(lines.size == count + 1 && lines.all { it.isFinite() } &&
            lines.zipWithNext().all { (a, b) -> b >= a } && grid.speed.isFinite() && grid.speed > 0) {
            "The song's clock does not match its score."
        }
        val unit = grid.speed * 60.0 / sheet.bpm
        val opened = ScoreTiming.opening(timing.seam(bar), 0.08 / unit)
        val start = Math.rint((lines[bar] - opened * unit) * 25.0).toInt()
        require(start > 0 && start < frames) { "Bar line ${bar + 1} lies outside the song as it was sung." }
        val ticks = sheet.unit.denominator / 4
        val seam = AbcScore.Fraction(Math.rint((timing.starts[bar].toDouble() - opened) * ticks).toLong(), ticks)
        val words = Words.breakWords(lyrics, score, bar)
        return BreakPlan(bar, bars, start, seam, ScoreEdit.interludeHead(score, bar, seam), words.text,
            if (words.matched) emptyList() else listOf(BreakNotice.Unmatched))
    }

    /** Inserted bars at the grid's tempo/speed, at least one 25 Hz frame; port of track.break_frames (Apache-2.0). */
    fun breakFrames(plan: BreakPlan, newScore: String, grid: BarGrid): Int {
        val sheet = EditText.read(newScore)
        val voice = sheet.voices.getValue("Vocal")
        val starts = voice.bars.map { it.start } + voice.time
        require(plan.bar >= 0 && plan.bars > 0 && plan.bar + plan.bars < starts.size &&
            grid.speed.isFinite() && grid.speed > 0) { "The break is not a stretch of this score's bars." }
        val length = starts[plan.bar + plan.bars] + starts[plan.bar] * -1
        return maxOf(1, Math.rint(length.toDouble() * grid.speed * 60.0 / sheet.bpm * 25.0).toInt())
    }
    /** Separate from cut notices so existing consumers of Notice remain exhaustive. */
    sealed interface MoveNotice {
        data object Unmatched : MoveNotice
        data object Tagged : MoveNotice
        data class BareHead(val section: String) : MoveNotice
        data class BareTail(val section: String) : MoveNotice
    }
    data class MovePlan(val lyrics: String, val score: String, val pieces: List<Pair<Int, Int>>,
                        val sung: List<Pair<Int, Int>>, val grid: BarGrid, val notices: List<MoveNotice>)

    /** Whole-section move ported from YuE2-ComfyUI's track._move/_sung and grid.move_frames (Apache-2.0).
     * Uses the score's shared pickup shift; it does not require separated voice levels.
     * Only reached boundaries and positive-length frame pieces are accepted. */
    fun move(score: String, lyrics: String, grid: BarGrid, first: Int, stop: Int, to: Int, frames: Int): MovePlan {
        val sheet = EditText.read(score)
        val timing = ScoreTiming(sheet)
        val count = timing.vocal.bars.size
        val lines = grid.bars
        require(lines.size == count + 1 && lines.all { it.isFinite() } &&
            lines.zipWithNext().all { (a, b) -> b >= a } && grid.speed.isFinite() && grid.speed > 0) {
            "The song's clock does not match its score."
        }
        require(frames > 0) { "The song has no frames to move." }
        val ordered = ScoreEdit.order(first, stop, to, count)
        val named = timing.sections.associate { section ->
            timing.vocal.bars.indexOfFirst { it.start == section.start } to section.name
        }
        for (bar in listOf(first, stop, to)) {
            require(bar == count || bar in named) { "A move starts, ends and lands at section boundaries; bar ${bar + 1} is inside a section." }
            // The song's last bar may ring on, or stop short, past its closing line: it only has to start inside the sound.
            require(bar == 0 || if (bar == count) lines[bar - 1] < frames / 25.0 else lines[bar] > 0 && lines[bar] < frames / 25.0) {
                "Bar line ${bar + 1} lies outside the song as it was sung."
            }
        }
        val touched = ordered.flatMap { listOf(it.first, it.second) }.distinct().filter { it > 0 && it < count }
        val unit = grid.speed * 60.0 / sheet.bpm
        val margin = 0.08 / unit
        val shift = ScoreTiming.sharedShift(touched.map(timing::seam), margin)
        fun frame(bar: Int): Int {
            if (bar == 0) return 0
            if (bar == count) return frames
            val at = Math.rint((lines[bar] - shift * unit) * 25.0).toInt()
            require(at > 0 && at < frames) { "The opening at bar ${bar + 1} lies outside the song as it was sung." }
            return at
        }
        val stretches = mutableListOf<Pair<Int, Int>>()
        val pieces = mutableListOf<Pair<Int, Int>>()
        for ((a, b) in ordered) {
            val start = frame(a); val end = frame(b)
            require(start < end) { "Bars ${a + 1} to $b are not a stretch of the song as it was sung." }
            if (pieces.lastOrNull()?.second == start) {
                pieces[pieces.lastIndex] = pieces.last().first to end
                stretches[stretches.lastIndex] = stretches.last().first to b
            } else { pieces += start to end; stretches += a to b }
        }
        val sorted = pieces.sortedBy { it.first }
        require(sorted.first().first == 0 && sorted.last().second == frames &&
            sorted.zipWithNext().all { (a, b) -> a.second == b.first } && pieces.sumOf { it.second - it.first } == frames) {
            "The pieces of a move must hold every frame exactly once."
        }
        val moved = ScoreEdit.moved(score, first, stop, to)
        val clock = grid.afterMove(moved, stretches, pieces, shift * unit)
        val next = ScoreTiming(EditText.read(moved))
        val newLines = clock.bars
        val sung = mutableListOf<Pair<Int, Int>>()
        var at = 0; var bar = 0; var last = 0
        for ((i, piece) in pieces.withIndex()) {
            if (i > 0 && pieces[i - 1].second != piece.first && at > 0 && at < frames) {
                val opens = if (bar <= 1) 0 else Math.rint((newLines[bar - 1] -
                    ScoreTiming.opening(next.seam(bar - 1), margin) * unit) * 25.0).toInt()
                val low = maxOf(last, minOf(opens, at - 25))
                if (low < at) sung += low to at
                last = at
            }
            at += piece.second - piece.first
            bar += stretches[i].second - stretches[i].first
        }
        val words = Words.moveWords(lyrics, score, first, stop, to)
        val notices = mutableListOf<MoveNotice>()
        if (!words.matched) notices += MoveNotice.Unmatched
        if (words.tagged) notices += MoveNotice.Tagged
        if (pieces.first().first > 0) notices += MoveNotice.BareHead(named[stretches.first().first].orEmpty())
        if (pieces.last().second < frames) {
            val lastBar = stretches.last().second
            notices += MoveNotice.BareTail(named.entries.filter { it.key < lastBar }.maxByOrNull { it.key }?.value.orEmpty())
        }
        return MovePlan(words.text, moved, pieces.toList(), sung.toList(), clock, notices.toList())
    }

    sealed interface Notice {
        data class Dropped(val tags: List<String>) : Notice
        /** Extra bars removed at the sound's end, numbered from one, both ends inclusive for display. */
        data class Tail(val firstBar: Int, val lastBar: Int) : Notice
        data object Unmatched : Notice
    }
    data class CutPlan(val first: Int, val stop: Int, val firstBar: Int, val stopBar: Int,
                       val lyrics: String, val score: String, val grid: BarGrid, val notices: List<Notice>)

    /** Plan a score-aware cut, extending its score range through any bars past the sound's end. */
    fun cut(score: String, lyrics: String, grid: BarGrid, firstBar: Int, stopBar: Int,
            first: Int, stop: Int, frames: Int): CutPlan {
        val count = EditText.read(score).voices.getValue("Vocal").bars.size
        require(grid.bars.size == count + 1 && grid.bars.all { it.isFinite() } &&
            grid.bars.zipWithNext().all { (a, b) -> b >= a } && grid.speed.isFinite() && grid.speed > 0) {
            "The song's clock does not match its score."
        }
        require(firstBar >= 0 && firstBar < stopBar && stopBar <= count) {
            "Bars ${firstBar + 1} to $stopBar are not bars of this score, which has $count."
        }
        require(frames > 0 && first >= 0 && first < stop && stop <= frames) { "The cut is not a stretch of the song." }
        require(frames - (stop - first) >= 25) { "A cut leaves less than a second of the song." }
        val last = if (stop == frames) count else stopBar
        val notices = mutableListOf<Notice>()
        if (last > stopBar) notices += Notice.Tail(stopBar + 1, count)
        val next = ScoreEdit.without(score, firstBar, last)
        val words = Words.cutWords(lyrics, score, firstBar, last)
        if (words.dropped.isNotEmpty()) notices += Notice.Dropped(words.dropped)
        else if (!words.matched) notices += Notice.Unmatched
        return CutPlan(first, stop, firstBar, last, words.text, next,
            grid.cutBars(firstBar, last, first, stop, next), notices.toList())
    }
}
