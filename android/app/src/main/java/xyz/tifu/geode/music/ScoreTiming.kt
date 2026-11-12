package xyz.tifu.geode.music

import xyz.tifu.geode.music.AbcScore.Fraction

/** Phrase-aware seams ported from YuE2-ComfyUI's inpaint/grid.py (Apache-2.0).
 * No audio/model dependencies. Every score distance here is in quarter notes. */
internal class ScoreTiming(val score: AbcScore.Score) {
    val vocal = score.voices.getValue("Vocal")
    val starts = vocal.bars.map { it.start } + vocal.time
    val sections = AbcScore.sections(score.text)
    private data class Phrase(val start: Fraction, val end: Fraction, val onsets: List<Fraction>)
    data class Seam(val low: Double, val high: Double)
    private operator fun Fraction.minus(other: Fraction) = this + other * -1
    private val phrases = buildList<Phrase> {
        for (n in vocal.notes.sortedBy { it.start }) {
            val end = n.start + n.duration
            val last = lastOrNull()
            if (last != null && n.start <= last.end)
                set(lastIndex, Phrase(last.start, maxOf(last.end, end), last.onsets + n.start))
            else add(Phrase(n.start, end, listOf(n.start)))
        }
    }
    private fun usualPickup(bar: Int): Fraction? {
        val section = sections.indexOfLast { it.start <= starts[bar] }
        val first = sections.getOrNull(section)?.start ?: starts.first()
        val last = sections.getOrNull(section + 1)?.start ?: starts.last()
        val leads = phrases.mapNotNull { run ->
            if (run.start < first || run.start >= last) null else {
                val line = starts.firstOrNull { it >= run.start }
                when {
                    line == run.start -> Fraction(0)
                    line != null && run.end > line -> line - run.start
                    else -> null
                }
            }
        }.groupingBy { it }.eachCount()
        return leads.keys.minWithOrNull(compareByDescending<Fraction> { leads.getValue(it) }.thenBy { it })
    }
    fun seam(bar: Int): Seam {
        if (bar <= 0 || bar >= starts.lastIndex) return Seam(0.0, 0.0)
        val down = starts[bar]
        val index = phrases.indexOfFirst { it.end > down }.let { if (it < 0) phrases.size else it }
        val before = phrases.getOrNull(index - 1)?.end ?: starts.first()
        fun at(low: Fraction, high: Fraction) = Seam((down - low).toDouble(), (down - high).toDouble())
        if (index == phrases.size) return at(starts.last(), before)
        val run = phrases[index]
        if (run.start >= starts[bar - 1]) return at(run.start, before)
        val following = phrases.getOrNull(index + 1)
        if (down !in run.onsets && following != null && run.end <= starts[bar + 1]) return at(following.start, run.end)
        val lead = usualPickup(bar)
        if (lead == null) return at(following?.start ?: starts.last(), run.end)
        return if (down - lead in run.onsets) Seam(lead.toDouble(), lead.toDouble()) else Seam(0.0, 0.0)
    }
    companion object {
        fun opening(seam: Seam, margin: Double): Double {
            val room = seam.high - seam.low
            return if (room >= 2 * margin)
                minOf(maxOf(0.0, seam.low + minOf(maxOf(margin, 1.0), room / 2)), seam.high - margin)
            else seam.low + margin
        }
        fun sharedShift(seams: List<Seam>, margin: Double): Double {
            if (seams.isEmpty()) return 0.0
            val low = seams.maxOf { it.low }; val high = seams.minOf { it.high }
            return if (high - low >= 2 * margin)
                minOf(maxOf(0.0, low + minOf(maxOf(margin, 1.0), (high - low) / 2)), high - margin)
            else low + margin
        }
    }
}
