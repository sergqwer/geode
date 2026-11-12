/** Exact score events and ABC rebuilding used by YuE2-ComfyUI's midi/score.py and phrasing.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction

internal operator fun Fraction.minus(b: Fraction) = this + b * -1
internal operator fun Fraction.div(n: Long) = Fraction(numerator, denominator * n)
internal fun Fraction.value() = numerator.toDouble() / denominator
internal val ZERO = Fraction(0)

data class TimedNote(val start: Fraction, val end: Fraction, val pitch: Int, val weight: Double = 1.0)
data class Harmony(val start: Fraction, var end: Fraction, val label: String)
data class ScoreBar(val start: Fraction, val end: Fraction, val numerator: Int, val denominator: Int)

/** Writes the native two voices in groups of at most four bars, splitting ties at event boundaries. */
internal object ScoreEvents {
    private val names = mapOf(
        7 to "B# C# C## D# D## E# F# F## G# G## A# B", 6 to "B# C# C## D# E E# F# F## G# G## A# B",
        5 to "B# C# C## D# E E# F# F## G# A A# B", 4 to "B# C# D D# E E# F# F## G# A A# B",
        3 to "B# C# D D# E E# F# G G# A A# B", 2 to "C C# D D# E E# F# G G# A A# B",
        1 to "C C# D D# E F F# G G# A A# B", 0 to "C C# D D# E F F# G G# A Bb B",
        -1 to "C C# D Eb E F F# G G# A Bb B", -2 to "C C# D Eb E F F# G Ab A Bb B",
        -3 to "C Db D Eb E F F# G Ab A Bb B", -4 to "C Db D Eb E F Gb G Ab A Bb B",
        -5 to "C Db D Eb E F Gb G Ab A Bb Cb", -6 to "C Db D Eb Fb F Gb G Ab A Bb Cb",
        -7 to "C Db D Eb Fb F Gb G Ab Bbb Bb Cb"
    ).mapValues { it.value.split(' ') }
    private val lengths = listOf(48L, 32L, 24L, 16L, 12L, 8L, 6L, 4L, 3L, 2L, 1L)

    fun build(bars: List<ScoreBar>, vocal: List<TimedNote>, instrument: List<TimedNote>, keys: List<Pair<Fraction, String>>,
              labels: List<Pair<Fraction, String>>, harmony: List<Harmony>, bpm: Int, unit: Int): String {
        if (bars.isEmpty()) fail(ErrorCode.NOTATION)
        fun keyAt(t: Fraction) = keys.lastOrNull { it.first <= t }?.second ?: "C"
        val first = bars.first()
        val out = mutableListOf("X:1", "T:", "M:${first.numerator}/${first.denominator}", "L:1/$unit", "Q:1/4=$bpm",
            "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"", "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"", "K:${keyAt(ZERO)}")
        // Changes and section labels start a new group; an inline key change remains inside its bar.
        var index = 0
        var previousMeter = first.numerator to first.denominator; var previousKey = keyAt(ZERO)
        while (index < bars.size) {
            val b = bars[index]; val meter = b.numerator to b.denominator; val key = keyAt(b.start)
            val label = labels.lastOrNull { it.first >= b.start && it.first < b.end }?.second
            if (label != null) out += "% $label"
            var stop = index + 1
            while (stop < minOf(bars.size, index + 4)) {
                val next = bars[stop]
                if (next.numerator to next.denominator != meter || keyAt(next.start) != key || labels.any { it.first >= next.start && it.first < next.end }) break
                stop++
            }
            for ((name, notes) in listOf("Vocal" to vocal, "Ins" to instrument)) {
                out += "V: $name"
                if (meter != previousMeter) out += "M:${meter.first}/${meter.second}"
                if (key != previousKey) out += "K:$key"
                out += (index until stop).joinToString("") { render(bars[it], notes, keys, if (name == "Vocal") harmony else emptyList(), unit, ::keyAt) + "|" }
            }
            previousMeter = meter; previousKey = key; index = stop
        }
        return out.joinToString("\n", postfix = "\n")
    }

    private fun render(bar: ScoreBar, notes: List<TimedNote>, keys: List<Pair<Fraction, String>>, harmony: List<Harmony>, unit: Int,
                       keyAt: (Fraction) -> String): String {
        val sounding = notes.filter { it.start < bar.end && it.end > bar.start }
        // abc_rebuild repeats a held chord at the next bar's start, even after equal spans merged.
        val chords = harmony.filter { it.start < bar.end && it.end > bar.start && it.label != "N" }
            .map { it.copy(start = maxOf(it.start, bar.start)) }
        val changes = keys.filter { it.first > bar.start && it.first < bar.end }
        if (sounding.isEmpty() && chords.isEmpty() && changes.isEmpty()) return "Z"
        val edges = (listOf(bar.start, bar.end) + sounding.flatMap { listOf(maxOf(bar.start, it.start), minOf(bar.end, it.end)) } +
            chords.map { it.start } + changes.map { it.first }).distinct().sorted()
        val local = mutableMapOf<Char, Int>(); var key = keyAt(bar.start)
        return buildString {
            for (i in 0 until edges.lastIndex) {
                val start = edges[i]; val end = edges[i + 1]
                changes.lastOrNull { it.first == start }?.let { key = it.second; local.clear(); append("[K:$key]") }
                chords.filter { it.start == start }.forEach { append('"'); append(MidiChords.symbol(it.label, key)); append('"') }
                val note = sounding.firstOrNull { it.start <= start && it.end > start }
                val count = (end - start) * unit.toLong() / 4
                if (count.denominator != 1L) fail(ErrorCode.NOTATION)
                var left = count.numerator
                while (left > 0) {
                    val length = lengths.first { it <= left }; left -= length
                    if (note == null) append('z') else {
                        val spelling = names.getValue(AbcScore.KEYS.getValue(key))[Math.floorMod(note.pitch, 12)]
                        val letter = spelling[0]; val alteration = spelling.count { it == '#' } - spelling.count { it == 'b' }
                        val current = local[letter] ?: AbcScore.keyAccidentals(key).getValue(letter)
                        if (current != alteration) {
                            append(when (alteration) { -2 -> "__"; -1 -> "_"; 0 -> "="; 1 -> "^"; else -> "^^" })
                            local[letter] = alteration
                        }
                        val octave = Math.floorDiv(note.pitch - AbcScore.NATURAL.getValue(letter) - alteration, 12)
                        if (octave >= 6) { append(letter.lowercaseChar()); append("'".repeat(octave - 6)) }
                        else { append(letter); append(",".repeat(5 - octave)) }
                    }
                    if (length != 1L) append(length)
                    if (note != null && (left > 0 || note.end > end)) append('-')
                }
            }
        }
    }
}
