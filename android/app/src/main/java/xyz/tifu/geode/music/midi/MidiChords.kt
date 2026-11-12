/** Harmony read and guessed from held notes, ported from YuE2-ComfyUI's midi/chords.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
import kotlin.math.sqrt

object MidiChords {
    private val sharp = "C C# D D# E F F# G G# A A# B".split(' ')
    private val flat = "C Db D Eb E F Gb G Ab A Bb B".split(' ')
    val qualities = linkedMapOf("maj" to "", "min" to "m", "dim" to "dim", "aug" to "aug", "7" to "7", "maj7" to "maj7",
        "min7" to "m7", "dim7" to "dim7", "hdim7" to "m7b5", "sus4" to "sus4", "sus2" to "sus2", "maj6" to "6",
        "min6" to "m6", "sus4(b7)" to "7sus4", "minmaj7" to "m(maj7)")
    private val steps = listOf(listOf(0,4,7), listOf(0,3,7), listOf(0,3,6), listOf(0,4,8), listOf(0,4,7,10), listOf(0,4,7,11),
        listOf(0,3,7,10), listOf(0,3,6,9), listOf(0,3,6,10), listOf(0,5,7), listOf(0,2,7), listOf(0,4,7,9), listOf(0,3,7,9), listOf(0,5,7,10), listOf(0,3,7,11))
    private val shapes = qualities.keys.zip(steps).toMap()
    private val fifthSteps = listOf(listOf(0,4,1), listOf(0,-3,1), listOf(0,-3,-6), listOf(0,4,8), listOf(0,4,1,-2), listOf(0,4,1,5),
        listOf(0,-3,1,-2), listOf(0,-3,-6,-9), listOf(0,-3,-6,-2), listOf(0,-1,1), listOf(0,2,1), listOf(0,4,1,3), listOf(0,-3,1,3), listOf(0,-1,1,-2), listOf(0,-3,1,5))
    private val degrees = listOf("1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7")
    fun label(weights: List<Double>, bass: Int, flats: Boolean = false, scale: Set<Int>? = null): String {
        val total = weights.sum()
        if (total <= 0 || weights.count { it >= 0.1 * total } < 2) return "N"
        val w = weights.toMutableList(); w[bass] += 0.35 * total
        val norm = sqrt(w.sumOf { it * it }); var best = "N"; var score = Double.NEGATIVE_INFINITY
        for (root in 0..11) for ((q, cost) in listOf("maj" to 0.0, "min" to 0.0, "7" to 0.03, "maj7" to 0.03, "min7" to 0.03, "dim" to 0.03, "sus4" to 0.05)) {
            val shape = shapes.getValue(q)
            val value = shape.sumOf { w[(root + it) % 12] } / (norm * sqrt(shape.size.toDouble())) - cost +
                (if (root == bass) 0.05 else 0.0) + (if (scale != null && shape.all { (root + it) % 12 in scale }) 0.02 else 0.0)
            if (value > score) { score = value; best = (if (flats) flat else sharp)[root] + ":$q" }
        }
        return best
    }
    fun spans(bars: List<ScoreBar>): List<Pair<Fraction, Fraction>> = bars.flatMap {
        if (it.numerator >= 4 && it.numerator % 2 == 0) { val middle = (it.start + it.end) / 2; listOf(it.start to middle, middle to it.end) }
        else listOf(it.start to it.end)
    }
    private fun merge(rows: MutableList<Harmony>, start: Fraction, end: Fraction, chord: String) {
        if (rows.lastOrNull()?.let { it.label == chord && it.end == start } == true) rows.last().end = end
        else rows += Harmony(start, end, chord)
    }
    fun guess(notes: List<TimedNote>, pieces: List<Pair<Fraction, Fraction>>, flats: Boolean = false, scale: Set<Int>? = null): List<Harmony> {
        val rows = mutableListOf<Harmony>()
        for ((start, end) in pieces) {
            val sounding = notes.filter { it.start < end && it.end > start }
            val weights = MutableList(12) { 0.0 }
            for (n in sounding) weights[n.pitch % 12] += (minOf(end, n.end) - maxOf(start, n.start)).value() * n.weight
            val together = sounding.any { a -> sounding.any { b -> a.pitch % 12 != b.pitch % 12 && maxOf(start, a.start, b.start) < minOf(end, a.end, b.end) } }
            merge(rows, start, end, if (together) label(weights, sounding.minOf { it.pitch } % 12, flats, scale) else "N")
        }
        return rows
    }
    private fun shape(classes: Set<Int>, prefer: List<Int>): Pair<Int, String>? {
        val found = shapes.flatMap { (q, shape) -> (0..11).filter { r -> shape.map { (r + it) % 12 }.toSet() == classes }.map { it to q } }
        for (root in prefer) found.firstOrNull { it.first == root }?.let { return it }
        return found.firstOrNull()
    }
    fun spelled(input: List<Int>): String? {
        val pitches = input.distinct().sorted(); val classes = pitches.map { it % 12 }.toSet()
        if (classes.size < 3) return null
        val bass = pitches[0]; val upper = pitches.drop(1); val above = upper.map { it % 12 }.toSet()
        val tries = mutableListOf<Pair<Set<Int>, List<Int>>>()
        if (bass < 48 && upper[0] >= 48) tries += above to listOf(upper[0] % 12, bass % 12)
        tries += classes to listOf(bass % 12, upper[0] % 12)
        if (bass % 12 !in above) tries += above to listOf(upper[0] % 12)
        for ((wanted, prefer) in tries) {
            val (root, quality) = shape(wanted, prefer) ?: continue
            val step = Math.floorMod(bass - root, 12)
            val degree = when {
                quality in listOf("dim", "hdim7", "dim7") && step == 6 -> "b5"
                quality == "dim7" && step == 9 -> "bb7"
                quality == "aug" && step == 8 -> "#5"
                else -> degrees[step]
            }
            return sharp[root] + ":$quality" + if (step == 0) "" else "/$degree"
        }
        return null
    }
    fun read(notes: List<TimedNote>, end: Fraction, scale: Set<Int>? = null): Pair<List<Harmony>, List<Fraction>> {
        val onsets = notes.map { it.start }.distinct().sorted(); val rows = mutableListOf<Harmony>(); val unnamed = mutableListOf<Fraction>()
        for ((i, onset) in onsets.withIndex()) {
            val stop = onsets.getOrNull(i + 1) ?: maxOf(end, onset + Fraction(1))
            val sounding = notes.filter { it.start == onset || (it.start < onset && onset < it.end && (minOf(it.end, stop) - onset) * 2 >= stop - onset) }.map { it.pitch }
            val chord = spelled(sounding) ?: run {
                unnamed += onset
                label((0..11).map { pc -> sounding.count { it % 12 == pc }.toDouble() }, sounding.minOrNull()?.rem(12) ?: 0, scale = scale)
            }
            merge(rows, onset, stop, chord)
        }
        return rows to unnamed
    }
    private fun pc(name: String) = Math.floorMod(AbcScore.NATURAL.getValue(name[0]) + name.count { it == '#' } - name.count { it == 'b' }, 12)

    /** A model label becomes an ABC symbol, with the root spelled by its chord tones' distance from the key. */
    fun symbol(label: String, key: String? = null): String {
        if (label == "N") return ""
        val root = label.substringBefore(':'); val descriptor = label.substringAfter(':'); val quality = descriptor.substringBefore('/')
        val bass = descriptor.substringAfter('/', "")
        val signature = if (key == null) 0 else AbcScore.KEYS.getValue(key); val fifths = listOf(0,2,4,-1,1,3,5)
        val letters = "CDEFGAB"; val tones = fifthSteps[qualities.keys.indexOf(quality)]
        var chosen = root; var best = Int.MAX_VALUE
        for ((i, letter) in letters.withIndex()) {
            if (key == null) break
            val alter = Math.floorMod(pc(root) - AbcScore.NATURAL.getValue(letter) + 6, 12) - 6
            val place = fifths[i] + 7 * alter
            fun distance(p: Int) = maxOf(0, signature - 1 - p, p - signature - 5)
            val cost = tones.sumOf { distance(place + it) } + distance(place)
            if (cost < best) { best = cost; chosen = letter + if (alter >= 0) "#".repeat(alter) else "b".repeat(-alter) }
        }
        if (chosen.drop(1).length > 2) chosen = root
        var suffix = ""
        if (bass.isNotEmpty()) {
            if (bass[0] in 'A'..'G') suffix = "/$bass" else {
                val degree = bass.filter { it.isDigit() }.toInt(); val alteration = bass.count { it == '#' } - bass.count { it == 'b' }
                val target = (pc(chosen) + listOf(0,2,4,5,7,9,11)[(degree - 1) % 7] + alteration) % 12
                val letter = letters[(letters.indexOf(chosen[0]) + degree - 1) % 7]
                val a = Math.floorMod(target - AbcScore.NATURAL.getValue(letter) + 6, 12) - 6
                suffix = "/" + letter + if (a >= 0) "#".repeat(a) else "b".repeat(-a)
            }
        }
        return chosen + qualities.getValue(quality) + suffix
    }
}
