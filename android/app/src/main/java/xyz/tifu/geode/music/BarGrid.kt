package xyz.tifu.geode.music

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.withSign

/**
 * A sung song's waveform and pitch classes every 10 ms (the engine's song.view, yue2_view.hpp): what the track editor
 * draws, and what [BarGrid] places the score's bars with.
 */
class SongView(val hop: Int, val rate: Int, private val peaks: ByteArray, private val classes: ByteArray) {
    val hops: Int get() = peaks.size
    val hopSeconds: Double get() = hop.toDouble() / rate
    val seconds: Double get() = hops * hopSeconds

    /** The largest |sample| of hop [h], 0..1. */
    fun peak(h: Int): Float = (peaks[h].toInt() and 0xFF) / 255f

    /** Pitch class [c] (0 = C) of hop [h], the hop's 12 over their norm. */
    fun pitchClass(h: Int, c: Int): Double = (classes[h * 12 + c].toInt() and 0xFF) / 255.0

    companion object {
        /** [f] read; null: not a song's view (missing, cut short, foreign). */
        fun read(f: File, rate: Int = 48000): SongView? = runCatching {
            RandomAccessFile(f, "r").use { r ->
                val head = ByteArray(12)
                r.readFully(head)
                if (String(head, 0, 4, Charsets.US_ASCII) != "GYV1") return null
                val b = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN)
                val hops = b.getInt(4)
                val hop = b.getInt(8)
                if (hops < 0 || hops > (1 shl 24) || hop <= 0 || r.length() != 12L + hops.toLong() * 13) return null
                val peaks = ByteArray(hops)
                val classes = ByteArray(hops * 12)
                r.readFully(peaks)
                r.readFully(classes)
                SongView(hop, rate, peaks, classes)
            }
        }.getOrNull()
    }
}

/**
 * The score's clock laid on the song: a second of the song = [offset] + [rate] x the score's second - YuE2-ComfyUI's
 * inpaint/grid.py measured, without its voice separator (too big for a phone): the score's sung notes and chord tones
 * as pitch classes slid along the song's to find its start (10 s either way), then every section that sings slid
 * within a second of that and a line drawn through them, weighted by how much each sings, the score stretched by the
 * speed the line found - four passes. Right as long as the song follows its score's lengths; the reference misplaced
 * one rock song of seven this way, so the editor shows these bars as estimated.
 */
class BarGrid(
    val offset: Double,
    val rate: Double,
    /** Seconds of the score (its own tempo) at every bar line, and the score's end after the last. */
    private val scoreBars: List<Double>,
    /** Each section's first bar and label. */
    val sections: List<Pair<Int, String>>,
    /** Song seconds a second of the score lasts - the measured [rate], kept when an edit fixes the lines (a song going
     *  on lays its new bars at it). */
    val speed: Double = rate,
) {
    /** The song's second of every bar line, and of the score's end after the last. */
    val bars: List<Double> get() = scoreBars.map { offset + rate * it }

    /**
     * The bars after [first, stop) (frames, 25 a second) is cut out: the lines inside it gone, those after it moved up
     * by its length, the sections moved with their bars (one whose first bar went starts at the next) - the edited
     * song's own lines (YuE2-ComfyUI's grid "lines" once an edit moves them off the clock).
     */
    fun cut(first: Int, stop: Int): BarGrid {
        val a = first / 25.0
        val b = stop / 25.0
        val keep = bars.indices.filter { bars[it] < a || bars[it] >= b }
        val lines = keep.map { val s = bars[it]; if (s >= b) s - (b - a) else s }
        val sections = this.sections.mapNotNull { (bar, name) ->
            keep.indexOfFirst { it >= bar }.takeIf { it >= 0 }?.let { it to name }
        }.distinctBy { it.first }
        return fixed(lines, sections, speed)
    }

    /** Score-indexed cut, ported from YuE2-ComfyUI's inpaint/grid.py after_cut (Apache-2.0).
     * [score] is the new score; frame and bar stops are exclusive. Keep the seam's line even for a pickup. */
    fun cutBars(firstBar: Int, stopBar: Int, first: Int, stop: Int, score: String): BarGrid {
        val old = bars
        require(firstBar >= 0 && firstBar < stopBar && stopBar < old.size && first >= 0 && first < stop) {
            "The cut is not a stretch of the song's bars."
        }
        require(old.all { it.isFinite() } && old.zipWithNext().all { (a, b) -> b >= a } && speed.isFinite() && speed > 0) {
            "The song's clock does not match its score."
        }
        val sheet = EditText.read(score)
        val vocal = sheet.voices.getValue("Vocal")
        val lines = old.take(firstBar) + old.drop(stopBar).map { maxOf(it - (stop - first) / 25.0, first / 25.0) }
        require(lines.size == vocal.bars.size + 1) { "The song's clock does not match its score." }
        val sections = AbcScore.sections(sheet.text).map { s ->
            vocal.bars.indexOfFirst { it.start == s.start } to s.name
        }
        return fixed(lines, sections, speed)
    }

    /** A moved score's lines carried by its original bar stretches and frame pieces, before seam retakes.
     * Ported from YuE2-ComfyUI's inpaint/grid.py after_move (Apache-2.0). All stops are exclusive.
     * [lead]: the seconds each piece is cut ahead of its bar line. The song's first bar may begin before its sound
     * (a pickup the model never sang): moved behind another stretch, that bar starts where the stretch before ends. */
    fun afterMove(score: String, stretches: List<Pair<Int, Int>>, pieces: List<Pair<Int, Int>>, lead: Double = 0.0): BarGrid {
        val old = bars
        require(stretches.isNotEmpty() && stretches.size == pieces.size && old.size >= 2 &&
            old.all { it.isFinite() } && old.zipWithNext().all { (a, b) -> b >= a } && speed.isFinite() && speed > 0) {
            "The song's clock does not match its score."
        }
        fun partition(ranges: List<Pair<Int, Int>>, end: Int): Boolean {
            val sorted = ranges.sortedBy { it.first }
            return sorted.first().first == 0 && sorted.last().second == end &&
                sorted.all { (a, b) -> a >= 0 && a < b && b <= end } &&
                sorted.zipWithNext().all { (a, b) -> a.second == b.first }
        }
        require(partition(stretches, old.size - 1) && partition(pieces, pieces.maxOf { it.second })) {
            "The pieces of a move must hold every bar and frame exactly once."
        }
        val sheet = EditText.read(score)
        val vocal = sheet.voices.getValue("Vocal")
        require(vocal.bars.size == old.size - 1) { "The song's clock does not match its score." }
        val lines = mutableListOf<Double>()
        var at = 0
        var end = old.last()
        for ((stretch, piece) in stretches.zip(pieces)) {
            val (low, high) = stretch; val (start, stop) = piece
            val moved = (at - start) / 25.0
            for (bar in low until high) lines += if (bar == 0 && start == 0 && at > 0) {
                // Where the stretch before ends; at the join itself when the first bar sounds for less than [lead].
                val join = at / 25.0
                maxOf(old[0] + moved, if (join + lead < old[1] + moved) join + lead else join)
            } else old[bar] + moved
            end = old[high] + moved
            at += stop - start
        }
        lines += end
        require(lines.zipWithNext().all { (a, b) -> b >= a }) { "The moved bars do not keep their order in time." }
        val sections = AbcScore.sections(sheet.text).map { s -> vocal.bars.indexOfFirst { it.start == s.start } to s.name }
        return fixed(lines, sections, speed)
    }

    /** The grid of a retake of frames [first, stop) sung as [count] new frames: the bars before it as they were, the
     *  ones inside stretched over the new part, the ones after moved by its change of length. */
    fun retake(first: Int, stop: Int, count: Int): BarGrid {
        val a = first / 25.0
        val b = stop / 25.0
        val n = count / 25.0
        val lines = bars.map { s -> if (s < a) s else if (s >= b) s + n - (b - a) else a + (s - a) * n / (b - a) }
        return fixed(lines, sections, speed)
    }

    /** Proportional inserted lines, later lines shifted by actual 25 Hz frames.
     * Port of YuE2-ComfyUI's grid.after_break (Apache-2.0); the original boundary stays as the first new downbeat. */
    fun afterBreak(newScore: String, bar: Int, bars: Int, count: Int): BarGrid {
        val old = this.bars
        val sheet = EditText.read(newScore)
        val vocal = sheet.voices.getValue("Vocal")
        require(bar > 0 && bar < old.size - 1 && bars > 0 && count > 0 &&
            vocal.bars.size + 1 == old.size + bars && old.all { it.isFinite() } &&
            old.zipWithNext().all { (a, b) -> b >= a } && speed.isFinite() && speed > 0) {
            "The song's clock does not match its score."
        }
        val starts = vocal.bars.map { it.start.toDouble() } + vocal.time.toDouble()
        val span = starts[bar + bars] - starts[bar]
        val moved = count / 25.0
        val inside = (0 until bars).map { old[bar] + (starts[bar + it] - starts[bar]) * moved / span }
        val lines = old.take(bar) + inside + old.drop(bar).map { it + moved }
        val named = AbcScore.sections(sheet.text).map { s -> vocal.bars.indexOfFirst { it.start == s.start } to s.name }
        return fixed(lines, named, speed)
    }

    /** As JSON (kept in an edited song's job): the bar lines' seconds, the sections and the speed. */
    fun json(): org.json.JSONObject = org.json.JSONObject()
        .put("bars", org.json.JSONArray().apply { bars.forEach { put(it) } })
        .put("speed", speed)
        .put("sections", org.json.JSONArray().apply { sections.forEach { (bar, name) -> put(org.json.JSONArray().put(bar).put(name)) } })

    companion object {
        /** Bars at these seconds (an edited song's), no clock behind them. */
        fun fixed(lines: List<Double>, sections: List<Pair<Int, String>>, speed: Double = 1.0) = BarGrid(0.0, 1.0, lines, sections, speed)

        /** [json]'s grid back; null: none, or not one. */
        fun of(o: org.json.JSONObject?): BarGrid? = runCatching {
            val bars = o?.optJSONArray("bars") ?: return null
            if (bars.length() < 2) return null
            val lines = List(bars.length()) { bars.getDouble(it) }
            val s = o.optJSONArray("sections")
            val sections = if (s == null) emptyList() else List(s.length()) { s.getJSONArray(it).let { p -> p.getInt(0) to p.getString(1) } }
            fixed(lines, sections, o.optDouble("speed", 1.0).takeIf { it.isFinite() && it > 0 } ?: 1.0)
        }.getOrNull()

        const val HOP = 0.01
        private const val CHORD_WEIGHT = 0.5
        private const val SEARCH_SECONDS = 10.0
        private const val SECTION_SECONDS = 1.0
        private const val PASSES = 4
        private const val OUTLIER_SECONDS = 0.25
        private const val STEEPEST = 0.03

        /** [abc]'s bars on the song [view] shows; null: no score, or one that cannot be read. */
        fun measure(abc: String, view: SongView): BarGrid? {
            if (abc.isBlank() || view.hops < 2) return null
            val score = runCatching { AbcScore.parse(abc) }.getOrNull() ?: return null
            val vocal = score.voices["Vocal"] ?: return null
            val quarter = 60.0 / score.bpm   // (the score's seconds a quarter note)
            val end = vocal.time.toDouble()
            val notes = vocal.notes.map { Triple(it.start.toDouble(), it.duration.toDouble(), Math.floorMod(it.pitch, 12)) }
            val chords = vocal.chords.mapIndexedNotNull { i, c ->
                val stop = vocal.chords.getOrNull(i + 1)?.start?.toDouble() ?: end
                val tones = runCatching { ScoreMidi.chordPitches(c.name).map { Math.floorMod(it, 12) }.toSet() }.getOrNull()
                tones?.let { Triple(c.start.toDouble(), stop, it) }
            }
            val sectionStarts = runCatching { AbcScore.sections(abc) }.getOrNull().orEmpty()
            val barStarts = vocal.bars.map { it.start.toDouble() }
            val count = view.hops
            val signal = Array(12) { c -> DoubleArray(count) { h -> view.pitchClass(h, c) } }

            fun picture(rate: Double, width: Int): Array<DoubleArray> {
                val step = quarter * rate / HOP
                val out = Array(12) { DoubleArray(width) }
                for ((start, length, pc) in notes) {
                    val a = Math.rint(start * step).toInt()
                    val b = Math.rint((start + length) * step).toInt()
                    if (a < width) for (h in maxOf(0, a) until maxOf(0, minOf(width, b))) out[pc][h] += 1.0
                }
                for ((start, stop, tones) in chords) {
                    val a = Math.rint(start * step).toInt()
                    val b = Math.rint(stop * step).toInt()
                    for (t in tones) for (h in maxOf(0, a) until maxOf(0, minOf(width, b))) out[t][h] += CHORD_WEIGHT
                }
                return out
            }

            // each section: its span in quarters and how much it sings (grid.py sung_ticks)
            val spans = sectionStarts.mapIndexed { i, s ->
                val first = s.start.toDouble()
                val last = sectionStarts.getOrNull(i + 1)?.start?.toDouble() ?: end
                first to last
            }
            var weights = spans.map { (first, last) ->
                notes.sumOf { (s, l, _) -> if (s < last && s + l > first) minOf(s + l, last) - maxOf(s, first) else 0.0 }
            }
            if (weights.none { it > 0 }) weights = spans.map { 1.0 }

            val reach = Math.rint(SEARCH_SECONDS / HOP).toInt()
            val near = Math.rint(SECTION_SECONDS / HOP).toInt()
            val start = argmax(correlation(signal, picture(1.0, count), -reach, reach, 0)) - reach
            var offset = start * HOP
            var rate = 1.0
            repeat(PASSES) {
                val pic = picture(rate, count + reach)
                val perHop = quarter * rate / HOP
                val center = Math.rint(offset / HOP).toInt()
                val points = ArrayList<Triple<Double, Double, Double>>()
                spans.forEachIndexed { i, (first, last) ->
                    val a = Math.rint(first * perHop).toInt()
                    val b = Math.rint(last * perHop).toInt()
                    val cut = minOf(b, pic[0].size)   // (the kernel sliced as the picture allows; the midpoint the section's)
                    if (b <= a || cut <= a || weights[i] <= 0) return@forEachIndexed
                    val kernel = Array(12) { c -> pic[c].copyOfRange(a, cut) }
                    if (kernel.all { row -> row.all { it == 0.0 } }) return@forEachIndexed
                    val best = argmax(correlation(signal, kernel, center - near, center + near, a)) + center - near
                    points += Triple((a + b) / 2.0 * HOP, best * HOP, weights[i])
                }
                if (points.isEmpty()) return@repeat
                val (o, drift) = line(points)
                offset = o
                rate *= 1.0 + drift
            }
            val sections = sectionStarts.map { s -> barStarts.indexOfFirst { it >= s.start.toDouble() - 1e-9 }.coerceAtLeast(0) to s.name }
            return BarGrid(offset, rate, (barStarts + end).map { it * quarter }, sections)
        }

        private fun argmax(v: DoubleArray): Int {
            var best = 0
            for (i in v.indices) if (v[i] > v[best]) best = i
            return best
        }

        /** (offset, drift) of the weighted least-squares line through (score second, offset, weight), its outliers
         *  further than OUTLIER_SECONDS dropped while more than two are left, its slope held to STEEPEST (grid.py line). */
        fun line(points: List<Triple<Double, Double, Double>>): Pair<Double, Double> {
            fun fitted(chosen: List<Triple<Double, Double, Double>>): Pair<Double, Double> {
                val total = chosen.sumOf { it.third }
                val meanT = chosen.sumOf { it.first * it.third } / total
                val meanO = chosen.sumOf { it.second * it.third } / total
                val spread = chosen.sumOf { it.third * (it.first - meanT) * (it.first - meanT) }
                if (spread <= 1e-9) return meanO to 0.0
                var drift = chosen.sumOf { it.third * (it.first - meanT) * (it.second - meanO) } / spread
                if (abs(drift) > STEEPEST) drift = STEEPEST.withSign(drift)
                return (meanO - drift * meanT) to drift
            }
            var kept = points.filter { it.third > 0 }
            if (kept.isEmpty()) return 0.0 to 0.0
            var (offset, drift) = fitted(kept)
            while (kept.size > 2) {
                val worst = kept.maxBy { abs(it.second - (offset + drift * it.first)) }
                if (abs(worst.second - (offset + drift * worst.first)) <= OUTLIER_SECONDS) break
                kept = kept.filter { it !== worst }
                val f = fitted(kept)
                offset = f.first
                drift = f.second
            }
            return offset to drift
        }

        /**
         * For every shift s in [first, last]: the sum over the 12 rows and m of signal[row][start + s + m] x
         * kernel[row][m], the signal zero outside itself (grid.py _correlation), through FFTs.
         */
        fun correlation(signal: Array<DoubleArray>, kernel: Array<DoubleArray>, first: Int, last: Int, start: Int): DoubleArray {
            val width = kernel[0].size
            val low = start + first
            val span = last - first + width
            var n = 1
            while (n < span) n = n shl 1   // (no wrap: the last shift reads the window's last value)
            val accRe = DoubleArray(n)
            val accIm = DoubleArray(n)
            val wr = DoubleArray(n)
            val wi = DoubleArray(n)
            val kr = DoubleArray(n)
            val ki = DoubleArray(n)
            for (row in 0 until 12) {
                wr.fill(0.0); wi.fill(0.0); kr.fill(0.0); ki.fill(0.0)
                val s = signal[row]
                for (i in 0 until span) {
                    val at = low + i
                    if (at >= 0 && at < s.size) wr[i] = s[at]
                }
                System.arraycopy(kernel[row], 0, kr, 0, width)
                fft(wr, wi, false)
                fft(kr, ki, false)
                for (i in 0 until n) {   // w x conj(k)
                    accRe[i] += wr[i] * kr[i] + wi[i] * ki[i]
                    accIm[i] += wi[i] * kr[i] - wr[i] * ki[i]
                }
            }
            fft(accRe, accIm, true)
            return DoubleArray(last - first + 1) { accRe[it] / n }
        }

        /** In place, n a power of two; [inverse] without the 1 / n. */
        private fun fft(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
            val n = re.size
            var j = 0
            for (i in 1 until n) {
                var bit = n shr 1
                while (j and bit != 0) {
                    j = j xor bit
                    bit = bit shr 1
                }
                j = j xor bit
                if (i < j) {
                    var t = re[i]; re[i] = re[j]; re[j] = t
                    t = im[i]; im[i] = im[j]; im[j] = t
                }
            }
            var len = 2
            while (len <= n) {
                val ang = 2 * PI / len * (if (inverse) 1 else -1)
                val wRe = cos(ang)
                val wIm = sin(ang)
                var i = 0
                while (i < n) {
                    var cr = 1.0
                    var ci = 0.0
                    for (k in 0 until len / 2) {
                        val a = i + k
                        val b = a + len / 2
                        val vr = re[b] * cr - im[b] * ci
                        val vi = re[b] * ci + im[b] * cr
                        re[b] = re[a] - vr; im[b] = im[a] - vi
                        re[a] += vr; im[a] += vi
                        val nr = cr * wRe - ci * wIm
                        ci = cr * wIm + ci * wRe
                        cr = nr
                    }
                    i += len
                }
                len = len shl 1
            }
        }
    }
}
