package xyz.tifu.geode.music

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt
import xyz.tifu.geode.tokenizer.Tokenizer

class BarGridTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private fun score(name: String): String {
        val text = javaClass.classLoader!!.getResource("abc_transpose.json")!!.readText()
        return JSONObject(text).getJSONObject("scores").getString(name)
    }

    /** A view whose pitch classes are [abc]'s notes and chords, its clock at [offset] + [rate] x the score's, with
     *  a little noise; written as the engine writes song.view. */
    private fun viewOf(abc: String, offset: Double, rate: Double, seconds: Double): File {
        val s = AbcScore.parse(abc)
        val vocal = s.voices.getValue("Vocal")
        val quarter = 60.0 / s.bpm
        val hops = (seconds / 0.01).toInt()
        val pc = Array(hops) { DoubleArray(12) }
        fun lay(start: Double, stop: Double, tone: Int, w: Double) {
            val a = Math.rint((offset + rate * start * quarter) / 0.01).toInt()
            val b = Math.rint((offset + rate * stop * quarter) / 0.01).toInt()
            for (h in maxOf(0, a) until minOf(hops, b)) pc[h][tone] += w
        }
        for (n in vocal.notes) lay(n.start.toDouble(), (n.start + n.duration).toDouble(), Math.floorMod(n.pitch, 12), 1.0)
        vocal.chords.forEachIndexed { i, c ->
            val stop = vocal.chords.getOrNull(i + 1)?.start?.toDouble() ?: vocal.time.toDouble()
            for (t in ScoreMidi.chordPitches(c.name).map { Math.floorMod(it, 12) }.toSet()) lay(c.start.toDouble(), stop, t, 0.5)
        }
        var x = 99L
        val f = tmp.newFile()
        val b = ByteBuffer.allocate(12 + hops * 13).order(ByteOrder.LITTLE_ENDIAN)
        b.put("GYV1".toByteArray()).putInt(hops).putInt(480)
        for (h in 0 until hops) b.put(100.toByte())
        for (h in 0 until hops) {
            val v = DoubleArray(12) {
                x = x * 6364136223846793005L + 1442695040888963407L
                pc[h][it] + 0.15 * ((x ushr 11).toDouble() / (1L shl 53))
            }
            val norm = sqrt(v.sumOf { it * it }).coerceAtLeast(1e-9)
            for (c in 0 until 12) b.put(Math.rint(v[c] / norm * 255).toInt().toByte())
        }
        f.writeBytes(b.array())
        return f
    }

    /** The song's length on its clock: the score's end, and a second more. */
    private fun lengthOf(abc: String, offset: Double, rate: Double): Double {
        val s = AbcScore.parse(abc)
        return offset + rate * s.voices.getValue("Vocal").time.toDouble() * 60.0 / s.bpm + 1.0
    }

    /** One section that sings: where it sits (the line through one point is flat, as in grid.py: no tempo from it). */
    @Test
    fun aSongOffItsScoreIsFound() {
        val abc = score("pop, cot full, seed 831001")
        val view = SongView.read(viewOf(abc, 2.37, 1.0, lengthOf(abc, 2.37, 1.0)))!!
        val g = BarGrid.measure(abc, view)!!
        assertEquals(2.37, g.offset, 0.03)
        assertEquals(1.0, g.rate, 0.004)
        assertEquals(listOf("intro", "verse"), g.sections.map { it.second })
        assertEquals(listOf(0, 5), g.sections.map { it.first })
    }

    /** Eleven sections: the start and the tempo both, a song sung a few percent off its score's (grid.py's passes). */
    @Test
    fun aSongOffItsScoresTempoIsFound() {
        val abc = score("pop rock with seven sections, cot full, seed 7")
        for ((offset, rate) in listOf(2.37 to 1.0, -0.8 to 1.02, 5.1 to 0.975, 0.0 to 1.01)) {
            val view = SongView.read(viewOf(abc, offset, rate, lengthOf(abc, offset, rate)))
            assertNotNull(view)
            val g = BarGrid.measure(abc, view!!)
            assertNotNull(g)
            assertTrue("offset ${g!!.offset} for $offset", abs(g.offset - offset) < 0.03)
            assertTrue("rate ${g.rate} for $rate", abs(g.rate - rate) < 0.004)
            // the bars where the song's clock puts them
            val quarter = 60.0 / AbcScore.parse(abc).bpm
            val bars = AbcScore.parse(abc).voices.getValue("Vocal").bars
            for ((i, bar) in bars.withIndex()) assertEquals(offset + rate * bar.start.toDouble() * quarter, g.bars[i], 0.05)
            assertEquals(11, g.sections.size)
        }
    }

    /** A cut song's bars: the cut ones gone, the rest moved up, the sections with them; kept as JSON and read back. */
    @Test
    fun aCutSongsBars() {
        val g = BarGrid.fixed(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 10.0), listOf(0 to "intro", 2 to "verse", 4 to "chorus"))
        val c = g.cut(100, 150)   // (4 s to 6 s: the bar from 4 s out)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0), c.bars)
        assertEquals(listOf(0 to "intro", 2 to "verse", 3 to "chorus"), c.sections)
        val back = BarGrid.of(c.json())!!
        assertEquals(c.bars, back.bars)
        assertEquals(c.sections, back.sections)
        // a trim at the start: everything moves up
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0), g.cut(0, 100).bars)
        assertNull(BarGrid.of(null))
        assertNull(BarGrid.of(JSONObject().put("bars", org.json.JSONArray().put(1.0))))
    }

    /** A retake's bars: the ones before it as they were, the ones inside stretched over the new part, the ones after
     *  moved by its change of length; the sections where they were. */
    @Test
    fun aRetakesBars() {
        val g = BarGrid.fixed(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 10.0), listOf(0 to "intro", 2 to "verse"))
        // bars 2-3 (2 s to 6 s, 100 frames) sung as 110 frames: 0.4 s longer
        val r = g.retake(50, 150, 110)
        val want = listOf(0.0, 2.0, 2.0 + 2.0 * 4.4 / 4.0, 6.4, 8.4, 10.4)
        assertEquals(want.size, r.bars.size)
        for (i in want.indices) assertEquals(want[i], r.bars[i], 1e-9)
        assertEquals(g.sections, r.sections)
        // the same length: the same bars; shorter at the end: nothing after it moves
        assertEquals(g.bars, g.retake(50, 150, 100).bars)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 9.5), g.retake(200, 300, 75).bars.map { Math.round(it * 1e9) / 1e9 })
    }

    @Test
    fun noScoreNoGrid() {
        val view = SongView.read(viewOf(score("pop, cot full, seed 831001"), 0.0, 1.0, 5.0))!!
        assertNull(BarGrid.measure("", view))
        assertNull(BarGrid.measure("X:1\nnot a score", view))
    }

    @Test
    fun aBrokenViewIsNone() {
        val f = tmp.newFile()
        f.writeBytes(byteArrayOf(1, 2, 3))
        assertNull(SongView.read(f))
        assertNull(SongView.read(File(tmp.root, "none.view")))
    }

    /**
     * A song the Redmi sang (work/tok-yue2/edit_song_result.json: its score's ids; edit_song.view: its view, as the
     * engine makes it): its bars land inside the song, its tempo within a few percent of the score's. Skipped
     * without them.
     */
    @Test
    fun aPhoneSongsBars() {
        val dir = File(System.getProperty("geode.yue2tokdir") ?: "")
        val bin = File(dir, Tokenizer.FILE)
        val result = File(dir, "edit_song_result.json")
        val viewFile = File(dir, "edit_song.view")
        assumeTrue("work/tok-yue2: tokenizer.bin, edit_song_result.json, edit_song.view", bin.isFile && result.isFile && viewFile.isFile)
        val ids = JSONObject(result.readText()).getJSONArray("abc").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val abc = Tokenizer.load(bin).decode(ids).trim()
        val view = SongView.read(viewFile)!!
        val g = BarGrid.measure(abc, view)
        assertNotNull(g)
        println("phone song: offset ${"%.3f".format(g!!.offset)} s, rate ${"%.4f".format(g.rate)}, ${g.bars.size - 1} bars, " +
            "sections ${g.sections}, bars at ${g.bars.take(6).map { "%.2f".format(it) }}")
        assertTrue(abs(g.rate - 1) <= 0.12)
        assertTrue(g.offset > -10.5 && g.offset < 10.5)
    }
}
