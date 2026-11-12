package xyz.tifu.geode.music

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/** A score as a MIDI file, read back: the tracks, the tempo, keys, sections, notes with their ties joined, chords. */
class ScoreMidiTest {
    private val awkward: String = JSONObject(javaClass.classLoader!!.getResource("abc_transpose.json")!!.readText())
        .getJSONObject("scores").getString("awkward")

    /** One event of a file read back: its track, tick, status byte (or 0xFF and the meta kind), data. */
    private data class Ev(val track: Int, val tick: Int, val status: Int, val kind: Int, val data: ByteArray)

    private fun read(b: ByteArray): Pair<Int, List<Ev>> {
        var at = 0
        fun u8() = b[at++].toInt() and 0xFF
        fun u16() = (u8() shl 8) or u8()
        fun u32() = (u16() shl 16) or u16()
        fun vlq(): Int {
            var v = 0
            while (true) {
                val c = u8()
                v = (v shl 7) or (c and 0x7F)
                if (c and 0x80 == 0) return v
            }
        }
        check(String(b, 0, 4) == "MThd")
        at = 8
        check(u16() == 1)
        val tracks = u16()
        val division = u16()
        assertEquals(ScoreMidi.DIVISION, division)
        val out = ArrayList<Ev>()
        for (t in 0 until tracks) {
            check(String(b, at, 4) == "MTrk")
            at += 4
            val end = u32() + at
            var tick = 0
            while (at < end) {
                tick += vlq()
                val status = u8()
                if (status == 0xFF) {
                    val kind = u8()
                    val n = vlq()
                    out += Ev(t, tick, status, kind, b.copyOfRange(at, at + n))
                    at += n
                } else {
                    val n = if (status and 0xF0 == 0xC0) 1 else 2
                    out += Ev(t, tick, status, 0, b.copyOfRange(at, at + n))
                    at += n
                }
            }
        }
        return tracks to out
    }

    @Test
    fun aScoreAsAMidiFile() {
        val (tracks, ev) = read(ScoreMidi.of(awkward))
        assertEquals(4, tracks)   // the conductor, Vocal, Ins, Chords
        val meta = ev.filter { it.track == 0 && it.status == 0xFF }
        // 90 BPM: 666667 microseconds a quarter
        val tempo = meta.single { it.kind == 0x51 }.data
        assertEquals(666_667, ((tempo[0].toInt() and 0xFF) shl 16) or ((tempo[1].toInt() and 0xFF) shl 8) or (tempo[2].toInt() and 0xFF))
        // D major at the start, B minor where the chorus starts (bar 4: 12 quarters in)
        assertEquals(listOf(0 to listOf(2, 0), 5760 to listOf(2, 1)),
            meta.filter { it.kind == 0x59 }.map { it.tick to it.data.map { b -> b.toInt() } })
        assertEquals(listOf(0 to "verse", 5760 to "chorus"), meta.filter { it.kind == 0x06 }.map { it.tick to String(it.data) })
        fun ons(track: Int) = ev.filter { it.track == track && it.status and 0xF0 == 0x90 }
        fun offs(track: Int) = ev.filter { it.track == track && it.status and 0xF0 == 0x80 }
        // the voice: 13 notes, ties joined (C# over the bar line, E tied to itself)
        assertEquals(13, ons(1).size)
        assertEquals(13, offs(1).size)
        val first = ons(1).first()
        assertEquals(0, first.tick)
        assertArrayEquals(byteArrayOf(66, 100), first.data)   // F sharp (the key's) on the voice's channel 1
        val cSharp = ons(1)[3]
        assertEquals(1440, cSharp.tick)
        assertEquals(73, cSharp.data[0].toInt())
        assertEquals(2400, offs(1).first { it.data[0].toInt() == 73 }.tick)
        // the piano: 4 notes on channel 2; the chords: 6 symbols, 21 notes on channel 3, the last held to the end
        assertEquals(4, ons(2).size)
        assertEquals(0x91, ons(2).first().status)
        assertEquals(21, ons(3).size)
        assertEquals(9600, offs(3).maxOf { it.tick })
        // every track ends at the score's end (5 bars of 4/4)
        assertEquals(List(4) { 9600 }, (0 until 4).map { t -> ev.last { it.track == t }.tick })
    }

    @Test
    fun chordSymbolsAsPitches() {
        assertEquals(listOf(50, 54, 57), ScoreMidi.chordPitches("D"))
        assertEquals(listOf(47, 55, 59, 62), ScoreMidi.chordPitches("G/B"))
        assertEquals(listOf(54, 58, 61, 64), ScoreMidi.chordPitches("F#7"))
        assertEquals(listOf(58, 61, 65, 69), ScoreMidi.chordPitches("Bbm(maj7)"))
    }

    @Test
    fun ticksRoundHalfToEven() {
        assertEquals(240, AbcScore.Fraction(1, 2).ticks(480))
        assertEquals(160, AbcScore.Fraction(1, 3).ticks(480))
        assertEquals(2, AbcScore.Fraction(5, 2).ticks(1))
        assertEquals(4, AbcScore.Fraction(7, 2).ticks(1))
    }
}
