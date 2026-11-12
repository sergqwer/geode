package xyz.tifu.geode.music.midi

import org.junit.Assert.assertEquals
import org.junit.Test
import xyz.tifu.geode.music.SungScore

class SungScoreLengthTest {
    @Test
    fun aBareTuneKeepsItsLength() {
        // 16 quarter notes at 120 BPM: 8 s; with the ceiling's margin 8 x 1.1 + 2 = 10.8 -> 11
        val bytes = MidiFixtures.file(MidiFixtures.P("Vocal", notes = MidiFixtures.line(List(16) { 60 + it % 8 })))
        val r = MidiImport.read(bytes, "melody")
        assertEquals(false, Phrasing.named(r.abc))
        assertEquals(11, SungScore.tuneSeconds(r.abc))
    }
}
