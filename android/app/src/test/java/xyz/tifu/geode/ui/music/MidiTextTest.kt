package xyz.tifu.geode.ui.music

import org.junit.Assert.assertEquals
import org.junit.Test

class MidiTextTest {
    @Test
    fun barsAsRuns() {
        assertEquals("1–3, 5", MidiText.bars(listOf("3", "1", "2", "5")))
        assertEquals("7", MidiText.bars(listOf("7", "7")))
        // (a range given already: as it is)
        assertEquals("1-4, 9", MidiText.bars(listOf("1-4", "9")))
        assertEquals("", MidiText.bars(emptyList()))
    }

    @Test
    fun aFileTooBigIsNotRead() {
        val big = ByteArray(MidiText.MAX_BYTES + 1)
        assertEquals(true, runCatching { MidiText.read(big.inputStream()) }.exceptionOrNull() is MidiText.TooBig)
        assertEquals(10, MidiText.read(ByteArray(10).inputStream()).size)
    }
}
