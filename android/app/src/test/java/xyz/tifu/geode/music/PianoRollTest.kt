package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.AbcScore.Fraction

/** The piano roll's write-back (ScoreEdit.withNotes): one part's notes replaced, the rest of the score kept. */
class PianoRollTest {
    private val score = EditFixtures.header + EditFixtures.group("verse", "\"C\"C8D8|E16|", "G16|A16|") +
        EditFixtures.group("chorus", "F8G8|Z|", "c16|d16|")
    private fun notes(text: String, part: String) = AbcScore.parse(text.trim()).voices.getValue(part).notes

    @Test fun unchangedNotesGiveTheSameScore() {
        for (part in listOf("Vocal", "Ins")) assertEquals(score, ScoreEdit.withNotes(score, part, notes(score, part)))
    }

    @Test fun aMovedNoteIsWrittenAndTheChordStays() {
        val moved = notes(score, "Vocal").map { if (it.pitch == 62) AbcScore.Note(it.start, 64, it.duration) else it }
        val out = ScoreEdit.withNotes(score, "Vocal", moved)
        assertEquals(moved.map { Triple(it.start, it.pitch, it.duration) }, notes(out, "Vocal").map { Triple(it.start, it.pitch, it.duration) })
        assertTrue(out.contains("\"C\"C8E8|"))
        assertEquals(notes(score, "Ins").map { it.pitch }, notes(out, "Ins").map { it.pitch })
    }

    @Test fun aNoteAcrossTheBarLineIsTied() {
        val long = listOf(AbcScore.Note(Fraction(2), 67, Fraction(4)))
        val out = ScoreEdit.withNotes(score, "Ins", long)
        val back = notes(out, "Ins")
        assertEquals(1, back.size)
        assertEquals(Fraction(4), back.single().duration)
        assertTrue(out.contains("-|"))
    }

    @Test fun overlappingNotesAreRefused() {
        val two = listOf(AbcScore.Note(Fraction(0), 60, Fraction(2)), AbcScore.Note(Fraction(1), 62, Fraction(1)))
        EditFixtures.error("A part sings one note at a time") { ScoreEdit.withNotes(score, "Vocal", two) }
    }
}
