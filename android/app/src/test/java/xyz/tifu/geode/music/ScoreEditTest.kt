package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test

/** Fixtures and expectations ported by reading YuE2-ComfyUI's notation.py (Apache-2.0).
 * The reference is never executed. */
class ScoreEditTest {
    @Test fun partialGroupPreservesChordsAndOtherGroups() {
        val opening = EditFixtures.group("intro", "Z4|", "Z4|")
        val verse = EditFixtures.group("verse", "\"C\"C16|D16|E16|F16|", "C16|D16|E16|F16|")
        val score = EditFixtures.header + opening + verse
        val want = EditFixtures.header + opening + EditFixtures.group("verse", "\"C\"C16|F16|", "C16|F16|")
        assertEquals(want, ScoreEdit.without(score, 5, 7))
    }

    @Test fun splitAndFoldCompressedRests() {
        val s = EditFixtures.header + EditFixtures.group("verse", "Z|Z2|C16|", "Z|Z2|G16|")
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "Z2|C16|", "Z2|G16|"), ScoreEdit.without(s, 1, 2))
        val four = EditFixtures.header + EditFixtures.group("intro", " Z4 |", "Z4|")
        assertEquals(EditFixtures.header + EditFixtures.group("intro", "Z2|", "Z2|"), ScoreEdit.without(four, 1, 3))
        val unicode = EditFixtures.header + EditFixtures.group("intro", "\u00a0Z4\u3000|", "Z4|")
        assertEquals(EditFixtures.header + EditFixtures.group("intro", "Z2|", "Z2|"), ScoreEdit.without(unicode, 1, 3))
    }

    @Test fun wholeGroupsLoseDeclarationsAndAllTheirComments() {
        val s = EditFixtures.header + "% intro\n" + EditFixtures.group("playing", "Z2|", "Z2|") +
            EditFixtures.group("verse", "C16|D16|", "G16|A16|")
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "C16|D16|", "G16|A16|"), ScoreEdit.without(s, 0, 2))
    }

    @Test fun cutsAcrossGroupsAndKeepsSurvivingComments() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C16|D16|", "G16|A16|") +
            EditFixtures.group("chorus", "E16|F16|", "B16|c16|")
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "C16|", "G16|") +
            EditFixtures.group("chorus", "F16|", "c16|"), ScoreEdit.without(s, 1, 3))
    }

    @Test fun outgoingTiesAreRemovedInBothVoicesEvenAcrossGroups() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C16-|", "G16-|") +
            EditFixtures.group("chorus", "C16|D16|", "G16|A16|")
        val got = ScoreEdit.without(s, 1, 2)
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "C16|", "G16|") +
            EditFixtures.group("chorus", "D16|", "A16|"), got)
        assertEquals(2, AbcScore.parse(got.trim()).voices.getValue("Vocal").notes.size)
    }

    @Test fun tiesWhollyOutsideTheCutStayWritten() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C16-|C16|D16|E16|", "G16-|G16|A16|B16|")
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "C16-|C16|E16|", "G16-|G16|B16|"),
            ScoreEdit.without(s, 2, 3))
    }

    @Test fun preservesOuterWhitespaceAndMixedLineEndings() {
        val head = EditFixtures.header.replace("\n", "\r\n")
        val group = "% verse\r\nV: Vocal\nC16|D16|\r\nV: Ins\r\nG16|A16|"
        val s = " \t\r\n" + head + group + "\r\n\t "
        assertEquals(" \t\r\n" + head + group.replace("C16|D16|", "C16|").replace("G16|A16|", "G16|") + "\r\n\t ",
            ScoreEdit.without(s, 1, 2))
        assertFalse(ScoreEdit.without((EditFixtures.header + EditFixtures.group("verse", "Z2|", "Z2|")).trimEnd(), 0, 1).endsWith("\n"))
    }

    @Test fun keepsTwoFourAndExactQuarterNoteLengths() {
        val short = EditFixtures.group("bridge", "C8|D8|", "G8|A8|", "M:2/4\nK:G\n")
        val s = EditFixtures.header + EditFixtures.group("verse", "Z2|", "Z2|") + short
        val want = EditFixtures.header + EditFixtures.group("verse", "Z|", "Z|") + short
        val got = ScoreEdit.without(s, 0, 1)
        assertEquals(want, got)
        val score = AbcScore.parse(got.trim())
        assertEquals(listOf(4.0, 2.0, 2.0), score.voices.getValue("Vocal").bars.map { it.duration.toDouble() })
        assertEquals(120, score.bpm); assertEquals(AbcScore.Fraction(1, 16), score.unit)
        val fractional = EditFixtures.header + EditFixtures.group("verse", "Z2|", "Z2|") +
            EditFixtures.group("bridge", "C6|D6|", "G6|A6|", "M:3/8\n")
        val bars = AbcScore.parse(ScoreEdit.without(fractional, 0, 1).trim()).voices.getValue("Vocal").bars
        assertEquals(listOf(AbcScore.Fraction(4), AbcScore.Fraction(3, 2), AbcScore.Fraction(3, 2)), bars.map { it.duration })
    }

    @Test fun refusesLostMeterOrKeyDeclarations() {
        for (field in listOf("M:2/4\n", "K:G\n")) {
            val music = if (field.startsWith("M:")) "C8|" else "C16|"
            val s = EditFixtures.header + EditFixtures.group("verse", "Z|", "Z|") +
                EditFixtures.group("bridge", music, music, field) + EditFixtures.group("chorus", music, music)
            EditFixtures.error("The bars could not be cut out of the score:") { ScoreEdit.without(s, 1, 2) }
        }
    }

    @Test fun declarationCanGoWhenTheFollowingGroupRestoresIt() {
        val s = EditFixtures.header + EditFixtures.group("verse", "Z|", "Z|") +
            EditFixtures.group("bridge", "C8|", "G8|", "M:2/4\nK:G\n") +
            EditFixtures.group("chorus", "C16|", "G16|", "M:4/4\nK:C\n")
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "Z|", "Z|") +
            EditFixtures.group("chorus", "C16|", "G16|", "M:4/4\nK:C\n"), ScoreEdit.without(s, 1, 2))
    }

    @Test fun refusesInlineKeyChangeEvenIfItWouldReadBack() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C8[K:G]D8|E16|", "G8[K:G]A8|B16|")
        EditFixtures.error("Bar 1 changes key halfway through") { ScoreEdit.without(s, 0, 1) }
    }

    @Test fun invalidRangesAndUnreadableScoresFail() {
        val s = EditFixtures.header + EditFixtures.group("verse", "Z2|", "Z2|")
        for ((a, b) in listOf(-1 to 1, 0 to 0, 1 to 3)) EditFixtures.error("Bars ") { ScoreEdit.without(s, a, b) }
        EditFixtures.error("That is every bar") { ScoreEdit.without(s, 0, 2) }
        EditFixtures.error("The song has no score.") { ScoreEdit.without("", 0, 1) }
        EditFixtures.error("This score cannot be read note by note:") { ScoreEdit.without("X:1", 0, 1) }
    }
}

internal object EditFixtures {
    val header = "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n" +
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n" +
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\n"
    fun group(name: String, vocal: String, ins: String = "Z|".repeat(vocal.count { it == '|' }), fields: String = "") =
        (if (name.isEmpty()) "" else "% $name\n") + "V: Vocal\n$fields$vocal\nV: Ins\n$fields$ins\n"
    fun error(prefix: String, body: () -> Unit) {
        try { body(); fail("Expected IllegalArgumentException: $prefix") }
        catch (e: IllegalArgumentException) { assertTrue("Expected '$prefix', got '${e.message}'", e.message.orEmpty().startsWith(prefix)) }
    }
}
