package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test

/** Fixtures and expectations ported by reading YuE2-ComfyUI's inpaint/ops.py (Apache-2.0).
 * The reference is never executed. */
class WordsTest {
    private val score = EditFixtures.header + EditFixtures.group("intro", "C16|") +
        EditFixtures.group("verse", "C4D4E4F4|G16|") + EditFixtures.group("chorus", "A8B8|c8d8|") +
        EditFixtures.group("outro", "Z|", "Z|")
    private val lyrics = "[Intro]\n\n[Verse]\none two\n\n[Chorus]\nthree four\n\n[Outro]"

    @Test fun strictlyMoreThanHalfDropsWholeBlock() {
        val expected = "[Intro]\n\n[Chorus]\nthree four\n\n[Outro]"
        assertEquals(Words.Cut(expected, listOf("Verse"), true), Words.cutWords(lyrics, score, 1, 2))
        assertEquals(Words.Cut(lyrics, emptyList(), true), Words.cutWords(lyrics, score, 3, 4)) // two of four
        assertEquals(Words.Cut(lyrics, emptyList(), true), Words.cutWords(lyrics, score, 5, 6)) // no onsets
    }

    @Test fun aliasesAndNamesStepOverSingingIntro() {
        val words = lyrics.replace("[Chorus]", "[Hook 2]")
        assertEquals(listOf("Hook 2"), Words.cutWords(words, score, 3, 5).dropped)
        assertEquals(listOf("Verse", "Chorus"), Words.cutWords(lyrics, score, 1, 5).dropped)
    }

    @Test fun fallbackRequiresExactlyAsManySoundingSections() {
        val three = "[Bridge]\na\n[Bridge]\nb\n[Bridge]\nc"
        assertEquals(Words.Cut("[Bridge]\na\n[Bridge]\nc", listOf("Bridge"), true), Words.cutWords(three, score, 1, 3))
        val unmatched = " \r\n[Bridge]\na\r\n[Bridge]\nb \n"
        assertEquals(Words.Cut(unmatched, emptyList(), false), Words.cutWords(unmatched, score, 1, 3))
    }

    @Test fun repeatedAndEmptyTagsKeepTheirOriginalBlocks() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C16|") + EditFixtures.group("verse", "D16|")
        val l = "[Verse]\none\n\n[Outro]\n\n[Verse]\ntwo\n"
        assertEquals(Words.Cut("[Outro]\n\n[Verse]\ntwo", listOf("Verse"), true), Words.cutWords(l, s, 0, 1))
        assertEquals(listOf("Verse"), Words.cutWords(l, s, 1, 2).dropped)
    }

    @Test fun untaggedOpeningAndUnicodeSpacesPairAsVerse() {
        val s = EditFixtures.header + EditFixtures.group("  VERSE\u00a0 ", "C16|") + EditFixtures.group("outro", "Z|", "Z|")
        assertEquals(Words.Cut("[Outro]", listOf(""), true), Words.cutWords("hello\n\n[Outro]", s, 0, 1))
    }

    @Test fun tieContinuationIsNotAnotherOnset() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C16-|C16|D16|")
        val l = "[Verse]\nhello"
        assertEquals(Words.Cut(l, emptyList(), true), Words.cutWords(l, s, 0, 1)) // one of two, not a majority
        assertEquals(Words.Cut(l, emptyList(), true), Words.cutWords(l, s, 1, 2)) // zero onsets
        assertEquals(Words.Cut("", listOf("Verse"), true), Words.cutWords(l, s, 0, 3))
    }

    @Test fun survivingBlocksKeepTheirCharactersAndEndings() {
        val l = "[Verse]\r\none two\r\n\r\n[Chorus]\r\n  three four  \r\n\r\n[Outro]\r\n"
        assertEquals("[Chorus]\r\n  three four  \r\n\r\n[Outro]", Words.cutWords(l, score, 1, 3).text)
        assertEquals(Words.Cut("", emptyList(), true), Words.cutWords("", score, 0, 1))
        EditFixtures.error("Bars ") { Words.cutWords(l, score, 2, 99) }
    }

    @Test fun rewritesOriginalLineIndicesAndKeepsIndentation() {
        val l = "[Verse]\none\n\ntwo\n[Outro]\n"
        val r = Words.changeWords(l, 1, 4, "\n  red  \n\nblue\n\n")
        assertEquals("[Verse]\n  red\n\nblue\n[Outro]\n", r.text)
        assertEquals(listOf("one", "", "two"), r.before)
        assertEquals(listOf("  red", "", "blue"), r.after)
        assertEquals("Verse", r.tag)
        assertEquals(2, r.oldSyllables); assertEquals(2, r.newSyllables)
    }

    @Test fun rewriteKeepsCrLfAndTheReplacedRangesFinalEnding() {
        val l = "[Verse]\r\none\r\ntwo\n[Outro]\r\n"
        assertEquals("[Verse]\r\nred\r\nblue\r\ngreen\n[Outro]\r\n", Words.changeWords(l, 1, 3, "red\nblue\ngreen").text)
        // No selected line has an ending here: the reference defaults inserted separators to LF.
        assertEquals("[Verse]\r\nred\nblue", Words.changeWords("[Verse]\r\none", 1, 2, "red\nblue\n").text)
        assertEquals("red\nblue", Words.changeWords("one", 0, 1, "red\nblue").text)
        assertEquals("", Words.changeWords("one", 0, 1, "two").tag)
    }

    @Test fun pythonLineSeparatorsAndNoPhantomTrailingLine() {
        // The replacement's own separators are discarded: both inserted lines use the selected line's CRLF.
        assertEquals("[Verse]\u2028red\r\nblue\r\ntwo\n", Words.changeWords("[Verse]\u2028one\r\ntwo\n", 1, 2, "red\u0085blue").text)
        EditFixtures.error("Lines 3 to 3 are not lines") { Words.changeWords("one\ntwo\n", 2, 3, "three") }
    }

    @Test fun rejectsTagsSilenceAndInvalidRanges() {
        val l = "[Verse]\none\n\n[Outro]"
        EditFixtures.error("A section tag is not a line") { Words.changeWords(l, 1, 4, "two") }
        EditFixtures.error("Nothing is sung") { Words.changeWords(l, 2, 3, "two") }
        EditFixtures.error("New words cannot name a section") { Words.changeWords(l, 1, 2, "two\n[Bridge]") }
        for (text in listOf("", " \n ", "123 ...")) EditFixtures.error("The new words say nothing") { Words.changeWords(l, 1, 2, text) }
        for ((a, b) in listOf(-1 to 1, 1 to 1, 1 to 8)) EditFixtures.error("Lines ") { Words.changeWords(l, a, b, "two") }
    }

    @Test fun countsUnicodeSyllablesAndFitThresholds() {
        val r = Words.changeWords("[Verse]\nhello", 1, 2, "你好\nмама")
        assertEquals(2, r.oldSyllables); assertEquals(4, r.newSyllables)
        assertTrue(Words.needsFitNotice(r))
        fun notice(old: Int, new: Int) = Words.needsFitNotice(Words.Rewrite("", emptyList(), emptyList(), "", old, new))
        assertFalse(notice(4, 5)); assertFalse(notice(8, 10)); assertFalse(notice(8, 6))
        assertTrue(notice(7, 9)); assertTrue(notice(7, 5)); assertFalse(notice(0, 1)); assertTrue(notice(0, 2))
    }

    @Test fun unicodeWhitespaceInsideTagsIsNotPartOfTheirNames() {
        val words = "[\u00a0 Verse\u3000]\none"
        assertEquals("Verse", Words.changeWords(words, 1, 2, "two").tag)
        assertEquals(listOf("Verse"), Words.cutWords(words, score, 1, 3).dropped)
    }

    @Test fun twoFourSectionsCountNoteOnsetsInQuarterUnits() {
        val s = EditFixtures.header + EditFixtures.group("verse", "Z|", "Z|") +
            EditFixtures.group("bridge", "C4D4|E8|", "G4A4|B8|", "M:2/4\n")
        assertEquals(Words.Cut("", listOf("Bridge"), true), Words.cutWords("[Bridge]\nhello", s, 1, 2))
    }
}
