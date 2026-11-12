package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test

/** Fixtures worked out by reading YuE2-ComfyUI's notation.py and inpaint/{ops,grid,track}.py
 * (Apache-2.0). The reference is never executed. */
class MoveTest {
    private val header = EditFixtures.header
    private fun group(name: String, vocal: String, ins: String, fields: String = "") = EditFixtures.group(name, vocal, ins, fields)
    private val a = group("verse", "C16|D16|", "G16|A16|")
    private val b = group("chorus", "E16|F16|", "B16|c16|")
    private val c = group("bridge", "G16|A16|", "d16|e16|")
    private val d = group("outro", "B16|c16|", "f16|g16|")
    private val score = header + a + b + c + d
    private val silent = header + listOf("verse", "chorus", "bridge", "outro").joinToString("") { group(it, "Z2|", "Z2|") }
    private val grid = BarGrid.fixed(List(9) { 2.0 * it }, listOf(0 to "verse", 2 to "chorus", 4 to "bridge", 6 to "outro"))
    private val lyrics = "[Verse]\none\n\n[Chorus]\ntwo\n\n[Bridge]\nthree\n\n[Outro]"

    @Test fun originalIndexOrderInBothDirectionsAndAtEnds() {
        assertEquals(listOf(0 to 2, 4 to 6, 2 to 4, 6 to 8), ScoreEdit.order(2, 4, 6, 8))
        assertEquals(listOf(0 to 2, 4 to 6, 2 to 4, 6 to 8), ScoreEdit.order(4, 6, 2, 8))
        assertEquals(listOf(2 to 8, 0 to 2), ScoreEdit.order(0, 2, 8, 8))
        assertEquals(listOf(6 to 8, 0 to 6), ScoreEdit.order(6, 8, 0, 8))
        assertEquals(header + a + c + b + d, ScoreEdit.moved(score, 2, 4, 6))
        assertEquals(header + a + c + b + d, ScoreEdit.moved(score, 4, 6, 2))
        assertEquals(header + b + c + d + a, ScoreEdit.moved(score, 0, 2, 8))
        assertEquals(header + d + a + b + c, ScoreEdit.moved(score, 6, 8, 0))
    }

    @Test fun invalidRangesAndNoOpDestinationsAreRefused() {
        for ((first, stop, to) in listOf(Triple(-1, 2, 4), Triple(2, 2, 4), Triple(2, 9, 0)))
            EditFixtures.error("Bars ") { ScoreEdit.moved(score, first, stop, to) }
        for (to in listOf(2, 3, 4)) EditFixtures.error("Bars 3 to 4 would be moved") { ScoreEdit.moved(score, 2, 4, to) }
        for (to in listOf(-1, 9)) EditFixtures.error("There is no bar line") { ScoreEdit.moved(score, 2, 4, to) }
    }

    @Test fun anUnnamedOpeningStaysFirst() {
        val s = header + "V: Vocal\nC16|D16|\nV: Ins\nG16|A16|\n" + b + c
        EditFixtures.error("The song's opening has no section name") { ScoreEdit.moved(s, 0, 2, 6) }
        EditFixtures.error("The song's opening has no section name") { ScoreEdit.moved(s, 2, 4, 0) }
        assertEquals(header + "V: Vocal\nC16|D16|\nV: Ins\nG16|A16|\n" + c + b, ScoreEdit.moved(s, 2, 4, 6))
    }

    @Test fun splitsCompressedRestsAndRestatesSectionNames() {
        val s = header + group("intro", "Z4|", "Z4|") + group("verse", "C16|D16|", "G16|A16|")
        val got = ScoreEdit.moved(s, 1, 3, 6)
        val want = header + group("intro", "Z|", "Z|") + group("intro", "Z|", "Z|") +
            group("verse", "C16|D16|", "G16|A16|") + group("intro", "Z2|", "Z2|")
        assertEquals(want, got)
        for (name in listOf("Vocal", "Ins")) assertEquals(6, AbcScore.parse(got.trim()).voices.getValue(name).bars.size)
    }

    @Test fun tiesAtEveryDisplacedBoundaryAreRemovedInBothVoices() {
        val s = header + group("verse", "C16-|C16-|", "G16-|G16-|") +
            group("chorus", "C16-|C16-|", "G16-|G16-|") + group("outro", "C16|D16|", "G16|A16|")
        val got = ScoreEdit.moved(s, 2, 4, 6)
        assertEquals(header + group("verse", "C16-|C16|", "G16-|G16|") +
            group("outro", "C16|D16|", "G16|A16|") + group("chorus", "C16-|C16|", "G16-|G16|"), got)
        assertEquals(4, AbcScore.parse(got.trim()).voices.getValue("Vocal").notes.size)
    }

    @Test fun meterAndKeyAreRestoredWhenGroupsChangePredecessors() {
        val short = group("bridge", "C8|D8|", "G8|A8|", "M:2/4\nK:G\n")
        val later = group("chorus", "E8|F8|", "B8|c8|")
        val s = header + a + short + later
        val got = ScoreEdit.moved(s, 2, 4, 6)
        val want = header + a + group("chorus", "E8|F8|", "B8|c8|", "K:G\nM:2/4\n") + short
        assertEquals(want, got)
        val original = AbcScore.parse(s.trim()); val after = AbcScore.parse(got.trim())
        val indices = listOf(0, 1, 4, 5, 2, 3)
        for (name in listOf("Vocal", "Ins")) {
            val old = original.voices.getValue(name); val new = after.voices.getValue(name)
            assertEquals(indices.map { old.bars[it].duration }, new.bars.map { it.duration })
            assertEquals(indices.map { i -> old.keys.last { it.start <= old.bars[i].start }.name },
                new.bars.map { bar -> new.keys.last { it.start <= bar.start }.name })
        }
        // Moving the first section past G must explicitly restore C and 4/4 there.
        val firstAtEnd = ScoreEdit.moved(s, 0, 2, 6)
        assertTrue(firstAtEnd.contains("% verse\nV: Vocal\nK:C\nM:4/4\nC16|D16|"))
    }

    @Test fun inlineKeysInTheReorderedIntervalAreRefused() {
        val intro = group("intro", "C8[K:G]D8|E16|", "G8[K:G]A8|B16|")
        val s = header + intro + b + c
        EditFixtures.error("Bar 1 changes key halfway through") { ScoreEdit.moved(s, 0, 2, 6) }
        // The unchanged intro may keep an inline change while later sections move.
        assertEquals(6, AbcScore.parse(ScoreEdit.moved(s, 2, 4, 6).trim()).voices.getValue("Vocal").bars.size)
    }

    @Test fun chordsOuterWhitespaceAndCrLfSurvive() {
        val s = " \r\n" + (header + a.replace("C16|", "\"C\"C16|") + b + c + d).replace("\n", "\r\n") + "\t "
        val want = " \r\n" + (header + a.replace("C16|", "\"C\"C16|") + c + b + d).replace("\n", "\r\n") + "\t "
        assertEquals(want, ScoreEdit.moved(s, 2, 4, 6))
        assertFalse(ScoreEdit.moved(score.trimEnd(), 2, 4, 6).endsWith("\n"))
    }

    @Test fun lyricsMoveByNameAndAliasInBothDirections() {
        val want = "[Verse]\none\n\n[Bridge]\nthree\n\n[Chorus]\ntwo\n\n[Outro]"
        assertEquals(Words.Moving(want, listOf("Chorus"), true), Words.moveWords(lyrics, score, 2, 4, 6))
        assertEquals(Words.Moving(want, listOf("Bridge"), true), Words.moveWords(lyrics, score, 4, 6, 2))
        assertEquals(want.replace("[Chorus]", "[Hook 2]"), Words.moveWords(lyrics.replace("[Chorus]", "[Hook 2]"), score, 2, 4, 6).text)
        val end = Words.moveWords(lyrics, score, 0, 2, 8)
        assertTrue(end.text.endsWith("[Verse]\none\n\n[Outro]"))
    }

    @Test fun repeatedLabelsCountFallbackAndUnmatchedLyrics() {
        val s = header + a + b.replace("% chorus", "% verse") + c + d
        val l = "[Verse]\none\n[Verse]\ntwo\n[Bridge]\nthree\n[Outro]\nfour"
        assertEquals("[Verse]\none\n[Bridge]\nthree\n[Verse]\ntwo\n[Outro]\nfour", Words.moveWords(l, s, 2, 4, 6).text)
        val four = "[Bridge]\none\n[Bridge]\ntwo\n[Bridge]\nthree\n[Bridge]\nfour"
        assertEquals("[Bridge]\none\n[Bridge]\nthree\n[Bridge]\ntwo\n[Bridge]\nfour", Words.moveWords(four, score, 2, 4, 6).text)
        val unmatched = " [Pre-Chorus]\nhello\r\n"
        assertEquals(Words.Moving(unmatched, emptyList(), false), Words.moveWords(unmatched, score, 2, 4, 6))
    }

    @Test fun untaggedSingingGetsATagWhenItMovesBehindAnotherSection() {
        val l = "one\n\n[Chorus]\ntwo\n\n[Bridge]\nthree\n\n[Outro]"
        val r = Words.moveWords(l, score, 0, 2, 6)
        assertEquals("[Chorus]\ntwo\n\n[Bridge]\nthree\n\n[Verse]\none\n\n[Outro]", r.text)
        assertEquals(listOf(""), r.moved); assertTrue(r.tagged); assertTrue(r.matched)
    }

    @Test fun blankGapsStayAtPositionsAndEmptyTagsDoNotSing() {
        val l = "[Verse]\r\n one \r\n\r\n\r\n[Chorus]\r\n two \r\n\r\n[Bridge]\r\n three \r\n[Outro]\r\n"
        val expected = "[Verse]\r\n one \r\n\r\n\r\n[Bridge]\r\n three \r\n\r\n[Chorus]\r\n two \r\n[Outro]\r\n"
        assertEquals(expected, Words.moveWords(l, score, 2, 4, 6).text)
        assertEquals(Words.Moving("[Intro]\n[Outro]", emptyList(), true), Words.moveWords("[Intro]\n[Outro]", score, 2, 4, 6))
    }

    @Test fun planPiecesAndSungAreInAssembledCoordinates() {
        val p = TrackEdit.move(silent, lyrics, grid, 2, 4, 6, 400)
        assertEquals(listOf(0 to 100, 200 to 300, 100 to 200, 300 to 400), p.pieces)
        assertEquals(listOf(50 to 100, 150 to 200, 250 to 300), p.sung)
        assertEquals(grid.bars, p.grid.bars)
        assertEquals(listOf(0 to "verse", 2 to "bridge", 4 to "chorus", 6 to "outro"), p.grid.sections)
        assertEquals(TrackEdit.move(silent, lyrics, grid, 4, 6, 2, 400).pieces, p.pieces)
        assertContract(p, 400)
    }

    @Test fun firstSectionAndEndMovesKeepTailFrames() {
        val p = TrackEdit.move(silent, "", grid, 0, 2, 8, 425)
        assertEquals(listOf(100 to 425, 0 to 100), p.pieces)
        assertEquals(listOf(250 to 325), p.sung)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0), p.grid.bars)
        assertTrue(p.notices.any { it is TrackEdit.MoveNotice.BareHead })
        assertTrue(p.notices.any { it is TrackEdit.MoveNotice.BareTail })
        assertContract(p, 425)
        val q = TrackEdit.move(silent, "", grid, 6, 8, 0, 400)
        assertEquals(listOf(300 to 400, 0 to 300), q.pieces)
        assertEquals(listOf(50 to 100), q.sung)
        assertContract(q, 400)
    }

    @Test fun pickupShiftIsSharedAcrossEveryTouchedBoundary() {
        val s = header + group("verse", "Z|z8C8-|", "Z2|") + group("chorus", "C8z8|z8D8-|", "Z2|") +
            group("bridge", "D8z8|z8E8-|", "Z2|") + group("outro", "E8z8|Z|", "Z2|")
        val p = TrackEdit.move(s, "", grid, 2, 4, 6, 400)
        // Each boundary has a two-quarter pickup and room for another quarter: 1.5 seconds early.
        assertEquals(listOf(0 to 62, 162 to 262, 62 to 162, 262 to 400), p.pieces)
        assertContract(p, 400)
    }

    @Test fun shortPiecesNeverBorrowSingingFromThePreviousPiece() {
        val h = header.replace("M:4/4", "M:1/4")
        val s = h + listOf("verse", "chorus", "bridge", "outro").joinToString("") { group(it, "Z|", "Z|") }
        val g = BarGrid.fixed(List(5) { it * 0.5 }, emptyList())
        val p = TrackEdit.move(s, "", g, 1, 2, 3, 50)
        assertEquals(listOf(0 to 12, 25 to 38, 12 to 25, 38 to 50), p.pieces)
        assertEquals(listOf(0 to 12, 12 to 25, 25 to 38), p.sung)
        assertContract(p, 50)
    }

    @Test fun variableMeterOffsetAndSpeedArePreserved() {
        val s = header + group("verse", "Z2|", "Z2|") + group("bridge", "Z2|", "Z2|", "M:2/4\n") +
            group("chorus", "Z2|", "Z2|")
        val g = BarGrid.fixed(listOf(-0.2, 1.8, 3.8, 4.8, 5.8, 6.8, 7.8), emptyList(), 1.2)
        val p = TrackEdit.move(s, "", g, 2, 4, 6, 220)
        assertEquals(1.2, p.grid.speed, 0.0)
        assertEquals(7, p.grid.bars.size)
        assertEquals(-0.2, p.grid.bars.first(), 1e-9)
        assertContract(p, 220)
    }

    @Test fun unsungPickupAndRingingEndKeepTheMovedClockInOrder() {
        // The model began 2.5 s into bar 1 and stopped 0.5 s before the last line, as YuE2 songs often do.
        val g = BarGrid.fixed(listOf(-2.5, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.5), grid.sections)
        val p = TrackEdit.move(silent, "", g, 0, 2, 6, 400)
        assertEquals(listOf(100 to 300, 0 to 100, 300 to 400), p.pieces)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.5), p.grid.bars)
        assertContract(p, 400)
        val q = TrackEdit.move(silent, "", g, 6, 8, 0, 400)
        assertEquals(listOf(300 to 400, 0 to 300), q.pieces)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0), q.grid.bars)
        assertContract(q, 400)
    }

    @Test fun aMovedFirstBarStartsWhereTheStretchBeforeItEnds() {
        val s = header + group("verse", "Z|z8C8-|", "Z2|") + group("chorus", "C8z8|z8D8-|", "Z2|") +
            group("bridge", "D8z8|z8E8-|", "Z2|") + group("outro", "E8z8|Z|", "Z2|")
        for ((second, early) in listOf(3.0 to true, 1.0 to false)) {
            // Bar 1 sounds for [second] seconds; every piece is cut 1.5 s ahead of its bar line.
            val g = BarGrid.fixed(listOf(-2.0, second, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0), grid.sections)
            val p = TrackEdit.move(s, "", g, 0, 2, 6, 400)
            val join = (p.pieces[0].second - p.pieces[0].first) / 25.0
            assertEquals(1.5, 4.0 - p.pieces[0].first / 25.0, 0.03)
            assertEquals(if (early) join + 1.5 else join, p.grid.bars[4], 0.03)
            assertEquals(join + second, p.grid.bars[5], 1e-9)
            assertContract(p, 400)
        }
    }

    @Test fun audioMoveRequiresWholeReachedSectionsAndMatchingClock() {
        EditFixtures.error("A move starts, ends and lands") { TrackEdit.move(score, lyrics, grid, 1, 2, 6, 400) }
        EditFixtures.error("A move starts, ends and lands") { TrackEdit.move(score, lyrics, grid, 2, 4, 5, 400) }
        EditFixtures.error("Bar line 7 lies outside") { TrackEdit.move(score, lyrics, grid, 2, 4, 6, 250) }
        EditFixtures.error("Bar line 9 lies outside") { TrackEdit.move(score, lyrics, grid, 2, 4, 8, 350) }
        EditFixtures.error("The song's clock does not match") { TrackEdit.move(score, lyrics, grid.cut(100, 200), 2, 4, 6, 400) }
        EditFixtures.error("The song has no frames") { TrackEdit.move(score, lyrics, grid, 2, 4, 6, 0) }
        EditFixtures.error("The song has no score") { TrackEdit.move("", lyrics, grid, 2, 4, 6, 400) }
    }

    @Test fun resultRetakesCanBeAppliedInExecutionOrder() {
        val p = TrackEdit.move(silent, "", grid, 2, 4, 6, 400)
        val after = p.grid.retake(50, 100, 55).retake(155, 205, 47).retake(252, 302, 50)
        assertEquals(16.08, after.bars.last(), 1e-9)
        assertEquals(p.grid.sections, after.sections)
        assertEquals(p.grid.speed, after.speed, 0.0)
    }

    @Test fun afterMoveRejectsMissingOrDuplicatedSourceRanges() {
        val moved = ScoreEdit.moved(silent, 2, 4, 6)
        EditFixtures.error("The pieces of a move") { grid.afterMove(moved, listOf(0 to 4, 0 to 4), listOf(0 to 200, 200 to 400)) }
        EditFixtures.error("The pieces of a move") { grid.afterMove(moved, listOf(0 to 4, 4 to 8), listOf(0 to 200, 190 to 400)) }
    }

    @Test fun everyLegalWholeSectionOrderHonoursTheEngineContract() {
        val boundaries = listOf(0, 2, 4, 6, 8)
        for (first in boundaries) for (stop in boundaries) for (to in boundaries) {
            if (first >= stop || to in first..stop) continue
            val p = TrackEdit.move(silent, lyrics, grid, first, stop, to, 400)
            assertContract(p, 400)
        }
    }

    private fun assertContract(p: TrackEdit.MovePlan, frames: Int) {
        val sorted = p.pieces.sortedBy { it.first }
        assertEquals(0, sorted.first().first); assertEquals(frames, sorted.last().second)
        assertTrue(sorted.all { it.first < it.second })
        assertTrue(sorted.zipWithNext().all { (a, b) -> a.second == b.first })
        assertTrue(p.pieces.zipWithNext().all { (a, b) -> a.second != b.first })
        assertEquals(frames, p.pieces.sumOf { it.second - it.first })
        var at = 0
        val junctions = p.pieces.dropLast(1).map { piece -> at += piece.second - piece.first; at }
        assertEquals(junctions, p.sung.map { it.second })
        var last = 0
        for ((low, high) in p.sung) { assertTrue(low >= last && low < high); last = high }
        assertEquals(AbcScore.parse(p.score.trim()).voices.getValue("Vocal").bars.size + 1, p.grid.bars.size)
        assertTrue(p.grid.bars.zipWithNext().all { (a, b) -> b >= a })
    }
}
