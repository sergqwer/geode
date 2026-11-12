package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test

/** Fixtures and expected answers ported from YuE2-ComfyUI tests/test_inpaint_extend.py,
 * test_inpaint_grid.py and test_inpaint_track.py (Apache-2.0); the reference is never executed. */
class GoOnTest {
    private val header = "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n" +
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n" +
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\n"
    private val rap = header + """
        % intro
        V: Vocal
        z16|z8z2DDA2AA|
        V: Ins
        Z|Z|
        % verse
        V: Vocal
        A2DDAAA2AGGGG2GG|AGGGGFFFFDD2B4|
        V: Ins
        Z|Z|
        % chorus
        V: Vocal
        A2z8D2B4|A2z4FG3F2B2B2|A2z8DDBB3|A2z8z2G2G2|
        V: Ins
        Z|Z|Z|Z|
        % outro
        V: Vocal
        z16|z16|
        V: Ins
        Z|Z|
    """.trimIndent() + "\n"
    private val ballad = header + """
        % intro
        V: Vocal
        z16|z16|
        V: Ins
        Z|Z|
        % verse
        V: Vocal
        z2AAA2A2A2e4d2|B4z12|z2AAA2A2A2e4d2|z6aaa4f2e2|
        V: Ins
        Z|Z|Z|Z|
        % chorus
        V: Vocal
        B4z12|z6ggg4e2d2|A4z12|z16|
        V: Ins
        Z|Z|Z|Z|
    """.trimIndent() + "\n"
    private val dance = header + """
        % verse
        V: Vocal
        z8c4e4|e4d4d4B4|c4e4e4f4-|f4g4e4c4-|
        V: Ins
        Z|Z|Z|Z|
        % chorus
        V: Vocal
        c4z4b2b2g2b2-|b4f4z8|z16|z16|
        V: Ins
        Z|Z|Z|Z|
    """.trimIndent() + "\n"
    private val ending = header + """
        % chorus
        V: Vocal
        z8c4d4|e4d4c4d4|e8d4c4|
        V: Ins
        Z|Z|Z|
        % interlude
        V: Vocal
        d8z8|z16|
        V: Ins
        Z|Z|
    """.trimIndent() + "\n"
    private val tail = rap.replace("A2z8z2G2G2|", "A2z8z2G2G2-|").replace("z16|z16|\nV: Ins\nZ|Z|\n", "G4z12|z16|\nV: Ins\nZ|Z|\n")
    private val lyrics = "[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"
    private val bridge = "[Bridge]\nnine ten eleven\ntwelve and more"
    private val clock = GoOn.Clock(List(11) { 0.5 + 2 * it }, 1.0)
    private fun plan() = GoOn.plan(lyrics, rap, bridge, clock, 500)!!
    private fun written(p: GoOn.Plan, music: String = "A2z8D2B4|z16|z16|") =
        p.head + "% bridge\nV: Vocal\n$music\nV: Ins\n" + "Z|".repeat(music.count { it == '|' }) + "\n"
    private fun error(expected: String, body: () -> Unit) {
        try { body(); fail("Expected IllegalArgumentException") }
        catch (e: IllegalArgumentException) { assertEquals(expected, e.message) }
    }

    @Test fun constantsMatchReference() {
        assertEquals("[Verse]", GoOn.TAG)
        assertEquals(4000, GoOn.LONGEST_WORDS)
        assertEquals(0.5, GoOn.SOUND_REACHED, 0.0)
        assertEquals(50, GoOn.EARLIER); assertEquals(250, GoOn.LATER)
        assertEquals(3, GoOn.SCORE_TRIES); assertEquals(7919L, GoOn.SCORE_SEED_STEP)
    }

    @Test fun emptyTextKeepsTheLyricsAndAddsNothing() {
        assertEquals(GoOn.Going(lyrics, emptyList(), emptyList(), false), GoOn.goOn(lyrics, "  \n "))
        assertEquals(GoOn.Going("", emptyList(), emptyList(), false), GoOn.goOn("", ""))
    }

    @Test fun untaggedWordsBecomeAVerse() {
        assertEquals(GoOn.Going("[Verse]\nnine ten", listOf("[Verse]", "nine ten"), listOf("nine ten"), true),
            GoOn.goOn("", "\n  nine ten  \n"))
    }

    @Test fun newLinesGoBeforeTrailingOutro() {
        val g = GoOn.goOn(lyrics, bridge)
        assertEquals(lyrics.removeSuffix("\n\n[Outro]") + "\n\n" + bridge + "\n\n[Outro]", g.text)
        assertEquals(listOf("[Bridge]", "nine ten eleven", "twelve and more"), g.added)
        assertEquals(listOf("nine ten eleven", "twelve and more"), g.sung)
        assertFalse(g.tagged)
    }

    @Test fun tagsOnlySingNothingAndKeepTrailingTags() {
        val g = GoOn.goOn(lyrics + "\n[End]\n", "[Outro]")
        assertTrue(g.sung.isEmpty())
        assertTrue(g.text.endsWith("[Outro]\n\n[Outro]\n[End]"))
    }

    @Test fun crlfIsPreserved() {
        val crlf = lyrics.replace("\n", "\r\n")
        assertEquals(crlf.removeSuffix("\r\n\r\n[Outro]") + "\r\n\r\n[Bridge]\r\nnine ten\r\n\r\n[Outro]",
            GoOn.goOn(crlf, "[Bridge]\nnine ten").text)
    }

    @Test fun sungLinesUseLettersAndTrimButKeepBlankAddedLines() {
        val g = GoOn.goOn("", "[Bridge]\n123\n...\n\nПривіт\n你好")
        assertEquals(listOf("Привіт", "你好"), g.sung)
        assertEquals(listOf("[Bridge]", "123", "...", "", "Привіт", "你好"), g.added)
    }

    @Test fun refusesABookWithReferenceMessage() {
        error("The new words run to more than 4000 characters. A song goes on by a few sections at a time.") {
            GoOn.goOn(lyrics, "la ".repeat(2000))
        }
        assertEquals(4000, GoOn.goOn("", "a".repeat(4000)).sung.single().length)
        // Python len counts Unicode code points, not UTF-16 code units.
        assertFalse(GoOn.goOn("", "😀".repeat(3000)).added.isEmpty())
    }

    @Test fun lastSungPairsNamesAndAliases() {
        assertEquals(4, GoOn.lastSung(lyrics, rap))
        assertEquals(4, GoOn.lastSung(lyrics.replace("[Chorus]", "[Hook 2]"), rap))
        assertNull(GoOn.lastSung("[Intro]\n\n[Outro]", rap))
    }

    @Test fun lastSungFallsBackOnlyWhenCountsMatch() {
        // RAP has notes in intro, verse and chorus: three unmatched blocks can pair in order.
        assertEquals(4, GoOn.lastSung("[Bridge]\na\n[Bridge]\nb\n[Bridge]\nc", rap))
        assertNull(GoOn.lastSung("[Bridge]\na\n[Bridge]\nb", rap))
        assertEquals(2, GoOn.lastSung("one", rap))
    }

    @Test fun repeatedTagsPairInOrderAndEmptyTagsDoNotSing() {
        val twice = rap.replace("% chorus", "% verse")
        assertEquals(4, GoOn.lastSung("[Verse]\none\n[Outro]\n\n[Verse]\ntwo", twice))
    }

    @Test fun headOfWholeScoreIsIdentical() {
        assertEquals(rap, GoOn.head(rap, 10))
        assertEquals(rap.replace("\n", "\r\n"), GoOn.head(rap.replace("\n", "\r\n"), 10))
    }

    @Test fun headCutsInsideAGroupAndRetainsEveryBar() {
        val h = GoOn.head(rap, 5)
        assertTrue(h.endsWith("% chorus\nV: Vocal\nA2z8D2B4|\nV: Ins\nZ|\n"))
        val before = AbcScore.parse(rap); val after = AbcScore.parse(h)
        for (voice in listOf("Vocal", "Ins")) assertEquals(before.voices.getValue(voice).bars.take(5), after.voices.getValue(voice).bars)
        assertEquals(before.unit, after.unit); assertEquals(before.bpm, after.bpm)
    }

    @Test fun headCutsInsideZ3AndKeepsHeadersAndComments() {
        val s = header + "% intro\n% verse\nV: Vocal\nM:3/4\nK:G\n Z3 |\nV: Ins\nM:3/4\nK:G\nZ3|\n"
        val h = GoOn.head(s, 2)
        assertEquals(header + "% intro\n% verse\nV: Vocal\nM:3/4\nK:G\nZ2|\nV: Ins\nM:3/4\nK:G\nZ2|\n", h)
        val old = AbcScore.parse(s); val new = AbcScore.parse(h)
        assertEquals(old.voices.getValue("Vocal").bars.take(2), new.voices.getValue("Vocal").bars)
        assertEquals(old.voices.getValue("Vocal").keys, new.voices.getValue("Vocal").keys)
    }

    @Test fun headRemovesOutgoingTiesInBothVoices() {
        val s = header + "% verse\nV: Vocal\nC16-|C16|\nV: Ins\nG16-|G16|\n"
        val h = GoOn.head(s, 1)
        assertEquals(header + "% verse\nV: Vocal\nC16|\nV: Ins\nG16|\n", h)
        assertEquals(1, AbcScore.parse(h).voices.getValue("Ins").bars.size)
        val t = GoOn.head(tail, 8)
        assertFalse(t.contains("G2G2-")); assertTrue(t.contains("A2z8z2G2G2|"))
        assertEquals(8, AbcScore.parse(t).voices.getValue("Vocal").bars.size)
    }

    @Test fun headDropsOnlyTheRequestedSectionNames() {
        val h = GoOn.head(rap, 9, setOf(8))
        assertFalse(h.contains("% outro"))
        assertEquals(listOf("intro", "verse", "chorus"), AbcScore.sections(h).map { it.name })
        assertEquals(9, AbcScore.parse(h).voices.getValue("Vocal").bars.size)
    }

    @Test fun headRejectsInvalidBoundariesWithReferenceMessages() {
        for (stop in listOf(0, 11)) error("The score cannot end after bar $stop: it has 10 bars.") { GoOn.head(rap, stop) }
        error("The song has no score.") { GoOn.head("", 1) }
    }

    @Test fun goesOnAfterLastSungNoteAtReferenceFrame414() {
        assertEquals(GoOn.From(8, 10, 414), GoOn.goingOnFrom(rap, clock, 500))
        val p = plan()
        assertEquals(8, p.bar); assertEquals(10, p.bars); assertEquals(414, p.start)
        assertEquals(GoOn.head(rap, 8), p.head)
        assertEquals(listOf("nine ten eleven", "twelve and more"), p.sung)
    }

    @Test fun songStoppedShortUsesLastReachedBoundary() {
        val found = GoOn.goingOnFrom(rap, clock, 300)!!
        assertEquals(5, found.bar)
        assertEquals(GoOn.head(rap, 5), GoOn.plan(lyrics, rap, bridge, clock, 300)!!.head)
    }

    @Test fun pickupsOpenInRestOrBeforeContinuousSinging() {
        // test_inpaint_grid: RAP verse pickup six sixteenths early, with room for a beat;
        // chorus pickup six early without a rest, protected by the 0.08 s margin.
        assertEquals(GoOn.From(2, 10, 81), GoOn.goingOnFrom(rap, clock, 150))
        assertEquals(GoOn.From(4, 10, 192), GoOn.goingOnFrom(rap, clock, 250))
        assertEquals(GoOn.From(2, 10, 106), GoOn.goingOnFrom(ballad, clock, 150))
        assertEquals(GoOn.From(6, 10, 272), GoOn.goingOnFrom(ballad, clock, 350))
    }

    @Test fun heldNoteAndLastWordOnInstrumentalStayBeforeNewSection() {
        assertEquals(GoOn.From(4, 8, 227), GoOn.goingOnFrom(dance, GoOn.Clock(clock.lines.take(9), 1.0), 250))
        assertEquals(GoOn.From(3, 5, 190), GoOn.goingOnFrom(ending, GoOn.Clock(clock.lines.take(6), 1.0), 200))
    }

    @Test fun heldChorusTailMergesOutroComment() {
        val p = GoOn.plan(lyrics, tail, bridge, clock, 500)!!
        assertEquals(9, p.bar)
        assertFalse(p.head.contains("% outro"))
        assertEquals(listOf("intro", "verse", "chorus"), AbcScore.sections(p.head).map { it.name })
    }

    @Test fun noVocalsGoesOnAfterLastBarAndClampsToAudio() {
        val silent = header + "% instrumental\nV: Vocal\nZ3|\nV: Ins\nZ3|\n"
        val c = GoOn.Clock(listOf(0.0, 2.0, 4.0, 6.0), 1.0)
        assertEquals(GoOn.From(3, 3, 150), GoOn.goingOnFrom(silent, c, 200))
        val one = header + "% instrumental\nV: Vocal\nZ|\nV: Ins\nZ|\n"
        assertEquals(GoOn.From(1, 1, 99), GoOn.goingOnFrom(one, GoOn.Clock(listOf(0.0, 6.0), 1.0), 100))
    }

    @Test fun noScoreOrClockReturnsNull() {
        assertNull(GoOn.goingOnFrom("", clock, 500))
        assertNull(GoOn.goingOnFrom(rap, GoOn.Clock(emptyList(), 1.0), 500))
        assertNull(GoOn.plan(lyrics, "", bridge, clock, 500))
    }

    @Test fun newEndingAloneKeepsWordsAndUntaggedPlanReportsTag() {
        val p = GoOn.plan(lyrics, rap, "", clock, 500)!!
        assertEquals(lyrics, p.lyrics); assertTrue(p.sung.isEmpty()); assertFalse(p.tagged)
        val tagged = GoOn.plan(lyrics, rap, "nine ten eleven", clock, 500)!!
        assertTrue(tagged.tagged); assertTrue(tagged.lyrics.endsWith("[Verse]\nnine ten eleven\n\n[Outro]"))
    }

    @Test fun extensionKeepsTailAndOpeningLeadIn() {
        val p = plan(); val w = written(p)
        // reference formula: round(16.5/.04) - 414 + 150 = 148; tail adds 30.
        assertEquals(148, GoOn.extensionFrames(rap, clock, 500, p, w))
        assertEquals(178, GoOn.extensionFrames(rap, clock, 542, p, w))
    }

    @Test fun extensionUsesTempoSpeedAndVariableMeterLengths() {
        val p = plan()
        val w = p.head + "% bridge\nV: Vocal\nM:3/4\nZ|\nV: Ins\nM:3/4\nZ|\n"
        val faster = GoOn.Clock(clock.lines, 1.2)
        assertEquals(43, GoOn.extensionFrames(rap, faster, 500, p, w)) // -2 + 3 quarters * .5 * 1.2 / .04
    }

    @Test fun linesAfterKeepOldLinesAndExtrapolateAtTempo() {
        // test_inpaint_extend's after-cut clock: the four new boundaries are 18,20,22,24.
        val lined = GoOn.Clock(listOf(0.5, 2.5, 4.5, 6.5, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0), 1.0)
        val p = plan(); val w = written(p, "A2z8D2B4|z16|z16|z16|")
        val (lines, sections) = GoOn.linesAfter(lined, p, w)
        assertEquals(lined.lines.take(9), lines.take(9))
        assertEquals(listOf(18.0, 20.0, 22.0, 24.0), lines.drop(9))
        assertEquals(listOf(0 to "intro", 2 to "verse", 4 to "chorus", 8 to "bridge"), sections)
    }

    @Test fun plainClockNewBoundariesAndSpeedArePreserved() {
        val p = plan()
        val (lines, _) = GoOn.linesAfter(GoOn.Clock(clock.lines, 1.1), p, written(p))
        assertEquals(clock.lines.take(9), lines.take(9))
        assertEquals(23.1, lines.last(), 1e-9)
    }

    @Test fun rejectsMismatchedClockRatherThanIndexingWrongBars() {
        error("The song's clock does not match its score.") { GoOn.goingOnFrom(rap, GoOn.Clock(listOf(0.0, 2.0), 1.0), 500) }
    }
}
