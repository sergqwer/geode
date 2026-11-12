package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.AbcScore.Fraction

/** Read-only ports of YuE2-ComfyUI's break/notation/grid/edit_track fixtures (Apache-2.0).
 * Each test identifies its reference case; no reference code is executed. */
class BreakTest {
    private val lyrics = "[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"
    private val pickup = EditFixtures.header +
        EditFixtures.group("intro", "z16|z16|") +
        EditFixtures.group("verse", "C4D4E4F4|G8z4C2D2|") +
        EditFixtures.group("chorus", "E8z8|F16|") +
        EditFixtures.group("outro", "z16|z16|")
    private val rap = EditFixtures.header +
        EditFixtures.group("intro", "z16|z8z2DDA2AA|") +
        EditFixtures.group("verse", "A2DDAAA2AGGGG2GG|AGGGGFFFFDD2B4|") +
        EditFixtures.group("chorus", "A2z8D2B4|A2z4FG3F2B2B2|A2z8DDBB3|A2z8z2G2G2|") +
        EditFixtures.group("outro", "z16|z16|")
    private val played = "V: Vocal\nc4d4e4f4|g16|\nV: Ins\nC4E4G4c4|B16|\n"

    private fun sheet(s: String) = AbcScore.parse(s.trim())
    private fun clock(s: String = pickup, speed: Double = 1.0, offset: Double = 0.5): BarGrid {
        val score = sheet(s)
        val v = score.voices.getValue("Vocal")
        val starts = v.bars.map { it.start } + v.time
        val names = AbcScore.sections(score.text).map { e -> v.bars.indexOfFirst { it.start == e.start } to e.name }
        return BarGrid.fixed(starts.map { offset + it.toDouble() * speed * 60.0 / score.bpm }, names, speed)
    }
    private fun plan(s: String = pickup, words: String = lyrics) = TrackEdit.breakPlan(s, words, clock(s), 4, 2, 500)
    private fun notes(s: String, voice: String = "Vocal") = sheet(s).voices.getValue(voice).notes
        .map { Triple(it.start, it.duration, it.pitch) }.sortedWith(compareBy({ it.first }, { it.third }))
    private fun broken(s: String = pickup): String {
        val p = plan(s)
        return ScoreEdit.interluded(s, p.head + played, p.bar, p.bars, p.seam)
    }

    // test_inpaint_break: test_the_score_the_model_writes_a_break_from_ends_before_the_pickup_under_an_interlude
    @Test fun headClipsPickupAndKeepsInstrumentalAndHeader() {
        val p = plan()
        assertEquals(Fraction(29, 2), p.seam) // 58 L:1/16 ticks, NOT 58 quarter notes.
        assertTrue(p.head.endsWith("% interlude\n"))
        assertEquals(1, Regex("% interlude").findAll(p.head).count())
        val head = p.head.removeSuffix("% interlude\n")
        assertEquals(4, sheet(head).voices.getValue("Vocal").bars.size)
        assertEquals(notes(pickup).filter { it.first < p.seam }, notes(head))
        assertEquals(notes(GoOn.head(pickup, 4), "Ins"), notes(head, "Ins"))
        assertTrue(head.startsWith(EditFixtures.header))
    }

    // test_inpaint_break: test_the_break_is_played_and_the_pickup_goes_on_after_it_with_its_section
    @Test fun parentsVocalsReturnAfterBreakAndGeneratedVocalsDisappear() {
        val p = plan()
        val result = broken()
        val voice = sheet(result).voices.getValue("Vocal")
        assertEquals(10, voice.bars.size)
        assertEquals(listOf("intro", "verse", "interlude", "chorus", "outro"),
            AbcScore.sections(result.trim()).map { it.name })
        assertEquals(listOf(0.0, 8.0, 16.0, 24.0, 32.0), AbcScore.sections(result.trim()).map { it.start.toDouble() })
        val expected = notes(pickup).map { (start, duration, pitch) ->
            Triple(if (start >= p.seam) start + Fraction(8) else start, duration, pitch)
        }
        assertEquals(expected, notes(result))
        val restored = voice.notes.filter { it.start >= Fraction(16) && it.start < Fraction(24) }
        assertEquals(listOf(Fraction(23), Fraction(47, 2)), restored.map { it.start })
        assertTrue(restored.all { it.start >= voice.bars[5].start })
        assertEquals(5, notes(result, "Ins").count { it.first >= Fraction(16) && it.first < Fraction(24) })
        assertTrue(ScoreEdit.interluded(pickup + "\n\n", p.head + played, 4, 2, p.seam).endsWith("\n\n\n"))
    }

    // test_inpaint_break: test_a_tie_into_the_break_comes_off_and_what_it_held_is_struck_again_after_it
    @Test fun heldNoteIsSplitAndStruckAgain() {
        val tied = EditFixtures.header + EditFixtures.group("intro", "z16|z16|") +
            EditFixtures.group("verse", "C16|C8z6D2-|") +
            EditFixtures.group("chorus", "D8E8|F16|") + EditFixtures.group("outro", "z16|")
        val seam = Fraction(16)
        val head = ScoreEdit.interludeHead(tied, 4, seam)
        assertTrue(Triple(Fraction(31, 2), Fraction(1, 2), 62) in notes(head.removeSuffix("% interlude\n")))
        val result = ScoreEdit.interluded(tied, head + played, 4, 2, seam)
        assertTrue(Triple(Fraction(31, 2), Fraction(1, 2), 62) in notes(result))
        assertTrue(Triple(Fraction(24), Fraction(2), 62) in notes(result))
    }

    // notation.test_a_chord_on_a_resting_bar_keeps_the_rest_written_out and
    // test_an_emptied_bar_is_a_full_rest_and_a_long_rest_opens_for_one_note, applied to a break.
    @Test fun chordsStayAndCompressedGeneratedRestsOpenForParentsPickup() {
        val s = pickup.replace("G8z4C2D2|", "\"C\"G8\"F\"z4C2D2|")
        val p = plan(s)
        val result = ScoreEdit.interluded(s, p.head + "V: Vocal\nZ2|\nV: Ins\nZ2|\n", 4, 2, p.seam)
        assertEquals(notes(broken()), notes(result))
        assertEquals(sheet(s).voices.getValue("Vocal").chords, sheet(result).voices.getValue("Vocal").chords)
        assertTrue(result.contains(EditFixtures.group("intro", "z16|z16|")))
        assertTrue(notes(result, "Ins").isEmpty())
    }

    // Same held-note case, with the seam inside a bar: verify both halves' exact lengths.
    @Test fun headAndInsertionSplitHeldNoteInsideBar() {
        val tied = EditFixtures.header + EditFixtures.group("verse", "C16|D16-|D16|E16|") +
            EditFixtures.group("chorus", "F16|G16|")
        val seam = Fraction(13, 2)
        val head = ScoreEdit.interludeHead(tied, 4, seam)
        assertEquals(listOf(Triple(Fraction(0), Fraction(4), 60), Triple(Fraction(4), Fraction(5, 2), 62)),
            notes(head.removeSuffix("% interlude\n")))
        val result = ScoreEdit.interluded(tied, head + played, 4, 2, seam)
        assertTrue(Triple(Fraction(4), Fraction(5, 2), 62) in notes(result))
        assertTrue(Triple(Fraction(29, 2), Fraction(11, 2), 62) in notes(result))
    }

    // test_inpaint_break: test_a_break_the_model_stopped_writing_halfway_is_refused_to_be_written_again
    @Test fun exactlyRequestedBarsTakenAndTooFewCompleteBarsRefused() {
        val p = plan()
        for (cut in listOf(played.substringBefore("V: Ins"), played.dropLast(6), "")) {
            try {
                ScoreEdit.interluded(pickup, p.head + cut, 4, 2, p.seam)
                fail("Expected too few whole bars")
            } catch (e: IllegalArgumentException) { assertTrue(e.message.orEmpty().contains("whole bars of the break")) }
        }
        val longer = "V: Vocal\nc4d4e4f4|g16|a16|b16|\nV: Ins\nZ4|\n% chorus\nV: Vocal\nc16|\nV: Ins\nZ|\n"
        val result = ScoreEdit.interluded(pickup, p.head + longer, 4, 2, p.seam)
        assertEquals(10, sheet(result).voices.getValue("Vocal").bars.size)
        assertEquals(notes(broken()), notes(result))
        assertTrue(notes(result, "Ins").isEmpty())
    }

    // notation.test_a_score_cut_off_mid_line_is_drawn_up_to_its_last_whole_group + break's complete-group rule.
    @Test fun unfinishedMaterialAfterUsableGroupsIsTolerated() {
        val p = plan()
        val expected = broken()
        for (tail in listOf("% chorus\n", "% chorus\nV: Vocal\nc8d", "% chorus\nV: Vocal\nc16|\nV: Ins\nC8")) {
            assertEquals(expected, ScoreEdit.interluded(pickup, p.head + played + tail, 4, 2, p.seam))
        }
    }

    // notation.interluded's _broken_back checks (following the declaration fixtures in notation's moved/restyled tests).
    @Test fun restoresParentsMeterAndKeyAfterGeneratedDeclarations() {
        val p = plan()
        val generated = EditFixtures.group("different name", "F8|G8|", "F8|G8|", "M:2/4\nK:G\n")
        val result = ScoreEdit.interluded(pickup, p.head + generated, 4, 2, p.seam)
        val old = sheet(pickup)
        val now = sheet(result)
        for (name in listOf("Vocal", "Ins")) {
            val a = old.voices.getValue(name); val b = now.voices.getValue(name)
            assertEquals(listOf(Fraction(2), Fraction(2)), b.bars.slice(4..5).map { it.duration })
            for (i in a.bars.indices) {
                val n = b.bars[if (i < 4) i else i + 2]
                assertEquals(a.bars[i].duration, n.duration)
                assertEquals(a.bars[i].meter, n.meter)
                assertEquals("C", b.keys.last { it.start <= n.start }.name)
            }
        }
        assertEquals(listOf("intro", "verse", "interlude", "chorus", "outro"), AbcScore.sections(result.trim()).map { it.name })
    }

    // notation.interluded: KEY_CHANGE_BREAK; neighbouring edit refusal fixtures.
    @Test fun refusesInlineKeyAtEitherSideAndInvalidScoresOrRanges() {
        // Both voices must carry the same key timeline, so construct the two-voice inline change directly.
        val inline = EditFixtures.header + EditFixtures.group("verse", "C16|D8[K:G]E8|", "C16|D8[K:G]E8|") +
            EditFixtures.group("chorus", "F16|G16|")
        EditFixtures.error("Bar 2 changes key halfway through") {
            ScoreEdit.interluded(inline, "", 2, 2, Fraction(8))
        }
        val at = EditFixtures.header + EditFixtures.group("verse", "C16|D16|") +
            EditFixtures.group("chorus", "E8[K:G]F8|G16|", "E8[K:G]F8|G16|")
        EditFixtures.error("Bar 3 changes key halfway through") { ScoreEdit.interluded(at, "", 2, 2, Fraction(8)) }
        for ((bar, bars) in listOf(0 to 2, 8 to 2, 4 to 0, -1 to 2)) {
            EditFixtures.error("A break of") { ScoreEdit.interluded(pickup, "", bar, bars, Fraction(16)) }
        }
        EditFixtures.error("The song has no score") { ScoreEdit.interludeHead("", 4, Fraction(16)) }
        EditFixtures.error("This score cannot be read") { ScoreEdit.interluded("X:1", "", 4, 2, Fraction(16)) }
    }

    // test_inpaint_break: test_the_words_get_an_interlude_where_the_break_is
    @Test fun namedWordsAndClosingEmptyTags() {
        val expected = lyrics.replace("[Chorus]", "[Interlude]\n\n[Chorus]")
        assertEquals(Words.Breaking(expected, true), Words.breakWords(lyrics, rap, 4))
        assertTrue(Words.breakWords(lyrics, rap, 8).text.endsWith("seven eight\n\n[Interlude]\n\n[Outro]"))
        val bare = "[Verse]\none two three\n\n[Chorus]\nseven eight"
        assertEquals(bare + "\n\n[Interlude]", Words.breakWords(bare, rap, 8).text)
        assertEquals(expected.replace("\n", "\r\n"), Words.breakWords(lyrics.replace("\n", "\r\n"), rap, 4).text)
    }

    // test_inpaint_break: test_words_paired_only_as_far_as_the_break_still_take_its_tag; count fallback from ops._paired.
    @Test fun partialNamePairingCountFallbackAndUnmatched() {
        val later = lyrics.replace("[Outro]", "[Bridge]\nnine ten\n\n[Outro]")
        assertEquals(Words.Breaking(later.replace("[Chorus]", "[Interlude]\n\n[Chorus]"), true), Words.breakWords(later, rap, 4))
        val counted = "[One]\nfirst line\n\n[Two]\nsecond line"
        assertEquals(counted.replace("[Two]", "[Interlude]\n\n[Two]"), Words.breakWords(counted, pickup, 4).text)
        val lost = "[Bridge]\nnine ten\n\n[Chorus]\nseven eight"
        assertEquals(Words.Breaking(lost, false), Words.breakWords(lost, rap, 4))
        assertEquals(Words.Breaking("[Outro]", false), Words.breakWords("[Outro]", rap, 4))
        val p = plan(rap, "[Bridge]\nnine ten")
        assertEquals("[Bridge]\nnine ten", p.lyrics)
        assertEquals(listOf(TrackEdit.BreakNotice.Unmatched), p.notices)
    }

    // test_inpaint_break: test_a_break_opens_where_a_retake_of_the_section_after_it_would_and_takes_nothing_out
    @Test fun planOpensInRestBeforePickup() {
        val p = plan()
        assertEquals(194, p.start) // 0.5 + 8 - 0.75 seconds, at 25 Hz, ties-to-even.
        assertEquals(4, p.bar); assertEquals(2, p.bars)
        assertEquals(Fraction(29, 2), p.seam)
        assertTrue(p.notices.isEmpty())
        assertTrue(p.start < Math.rint(clock().bars[4] * 25).toInt())
    }

    // test_inpaint_grid: continuous rap pickup / delayed dance phrase; opening is measured in quarters here.
    @Test fun continuousPickupRoundsToLGridAndDelayedPhraseOpensAfterBoundary() {
        val p = plan(rap)
        assertEquals(192, p.start)
        assertEquals(Fraction(57, 4), p.seam) // 64 - round(6.64) = 57 sixteenths.
        val dance = EditFixtures.header + EditFixtures.group("verse", "z8c4e4|e4d4d4B4|c4e4e4f4-|f4g4e4c4-|") +
            EditFixtures.group("chorus", "c4z4b2b2g2b2-|b4f4z8|z16|z16|")
        val d = plan(dance)
        assertTrue(d.start > clock(dance).bars[4] * 25)
        // grid.opening of seam (-8, -8, -4): min(max(0, -8 + 2), -4 - margin) = -4.64 sixteenths (margin 0.08 s at
        // 0.125 s a sixteenth): the line + round(4.64) = 5 sixteenths, a margin before the chorus's first note.
        assertEquals(Fraction(69, 4), d.seam)
    }

    // test_inpaint_break: test_a_break_goes_before_a_section_and_inside_the_song + no-score/length refusals.
    @Test fun planRefusesEdgesInsideSectionsUnreachedAndBadLengthOrClock() {
        for (bar in listOf(0, 10, -1, 11)) EditFixtures.error("A break goes between") {
            TrackEdit.breakPlan(rap, lyrics, clock(rap), bar, 2, 500)
        }
        for (bar in listOf(3, 5)) EditFixtures.error("Bar line") {
            TrackEdit.breakPlan(rap, lyrics, clock(rap), bar, 2, 500)
        }
        for (frames in listOf(0, 100, 194)) EditFixtures.error("Bar line 5 lies outside") {
            TrackEdit.breakPlan(pickup, lyrics, clock(), 4, 2, frames)
        }
        EditFixtures.error("Bar line 5 lies outside") { TrackEdit.breakPlan(pickup, lyrics, clock(offset = -20.0), 4, 2, 500) }
        for (bars in listOf(0, -1, 17)) EditFixtures.error("A break must have 1 to 16") {
            TrackEdit.breakPlan(pickup, lyrics, clock(), 4, bars, 500)
        }
        for (bars in listOf(1, 4, 16)) assertEquals(bars, TrackEdit.breakPlan(pickup, lyrics, clock(), 4, bars, 500).bars)
        EditFixtures.error("The song has no score") { TrackEdit.breakPlan("", lyrics, clock(), 4, 2, 500) }
        EditFixtures.error("This score cannot be read") { TrackEdit.breakPlan("X:1", lyrics, clock(), 4, 2, 500) }
        for (grid in listOf(BarGrid.fixed(listOf(0.0, 1.0), emptyList()), clock(speed = 0.0), clock(speed = Double.NaN))) {
            EditFixtures.error("The song's clock does not match") { TrackEdit.breakPlan(pickup, lyrics, grid, 4, 2, 500) }
        }
    }

    // test_inpaint_break: test_the_grid_after_a_break_moves_what_comes_after_it_by_what_came_in;
    // test_edit_track: test_a_break_is_played_before_its_section_and_what_comes_after_moves_by_it.
    @Test fun frameCountAndActualInsertedFramesPlaceLines() {
        val p = plan()
        val result = broken()
        val grid = clock()
        assertEquals(100, TrackEdit.breakFrames(p, result, grid))
        assertEquals(125, TrackEdit.breakFrames(p, result, clock(speed = 1.25)))
        val after = grid.afterBreak(result, 4, 2, 103)
        assertEquals(grid.bars.take(5), after.bars.take(5))
        assertEquals(grid.bars[4] + 103 / 50.0, after.bars[5], 1e-10)
        grid.bars.drop(4).zip(after.bars.drop(6)).forEach { (old, now) -> assertEquals(old + 103 / 25.0, now, 1e-10) }
        assertEquals(listOf(0, 2, 4, 6, 8), after.sections.map { it.first })
        assertEquals(grid.speed, after.speed, 0.0)
    }

    // grid.after_break's proportional-bar rule extended to differing generated meters and an already fixed grid.
    @Test fun unequalNewBarsUseTheirScoreProportions() {
        val p = plan()
        val generated = EditFixtures.group("", "C8|", "G8|", "M:2/4\n") +
            EditFixtures.group("", "D16|", "A16|", "M:4/4\n")
        val result = ScoreEdit.interluded(pickup, p.head + generated, 4, 2, p.seam)
        val old = clock()
        val grid = BarGrid.fixed(old.bars.mapIndexed { i, t -> t + i * 0.1 }, old.sections, 1.2)
        assertEquals(90, TrackEdit.breakFrames(p, result, grid))
        val after = grid.afterBreak(result, 4, 2, 93)
        assertEquals(grid.bars[4] + 93 / 75.0, after.bars[5], 1e-10)
        assertEquals(grid.bars[4] + 93 / 25.0, after.bars[6], 1e-10)
        assertEquals(1.2, after.speed, 0.0)
        EditFixtures.error("The song's clock does not match") { grid.afterBreak(result, 4, 1, 93) }
        assertEquals(1, TrackEdit.breakFrames(p, result, clock(speed = 0.00001)))
    }

    // Codex's review: a key change halfway through a bar next to the break is refused when the break is planned.
    @Test fun planRefusesAnInlineKeyChangeNextToTheBreak() {
        val s = EditFixtures.header + EditFixtures.group("verse", "C16|z16|", "C16|z16|") +
            EditFixtures.group("chorus", "E8[K:G]F8|G16|", "E8[K:G]F8|G16|")
        EditFixtures.error("Bar 3 changes key halfway through") {
            TrackEdit.breakPlan(s, "", BarGrid.fixed(List(5) { 2.0 * it }, listOf(0 to "verse", 2 to "chorus")), 2, 2, 200)
        }
    }
}
