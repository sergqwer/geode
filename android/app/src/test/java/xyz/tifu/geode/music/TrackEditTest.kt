package xyz.tifu.geode.music

import org.junit.Assert.*
import org.junit.Test

/** Fixtures and expectations ported by reading YuE2-ComfyUI's inpaint/{track,grid}.py (Apache-2.0).
 * The reference is never executed. */
class TrackEditTest {
    private val score = EditFixtures.header + EditFixtures.group("verse", "C16|D16|") +
        EditFixtures.group("chorus", "E16|F16|") + EditFixtures.group("outro", "Z2|", "Z2|")
    private val lyrics = "[Verse]\none\n[Chorus]\ntwo\n[Outro]"
    private val grid = BarGrid.fixed(List(7) { it * 2.0 }, listOf(0 to "verse", 2 to "chorus", 4 to "outro"), 1.1)

    @Test fun interiorCutPlansWordsScoreAndIndexedGrid() {
        val p = TrackEdit.cut(score, lyrics, grid, 2, 4, 95, 195, 300)
        assertEquals("[Verse]\none\n[Outro]", p.lyrics)
        assertEquals(EditFixtures.header + EditFixtures.group("verse", "C16|D16|") + EditFixtures.group("outro", "Z2|", "Z2|"), p.score)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0), p.grid.bars)
        assertEquals(listOf(0 to "verse", 2 to "outro"), p.grid.sections)
        assertEquals(1.1, p.grid.speed, 0.0)
        assertEquals(listOf(TrackEdit.Notice.Dropped(listOf("Chorus"))), p.notices)
        assertEquals(95, p.first); assertEquals(195, p.stop); assertEquals(2, p.firstBar); assertEquals(4, p.stopBar)
    }

    @Test fun tailCutAlsoRemovesUnreachedScoreBars() {
        val p = TrackEdit.cut(score, lyrics, grid, 2, 3, 100, 150, 150)
        assertEquals(6, p.stopBar)
        assertEquals(2, AbcScore.parse(p.score.trim()).voices.getValue("Vocal").bars.size)
        assertEquals(listOf(0.0, 2.0, 10.0), p.grid.bars) // reference after_cut shifts the old score end
        assertEquals(listOf(TrackEdit.Notice.Tail(4, 6), TrackEdit.Notice.Dropped(listOf("Chorus"))), p.notices)
    }

    @Test fun tailBoundaryClampsAtTheAudioSeam() {
        val p = TrackEdit.cut(score, lyrics, grid, 4, 6, 200, 300, 300)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0), p.grid.bars)
        assertTrue(p.notices.isEmpty())
        val next = ScoreEdit.without(score, 2, 4)
        assertEquals(listOf(0.0, 2.0, 4.0, 4.0, 4.0), grid.cutBars(2, 4, 100, 400, next).bars)
    }

    @Test fun headTrimUpdatesMetadataWithoutRequiringAModel() {
        val p = TrackEdit.cut(score, lyrics, grid, 0, 2, 0, 100, 300)
        assertEquals("[Chorus]\ntwo\n[Outro]", p.lyrics)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0), p.grid.bars)
        assertEquals(listOf(0 to "chorus", 2 to "outro"), p.grid.sections)
    }

    @Test fun pickupBeforeTheBarLineDoesNotDeleteAnExtraBoundary() {
        val next = ScoreEdit.without(score, 2, 3)
        val g = grid.cutBars(2, 3, 75, 125, next)
        assertEquals(listOf(0.0, 2.0, 4.0, 6.0, 8.0, 10.0), g.bars)
        assertEquals(AbcScore.parse(next.trim()).voices.getValue("Vocal").bars.size + 1, g.bars.size)
    }

    @Test fun variableMeterAndSectionsComeFromTheNewScore() {
        val s = EditFixtures.header + EditFixtures.group("verse", "Z2|", "Z2|") +
            EditFixtures.group("bridge", "C8|D8|", "G8|A8|", "M:2/4\n")
        val g = BarGrid.fixed(listOf(0.5, 2.5, 4.5, 5.5, 6.5), emptyList(), 0.9)
        val p = TrackEdit.cut(s, "[Bridge]\nhello", g, 0, 1, 0, 50, 200)
        assertEquals(listOf(0.5, 2.5, 3.5, 4.5), p.grid.bars)
        assertEquals(listOf(0 to "verse", 1 to "bridge"), p.grid.sections)
        assertEquals(0.9, p.grid.speed, 0.0)
    }

    @Test fun unmatchedLyricsAreKeptWithANotice() {
        val words = "  [Bridge]\nhello\n"
        val p = TrackEdit.cut(score, words, grid, 2, 4, 100, 200, 300)
        assertEquals(words, p.lyrics)
        assertEquals(listOf(TrackEdit.Notice.Unmatched), p.notices)
    }

    @Test fun furtherCutAndGoOnSeeCorrespondingBars() {
        val p = TrackEdit.cut(score, lyrics, grid, 2, 4, 100, 200, 350)
        val q = TrackEdit.cut(p.score, p.lyrics, p.grid, 0, 1, 0, 50, 250)
        assertEquals(4, q.grid.bars.size)
        assertNotNull(GoOn.goingOnFrom(q.score, GoOn.Clock(q.grid.bars, q.grid.speed), 200))
        assertEquals(q.grid.bars.size, q.grid.retake(10, 50, 45).bars.size)
    }

    @Test fun rejectsLegacyMismatchedGridsAndInvalidClocks() {
        val legacy = grid.cut(100, 200)
        EditFixtures.error("The song's clock does not match") { TrackEdit.cut(score, lyrics, legacy, 0, 1, 0, 50, 300) }
        for (g in listOf(BarGrid.fixed(listOf(0.0, Double.NaN) + grid.bars.drop(2), emptyList()),
            BarGrid.fixed(grid.bars.reversed(), emptyList()), BarGrid.fixed(grid.bars, emptyList(), 0.0))) {
            EditFixtures.error("The song's clock does not match") { TrackEdit.cut(score, lyrics, g, 0, 1, 0, 50, 300) }
        }
    }

    @Test fun rejectsInvalidRangesAllBarsAndTooLittleSound() {
        EditFixtures.error("Bars ") { TrackEdit.cut(score, lyrics, grid, 0, 7, 0, 50, 300) }
        EditFixtures.error("The cut is not a stretch") { TrackEdit.cut(score, lyrics, grid, 0, 1, -1, 50, 300) }
        EditFixtures.error("The cut is not a stretch") { TrackEdit.cut(score, lyrics, grid, 0, 1, 20, 20, 300) }
        EditFixtures.error("A cut leaves less than a second") { TrackEdit.cut(score, lyrics, grid, 0, 1, 0, 280, 300) }
        EditFixtures.error("That is every bar") { TrackEdit.cut(score, lyrics, grid, 0, 6, 0, 50, 100) }
        EditFixtures.error("That is every bar") { TrackEdit.cut(score, lyrics, grid, 0, 2, 25, 100, 100) } // tail extension
        EditFixtures.error("The song's clock does not match") { grid.cutBars(0, 1, 0, 50, score) }
    }
}
