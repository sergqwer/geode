package xyz.tifu.geode.media

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Where a video's frames are taken: one a second, at most 32 - spread over the whole of a longer one. */
class VideoFramesTest {
    @Test
    fun oneASecondUpToThirtyTwo() {
        assertEquals(listOf(0L), VideoDecoder.frameTimes(0))
        assertEquals(listOf(0L), VideoDecoder.frameTimes(400))
        assertEquals(listOf(0L, 1000L), VideoDecoder.frameTimes(1001))
        assertEquals((0L until 6L).map { it * 1000 }, VideoDecoder.frameTimes(6000))
        assertEquals((0L until 32L).map { it * 1000 }, VideoDecoder.frameTimes(32_000))
    }

    @Test
    fun aLongerVideoSpreadsItsThirtyTwo() {
        for (ms in listOf(32_001L, 100_000L, 3_600_000L)) {
            val t = VideoDecoder.frameTimes(ms)
            assertEquals(32, t.size)
            assertEquals(0L, t.first())
            // the last at (n - 1) / n of it (whole milliseconds, rounded down)
            assertTrue("$ms: ${t.last()}", t.last() < ms && t.last() >= ms - ms / 32 - 1)
            assertTrue(t.zipWithNext().all { (a, b) -> b > a })
        }
    }
}
