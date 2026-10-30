package xyz.tifu.geode.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MemoryTestTest {
    private fun step(heldMb: Long, swapMb: Long, ms: Long, availMb: Long = 1024) =
        ProbeStep(heldMb shl 20, availMb shl 20, 0, swapMb shl 20, ms)

    /** A real run on the Redmi Note 9 Pro: VmSwap starts at 21 MB, MemAvailable < 128 MB at 3648 MB, steps slow down
     *  at 3840 MB, killed after 4096 MB. */
    @Test
    fun redmiNote9Pro() {
        val lines = javaClass.classLoader!!.getResource("memprobe_redmi_note9pro.txt")!!.readText().lines()
        val steps = MemoryTest.parseSteps(lines)
        assertEquals(64, steps.size)
        assertEquals(3584L shl 20, MemoryTest.lastCleanStep(steps)!!.held)
    }

    /** A real run on the RedMagic 10 Pro (16 GB, a 10.9 GB swap): 0.12 GB of the probe goes to swap at 8.8 GB and
     *  stays there, one step at 9.5 GB is 2.7x slow, the rest fast up to ~10.5 GB (games hold ~10-11 GB); from
     *  ~10.6 GB the kernel swaps gigabytes and steps take seconds. */
    @Test
    fun redMagic10Pro() {
        val lines = javaClass.classLoader!!.getResource("memprobe_redmagic10pro.txt")!!.readText().lines()
        val steps = MemoryTest.parseSteps(lines)
        val clean = MemoryTest.lastCleanStep(steps)!!.held
        assert(clean in (10000L shl 20)..(10600L shl 20)) { "clean step at ${clean shr 20} MB" }
    }

    @Test
    fun swapGrowthCountsFromTheFirstStep() {
        val steps = listOf(step(64, 20, 60), step(128, 22, 60), step(192, 30, 70), step(256, 80, 65))
        assertEquals(192L shl 20, MemoryTest.lastCleanStep(steps)!!.held)
    }

    @Test
    fun aSlowStepEndsTheCleanRange() {
        val steps = listOf(step(64, 0, 60), step(128, 0, 50), step(192, 0, 70), step(256, 0, 400))
        assertEquals(192L shl 20, MemoryTest.lastCleanStep(steps)!!.held)
    }

    @Test
    fun oneSlowStepAloneDoesNot() {
        val steps = (1..8).map { step(64L * it, 0, 60) } +
            listOf(step(576, 0, 400), step(640, 0, 60), step(704, 0, 400), step(768, 0, 500))
        assertEquals(640L shl 20, MemoryTest.lastCleanStep(steps)!!.held)
    }

    @Test
    fun runningOutOfAvailableMemoryEndsTheCleanRange() {
        val steps = listOf(step(64, 0, 60), step(128, 0, 60, availMb = 300), step(192, 0, 60, availMb = 100))
        assertEquals(128L shl 20, MemoryTest.lastCleanStep(steps)!!.held)
    }

    @Test
    fun allCleanOrNone() {
        val steps = listOf(step(64, 0, 60), step(128, 0, 60))
        assertEquals(128L shl 20, MemoryTest.lastCleanStep(steps)!!.held)
        assertNull(MemoryTest.lastCleanStep(emptyList()))
    }
}
