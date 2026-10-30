package xyz.tifu.geode.bench

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextPlanTest {
    // the all-q2 Qwen3.6-35B-A3B pack (work/packs/q2)
    private val q2 = ModelShape(
        name = "q2", layers = 40, experts = 256, topK = 8, hidden = 2048, moeFf = 512,
        denseWeights = emptyMap(), denseBytesPerToken = 0, denseFileBytes = 683_671_552, otherWeights = 0,
        expertFormat = "q2", expertBlockBytes = 983_040, attnLayers = 10,
        kvBytesPerToken = 10L * 2 * 2 * 256 * 2, stateBytes = 65_863_680,
    )

    @Test
    fun theContextTakesItsStepsFromTheExpertCache() {
        val p = SpeedModel.plan(q2, 3300L shl 20, 32768)
        // 20 MiB of FP16 K and V per 1024 tokens: 22 slots of 960 KiB
        assertEquals(20L shl 20, p.chunkBytes)
        assertEquals(22L, p.slotsPerChunk)
        // what Model::open computes: (budget - dense - state - 32 MiB - one chunk) / slot
        assertEquals(((3300L shl 20) - 683_671_552 - 65_863_680 - (32L shl 20) - (20L shl 20)) / 983_040, p.startSlots)
        assertEquals(32768, p.maxContext)
        assertEquals(p.startSlots, p.slotsAt(1))
        assertEquals(p.startSlots, p.slotsAt(1024))
        assertEquals(p.startSlots - 22, p.slotsAt(1025))
        assertEquals(p.startSlots - 31 * 22, p.slotsAt(32768))
    }

    @Test
    fun aBudgetCapsTheLongestContext() {
        val p = SpeedModel.plan(q2, 3300L shl 20, 262144)
        assertTrue(p.maxContext < 262144)
        assertEquals(0, p.maxContext % ContextPlan.KV_CHUNK)
        // at the cap the cache keeps what one layer and the next one's guesses pin, and one more chunk would not fit
        assertTrue(p.slotsAt(p.maxContext) >= 2L * q2.topK + 2)
        assertTrue(p.slotsAt(p.maxContext + 1) < 2L * q2.topK + 2)
    }
}
