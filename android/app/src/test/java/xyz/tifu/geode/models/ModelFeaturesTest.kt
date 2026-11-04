package xyz.tifu.geode.models

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import xyz.tifu.geode.engine.GeodeNative

/**
 * What a model takes as the user left it (Model settings: InstalledModel.off) and what the engine then leaves unread
 * (InstalledModel.engineOff); the manifest's longest context and drafting (PackManifest.parse).
 */
class ModelFeaturesTest {
    @get:Rule
    val tmp = TemporaryFolder()

    /** A Gemma 4 pack with both encoders and the drafter installed. */
    private fun gemma(off: Int = 0): InstalledModel {
        val dir = tmp.newFolder()
        for (f in listOf("vision.bin", "audio.bin", "drafter.bin")) java.io.File(dir, f).writeText("x")
        return InstalledModel("e4b", dir, PackManifest("E4B", "", "gemma4", emptyList(), vision = true, audio = true,
            maxContext = 131072, draft = true), off)
    }

    @Test
    fun everythingOnLoadsEverything() {
        val m = gemma()
        assertTrue(m.vision && m.audio && m.video && m.draft)
        assertEquals(0, m.engineOff)
    }

    @Test
    fun imagesOffKeepsTheVisionEncoderForVideo() {
        val m = gemma(OFF_IMAGES)
        assertFalse(m.vision)
        assertTrue(m.video)
        assertEquals(0, m.engineOff and GeodeNative.OFF_VISION)
    }

    @Test
    fun imagesAndVideoOffLeaveTheVisionEncoderUnread() {
        val m = gemma(OFF_IMAGES or OFF_VIDEO)
        assertEquals(GeodeNative.OFF_VISION, m.engineOff and GeodeNative.OFF_VISION)
        // the audio encoder stays: clips are still on
        assertEquals(0, m.engineOff and GeodeNative.OFF_AUDIO)
    }

    @Test
    fun audioOffKeepsItsEncoderForAVideosSound() {
        assertEquals(0, gemma(OFF_AUDIO).engineOff and GeodeNative.OFF_AUDIO)
        assertEquals(GeodeNative.OFF_AUDIO, gemma(OFF_AUDIO or OFF_VIDEO).engineOff and GeodeNative.OFF_AUDIO)
    }

    @Test
    fun draftingOff() {
        val m = gemma(OFF_DRAFT)
        assertFalse(m.draft)
        assertEquals(GeodeNative.OFF_DRAFT, m.engineOff)
    }

    @Test
    fun aSwitchForWhatThePackLacksChangesNothing() {
        // a Qwen pack without encoders: turning images off leaves nothing for the engine to skip
        val dir = tmp.newFolder("q2b")
        val m = InstalledModel("q2b", dir, PackManifest("2B", "", "qwen35", emptyList(), draft = true), OFF_IMAGES or OFF_AUDIO)
        assertFalse(m.hasVision || m.hasAudio || m.hasVideo)
        assertEquals(0, m.engineOff)
    }

    @Test
    fun theManifestSaysItsLongestContextAndDrafting() {
        val qwen = PackManifest.parse("""{"format":"geode-pack","version":1,"arch":"qwen35","files":[],
            "config":{"max_position_embeddings":262144,"mtp_num_hidden_layers":1},"mtp":{}}""")!!
        assertEquals(262144, qwen.maxContext)
        assertTrue(qwen.draft)
        val plain = PackManifest.parse("""{"format":"geode-pack","version":1,"arch":"qwen35moe","files":[]}""")!!
        assertEquals(0, plain.maxContext)
        assertFalse(plain.draft)
    }
}
