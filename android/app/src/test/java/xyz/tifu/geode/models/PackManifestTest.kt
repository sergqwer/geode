package xyz.tifu.geode.models

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** A download or an import takes every pack the engine runs - the light 2B too, not only the 35B. */
class PackManifestTest {
    private fun manifest(arch: String) = PackManifest("m", "", arch, listOf(PackFile("dense.bin", 1, "aa")))

    @Test
    fun theEnginesArchitecturesRun() {
        assertTrue(manifest("qwen35moe").runnable)
        assertTrue(manifest("qwen35").runnable)
        assertTrue(manifest("gemma4").runnable)
        assertTrue(manifest("gemma4").gemma)
        assertFalse(manifest("qwen35").gemma)
    }

    @Test
    fun aGemmaPacksOptionalFilesAndEncoders() {
        val m = PackManifest.parse("""{"format": "geode-pack", "version": 1, "arch": "gemma4", "name": "Gemma 4 E2B",
            "files": [{"name": "dense.bin", "size": 10, "sha256": "aa"},
                      {"name": "vision.bin", "size": 3, "sha256": "bb", "optional": true},
                      {"name": "audio.bin", "size": 2, "sha256": "cc", "optional": true}],
            "vision": {"arch": "gemma4"}, "audio": {"arch": "gemma4"}}""")!!
        assertTrue(m.runnable && m.vision && m.audio)
        assertEquals(listOf(false, true, true), m.files.map { it.optional })
        assertEquals(15L, m.bytes)
        assertFalse(PackManifest.parse("""{"format": "geode-pack", "version": 1, "arch": "qwen35", "files": []}""")!!.audio)
    }

    @Test
    fun anotherArchitectureDoesNot() {
        assertFalse(manifest("llama").runnable)
        assertFalse(manifest("").runnable)
    }
}
