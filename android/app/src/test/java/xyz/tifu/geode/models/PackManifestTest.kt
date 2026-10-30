package xyz.tifu.geode.models

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
    }

    @Test
    fun anotherArchitectureDoesNot() {
        assertFalse(manifest("llama").runnable)
        assertFalse(manifest("").runnable)
    }
}
