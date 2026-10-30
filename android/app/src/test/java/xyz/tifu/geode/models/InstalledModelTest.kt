package xyz.tifu.geode.models

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
import java.io.File

/** A saved conversation state belongs to the weights, not to the folder they sit in (InstalledModel.stateId). */
class InstalledModelTest {
    private fun model(dir: String, vararg files: Pair<String, String>, description: String = "") =
        InstalledModel("q4", File(dir), PackManifest("Q4", description, "qwen35moe",
            files.map { (n, sha) -> PackFile(n, 1, sha) }))

    @Test
    fun theSameWeightsTheSameId() {
        val a = model("/m/q4", "dense.bin" to "aa", "experts-00.bin" to "bb")
        // the files listed in another order, another description: the same weights
        val b = model("/m/q4", "experts-00.bin" to "bb", "dense.bin" to "aa", description = "new text")
        assertEquals(a.stateId, b.stateId)
    }

    @Test
    fun newWeightsInTheSameFolderAnotherId() {
        val old = model("/m/q4", "dense.bin" to "aa", "experts-00.bin" to "bb")
        val new = model("/m/q4", "dense.bin" to "aa", "experts-00.bin" to "cc")
        assertNotEquals(old.stateId, new.stateId)
    }

    @Test
    fun anotherFolderAnotherId() {
        val a = model("/m/q4", "dense.bin" to "aa")
        val b = model("/m/q4copy", "dense.bin" to "aa")
        assertNotEquals(a.stateId, b.stateId)
    }
}
