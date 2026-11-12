package xyz.tifu.geode.music

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class SongStoreTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private fun store() = SongStore(File(tmp.root, "songs"), File(tmp.root, "shared"))

    @Test
    fun aSongChangedOnDiskIsReadAgain() {
        val a = store()
        val song = a.create("One", "pop", "la la", JSONObject().put("seed", 7L))
        assertEquals(SongStatus.WAITING, a.list().single().status)
        // another store (the server's, the :music process's) writes: the first one sees it at its next list
        val b = store()
        b.rename(song.id, "Two")
        assertEquals("Two", a.list().single().title)
        b.finish(song.id, JSONObject().put("ok", true).put("seconds", 12.5))
        val done = a.list().single()
        assertEquals(SongStatus.DONE, done.status)
        assertEquals(12.5, done.seconds, 0.0)
        assertEquals(7L, done.seed)
        // a result rewritten behind its back (the same size, another time): read again too
        val r = File(song.dir, SongStore.RESULT)
        r.writeText(r.readText().replace("\"ok\":true", "\"ok\":null"))
        r.setLastModified(r.lastModified() + 5000)
        assertEquals(SongStatus.FAILED, a.list().single().status)
    }

    @Test
    fun aDeletedSongLeavesTheList() {
        val a = store()
        val one = a.create("One", "pop", "", JSONObject())
        val two = a.create("Two", "rock", "", JSONObject())
        assertEquals(2, a.list().size)
        store().delete(one.id)
        assertEquals(listOf(two.id), a.list().map { it.id })
        assertNull(a.find(one.id))
        // a folder whose request is not there yet (mkdir before the write) is no song; once written, it is
        val d = File(a.root, "20260101-000000").apply { mkdirs() }
        assertEquals(1, a.list().size)
        File(d, SongStore.SONG).writeText(JSONObject().put("id", d.name).put("title", "Three").put("created", 1L).toString())
        assertTrue(a.list().any { it.title == "Three" })
    }
}
