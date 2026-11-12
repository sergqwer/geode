package xyz.tifu.geode.ui.music

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import xyz.tifu.geode.music.Song

class LibrarySearchTest {
    private fun song(id: String, title: String, style: String, lyrics: String) =
        Song(id, File(id), title, style, lyrics, 1L, JSONObject(), null)

    @Test
    fun everyWordInTitleStyleOrLyrics() {
        val a = song("a", "Нічне місто", "Ukrainian, indie pop", "[Verse]\nСвітло над мОрем")
        val b = song("b", "Summer night", "English, synthwave", "[Chorus]\nWe ride")
        val all = listOf(a, b)
        assertEquals(all, LibrarySearch.filter(all, "  "))
        assertEquals(listOf(b), LibrarySearch.filter(all, "SYNTH night"))
        // (the lyrics' stress capitals do not count)
        assertEquals(listOf(a), LibrarySearch.filter(all, "морем pop"))
        assertEquals(emptyList<Song>(), LibrarySearch.filter(all, "pop ride"))
    }
}
