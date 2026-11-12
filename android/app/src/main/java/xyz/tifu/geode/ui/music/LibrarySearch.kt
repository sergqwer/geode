package xyz.tifu.geode.ui.music

import java.util.Locale
import xyz.tifu.geode.music.Song

/** The library searched: a song is shown when every word of the query is in its title, style or lyrics. */
object LibrarySearch {
    /** A library this long gets the search box (a shorter one is read at a glance). */
    const val FROM = 6

    /** [songs] that match [query] (all of them for a blank one), in their order. Case and the lyrics' stress capitals
     *  do not count. */
    fun filter(songs: List<Song>, query: String): List<Song> {
        val words = query.lowercase(Locale.ROOT).split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.isEmpty()) return songs
        return songs.filter { s ->
            val text = (s.title + "\n" + s.style + "\n" + s.lyrics).lowercase(Locale.ROOT)
            words.all { it in text }
        }
    }
}
