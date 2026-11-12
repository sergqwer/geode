package xyz.tifu.geode.music.midi

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The JVM's regex engine is not the phone's: Android's (ICU) refuses Java's (?U) flag and an unescaped ] opening a
 * class - a pattern these tests pass would throw on the phone. The music sources are read for both.
 */
class AndroidRegexTest {
    @Test
    fun noJavaOnlyRegexInTheMusicSources() {
        val root = File("src/main/java/xyz/tifu/geode/music")
        val bad = root.walk().filter { it.isFile && it.extension == "kt" }.flatMap { f ->
            f.readLines().mapIndexedNotNull { i, line ->
                if ("(?U)" in line || "UNICODE_CHARACTER_CLASS" in line || "[^]]" in line || "[]]" in line) "${f.name}:${i + 1}" else null
            }
        }.toList()
        assertEquals(emptyList<String>(), bad)
    }

    @Test
    fun foldSplitsOnUnicodeSpaces() {
        assertEquals("a b c", MidiParts.fold(" a\u00a0 b\u3000c "))
    }
}
