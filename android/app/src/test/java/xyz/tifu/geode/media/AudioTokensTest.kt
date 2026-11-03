package xyz.tifu.geode.media

import org.junit.Assert.assertEquals
import org.junit.Test

/** A clip's <|audio|> tokens as the engine counts them (tools/gemma4_audio.py's n_tokens over its valid frames). */
class AudioTokensTest {
    @Test
    fun likeTheEngine() {
        val want = mapOf(100L to 0, 320L to 1, 321L to 1, 640L to 1, 1600L to 3, 16000L to 25, 16001L to 25,
            48000L to 75, 160000L to 250, 479999L to 750, 480000L to 750, 600000L to 750, 0L to 0)
        for ((n, t) in want) assertEquals("$n samples", t, AudioStore.tokens(n))
    }
}
