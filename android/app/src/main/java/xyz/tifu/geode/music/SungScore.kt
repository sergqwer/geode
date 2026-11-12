package xyz.tifu.geode.music

import xyz.tifu.geode.music.midi.Underlay
import xyz.tifu.geode.tokenizer.Tokenizer

/**
 * The score a song is sung from, as the engine takes it: its ids, and when the lyrics were laid along a tune the
 * seconds its singing is stopped at (0: the settings' own length). One for the Music screen's draft and the server's
 * requests alike.
 */
class SungScore(val abc: IntArray, val maxSeconds: Int = 0) {
    /** [p] with the laid tune's length when the settings name none (YuE2-ComfyUI's max_seconds 0). */
    fun prefs(p: MusicPrefs): MusicPrefs =
        if (maxSeconds > 0 && p.maxSeconds == 0) p.copy(maxSeconds = maxSeconds.coerceIn(MIN_SECONDS, YuE2Prompt.MAX_SECONDS)) else p

    companion object {
        const val MIN_SECONDS = 10

        /**
         * [text] (a score in YuE2's ABC) sung with [lyrics]: a score that names no sections - a MIDI file's tune - has
         * the lyrics laid along it (Underlay: YuE2-ComfyUI's "words on a MIDI tune") and the ceiling that gives; one
         * that names them (the model's own scores always do) as it is - [ids], its ids already when the text is
         * unchanged, else the text tokenized.
         */
        fun of(text: String, lyrics: String, tok: Tokenizer, ids: IntArray? = null): SungScore {
            Underlay.lay(text, lyrics)?.let { laid -> return SungScore(tok.encode(laid.abc), laid.maxSeconds) }
            // a bare tune with nothing laid along it (an instrumental): its own length, not the lyrics' guess
            return SungScore(ids ?: tok.encode(text), runCatching { if (!xyz.tifu.geode.music.midi.Phrasing.named(text)) tuneSeconds(text) else 0 }.getOrDefault(0))
        }

        /** How long [text] plays (its longest voice at its tempo), with the ceiling's margin (x 1.1 + 2 s), whole seconds. */
        fun tuneSeconds(text: String): Int {
            val s = AbcScore.parse(text)
            val beats = s.voices.values.maxOfOrNull { it.time.numerator.toDouble() / it.time.denominator } ?: return 0
            return kotlin.math.ceil(xyz.tifu.geode.music.midi.Phrasing.ceiling(beats * 60.0 / s.bpm)).toInt()
        }
    }
}
