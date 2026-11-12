package xyz.tifu.geode.ui.music

import android.content.Context
import xyz.tifu.geode.R
import xyz.tifu.geode.music.midi.Code
import xyz.tifu.geode.music.midi.ErrorCode
import xyz.tifu.geode.music.midi.MidiError
import xyz.tifu.geode.music.midi.Notice

/**
 * The MIDI import's notices and errors in the app's language. The midi package says what happened as codes and numbers
 * (YuE2-ComfyUI's English sentences would not translate); the sentences are here, a string resource each.
 */
object MidiText {
    /** A file larger than this is no song's tune (a karaoke file is some 100 KB): not read. */
    const val MAX_BYTES = 4 shl 20

    /** What a picker offers: MIDI by every name it goes under (a file the reader cannot read is refused by it). */
    val TYPES = arrayOf("audio/midi", "audio/x-midi", "audio/mid", "audio/sp-midi", "application/x-midi")

    /** [n] as a sentence. */
    fun notice(c: Context, n: Notice): String {
        val a = n.args
        fun arg(i: Int) = a.getOrElse(i) { "?" }
        return when (n.code) {
            Code.TEMPO_AVERAGED -> c.getString(R.string.music_midi_tempo, arg(0), arg(1), arg(2))
            Code.PLAYED_IN -> c.getString(R.string.music_midi_played_in)
            Code.OCTAVE_MOVED -> c.getString(if (arg(0) == "voice") R.string.music_midi_moved_voice else R.string.music_midi_moved_ins,
                arg(1), signed(arg(2)))
            Code.TOP_NOTES_KEPT -> c.getString(R.string.music_midi_top_notes, arg(0), arg(1))
            Code.CHORDS_GUESSED -> c.getString(R.string.music_midi_chords_guessed)
            Code.CHORDS_READ -> c.getString(R.string.music_midi_chords_read, arg(0), arg(1))
            Code.UNNAMED_CHORDS -> c.getString(R.string.music_midi_unnamed, arg(0),
                bars(arg(1).split(',')) + if ((a.getOrNull(2)?.toIntOrNull() ?: 0) > 0) ", \u2026" else "")
            Code.LYRICS_LAID -> {
                // (tag, its bars) pairs, then the song's seconds and the ceiling the singing is stopped at
                val pairs = a.dropLast(2).chunked(2).filter { it.size == 2 }
                    .joinToString("; ") { (tag, b) -> c.getString(R.string.music_midi_section, tag(tag), bars(b.split(','))) }
                c.getString(R.string.music_midi_laid, pairs, seconds(a.getOrElse(a.size - 2) { "0" }))
            }
            Code.UNSUNG_BARS -> c.getString(R.string.music_midi_unsung, bars(a))
            Code.CROWDED_NOTES -> c.getString(R.string.music_midi_crowded_notes, arg(0), arg(1), arg(2))
            Code.CROWDED_WORDS -> c.getString(R.string.music_midi_crowded_words, arg(0), arg(1))
        }
    }

    /** Why a file could not be imported, as the end of "This MIDI file cannot be sung: ...". */
    fun error(c: Context, e: Throwable): String = when (e) {
        is MidiError -> c.getString(when (e.code) {
            ErrorCode.SMPTE -> R.string.music_midi_err_frames
            ErrorCode.NO_NOTES, ErrorCode.NO_PITCHED_NOTES, ErrorCode.DRUM_TRACK -> R.string.music_midi_err_no_tune
            ErrorCode.TOO_LONG -> R.string.music_midi_err_long
            ErrorCode.NOTATION -> R.string.music_midi_err_score
            else -> R.string.music_midi_err_file
        })
        is TooBig -> c.getString(R.string.music_midi_err_big, "${MAX_BYTES shr 20} MB")
        else -> e.message ?: e.javaClass.simpleName
    }

    /** A file over MAX_BYTES. */
    class TooBig : java.io.IOException("larger than $MAX_BYTES bytes")

    /** [s]'s bytes, up to MAX_BYTES; more: TooBig. */
    fun read(s: java.io.InputStream): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        val buf = ByteArray(64 shl 10)
        while (true) {
            val n = s.read(buf)
            if (n < 0) break
            out.write(buf, 0, n)
            if (out.size() > MAX_BYTES) throw TooBig()
        }
        return out.toByteArray()
    }

    /** Bar numbers as runs ("1-8, 12"); anything else among them (a range given already) as it is. */
    internal fun bars(list: List<String>): String {
        val items = list.map { it.trim() }.filter { it.isNotEmpty() }
        val nums = items.map { it.toIntOrNull() }
        if (nums.any { it == null }) return items.joinToString(", ")
        val sorted = nums.filterNotNull().distinct().sorted()
        val runs = mutableListOf<String>()
        var i = 0
        while (i < sorted.size) {
            var j = i
            while (j + 1 < sorted.size && sorted[j + 1] == sorted[j] + 1) j++
            runs += if (j > i) "${sorted[i]}–${sorted[j]}" else "${sorted[i]}"
            i = j + 1
        }
        return runs.joinToString(", ")
    }

    private fun signed(s: String) = s.toIntOrNull()?.let { if (it > 0) "+$it" else "$it" } ?: s

    private fun seconds(s: String) = s.toDoubleOrNull()?.let { Math.round(it).toString() } ?: s

    /** A section tag as the lyrics write it ("intro": the bars before the first line). */
    private fun tag(t: String) = "[" + t.replaceFirstChar { it.uppercase() } + "]"
}
