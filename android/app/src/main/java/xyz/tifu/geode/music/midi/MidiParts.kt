/** Parts, automatic choices and text decoding from YuE2-ComfyUI's midi/tracks.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction
import kotlin.math.abs

object MidiParts {
    val families = listOf("Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass", "Reed",
        "Pipe", "Synth Lead", "Synth Pad", "Synth Effects", "Ethnic", "Percussive", "Sound Effects")
    private val voiceName = Regex("vocal|voice|vox|melod|sing|lead|вокал|голос|мелод", RegexOption.IGNORE_CASE)
    private val instrumentName = Regex("^ins\\b|instrument|solo", RegexOption.IGNORE_CASE)
    fun chordsName(name: String) = Regex("^\\s*chords?\\s*$", RegexOption.IGNORE_CASE).matches(name)
    // (Python's str.split: any Unicode space. Not Java's Unicode flag - Android's ICU regex refuses it; \p{Z} reads alike)
    internal fun fold(text: String) = text.trim().split(Regex("[\\s\\p{Z}\\u0085]+")).joinToString(" ")
    private fun raw(data: ByteArray) = data.filter { it != 0.toByte() }.toByteArray()
    private fun strict(data: ByteArray, charset: String) = Charset.forName(charset).newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(data)).toString()

    fun encodingOf(data: ByteArray): String {
        val bytes = raw(data)
        val value = try { strict(bytes, "UTF-8") } catch (_: java.nio.charset.CharacterCodingException) {
            val high = bytes.count { (it.toInt() and 255) >= 192 }
            val letters = high + bytes.count { it.toInt() in 65..90 || it.toInt() in 97..122 }
            return if (high >= 3 && high * 2 >= letters) "windows-1251" else "ISO-8859-1"
        }
        val wide = value.filter { it.code > 127 }
        if (wide.length >= 3 && wide.all { it.code <= 255 } && wide.length * 2 >= value.count { it.isLetter() }) {
            try { strict(value.toByteArray(Charsets.ISO_8859_1), "windows-1251"); return "utf-8-cp1251" }
            catch (_: java.nio.charset.CharacterCodingException) { }
        }
        return "UTF-8"
    }
    fun textOf(data: ByteArray, encoding: String): String {
        val bytes = raw(data)
        return if (encoding == "utf-8-cp1251") bytes.toString(Charsets.UTF_8).toByteArray(Charsets.ISO_8859_1).toString(Charset.forName("windows-1251"))
        else bytes.toString(Charset.forName(encoding))
    }
    fun decode(data: ByteArray) = fold(textOf(data, encodingOf(data)))

    class Part(val number: Int, val track: Int, val channel: Int, val notes: List<MidiReader.Note>, val program: Int, val name: String) {
        val median: Double get() = notes.map { it.pitch }.sorted().let { (it[(it.size - 1) / 2] + it[it.size / 2]) / 2.0 }
        val drums get() = channel == 9
        val family get() = if (drums) "Drums" else families[program / 8]
        val bass get() = !drums && (program in 32..39 || median < 48)
        val chordShare get() = notes.groupingBy { it.start }.eachCount().values.filter { it > 1 }.sum().toDouble() / notes.size
        val polyphony: Int get() {
            var sounding = 0; var most = 0
            (notes.map { it.start to 1 } + notes.map { it.end to -1 }).sortedWith(compareBy({ it.first }, { it.second })).forEach {
                sounding += it.second; most = maxOf(most, sounding)
            }
            return most
        }
    }
    data class Choice(val voice: Part, val instrument: Part?, val voiceWhy: Why, val instrumentWhy: Why)

    fun parts(song: MidiReader.Song): List<Part> {
        val programs = linkedMapOf<Int, Int>()
        for (t in song.tracks) for ((c, p) in t.programs) programs.putIfAbsent(c, p)
        val found = mutableListOf<Part>()
        for (t in song.tracks) {
            val channels = t.notes.map { it.channel }.distinct().sorted(); val name = decode(t.name)
            for (c in channels) found += Part(found.size + 1, t.index, c, t.notes.filter { it.channel == c },
                t.programs[c] ?: programs[c] ?: 0, if (channels.size > 1) (if (name.isEmpty()) "" else "$name ") + "channel ${c + 1}" else name)
        }
        return found
    }

    fun choose(found: List<Part>, vocal: Int? = null, instrument: Int? = null, syllables: List<Int> = emptyList(), division: Int = 480): Choice {
        val pitched = found.filter { !it.drums }
        if (pitched.isEmpty()) fail(ErrorCode.NO_PITCHED_NOTES)
        fun numbered(n: Int?, widget: String): Part? {
            if (n == null) return null
            val listing = found.flatMap { listOf(it.number.toString(), it.name, it.family) }
            val part = found.find { it.number == n } ?: fail(ErrorCode.TRACK_NUMBER, widget, n, found.size, *listing.toTypedArray())
            if (part.drums) fail(ErrorCode.DRUM_TRACK, widget, n, *listing.toTypedArray())
            return part
        }
        var why = Why.CHOSEN
        val voice = numbered(vocal, "vocal_track") ?: run {
            val usable = pitched.filter { !chordsName(it.name) }.ifEmpty { pitched }
            fun share(p: Part) = if (syllables.isEmpty()) 0.0 else syllables.count { s -> p.notes.any { abs(it.start.toLong() - s) <= maxOf(1, division / 8) } }.toDouble() / syllables.size
            val karaoke = usable.maxByOrNull { share(it) }!!
            val named = usable.filter { voiceName.containsMatchIn(it.name) && !it.bass }
            when {
                syllables.isNotEmpty() && share(karaoke) >= 0.5 -> { why = Why.KARAOKE; karaoke }
                named.isNotEmpty() -> { why = Why.NAME; named.maxByOrNull { it.notes.size }!! }
                else -> {
                    why = Why.HIGHEST
                    val busiest = usable.maxOf { it.notes.size }
                    usable.filter { !it.bass && it.notes.size >= busiest * 0.25 }.ifEmpty { usable }.maxByOrNull { it.median - 12 * it.chordShare }!!
                }
            }
        }
        var insWhy = Why.NOT_SELECTED
        val ins = if (instrument == -1) null else numbered(instrument, "instrument_track")?.also {
            if (it === voice) fail(ErrorCode.SAME_TRACK, voice.number)
            insWhy = Why.CHOSEN
        } ?: run {
            val usable = pitched.filter { it !== voice && !chordsName(it.name) }
            val named = usable.filter { instrumentName.containsMatchIn(it.name) }
            if (named.isNotEmpty()) { insWhy = Why.NAME; named.maxByOrNull { it.notes.size } }
            else {
                val busiest = usable.maxOfOrNull { it.notes.size } ?: 0
                usable.filter { !it.bass && it.median >= 55 && it.chordShare <= 0.5 && it.notes.size >= busiest * 0.25 }
                    .maxByOrNull { it.notes.size * (1 - it.chordShare) }?.also { insWhy = Why.BUSIEST }
            }
        }
        return Choice(voice, ins, why, insWhy)
    }
}
