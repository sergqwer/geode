package xyz.tifu.geode.music

import java.io.ByteArrayOutputStream

/**
 * A song's score as a MIDI file (YuE2-ComfyUI's midi/export.py, Apache-2.0): to hear the tune before it is sung (the
 * phone's own MIDI synthesizer plays it: MediaPlayer) and to take it to a sequencer. A conductor track carries the
 * tempo, every meter at the bar it starts in, every key and a marker at each section the score names; a track follows
 * for each line that sounds - "Vocal" on channel 1 with a voice sound, "Ins" on channel 2 with a piano - and, where the
 * score has chord symbols, "Chords" on channel 3 with strings holding each chord until the next one (the root between
 * C3 and B3, the chord stacked above it, a slash bass an octave below). Every track runs to the score's end, so closing
 * bars of rest are kept. The track names are the ones a MIDI import looks for.
 */
object ScoreMidi {
    const val DIVISION = 480
    private const val CONDUCTOR = "YuE2 score"

    /** A line of the score as a track: its name, channel, General MIDI program (from 0), velocity. */
    private class Line(val name: String, val channel: Int, val program: Int, val velocity: Int)

    private val LINES = listOf(Line("Vocal", 0, 53, 100), Line("Ins", 1, 0, 90))
    private val CHORDS = Line("Chords", 2, 48, 64)
    private const val CHORD_ROOT = 48

    /** The steps above the root of every chord quality the score format knows, as the score spells them. */
    val QUALITY_STEPS: Map<String, List<Int>> = linkedMapOf(
        "" to listOf(0, 4, 7), "m" to listOf(0, 3, 7), "dim" to listOf(0, 3, 6), "aug" to listOf(0, 4, 8),
        "7" to listOf(0, 4, 7, 10), "maj7" to listOf(0, 4, 7, 11), "m7" to listOf(0, 3, 7, 10),
        "dim7" to listOf(0, 3, 6, 9), "m7b5" to listOf(0, 3, 6, 10), "sus4" to listOf(0, 5, 7), "sus2" to listOf(0, 2, 7),
        "6" to listOf(0, 4, 7, 9), "m6" to listOf(0, 3, 7, 9), "7sus4" to listOf(0, 5, 7, 10), "m(maj7)" to listOf(0, 3, 7, 11),
    )
    private val SYMBOL = Regex("([A-G](?:bb|##|b|#)?)(.*?)(?:/([A-G](?:bb|##|b|#)?))?")

    private fun pitchClass(name: String): Int =
        Math.floorMod(AbcScore.NATURAL.getValue(name[0]) + name.count { it == '#' } - name.count { it == 'b' }, 12)

    private fun tick(quarters: AbcScore.Fraction): Int = quarters.ticks(DIVISION)

    /** The pitches a chord symbol is held as; IllegalArgumentException: not a symbol the score format knows. */
    fun chordPitches(symbol: String): List<Int> {
        val m = SYMBOL.matchEntire(symbol.trim()) ?: throw IllegalArgumentException("'$symbol' is not a chord symbol")
        val steps = QUALITY_STEPS[m.groupValues[2]] ?: throw IllegalArgumentException("'$symbol' is not a chord symbol")
        val root = CHORD_ROOT + pitchClass(m.groupValues[1])
        val pitches = steps.map { root + it }.toMutableList()
        if (m.groupValues[3].isNotEmpty()) pitches.add(0, CHORD_ROOT - 12 + pitchClass(m.groupValues[3]))
        return pitches
    }

    /** The score as a format-1 MIDI file; AbcScore.AbcError says why a score cannot be read. */
    fun of(text: String): ByteArray {
        val score = AbcScore.parse(text)
        val vocal = score.voices.getValue("Vocal")
        val end = tick(vocal.time)
        val conductor = mutableListOf(0 to Smf.text(Smf.NAME, CONDUCTOR), 0 to Smf.tempo(Math.round(60e6 / score.bpm).toInt()))
        var meter: AbcScore.Meter? = null
        for (bar in vocal.bars) if (bar.meter != meter) {
            conductor += tick(bar.start) to Smf.meter(bar.meter.numerator.toInt(), bar.meter.denominator.toInt())
            meter = bar.meter
        }
        var key: String? = null
        for (k in vocal.keys) if (k.name != key) {
            AbcScore.KEYS[k.name]?.let { sharps -> conductor += tick(k.start) to Smf.key(sharps, k.name.endsWith("m")) }
            key = k.name
        }
        for (s in AbcScore.sections(text)) if (s.name.isNotEmpty()) conductor += tick(s.start) to Smf.text(Smf.MARKER, s.name)
        val tracks = mutableListOf<List<Pair<Int, ByteArray>>>(conductor)
        for (line in LINES) {
            val notes = score.voices.getValue(line.name).notes
            if (notes.isEmpty()) continue
            val events = mutableListOf(0 to Smf.text(Smf.NAME, line.name), 0 to Smf.program(line.channel, line.program))
            for (n in notes) {
                events += tick(n.start) to Smf.noteOn(line.channel, n.pitch, line.velocity)
                events += tick(n.start + n.duration) to Smf.noteOff(line.channel, n.pitch)
            }
            tracks += events
        }
        if (vocal.chords.isNotEmpty()) {
            val events = mutableListOf(0 to Smf.text(Smf.NAME, CHORDS.name), 0 to Smf.program(CHORDS.channel, CHORDS.program))
            vocal.chords.forEachIndexed { i, c ->
                val stop = vocal.chords.getOrNull(i + 1)?.start ?: vocal.time
                if (stop <= c.start) return@forEachIndexed
                for (p in chordPitches(c.name)) {
                    events += tick(c.start) to Smf.noteOn(CHORDS.channel, p, CHORDS.velocity)
                    events += tick(stop) to Smf.noteOff(CHORDS.channel, p)
                }
            }
            tracks += events
        }
        return Smf.write(tracks, DIVISION, end)
    }
}

/** Standard MIDI files: the events a score needs, and a format-1 file of tracks (YuE2-ComfyUI's midi/smf.py). */
object Smf {
    const val NAME = 0x03
    const val MARKER = 0x06
    private const val TEMPO = 0x51
    private const val METER = 0x58
    private const val KEY = 0x59
    private const val END = 0x2F

    /** A variable-length quantity: delta times and meta lengths. */
    fun number(value: Int): ByteArray {
        require(value in 0..0x0FFFFFFF) { "$value does not fit a MIDI variable-length number" }
        val out = ArrayList<Byte>()
        var v = value
        out += (v and 0x7F).toByte()
        v = v shr 7
        while (v != 0) {
            out += ((v and 0x7F) or 0x80).toByte()
            v = v shr 7
        }
        return out.reversed().toByteArray()
    }

    fun meta(kind: Int, payload: ByteArray): ByteArray = byteArrayOf(0xFF.toByte(), kind.toByte()) + number(payload.size) + payload

    /** Microseconds per quarter note. */
    fun tempo(microseconds: Int) = meta(TEMPO, byteArrayOf((microseconds shr 16).toByte(), (microseconds shr 8).toByte(), microseconds.toByte()))

    /** A time signature, the metronome at every quarter, eight 32nds to a quarter. */
    fun meter(numerator: Int, denominator: Int) =
        meta(METER, byteArrayOf(numerator.toByte(), (31 - Integer.numberOfLeadingZeros(denominator)).toByte(), 24, 8))

    /** A key signature: sharps positive, flats negative. */
    fun key(sharps: Int, minor: Boolean) = meta(KEY, byteArrayOf(sharps.toByte(), if (minor) 1 else 0))

    /** A text-like meta event (a name, a marker, a lyric) in UTF-8. */
    fun text(kind: Int, value: String) = meta(kind, value.toByteArray(Charsets.UTF_8))

    fun program(channel: Int, value: Int) = byteArrayOf((0xC0 or channel).toByte(), value.toByte())
    fun noteOn(channel: Int, pitch: Int, velocity: Int) = byteArrayOf((0x90 or channel).toByte(), pitch.toByte(), velocity.toByte())
    fun noteOff(channel: Int, pitch: Int) = byteArrayOf((0x80 or channel).toByte(), pitch.toByte(), 0)

    /** Meta events first at a tick, then note-offs, then the rest in the order given. */
    private fun rank(e: ByteArray): Int = when {
        e[0] == 0xFF.toByte() -> 0
        (e[0].toInt() and 0xF0) == 0x80 -> 1
        else -> 2
    }

    /**
     * A format-1 file of [tracks], each a list of (tick, event). Sorted by tick; at one tick meta events first and
     * note-offs before the rest (a note ending where the same pitch starts again is not cut short), the same rank in
     * the order given. Every track ends at [end] when that is later than its last event.
     */
    fun write(tracks: List<List<Pair<Int, ByteArray>>>, division: Int = ScoreMidi.DIVISION, end: Int = 0): ByteArray {
        val out = ByteArrayOutputStream()
        fun u32(v: Int) = byteArrayOf((v shr 24).toByte(), (v shr 16).toByte(), (v shr 8).toByte(), v.toByte())
        fun u16(v: Int) = byteArrayOf((v shr 8).toByte(), v.toByte())
        out.write("MThd".toByteArray(Charsets.US_ASCII) + u32(6) + u16(1) + u16(tracks.size) + u16(division))
        for (events in tracks) {
            val ordered = events.withIndex().sortedWith(compareBy({ it.value.first }, { rank(it.value.second) }, { it.index }))
            val body = ByteArrayOutputStream()
            var last = 0
            for ((_, e) in ordered) {
                body.write(number(e.first - last))
                body.write(e.second)
                last = e.first
            }
            body.write(number(maxOf(0, end - last)))
            body.write(meta(END, ByteArray(0)))
            val b = body.toByteArray()
            out.write("MTrk".toByteArray(Charsets.US_ASCII) + u32(b.size) + b)
        }
        return out.toByteArray()
    }
}
