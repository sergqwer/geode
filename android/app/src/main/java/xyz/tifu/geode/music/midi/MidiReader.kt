/** The reader of YuE2-ComfyUI's midi/smf.py (Apache-2.0), keeping text bytes for karaoke decoding. */
package xyz.tifu.geode.music.midi

object MidiReader {
    data class Note(val start: Int, val end: Int, val pitch: Int, val velocity: Int, val channel: Int)
    data class Text(val tick: Int, val kind: Int, val bytes: ByteArray)
    data class Tempo(val tick: Int, val value: Int)
    data class Meter(val tick: Int, val numerator: Int, val denominator: Int)
    data class Key(val tick: Int, val sharps: Int, val minor: Boolean)
    class Track(val index: Int) {
        var name = byteArrayOf()
        val notes = mutableListOf<Note>()
        val programs = linkedMapOf<Int, Int>()
        val texts = mutableListOf<Text>()
        var end = 0
    }
    class Song(val format: Int, val division: Int) {
        val tracks = mutableListOf<Track>()
        val tempos = mutableListOf<Tempo>()
        val meters = mutableListOf<Meter>()
        val keys = mutableListOf<Key>()
        val markers = mutableListOf<Text>()
        val end get() = tracks.maxOfOrNull { it.end } ?: 0
    }

    private class Cursor(val data: ByteArray, val context: String) {
        var at = 0
        fun byte(): Int {
            if (at >= data.size) fail(ErrorCode.TRUNCATED, context)
            return data[at++].toInt() and 255
        }
        fun take(n: Int): ByteArray {
            if (n < 0 || n > data.size - at) fail(ErrorCode.TRUNCATED, context)
            return data.copyOfRange(at, at + n).also { at += n }
        }
        fun number(): Int {
            var n = 0
            repeat(4) { val b = byte(); n = (n shl 7) or (b and 127); if (b < 128) return n }
            fail(ErrorCode.LONG_NUMBER, context)
        }
        fun word() = (byte() shl 8) or byte()
        fun size(): Int {
            val n = (byte().toLong() shl 24) or (byte().toLong() shl 16) or (byte().toLong() shl 8) or byte().toLong()
            if (n > Int.MAX_VALUE) fail(ErrorCode.TRUNCATED, context)
            return n.toInt()
        }
    }

    fun read(bytes: ByteArray): Song {
        var data = bytes
        fun tag(at: Int, value: String) = at >= 0 && at + value.length <= data.size &&
            data.copyOfRange(at, at + value.length).toString(Charsets.US_ASCII) == value
        if (tag(0, "RIFF") && tag(8, "RMID")) {
            var at = 12
            var inner: ByteArray? = null
            while (at <= data.size - 8) {
                val n = (0..3).fold(0L) { a, i -> a or ((data[at + 4 + i].toLong() and 255) shl (8 * i)) }
                if (tag(at, "data")) { inner = data.copyOfRange(at + 8, minOf(data.size.toLong(), at + 8L + n).toInt()); break }
                if (n + 8 > data.size - at) break
                at += 8 + n.toInt() + (n.toInt() and 1)
            }
            data = inner ?: fail(ErrorCode.RIFF_NO_DATA)
        }
        if (!tag(0, "MThd")) fail(ErrorCode.NOT_MIDI)
        if (data.size < 14) fail(ErrorCode.SHORT_HEADER)
        val c = Cursor(data, "header"); c.take(4)
        val size = c.size()
        if (size < 6) fail(ErrorCode.HEADER_LENGTH, size)
        val format = c.word(); val count = c.word(); val division = c.word()
        if (division and 0x8000 != 0) fail(ErrorCode.SMPTE)
        if (division == 0) fail(ErrorCode.ZERO_DIVISION)
        val song = Song(format, division)
        c.take(size - 6)
        while (c.at <= data.size - 8 && song.tracks.size < count) {
            val kind = c.take(4).toString(Charsets.US_ASCII); val length = c.size()
            // Like the Python reader, accept a short chunk if its events still finish cleanly.
            val body = c.take(minOf(length, data.size - c.at))
            if (kind == "MTrk") song.tracks += track(body, song.tracks.size, song)
        }
        if (song.tracks.isEmpty()) fail(ErrorCode.NO_TRACK)
        song.tempos.sortBy { it.tick }; song.meters.sortBy { it.tick }; song.keys.sortBy { it.tick }; song.markers.sortBy { it.tick }
        return song
    }

    private fun track(body: ByteArray, index: Int, song: Song): Track {
        val t = Track(index); val c = Cursor(body, "track ${index + 1}")
        var tick = 0; var status = -1
        val sounding = linkedMapOf<Pair<Int, Int>, MutableList<Pair<Int, Int>>>()
        while (c.at < body.size) {
            val delta = c.number()
            if (tick > Int.MAX_VALUE - delta) fail(ErrorCode.TOO_LONG, 20000)
            tick += delta
            val first = c.byte()
            if (first == 255) {
                val kind = c.byte(); val p = c.take(c.number())
                fun u(i: Int) = p[i].toInt() and 255
                if (kind == 47) break
                when {
                    kind == 81 && p.size == 3 -> { val v = (u(0) shl 16) or (u(1) shl 8) or u(2); if (v > 0) song.tempos += Tempo(tick, v) }
                    kind == 88 && p.size >= 2 && u(0) > 0 && u(1) <= 6 -> song.meters += Meter(tick, u(0), 1 shl u(1))
                    kind == 89 && p.size >= 2 && p[0].toInt() in -7..7 && u(1) in 0..1 -> song.keys += Key(tick, p[0].toInt(), u(1) == 1)
                    kind == 6 -> song.markers += Text(tick, kind, p)
                    kind == 3 && t.name.isEmpty() -> t.name = p
                }
                if (kind in 1..7) t.texts += Text(tick, kind, p)
                continue
            }
            if (first == 240 || first == 247) { c.take(c.number()); continue }
            if (first >= 128) {
                if (first >= 240) fail(ErrorCode.SYSTEM_EVENT, index + 1, first.toString(16))
                status = first
            } else if (status < 0) fail(ErrorCode.NO_STATUS, index + 1)
            val kind = status shr 4; val channel = status and 15
            val a = if (first >= 128) c.byte() else first
            val b = if (kind == 12 || kind == 13) 0 else c.byte()
            if (a >= 128 || b >= 128) fail(ErrorCode.SHORT_EVENT, index + 1)
            val key = channel to a
            when {
                kind == 9 && b > 0 -> sounding.getOrPut(key) { mutableListOf() } += tick to b
                kind == 8 || kind == 9 -> sounding[key]?.let { if (it.isNotEmpty()) { val (s, v) = it.removeAt(0); t.notes += Note(s, tick, a, v, channel) } }
                kind == 12 -> t.programs.putIfAbsent(channel, a)
            }
        }
        t.end = tick
        for ((key, opened) in sounding) for ((s, v) in opened) t.notes += Note(s, tick, key.second, v, key.first)
        t.notes.sortWith(compareBy({ it.start }, { it.end }, { it.pitch }, { it.velocity }, { it.channel }))
        return t
    }
}
