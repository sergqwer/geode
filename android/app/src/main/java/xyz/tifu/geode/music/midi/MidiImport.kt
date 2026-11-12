/** Score conversion and the non-node choices of YuE2-ComfyUI's midi/score.py and load_midi.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

object MidiImport {
    enum class Role { VOICE, INSTRUMENT, CHORDS, DRUMS, NONE }
    data class Track(val number: Int, val name: String, val family: String, val notes: Int, val low: Int, val high: Int,
                     val role: Role, val why: Why)
    data class Result(val abc: String, val lyrics: String, val tracks: List<Track>, val bars: Int, val bpm: Int,
                      val meter: String, val key: String, val seconds: Double, val notices: List<Notice>)
    data class Beat(val tick: Fraction, val number: Int, val numerator: Int, val denominator: Int)
    private val major = listOf(6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88)
    private val minor = listOf(6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17)
    private val majorNames = "C Db D Eb E F F# G Ab A Bb B".split(' ')
    private val minorNames = "Cm C#m Dm D#m Em Fm F#m Gm G#m Am Bbm Bm".split(' ')

    fun secondsAt(song: MidiReader.Song, tick: Fraction): Double {
        var seconds = 0.0; var last = 0; var tempo = 500000
        for ((change, value) in song.tempos) {
            if (Fraction(change.toLong()) >= tick) break
            seconds += (change.toDouble() - last) * tempo / 1e6 / song.division
            last = change; tempo = value
        }
        return seconds + (tick.value() - last) * tempo / 1e6 / song.division
    }
    fun beats(song: MidiReader.Song, end: Int): List<Beat> {
        val meters = mutableListOf<MidiReader.Meter>()
        for (m in song.meters) {
            if (meters.lastOrNull()?.tick == m.tick) meters[meters.lastIndex] = m
            else if (meters.lastOrNull()?.let { it.numerator == m.numerator && it.denominator == m.denominator } != true) meters += m
        }
        if (meters.isEmpty() || meters[0].tick > 0) meters.add(0, MidiReader.Meter(0,4,4))
        val rows = mutableListOf<Beat>(); var tick = ZERO; var number = 1; var index = 0
        while (true) {
            while (index + 1 < meters.size && Fraction(meters[index + 1].tick.toLong()) <= tick) { index++; number = 1 }
            val m = meters[index]; rows += Beat(tick, number, m.numerator, m.denominator)
            if (number == 1 && tick >= Fraction(end.toLong()) && rows.size > 1) return rows
            if (rows.size > 20000) fail(ErrorCode.TOO_LONG, 20000)
            val step = Fraction(song.division.toLong() * 4, m.denominator.toLong())
            val next = meters.getOrNull(index + 1)?.let { Fraction(it.tick.toLong()) }
            tick = if (next != null && tick < next && next < tick + step) next else tick + step
            number = number % m.numerator + 1
        }
    }
    fun gridOf(notes: List<MidiReader.Note>, division: Int): Int {
        val edges = notes.flatMap { listOf(it.start.toLong(), it.end.toLong()) }
        if (edges.all { it * 4 % division == 0L }) return 16
        return if (edges.count { it * 8 % division == 0L } >= 0.9 * edges.size) 32 else 16
    }
    fun pointsOf(rows: List<Beat>, subbeats: Int): List<Fraction> = rows.zipWithNext().flatMap { (a,b) ->
        (0 until subbeats).map { a.tick + (b.tick - a.tick) * it.toLong() / subbeats.toLong() }
    } + rows.last().tick
    fun snap(points: List<Fraction>, tick: Fraction): Int {
        val found = points.binarySearch(tick)
        if (found >= 0) return found
        val at = -found - 1
        if (at == 0) return 0
        if (at == points.size) return points.lastIndex
        return if (points[at] - tick < tick - points[at - 1]) at else at - 1
    }
    fun octaveShift(median: Double, low: Int, high: Int) = when {
        median < low -> 12 * ceil((low - median) / 12).toInt()
        median > high -> -12 * ceil((median - high) / 12).toInt()
        else -> 0
    }
    fun skyline(notes: List<TimedNote>): Pair<List<TimedNote>, Int> {
        val kept = mutableListOf<TimedNote>(); var struck = 0
        for (n in notes.sortedWith(compareBy({ it.start }, { -it.pitch }, { it.end }))) {
            val last = kept.lastOrNull()
            when {
                last == null || n.start >= last.end -> kept += n
                n.start == last.start -> struck++
                n.pitch > last.pitch -> { kept[kept.lastIndex] = last.copy(end = n.start); kept += n }
                n.end - last.end > last.end - n.start -> kept += n.copy(start = last.end)
            }
        }
        return kept to struck
    }
    private fun correlation(weights: List<Double>, profile: List<Double>, root: Int): Double {
        val a = (0..11).map { weights[(root + it) % 12] }; val am = a.average(); val bm = profile.average()
        val top = (0..11).sumOf { (a[it] - am) * (profile[it] - bm) }
        val bottom = sqrt(a.sumOf { (it-am)*(it-am) } * profile.sumOf { (it-bm)*(it-bm) })
        return if (bottom == 0.0) 0.0 else top / bottom
    }
    fun signatureName(sharps: Int, minor: Boolean): String = AbcScore.KEYS.entries.first { it.value == sharps && it.key.endsWith('m') == minor }.key
    /** The first eight bar numbers and the number omitted, for a localized bar-list message. */
    internal fun barArguments(numbers: List<Int>) = listOf(numbers.take(8).joinToString(","), maxOf(0, numbers.size - 8).toString())
    private fun keys(song: MidiReader.Song, weights: List<Double>): List<Pair<Fraction,String>> {
        var best = Triple(Double.NEGATIVE_INFINITY, 0, false)
        for (r in 0..11) for (m in listOf(false,true)) {
            val c = correlation(weights, if (m) minor else major, r)
            if (c >= best.first) best = Triple(c,r,m)
        }
        val distinct = mutableListOf<MidiReader.Key>()
        for (k in song.keys) if (distinct.lastOrNull()?.let { it.sharps == k.sharps && it.minor == k.minor } != true) distinct += k
        if (distinct.size >= 2) return distinct.mapIndexed { i,k -> Fraction(if (i == 0) 0 else k.tick.toLong()) to signatureName(k.sharps,k.minor) }
        distinct.firstOrNull()?.let {
            val root = Math.floorMod(7 * it.sharps + if (it.minor) 9 else 0, 12)
            if (correlation(weights, if (it.minor) minor else major, root) >= best.first - 0.05) return listOf(ZERO to signatureName(it.sharps,it.minor))
        }
        return listOf(ZERO to (if (best.third) minorNames else majorNames)[best.second])
    }

    /** Track numbers count parts with notes; a shared track is split into channels in ascending order. */
    fun read(bytes: ByteArray, mode: String, vocalTrack: Int? = null, instrumentTrack: Int? = null, withoutSections: Boolean = true): Result {
        if (mode != "melody" && mode != "full") fail(ErrorCode.MODE, mode)
        val song = MidiReader.read(bytes); val found = MidiParts.parts(song)
        if (found.isEmpty()) fail(ErrorCode.NO_NOTES)
        val words = Karaoke.read(song); val chosen = MidiParts.choose(found,vocalTrack,instrumentTrack,words?.syllables.orEmpty(),song.division)
        val end = maxOf(song.end, found.maxOf { p -> p.notes.maxOf { it.end } })
        val rows = beats(song,end)
        val selected = listOfNotNull(chosen.voice, chosen.instrument)
        val grid = gridOf(selected.flatMap { it.notes },song.division)
        val subbeats = maxOf(1, grid / rows.minOf { it.denominator })
        val points = pointsOf(rows,subbeats); val last = rows.last().tick
        val length = secondsAt(song,last); val bpm = Math.rint(last.value() / song.division * 60 / length).toInt().coerceAtLeast(1)
        val notices = mutableListOf<Notice>()
        val tempos = song.tempos.filter { Fraction(it.tick.toLong()) < last }.map { 60e6 / it.value }.ifEmpty { listOf(120.0) }
        if (tempos.max() > tempos.min() * 1.01) notices += Notice(Code.TEMPO_AVERAGED, listOf(tempos.min(),tempos.max(),bpm.toDouble()).map { Math.rint(it).toInt().toString() })
        val edges = selected.flatMap { it.notes }.flatMap { listOf(it.start.toLong(),it.end.toLong()) }
        if (grid == 16 && edges.count { it * 4 % song.division == 0L } < 0.9 * edges.size) notices += Notice(Code.PLAYED_IN)
        fun quantized(part: MidiParts.Part) = part.notes.mapNotNull {
            val start = snap(points,Fraction(it.start.toLong())); val stop = minOf(maxOf(snap(points,Fraction(it.end.toLong())),start+1),points.lastIndex)
            if (stop > start) TimedNote(points[start],points[stop],it.pitch) else null
        }
        fun line(part: MidiParts.Part, high: Int, name: String): List<TimedNote> {
            val (kept,struck) = skyline(quantized(part))
            val pitches = kept.map { it.pitch }.sorted()
            val shift = if (pitches.isEmpty()) 0 else octaveShift((pitches[(pitches.size-1)/2]+pitches[pitches.size/2])/2.0,60,high)
            if (shift != 0) notices += Notice(Code.OCTAVE_MOVED,listOf(name,part.number.toString(),shift.toString()))
            if (struck > 0) notices += Notice(Code.TOP_NOTES_KEPT,listOf(part.number.toString(),struck.toString()))
            return kept.filter { it.pitch + shift in 0..127 }.map { it.copy(start = it.start / song.division.toLong(),end = it.end / song.division.toLong(),pitch = it.pitch+shift) }
        }
        val vocal = line(chosen.voice,82,"voice"); val instrument = chosen.instrument?.let { line(it,88,"instrument") }.orEmpty()
        val weights = MutableList(12) { 0.0 }
        for (part in found.filter { !it.drums }) for (n in part.notes) weights[n.pitch % 12] += n.end.toDouble() - n.start
        val keyed = keys(song,weights)
        val downbeats = rows.filter { it.number == 1 }
        val bars = downbeats.zipWithNext().map { (a,b) ->
            val duration = (b.tick-a.tick) / song.division.toLong()
            if (duration == Fraction(a.numerator.toLong()*4,a.denominator.toLong())) ScoreBar(a.tick,b.tick,a.numerator,a.denominator)
            else { val meter = duration / 4; ScoreBar(a.tick,b.tick,meter.numerator.toInt(),meter.denominator.toInt()) }
        }
        val markers = Karaoke.markerSections(song)
        val sections = markers.ifEmpty {
            val starts = words?.sections?.map { it.tick to it.label }.orEmpty()
            if (starts.isNotEmpty() && bars.isNotEmpty() && Fraction(starts[0].first.toLong()) >= bars[0].end) listOf(0 to "intro") + starts else starts
        }.filter { Fraction(it.first.toLong()) < last }
        val chordsPart = if (mode == "full") found.firstOrNull { !it.drums && MidiParts.chordsName(it.name) } else null
        var harmony = emptyList<Harmony>()
        if (mode == "full") {
            val signature = AbcScore.KEYS.getValue(keyed[0].second); val tonic = Math.floorMod(7*signature,12)
            val scale = listOf(0,2,4,5,7,9,11).map { (tonic+it)%12 }.toSet()
            harmony = if (chordsPart != null) {
                val (read,unnamed) = MidiChords.read(quantized(chordsPart),last,scale)
                notices += Notice(Code.CHORDS_READ,listOf(chordsPart.number.toString(),chordsPart.name))
                if (unnamed.isNotEmpty()) notices += Notice(Code.UNNAMED_CHORDS,listOf(chordsPart.number.toString()) + barArguments(unnamed.map { t -> maxOf(1,bars.count { it.start <= t }) }.distinct().sorted()))
                read
            } else {
                notices += Notice(Code.CHORDS_GUESSED)
                MidiChords.guess(found.filter { !it.drums }.flatMap { p -> p.notes.map { TimedNote(Fraction(it.start.toLong()),Fraction(it.end.toLong()),it.pitch,if (p === chosen.voice) 0.5 else 1.0) } },MidiChords.spans(bars),signature<0,scale)
            }
        }
        val scoreBars = bars.map { it.copy(start = it.start / song.division.toLong(), end = it.end / song.division.toLong()) }
        val keyEvents = keyed.map { points[snap(points,it.first)] / song.division.toLong() to it.second }
        val labels = sections.map { points[snap(points,Fraction(it.first.toLong()))] / song.division.toLong() to it.second }
        val scoreHarmony = harmony.map { it.copy(start = it.start / song.division.toLong(),end = it.end / song.division.toLong()) }
        val unit = maxOf(grid, scoreBars.maxOf { it.denominator } * subbeats)
        var abc = ScoreEvents.build(scoreBars,vocal,instrument,keyEvents,labels,scoreHarmony,bpm,unit)
        // The native parser also checks that ties, voice grids, and key timelines agree.
        try { AbcScore.parse(abc) } catch (_: AbcScore.AbcError) { fail(ErrorCode.NOTATION) }
        val written = abc
        abc = if (withoutSections) Phrasing.bare(abc) else Phrasing.labelled(abc)
        val lyrics = words?.lyrics() ?: run {
            val parsed = AbcScore.parse(written)
            val scoreSections = AbcScore.sections(written)
            scoreSections.mapIndexedNotNull { i,s ->
                val stop = scoreSections.getOrNull(i+1)?.start ?: parsed.voices.getValue("Vocal").time
                if (vocal.any { it.start >= s.start && it.start < stop }) "[${Karaoke.tags[s.name] ?: "Verse"}]" else null
            }.joinToString("\n\n")
        }
        val tracks = found.map { p ->
            val role = when { p === chosen.voice -> Role.VOICE; p === chosen.instrument -> Role.INSTRUMENT; p === chordsPart -> Role.CHORDS; p.drums -> Role.DRUMS; else -> Role.NONE }
            val why = when (role) { Role.VOICE -> chosen.voiceWhy; Role.INSTRUMENT -> chosen.instrumentWhy; Role.CHORDS -> Why.CHORDS; Role.DRUMS -> Why.DRUMS; else -> Why.NOT_SELECTED }
            Track(p.number,p.name,p.family,p.notes.size,p.notes.minOf { it.pitch },p.notes.maxOf { it.pitch },role,why)
        }
        return Result(abc,lyrics,tracks,bars.size,bpm,"${rows[0].numerator}/${rows[0].denominator}",keyed[0].second,Math.rint(length*10)/10,notices)
    }
}
