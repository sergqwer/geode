/** Lyrics laid along a sectionless tune, ported from YuE2-ComfyUI's phrasing.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import java.text.Normalizer
import java.util.Locale
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
import kotlin.math.abs
import kotlin.math.ln

object Phrasing {
    internal val word = Regex("\\p{L}+(?:['’]\\p{L}+)*")
    internal const val cyrillicVowels = "аеёиоуыэюяіїє"
    internal val soundedI = Regex("(?<=[a-z])(?<![aeiouytscxgn])(?<!ll)i(?=[aou])")
    private val shortened = "every everything everybody everywhere everyday evening evenings different difference differently several favorite favourite camera chocolate interest interesting business vegetable comfortable temperature".split(' ').toSet()
    internal fun wide(c: Char) = c in '一'..'鿿' || c in '㐀'..'䶿' || c in '぀'..'ヿ' || c in '가'..'힯'
    internal fun plain(text: String) = Normalizer.normalize(text,Normalizer.Form.NFD).filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
    fun syllables(line: String) = word.findAll(line).sumOf { wordSyllables(it.value) }
    private fun wordSyllables(word: String): Int {
        val lower = word.lowercase(Locale.ROOT)
        if (lower.any { it in '\u0400'..'\u04ff' }) return maxOf(1,lower.count { it in cyrillicVowels })
        val wide = lower.count { wide(it) }; if (wide > 0) return wide
        val plain = plain(Regex("(?<=[aeiouy])[ïëüÿ]").replace(lower) { "|${it.value}" })
        if (plain.any { it in '\u0370'..'\u03ff' }) return maxOf(1,Regex("[αεηιουω]+").findAll(plain).count())
        var count = Regex("[aeiouyæøœ]+").findAll(plain).count()
        if (plain.endsWith("ing") && plain.length > 3 && plain[plain.length-4] in "aeiouy") count++
        count += soundedI.findAll(plain).count()
        if (plain in shortened) count--
        if (count > 1 && plain.length > 2) {
            if (plain.endsWith('e') && lower.last() !in "éèêë" && listOf("ee","ie","ye").none { plain.endsWith(it) } && !(plain.endsWith("le") && plain[plain.length-3] !in "aeiouy")) count--
            else if (plain.endsWith("es") && plain.length > 3 && plain[plain.length-3] !in "aeiouysxzcgh") count--
            else if (plain.endsWith("ed") && plain.length > 3 && plain[plain.length-3] !in "aeiouytd") count--
        }
        return maxOf(1,count)
    }
    fun named(abc: String) = abc.split('\n').any { it.startsWith("% ") }
    fun bare(abc: String) = abc.split('\n').filterNot { it.startsWith("% ") }.joinToString("\n")
    fun labelled(abc: String, label: String = "verse"): String {
        if (named(abc)) return abc
        val lines = abc.split('\n').toMutableList(); val at = lines.indexOf("V: Vocal")
        if (at < 0) return abc
        lines.add(at,"% $label"); return lines.joinToString("\n")
    }
    fun labelOf(tag: String): String {
        var clean = MidiParts.fold(tag.lowercase(Locale.ROOT).replace('_',' ')).replace(Regex("[\\s\\d:.#-]+$"),"")
        clean = mapOf("prechorus" to "pre-chorus", "pre chorus" to "pre-chorus", "post chorus" to "post-chorus", "hook" to "chorus", "refrain" to "chorus")[clean] ?: clean
        return if (clean in Karaoke.tags) clean else "verse"
    }
    data class LyricSection(val tag: String, val label: String, val lines: MutableList<String> = mutableListOf())
    fun lyricSections(lyrics: String): List<LyricSection> {
        val found = mutableListOf<LyricSection>()
        for (raw in lyrics.lines()) {
            val line = raw.trim(); if (line.isEmpty()) continue
            val tag = Regex("\\[\\s*([^\\]]*?)\\s*\\]").matchEntire(line)
            if (tag != null) { found += LyricSection(tag.groupValues[1],labelOf(tag.groupValues[1])); continue }
            if (syllables(line) == 0) continue
            if (found.isEmpty()) found += LyricSection("Verse","verse")
            found.last().lines += line
        }
        return found
    }
    data class Piece(val start: Fraction, val end: Fraction, val notes: Int, val breath: Boolean)
    internal fun phrases(notes: List<AbcScore.Note>): List<List<AbcScore.Note>> {
        val groups = mutableListOf<MutableList<AbcScore.Note>>()
        for (n in notes.sortedWith(compareBy({ it.start }, { it.pitch }, { it.duration }))) {
            val last = groups.lastOrNull()?.last()
            if (last != null && n.start - (last.start + last.duration) < Fraction(1,2) && last.duration < Fraction(2)) groups.last() += n
            else groups += mutableListOf(n)
        }
        return groups
    }
    fun pieces(notes: List<AbcScore.Note>, barStarts: List<Fraction>): List<Piece> {
        val found = mutableListOf<Piece>()
        for (group in phrases(notes)) {
            val chunks = mutableListOf<Pair<Int,MutableList<AbcScore.Note>>>()
            for (n in group) {
                val bar = barStarts.count { it <= n.start } - 1
                if (chunks.lastOrNull()?.first == bar) chunks.last().second += n else chunks += bar to mutableListOf(n)
            }
            for ((i, chunk) in chunks.withIndex()) found += Piece(chunk.second[0].start,chunk.second.last().let { it.start+it.duration },chunk.second.size,i==0)
        }
        return found
    }
    private data class Step(val k: Int, val m: Int, val origin: Int)
    internal fun path(found: List<Piece>, counts: List<Int>, start: Int, first: Boolean, bpm: Int): List<Pair<Int,Int>> {
        val size = found.size; val total = mutableListOf(0)
        found.forEach { total += total.last()+it.notes }
        fun cost(k: Int,m: Int,count: Int): Double {
            val notes = total[k+m]-total[k]; val seconds = (found[k+m-1].end-found[k].start).value()*60/bpm
            val pace = if (seconds>0) count/seconds else 28.0
            return abs(ln(notes.toDouble()/count)) + 2*(maxOf(0.0,ln(pace/3.5))+maxOf(0.0,ln(0.8/pace))) +
                0.15*(k+1 until k+m).count { found[it].breath } + if (k+m<size && !found[k+m].breath) 0.2 else 0.0
        }
        var best = DoubleArray(size+1) { Double.POSITIVE_INFINITY }; best[minOf(start,size)] = 0.0
        val steps = mutableListOf<Array<Step?>>()
        for ((index,count) in counts.withIndex()) {
            val jump = if (index==0 && !first) 0.25 else 0.5
            val ahead = Array(size+1) { Double.POSITIVE_INFINITY to -1 }; var low = Double.POSITIVE_INFINITY to -1
            for (k in 0..size) { ahead[k]=low; if (best[k]-0.05*k<low.first) low=best[k]-0.05*k to k }
            val behind = Array(size+1) { Double.POSITIVE_INFINITY to -1 }; low=Double.POSITIVE_INFINITY to -1
            for (k in size downTo 0) { behind[k]=low; if (best[k]<low.first) low=best[k] to k }
            val next = DoubleArray(size+1) { Double.POSITIVE_INFINITY }; val chosen = arrayOfNulls<Step>(size+1)
            for (k in 0 until size) {
                var came=best[k]; var origin=k
                if (ahead[k].second>=0 && ahead[k].first+jump+0.05*k<came) { came=ahead[k].first+jump+0.05*k; origin=ahead[k].second }
                if (behind[k].second>=0 && behind[k].first+jump+0.2<came) { came=behind[k].first+jump+0.2; origin=behind[k].second }
                if (!came.isFinite()) continue
                for (m in 1..minOf(8,size-k)) {
                    val value=came+cost(k,m,count)
                    if (value<next[k+m]) { next[k+m]=value; chosen[k+m]=Step(k,m,origin) }
                }
            }
            steps+=chosen; best=next
        }
        var end=(0..size).minByOrNull { best[it] }!!; val spans=mutableListOf<Pair<Int,Int>>()
        for (chosen in steps.asReversed()) { val s=chosen[end] ?: return emptyList(); spans+=s.k to s.m; end=s.origin }
        return spans.asReversed()
    }
    fun chordLabel(symbol: String): String {
        val m=Regex("([A-G](?:bb|##|b|#)?)(.*?)(?:/([A-G](?:bb|##|b|#)?))?").matchEntire(symbol) ?: fail(ErrorCode.NOTATION)
        val quality=MidiChords.qualities.entries.firstOrNull { it.value==m.groupValues[2] }?.key ?: fail(ErrorCode.NOTATION)
        return m.groupValues[1]+":"+quality+if (m.groupValues[3].isEmpty()) "" else "/"+m.groupValues[3]
    }
    /** The upstream decimal ceiling, before adapting it to the app's whole-second API. */
    fun ceiling(seconds: Double) = Math.rint((seconds*1.1+2)*10)/10

    private data class Sung(val section: LyricSection,val counts: List<Int>,val spans: List<Pair<Int,Int>>)
    private data class Window(val start: Fraction,var end: Fraction)
    private data class Run(var b0: Int,var b1: Int,val windows: MutableList<Window>,val labels: MutableList<Pair<Fraction,String>>)
    internal fun lay(abc: String,lyrics: String): Underlay.Laid? {
        if (named(abc)) return null
        val parsed=try { AbcScore.parse(abc) } catch (_: AbcScore.AbcError) { return null }
        val vocal=parsed.voices.getValue("Vocal"); val words=lyricSections(lyrics)
        if (vocal.notes.isEmpty() || words.none { it.lines.isNotEmpty() }) return null
        val starts=vocal.bars.map { it.start }; val found=pieces(vocal.notes,starts)
        val sung=mutableListOf<Sung>(); val done=mutableMapOf<List<String>,List<Pair<Int,Int>>>(); var position=0
        for (section in words.filter { it.lines.isNotEmpty() }) {
            val counts=section.lines.map { maxOf(1,syllables(it)) }
            val key=section.lines.map { MidiParts.fold(it.lowercase(Locale.ROOT).replace(Regex("[^\\p{L}\\p{N}_\\s\\p{Z}']")," ")) }
            val spans=done.getOrPut(key) { path(found,counts,position,sung.isEmpty(),parsed.bpm) }
            if (spans.isEmpty()) return null
            position=spans.last().first+spans.last().second; sung+=Sung(section,counts,spans)
        }
        fun barAt(t: Fraction)=maxOf(0,starts.count { it<=t }-1)
        fun lastBar(t: Fraction)=maxOf(0,starts.count { it<t }-1)
        val runs=mutableListOf<Run>(); var previous: Int?=null
        for ((section,_,spans) in sung) for ((index,span) in spans.withIndex()) {
            val (k,m)=span; val begin=found[k].start; val end=found[k+m-1].end
            val b0=barAt(begin); val b1=lastBar(end)
            if (runs.isNotEmpty() && previous!=null && k==previous) { runs.last().b1=maxOf(runs.last().b1,b1); runs.last().windows.last().end=end }
            else if (runs.isNotEmpty() && previous!=null && k>previous && b0<=runs.last().b1) { runs.last().b1=maxOf(runs.last().b1,b1); runs.last().windows+=Window(begin,end) }
            else runs+=Run(b0,b1,mutableListOf(Window(begin,end)),mutableListOf())
            if (index==0) runs.last().labels+=begin to section.label
            previous=k+m
        }
        var intro=emptyList<Int>()
        if (sung[0].spans[0].first==0 && runs[0].b0>0) {
            val old=runs[0].b0; runs[0].b0=maxOf(0,old-8); intro=(runs[0].b0 until old).toList()
            val leading=words.takeWhile { it !== sung[0].section }.filter { it.lines.isEmpty() }
            runs[0].labels.add(0,vocal.bars[runs[0].b0].start to (leading.lastOrNull()?.label ?: "intro"))
        }
        var now=ZERO; val bars=mutableListOf<ScoreBar>(); val voices=List(2) { mutableListOf<TimedNote>() }
        val harmony=mutableListOf<Harmony>(); val keys=mutableListOf<Pair<Fraction,String>>(); val labels=mutableListOf<Pair<Fraction,String>>()
        for (run in runs) {
            val begin=vocal.bars[run.b0].start; val finish=vocal.bars[run.b1].let { it.start+it.duration }; val shift=now-begin
            for (b in vocal.bars.subList(run.b0,run.b1+1)) bars+=ScoreBar(b.start+shift,b.start+b.duration+shift,b.meter.numerator.toInt(),b.meter.denominator.toInt())
            for ((track,voice) in listOf(vocal,parsed.voices.getValue("Ins")).withIndex()) for (n in voice.notes) {
                if (n.start<begin || n.start>=finish || track==0 && run.windows.none { n.start>=it.start && n.start<it.end }) continue
                voices[track]+=TimedNote(n.start+shift,minOf(n.start+n.duration,finish)+shift,n.pitch)
            }
            for ((i,c) in vocal.chords.withIndex()) {
                val low=maxOf(c.start,begin); val high=minOf(vocal.chords.getOrNull(i+1)?.start ?: vocal.time,finish)
                if (low<high) harmony+=Harmony(low+shift,high+shift,chordLabel(c.name))
            }
            keys+=now to (vocal.keys.lastOrNull { it.start<=begin }?.name ?: vocal.keys[0].name)
            keys+=vocal.keys.filter { it.start>begin && it.start<finish }.map { it.start+shift to it.name }
            labels+=run.labels.map { it.first+shift to it.second }; now+=finish-begin
        }
        val meter=bars.last(); bars+=ScoreBar(now,now+Fraction(4L*meter.numerator,meter.denominator.toLong()),meter.numerator,meter.denominator)
        val merged=mutableListOf<Pair<Fraction,String>>()
        for (k in keys) {
            if (merged.lastOrNull()?.second==k.second) continue
            if (merged.lastOrNull()?.first==k.first) merged[merged.lastIndex]=k else merged+=k
        }
        val written=try { ScoreEvents.build(bars,voices[0],voices[1],merged,labels,harmony,parsed.bpm,parsed.unit.denominator.toInt()) }
        catch (_: MidiError) { return null }
        val laid=try { AbcScore.parse(written) } catch (_: AbcScore.AbcError) { return null }
        val seconds=laid.voices.getValue("Vocal").time.value()*60/laid.bpm
        val args=mutableListOf<String>()
        if (intro.isNotEmpty()) { args+="intro"; args+=intro.joinToString(",") { (it+1).toString() } }
        for ((section,_,spans) in sung) {
            args+=section.tag
            args+=spans.flatMap { (k,m) -> (barAt(found[k].start)..lastBar(found[k+m-1].end)).toList() }.distinct().sorted().joinToString(",") { (it+1).toString() }
        }
        args+=seconds.toString(); args+=ceiling(seconds).toString()
        val notices=mutableListOf(Notice(Code.LYRICS_LAID,args))
        val windows=runs.flatMap { it.windows }; val voiced=vocal.notes.map { barAt(it.start) }.toSet()
        val heard=vocal.notes.filter { n -> windows.any { n.start>=it.start && n.start<it.end } }.map { barAt(it.start) }.toSet()
        val unsung=(voiced-heard).sorted(); if (unsung.isNotEmpty()) notices+=Notice(Code.UNSUNG_BARS,unsung.map { (it+1).toString() })
        val ratios=sung.flatMap { s -> s.counts.zip(s.spans).map { (count,span) -> found.subList(span.first,span.first+span.second).sumOf { it.notes }.toDouble()/count } }
        if (ratios.map { abs(ln(it)) }.average()>ln(1.5)) {
            val middle=ratios.sorted()[ratios.size/2]; val phraseNotes=phrases(vocal.notes).map { it.size }.sorted(); val syllables=sung.flatMap { it.counts }.sorted()
            val counts=listOf(phraseNotes[phraseNotes.size/2].toString(),syllables[syllables.size/2].toString())
            notices+=if (middle>1) Notice(Code.CROWDED_NOTES,listOf(String.format(Locale.ROOT,"%.1f",middle))+counts) else Notice(Code.CROWDED_WORDS,counts)
        }
        return Underlay.Laid(written,kotlin.math.ceil(ceiling(seconds)).toInt(),notices)
    }
}
