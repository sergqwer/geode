/** In-memory MIDI and native ABC fixtures from YuE2-ComfyUI's tests/midi_files.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import xyz.tifu.geode.music.Smf
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
import org.junit.Assert.assertEquals
import org.junit.Assert.fail

internal object MidiFixtures {
    data class N(val start: Int,val end: Int,val pitch: Int)
    data class P(val name: String,val channel: Int=0,val program: Int=0,val notes: List<N>,val texts: List<MidiReader.Text> = emptyList())
    val scale=listOf(60,62,64,65,67,69,71,72)
    fun line(pitches: List<Int>,start: Int=0,step: Int=480)=pitches.mapIndexed { i,p -> N(start+i*step,start+(i+1)*step,p) }
    fun chord(pitches: List<Int>,start: Int,end: Int)=pitches.map { N(start,end,it) }
    fun kar(pieces: List<String>,start: Int=0,step: Int=240)=karBytes(pieces.map { it.toByteArray() },start,step)
    fun karBytes(pieces: List<ByteArray>,start: Int=0,step: Int=240)=listOf(MidiReader.Text(0,1,"@KMIDI KARAOKE FILE".toByteArray()),MidiReader.Text(0,1,"@TSong title".toByteArray()))+
        pieces.mapIndexed { i,p -> MidiReader.Text(start+i*step,1,p) }
    fun file(vararg parts: P,meters: List<MidiReader.Meter> = listOf(MidiReader.Meter(0,4,4)),tempos: List<MidiReader.Tempo> = listOf(MidiReader.Tempo(0,500000)),
             keys: List<MidiReader.Key> = emptyList(),markers: List<Pair<Int,String>> = emptyList(),division: Int=480): ByteArray {
        val conductor=tempos.map { it.tick to Smf.tempo(it.value) }+meters.map { it.tick to Smf.meter(it.numerator,it.denominator) }+
            keys.map { it.tick to Smf.key(it.sharps,it.minor) }+markers.map { it.first to Smf.text(6,it.second) }
        val tracks=parts.map { p -> listOf(0 to Smf.text(3,p.name),0 to Smf.program(p.channel,p.program))+
            p.notes.flatMap { listOf(it.start to Smf.noteOn(p.channel,it.pitch,90),it.end to Smf.noteOff(p.channel,it.pitch)) }+p.texts.map { it.tick to Smf.meta(it.kind,it.bytes) } }
        return Smf.write(listOf(conductor)+tracks,division)
    }
    fun song(vararg parts: P)=MidiReader.read(file(*parts))
    fun timed(notes: List<N>)=notes.map { TimedNote(Fraction(it.start.toLong()),Fraction(it.end.toLong()),it.pitch) }
    fun sung(abc: String,voice: String="Vocal")=AbcScore.parse(abc).voices.getValue(voice).notes.map { Triple(it.start,it.pitch,it.duration) }
    fun quarters(pitches: List<Int>,n: Long=1,d: Long=1)=pitches.mapIndexed { i,p -> Triple(Fraction(i*n,d),p,Fraction(n,d)) }
    fun error(code: ErrorCode,action: () -> Unit) { try { action(); fail("Expected $code") } catch (e: MidiError) { assertEquals(code,e.code) } }
    data class Group(val vocal: String,val ins: String?=null,val fields: List<String> = emptyList(),val comments: List<String> = emptyList())
    fun tune(vararg groups: Group,meter: String="4/4",unit: Int=16,bpm: Int=100,key: String="C"): String {
        val text=mutableListOf("X:1","T:","M:$meter","L:1/$unit","Q:1/4=$bpm","V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
            "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"","K:$key")
        for (g in groups) {
            val count=g.vocal.dropLast(1).split('|').sumOf { bar -> Regex("Z([2-4])?").matchEntire(bar.trim())?.let { it.groupValues[1].ifEmpty { "1" }.toInt() } ?: 1 }
            text+=g.comments.map { "% $it" }
            for ((v,music) in listOf("Vocal" to g.vocal,"Ins" to (g.ins ?: if (count>1) "Z$count|" else "Z|"))) { text+="V: $v"; text+=g.fields; text+=music }
        }
        return text.joinToString("\n",postfix="\n")
    }
    fun tune(vararg bars: String,bpm: Int=100)=tune(*bars.map { Group(it) }.toTypedArray(),bpm=bpm)
    fun comments(abc: String)=abc.lines().filter { it.startsWith("% ") }.map { it.drop(2) }
}
