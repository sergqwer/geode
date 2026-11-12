/** JUnit cases from YuE2-ComfyUI's tests/test_midi_smf.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.Smf
import xyz.tifu.geode.music.midi.MidiFixtures.error

class MidiSmfTest {
    private fun b(vararg values: Int)=values.map { it.toByte() }.toByteArray()
    private val end=b(0,255,47,0)
    private fun u32(n: Int)=b(n ushr 24,n ushr 16,n ushr 8,n)
    private fun chunk(kind: String,body: ByteArray)=kind.toByteArray()+u32(body.size)+body
    private fun file(vararg tracks: ByteArray,format: Int=1,division: Int=480)=chunk("MThd",b(0,format,tracks.size ushr 8,tracks.size,division ushr 8,division))+tracks.fold(byteArrayOf()) { a,t -> a+chunk("MTrk",t) }
    private fun notes(body: ByteArray)=MidiReader.read(file(body)).tracks[0].notes
    @Test fun runningStatusAndVelocityZero() {
        val song=MidiReader.read(file(b(0,144,60,100,96,62,90,96,60,0,96,128,62,0)+end))
        assertEquals(listOf(MidiReader.Note(0,192,60,100,0),MidiReader.Note(96,288,62,90,0)),song.tracks[0].notes)
        assertEquals(480,song.division); assertEquals(1,song.format)
    }
    @Test fun timingNamesAndTextsFromAnyTrack() {
        val conductor=b(0,255,81,3,7,161,32,0,255,88,4,6,3,24,8,0,255,89,2,255,1)+end
        val lead=b(0,255,3,4)+"Lead".toByteArray()+b(0,193,81,0,255,6,5)+"Verse".toByteArray()+b(0,255,5,2)+"la".toByteArray()+b(0,145,67,80,131,96,129,67,0)+end
        val s=MidiReader.read(file(conductor,lead)); val t=s.tracks[1]
        assertEquals(listOf(MidiReader.Tempo(0,500000)),s.tempos); assertEquals(listOf(MidiReader.Meter(0,6,8)),s.meters)
        assertEquals(listOf(MidiReader.Key(0,-1,true)),s.keys); assertEquals("Verse",s.markers[0].bytes.toString(Charsets.UTF_8))
        assertEquals("Lead",t.name.toString(Charsets.UTF_8)); assertEquals(mapOf(1 to 81),t.programs)
        assertEquals(listOf(0 to 3,0 to 6,0 to 5),t.texts.map { it.tick to it.kind }); assertEquals(listOf(MidiReader.Note(0,480,67,80,1)),t.notes); assertEquals(480,s.end)
    }
    @Test fun formatZeroChannels() { assertEquals(listOf(0,9),MidiReader.read(file(b(0,144,60,100,0,153,36,100,96,128,60,0,0,137,36,0)+end,format=0)).tracks[0].notes.map { it.channel }.sorted()) }
    @Test fun retriggerPairsInOrder() { assertEquals(listOf(MidiReader.Note(0,32,60,100,0),MidiReader.Note(16,48,60,90,0)),notes(b(0,144,60,100,16,60,90,16,128,60,0,16,60,0)+end)) }
    @Test fun unclosedNoteEndsWithTrack() { assertEquals(listOf(MidiReader.Note(0,192,64,100,0)),notes(b(0,144,64,100,129,64,255,47,0))) }
    @Test fun sysexAndUnknownChunks() {
        val data=file(b(0,240,3,126,127,247,0,144,60,100,64,128,60,0)+end)
        val s=MidiReader.read(data.take(14).toByteArray()+chunk("XFIH","abc".toByteArray())+data.drop(14).toByteArray())
        assertEquals(listOf(MidiReader.Note(0,64,60,100,0)),s.tracks[0].notes)
    }
    @Test fun riffWrapper() {
        val inner=file(b(0,144,60,100,64,128,60,0)+end)
        fun le(n: Int)=b(n,n ushr 8,n ushr 16,n ushr 24)
        assertEquals(listOf(MidiReader.Note(0,64,60,100,0)),MidiReader.read("RIFF".toByteArray()+le(12+inner.size)+"RMIDdata".toByteArray()+le(inner.size)+inner).tracks[0].notes)
    }
    @Test fun unreadableFiles() {
        for ((data,code) in listOf("RIFF\u0000\u0000\u0000\u0000WAVEfmt ".toByteArray() to ErrorCode.NOT_MIDI,
            file(b(0,144,60)) to ErrorCode.TRUNCATED,file(b(0,60,100)+end) to ErrorCode.NO_STATUS,
            file(end,division=0xE728) to ErrorCode.SMPTE,file() to ErrorCode.NO_TRACK,
            file(b(0,255,1,255,255,255,255,127)) to ErrorCode.LONG_NUMBER)) error(code) { MidiReader.read(data) }
    }
    @Test fun variableLengthNumbers() {
        for ((n,written) in listOf(0 to b(0),127 to b(127),128 to b(129,0),8192 to b(192,0),0x0fffffff to b(255,255,255,127))) assertArrayEquals(written,Smf.number(n))
    }
    @Test fun writerReadsBackAndDoesNotCutRepeatedPitch() {
        val conductor=listOf(0 to Smf.tempo(600000),0 to Smf.meter(3,4),0 to Smf.key(2,false),960 to Smf.text(6,"Chorus"))
        val melody=listOf(0 to Smf.text(3,"Vocal"),0 to Smf.program(0,53),0 to Smf.noteOn(0,62,90),480 to Smf.noteOff(0,62),480 to Smf.noteOn(0,62,90),960 to Smf.noteOff(0,62)).reversed()
        val s=MidiReader.read(Smf.write(listOf(conductor,melody)))
        assertEquals(listOf(MidiReader.Tempo(0,600000)),s.tempos); assertEquals(listOf(MidiReader.Meter(0,3,4)),s.meters); assertEquals(listOf(MidiReader.Key(0,2,false)),s.keys)
        assertEquals("Chorus",s.markers[0].bytes.toString(Charsets.UTF_8)); assertEquals(960,s.markers[0].tick)
        assertEquals("Vocal",s.tracks[1].name.toString(Charsets.UTF_8)); assertEquals(mapOf(0 to 53),s.tracks[1].programs)
        assertEquals(listOf(MidiReader.Note(0,480,62,90,0),MidiReader.Note(480,960,62,90,0)),s.tracks[1].notes)
    }
    @Test fun trackEndsAfterLastEvent() { val s=MidiReader.read(Smf.write(listOf(listOf(0 to Smf.noteOn(0,60,90),480 to Smf.noteOff(0,60))),end=1920)); assertEquals(1920,s.end); assertEquals(listOf(MidiReader.Note(0,480,60,90,0)),s.tracks[0].notes) }
}
