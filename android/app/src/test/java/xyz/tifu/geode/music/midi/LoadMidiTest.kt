/** The non-ComfyUI behavior in YuE2-ComfyUI's tests/test_load_midi.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.ScoreMidi
import xyz.tifu.geode.music.midi.MidiFixtures.P
import xyz.tifu.geode.music.midi.MidiFixtures.N
import xyz.tifu.geode.music.midi.MidiFixtures.Group
import xyz.tifu.geode.music.midi.MidiFixtures.scale
import xyz.tifu.geode.music.midi.MidiFixtures.line
import xyz.tifu.geode.music.midi.MidiFixtures.chord
import xyz.tifu.geode.music.midi.MidiFixtures.kar
import xyz.tifu.geode.music.midi.MidiFixtures.karBytes
import xyz.tifu.geode.music.midi.MidiFixtures.file
import xyz.tifu.geode.music.midi.MidiFixtures.song
import xyz.tifu.geode.music.midi.MidiFixtures.timed
import xyz.tifu.geode.music.midi.MidiFixtures.sung
import xyz.tifu.geode.music.midi.MidiFixtures.quarters
import xyz.tifu.geode.music.midi.MidiFixtures.error
import xyz.tifu.geode.music.midi.MidiFixtures.tune
import xyz.tifu.geode.music.midi.MidiFixtures.comments

class LoadMidiTest {
    private val tune=line(listOf(67,69,71,72,71,69,67,67))
    private fun bytes(marked: Boolean=false)=file(P("Vocal",notes=tune),P("Bass",1,33,line(listOf(43,43,48,48),step=960)),markers=if (marked) listOf(0 to "Chorus") else emptyList())
    @Test fun outputsAndTrackList() { val r=MidiImport.read(bytes(),"melody"); assertEquals("[Verse]",r.lyrics); assertEquals(listOf(MidiImport.Role.VOICE,MidiImport.Role.NONE),r.tracks.map { it.role }); assertEquals(2,r.bars) }
    @Test fun bareByDefaultOrNamedOnRequest() {
        val kept=MidiImport.read(bytes(true),"melody",withoutSections=false); val bare=MidiImport.read(bytes(true),"melody")
        assertTrue(kept.abc.lines().contains("% chorus")); assertEquals(Phrasing.bare(kept.abc),bare.abc)
        assertEquals(listOf("verse"),comments(MidiImport.read(bytes(),"melody",withoutSections=false).abc)); assertFalse(Phrasing.named(MidiImport.read(bytes(),"melody").abc))
    }
    @Test fun karaokeWords() { assertEquals("[Verse]\nSing a song of\nsixpence a pocket",MidiImport.read(file(P("Vocal",notes=tune,texts=kar(listOf("\\Sing"," a"," song"," of","/six","pence"," a"," pocket"),step=480))),"melody").lyrics) }
    @Test fun readAndSingErrors() { error(ErrorCode.SHORT_HEADER) { MidiImport.read("MThd".toByteArray(),"melody") }; error(ErrorCode.TRACK_NUMBER) { MidiImport.read(bytes(),"melody",vocalTrack=5) } }
    @Test fun savedScoreMidi() { val r=MidiImport.read(bytes(),"melody"); assertEquals(listOf(67,69,71,72,71,69,67,67),MidiReader.read(ScoreMidi.of(r.abc)).tracks[1].notes.map { it.pitch }) }
}
