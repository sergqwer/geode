/** JUnit cases from YuE2-ComfyUI's tests/test_midi_score.py and the score export promise (Apache-2.0). */
package xyz.tifu.geode.music.midi

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.AbcScore
import xyz.tifu.geode.music.AbcScore.Fraction
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

class MidiScoreTest {
    private fun read(vararg parts: P,mode: String="melody")=MidiImport.read(file(*parts),mode)
    @Test fun plainTune() {
        val r=read(P("Tune",notes=line(scale)))
        assertEquals(listOf("M:4/4","L:1/16","Q:1/4=120"),r.abc.lines().subList(2,5)); assertEquals(quarters(scale),sung(r.abc))
        assertEquals("[Verse]",r.lyrics); assertEquals(2,r.bars); assertEquals(4.0,r.seconds,0.0); assertTrue(r.notices.isEmpty())
    }
    @Test fun thirtySeconds() { val r=read(P("Tune",notes=line(scale,step=60))); assertTrue(r.abc.lines().contains("L:1/32")); assertEquals(quarters(scale,1,8),sung(r.abc)) }
    @Test fun playedIn() { val r=read(P("Tune",notes=listOf(N(0,470,60),N(490,950,62),N(950,1450,64),N(1430,1920,65)))); assertEquals(quarters(listOf(60,62,64,65)),sung(r.abc)); assertTrue(r.notices.contains(Notice(Code.PLAYED_IN))) }
    @Test fun octaveMoves() {
        for ((p,shift) in listOf(listOf(41,43,45,43) to 24,listOf(90,91,93,91) to -12,listOf(70,72,74,72) to 0)) {
            val r=read(P("Tune",notes=line(p))); assertEquals(quarters(p.map { it+shift }),sung(r.abc))
            if (shift==0) assertTrue(r.notices.isEmpty()) else assertTrue(r.notices.contains(Notice(Code.OCTAVE_MOVED,listOf("voice","1",shift.toString()))))
        }
    }
    @Test fun topChordNotesAndTheirOctave() {
        val r=read(P("Lead",notes=chord(listOf(60,72),0,480)+chord(listOf(62,86),480,960)+chord(listOf(64,88),960,1440)))
        assertEquals(quarters(listOf(60,74,76)),sung(r.abc)); assertTrue(r.notices.contains(Notice(Code.OCTAVE_MOVED,listOf("voice","1","-12")))); assertTrue(r.notices.contains(Notice(Code.TOP_NOTES_KEPT,listOf("1","3"))))
    }
    @Test fun lowerOverlappingNote() { assertEquals(listOf(Triple(Fraction(0),72,Fraction(2)),Triple(Fraction(2),67,Fraction(2))),sung(read(P("Tune",notes=listOf(N(0,960,72),N(100,520,64),N(800,1920,67)))).abc)) }
    @Test fun fileKeyOrEstimatedKey() {
        val kept=MidiImport.read(file(P("Tune",notes=line(listOf(62,66,69,74,69,66,62,62))),keys=listOf(MidiReader.Key(0,2,false))),"melody")
        assertTrue(kept.abc.lines().contains("K:D"))
        val wrong=MidiImport.read(file(P("Tune",notes=line(listOf(68,72,75,80,75,72,68,68))),keys=listOf(MidiReader.Key(0,0,false))),"melody")
        assertEquals("Ab",wrong.key)
    }
    @Test fun meterChangesAndCompoundMeter() {
        val changed=MidiImport.read(file(P("Tune",notes=line(scale.take(7))),meters=listOf(MidiReader.Meter(0,4,4),MidiReader.Meter(1920,3,4))),"melody")
        assertTrue(changed.abc.lines().contains("M:3/4")); assertEquals(2,changed.bars)
        val compound=MidiImport.read(file(P("Tune",notes=line(scale.take(6),step=240)),meters=listOf(MidiReader.Meter(0,6,8))),"melody")
        assertEquals(listOf("M:6/8","L:1/16"),compound.abc.lines().subList(2,4)); assertEquals(quarters(scale.take(6),1,2),sung(compound.abc))
    }
    @Test fun markerSectionsAndTags() { val r=MidiImport.read(file(P("Tune",notes=line(scale)),markers=listOf(0 to "Chorus")),"melody",withoutSections=false); assertTrue(r.abc.lines().contains("% chorus")); assertEquals("[Chorus]",r.lyrics) }
    @Test fun karaokeLyricsAndIntro() {
        val r=MidiImport.read(file(P("Tune",notes=line(scale,start=1920),texts=kar(listOf("\\Sing"," a"," song"," of","/six","pence"," a"," pocket"),start=1920,step=480))),"melody",withoutSections=false)
        assertEquals("[Verse]\nSing a song of\nsixpence a pocket",r.lyrics); assertTrue(r.abc.lines().indexOf("% intro")<r.abc.lines().indexOf("% verse"))
    }
    @Test fun fullGuessesAndMelodyHasNoChords() {
        val parts=arrayOf(P("Vocal",notes=line(listOf(64,67,72,67,65,69,72,69))),P("Piano",1,0,chord(listOf(48,52,55),0,1920)+chord(listOf(41,45,48),1920,3840)))
        val full=read(*parts,mode="full"); assertTrue(full.abc.contains("\"C\"")); assertTrue(full.abc.contains("\"F\"")); assertTrue(full.notices.contains(Notice(Code.CHORDS_GUESSED)))
        val melody=read(*parts); assertTrue(AbcScore.parse(melody.abc).voices.getValue("Vocal").chords.isEmpty()); assertEquals(sung(melody.abc),sung(full.abc))
    }
    @Test fun instrumentLine() { val r=read(P("Vocal",notes=line(listOf(67,69,71,72))),P("Ins",1,0,line(listOf(76,77,79,81)))); assertEquals(quarters(listOf(76,77,79,81)),sung(r.abc,"Ins")); assertEquals(MidiImport.Role.INSTRUMENT,r.tracks[1].role) }
    @Test fun averageTempo() {
        val r=MidiImport.read(file(P("Tune",notes=line(scale)),tempos=listOf(MidiReader.Tempo(0,500000),MidiReader.Tempo(1920,1000000))),"melody")
        assertTrue(r.abc.lines().contains("Q:1/4=80")); assertEquals(6.0,r.seconds,0.0); assertTrue(r.notices.contains(Notice(Code.TEMPO_AVERAGED,listOf("60","120","80"))))
    }
    @Test fun cannotBeSung() { error(ErrorCode.NO_PITCHED_NOTES) { read(P("Kit",9,0,line(listOf(36,38)))) }; error(ErrorCode.NO_NOTES) { read() }; error(ErrorCode.MODE) { read(P("Tune",notes=line(scale)),mode="loud") } }
    @Test fun strayThirtySecond() { val r=read(P("Tune",notes=line(scale)+listOf(N(3840,3900,74),N(3900,4320,72)))); assertTrue(r.abc.lines().contains("L:1/32")); assertEquals(listOf(Triple(Fraction(8),74,Fraction(1,8)),Triple(Fraction(65,8),72,Fraction(7,8))),sung(r.abc).takeLast(2)) }
    @Test fun chordsTrackAndKeySpelling() {
        val r=MidiImport.read(file(P("Vocal",notes=line(listOf(65,67,69,70,72,70,69,67))),P("Chords",2,48,chord(listOf(53,57,60),0,1920)+chord(listOf(58,62,65),1920,2880)+chord(listOf(41,48,52,55),2880,3840)),keys=listOf(MidiReader.Key(0,-1,false))),"full")
        assertEquals(listOf(Fraction(0) to "F",Fraction(4) to "Bb",Fraction(6) to "C/F"),AbcScore.parse(r.abc).voices.getValue("Vocal").chords.map { it.start to it.name })
        assertEquals(listOf(Notice(Code.CHORDS_READ,listOf("2","Chords"))),r.notices); assertEquals(listOf(MidiImport.Role.VOICE,MidiImport.Role.CHORDS),r.tracks.map { it.role })
    }
    @Test fun unnamedChordBars() {
        val r=read(P("Vocal",notes=line(listOf(64,67,72,67,64,67,72,67,65,69,72,69))),P("Chords",2,48,chord(listOf(48,52,55),0,1920)+chord(listOf(48,50,52,55),1920,3840)+chord(listOf(41,45,48),3840,5760)),mode="full")
        assertEquals(listOf("C","C","F"),AbcScore.parse(r.abc).voices.getValue("Vocal").chords.map { it.name }); assertTrue(r.notices.contains(Notice(Code.UNNAMED_CHORDS,listOf("2","2","0"))))
    }
    @Test fun melodyIgnoresChordsTrack() { val r=read(P("Vocal",notes=line(listOf(64,67,72,67))),P("Chords",2,48,chord(listOf(48,52,55),0,1920))); assertTrue(AbcScore.parse(r.abc).voices.getValue("Vocal").chords.isEmpty()); assertTrue(r.notices.isEmpty()); assertEquals(listOf(MidiImport.Role.VOICE,MidiImport.Role.NONE),r.tracks.map { it.role }) }
    @Test fun barArgumentsStayNumeric() {
        // test_the_bars_are_named_the_way_the_node_says_them: localization owns the English formatting.
        for ((numbers,expected) in listOf(listOf(5) to listOf("5","0"),listOf(5,9) to listOf("5,9","0"),listOf(2,3,4) to listOf("2,3,4","0"),(1..11).toList() to listOf("1,2,3,4,5,6,7,8","3"))) assertEquals(expected,MidiImport.barArguments(numbers))
    }
    @Test fun exportedScoreRoundTrip() {
        val scores=JSONObject(javaClass.classLoader!!.getResource("abc_transpose.json")!!.readText()).getJSONObject("scores")
        val original=scores.getString("awkward"); val before=AbcScore.parse(original)
        val r=MidiImport.read(ScoreMidi.of(original),"full"); val after=AbcScore.parse(r.abc)
        assertEquals(before.bpm,after.bpm)
        for (voice in listOf("Vocal","Ins")) {
            assertEquals(sung(original,voice),sung(r.abc,voice)); assertEquals(before.voices.getValue(voice).bars,after.voices.getValue(voice).bars)
        }
        assertEquals(before.voices.getValue("Vocal").chords,after.voices.getValue("Vocal").chords)
    }
}
