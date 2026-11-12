/** JUnit cases from YuE2-ComfyUI's tests/test_midi_parts.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import java.nio.charset.Charset
import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.AbcScore.Fraction
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

class MidiPartsTest {
    private fun choose(vararg p: P)=MidiParts.choose(MidiParts.parts(song(*p)))
    @Test fun textEncodings() {
        val cp=Charset.forName("windows-1251")
        for ((bytes,text) in listOf("Lead".toByteArray() to "Lead","Мелодия".toByteArray(cp) to "Мелодия",
            "Дорожка 1".toByteArray(cp).toString(Charsets.ISO_8859_1).toByteArray() to "Дорожка 1",
            "Müller".toByteArray(Charsets.ISO_8859_1) to "Müller","Müller".toByteArray() to "Müller","  two\u0000  words ".toByteArray() to "two words")) assertEquals(text,MidiParts.decode(bytes))
    }
    @Test fun numberingAndSharedChannels() {
        val s=song(P("Piano",notes=line(listOf(60,62))),P("",channel=1,notes=emptyList()))
        val shared=song(P("",notes=line(listOf(72))),P("",channel=9,notes=line(listOf(36)))).tracks.drop(1)
        val merged=MidiReader.Track(2); merged.notes+=shared.flatMap { it.notes }; s.tracks+=merged
        assertEquals(listOf(Triple(1,"Piano","Piano"),Triple(2,"channel 1","Piano"),Triple(3,"channel 10","Drums")),MidiParts.parts(s).map { Triple(it.number,it.name,it.family) })
    }
    @Test fun highestMelodyAndNoBassInstrument() {
        val c=choose(P("Bass",1,33,line(listOf(36,38,40,41))),P("Tune",0,80,line(listOf(67,69,71,72))),P("Drums",9,0,line(listOf(36,38,36,38))))
        assertEquals("Tune",c.voice.name); assertEquals(Why.HIGHEST,c.voiceWhy); assertNull(c.instrument)
    }
    @Test fun highPadTakesNeitherLine() { val c=choose(P("Pad",0,88,chord(listOf(72,76,79),0,1920)+chord(listOf(74,77,81),1920,3840)),P("Tune",1,0,line(listOf(65,67,69,70,72,70,69,67)))); assertEquals("Tune",c.voice.name); assertNull(c.instrument) }
    @Test fun vocalNames() { for (name in listOf("Vocal","Lead Voice","Мелодия")) { val c=choose(P("Strings",0,48,line(listOf(79,81,83,84))),P(name,1,0,line(listOf(60,62,64,65)))); assertEquals(2,c.voice.number); assertEquals(Why.NAME,c.voiceWhy) } }
    @Test fun karaokeChoosesVoice() {
        val c=MidiParts.choose(MidiParts.parts(song(P("Flute",0,73,line(listOf(84,86,88,89),step=240)),P("Singer?",1,0,line(listOf(60,62,64,65,67,65,64,62))))),syllables=listOf(0,480,960,1440))
        assertEquals(2,c.voice.number); assertEquals(Why.KARAOKE,c.voiceWhy)
    }
    @Test fun explicitChoicesAndRefusals() {
        val p=MidiParts.parts(song(P("Lead",0,80,line(listOf(67,69))),P("Bass",1,33,line(listOf(40,41))),P("Kit",9,0,line(listOf(36,38)))))
        val c=MidiParts.choose(p,2,1); assertEquals(2,c.voice.number); assertEquals(1,c.instrument!!.number); assertEquals(Why.CHOSEN,c.voiceWhy); assertEquals(Why.CHOSEN,c.instrumentWhy)
        assertNull(MidiParts.choose(p,instrument=-1).instrument)
        error(ErrorCode.TRACK_NUMBER) { MidiParts.choose(p,7) }; error(ErrorCode.DRUM_TRACK) { MidiParts.choose(p,3) }; error(ErrorCode.SAME_TRACK) { MidiParts.choose(p,1,1) }
    }
    @Test fun drumsAlone() { error(ErrorCode.NO_PITCHED_NOTES) { choose(P("Kit",9,0,line(listOf(36,38)))) } }
    @Test fun description() {
        val bytes=file(P("Lead",0,80,chord(listOf(67,79),0,480)+line(listOf(69),start=480)))
        assertEquals(listOf(MidiImport.Track(1,"Lead","Synth Lead",3,67,79,MidiImport.Role.VOICE,Why.NAME)),MidiImport.read(bytes,"melody").tracks)
        assertEquals(2,MidiParts.parts(MidiReader.read(bytes))[0].polyphony)
    }
    @Test fun karRepeatedParagraphIsChorus() {
        val pieces=listOf("\\Here"," we"," go","/Down"," the"," road","\\Sing"," it"," loud","/All"," night","\\Walk"," on","/by","\\Sing"," it"," loud","/All"," night")
        val w=Karaoke.read(song(P("Words",notes=line(listOf(60)),texts=kar(pieces))))!!
        assertEquals("[Verse]\nHere we go\nDown the road\n\n[Chorus]\nSing it loud\nAll night\n\n[Verse]\nWalk on\nby\n\n[Chorus]\nSing it loud\nAll night",w.lyrics())
        assertEquals(listOf(0,1440,2640,3360),w.sections.map { it.tick }); assertEquals(listOf("verse","chorus","verse","chorus"),w.sections.map { it.label }); assertEquals(pieces.size,w.syllables.size)
    }
    @Test fun lyricReturnsBlankLinesAndPauses() {
        val pieces=listOf("One ","line\r","two ","lines\r","\r","after ","blank\r","far ","later"); val ticks=listOf(0,240,480,720,720,960,1200,5040,5280)
        val w=Karaoke.read(song(P("Vocal",notes=line(listOf(60)),texts=pieces.zip(ticks).map { MidiReader.Text(it.second,5,it.first.toByteArray()) })))!!
        assertEquals(listOf(listOf("One line","two lines"),listOf("after blank"),listOf("far later")),w.sections.map { it.lines })
    }
    @Test fun russianSyllables() {
        val cp=Charset.forName("windows-1251"); val pieces=List(4) { listOf(" пес".toByteArray(cp),"ня".toByteArray(cp)) }.flatten()
        assertEquals(listOf("песня песня песня песня"),Karaoke.read(song(P("Words",notes=line(listOf(60)),texts=karBytes(pieces))))!!.sections[0].lines)
    }
    @Test fun sectionMarkers() {
        val s=MidiReader.read(file(P("Words",notes=line(listOf(60)),texts=kar(listOf("\\A"," b"," c"," d","\\E"," f"," g"," h"))),markers=listOf(0 to "Intro 1",960 to "Bridge")))
        assertEquals(listOf("Intro","Bridge"),Karaoke.read(s)!!.sections.map { it.tag }); assertEquals(listOf(0 to "intro",960 to "bridge"),Karaoke.markerSections(s))
        assertEquals("pre-chorus",Karaoke.sectionLabel("Pre-Chorus 2")); assertNull(Karaoke.sectionLabel("Bar 17"))
    }
    @Test fun titleIsNotWords() { assertNull(Karaoke.read(song(P("Words",notes=line(listOf(60)),texts=kar(listOf("Hello")))))) }
    @Test fun guessedLabels() {
        for ((pcs,flats,expected) in listOf(Triple(listOf(0,4,7),false,"C:maj"),Triple(listOf(9,0,4),false,"A:min"),Triple(listOf(7,11,2,5),false,"G:7"),Triple(listOf(10,2,5),true,"Bb:maj"),Triple(listOf(10,2,5),false,"A#:maj"),Triple(listOf(11,2,5),false,"B:dim"),Triple(listOf(0),false,"N")))
            assertEquals(expected,MidiChords.label((0..11).map { pc -> pcs.count { it==pc }.toDouble() },pcs[0],flats))
    }
    @Test fun halfBarsAndMerging() {
        val notes=timed(chord(listOf(48,52,55),0,1920)+chord(listOf(45,48,52),1920,2880)+chord(listOf(43,47,50,53),2880,3840))
        val rows=MidiChords.guess(notes,MidiChords.spans(listOf(ScoreBar(Fraction(0),Fraction(1920),4,4),ScoreBar(Fraction(1920),Fraction(3840),4,4))))
        assertEquals(listOf(Harmony(Fraction(0),Fraction(1920),"C:maj"),Harmony(Fraction(1920),Fraction(2880),"A:min"),Harmony(Fraction(2880),Fraction(3840),"G:7")),rows)
        assertEquals(listOf(Fraction(0) to Fraction(1440)),MidiChords.spans(listOf(ScoreBar(Fraction(0),Fraction(1440),3,4))))
    }
    @Test fun soloIsNoChord() { assertEquals(listOf(Harmony(Fraction(0),Fraction(960),"N")),MidiChords.guess(timed(line(listOf(43,39,41,43),step=240)),listOf(Fraction(0) to Fraction(960)))) }
    @Test fun openFifthUsesKeyThird() { val w=MutableList(12) { 0.0 }; w[2]=1.0; w[9]=1.0; assertEquals("D:maj",MidiChords.label(w,2)); assertEquals("D:min",MidiChords.label(w,2,scale=setOf(5,7,9,10,0,2,4))) }
    @Test fun exactHeldChordShapes() {
        val cases=listOf(listOf(48,52,55) to "C:maj",listOf(57,60,64,67) to "A:min7",listOf(48,52,55,57) to "C:maj6",listOf(57,60,64,67,69) to "A:min7",listOf(52,55,60) to "C:maj/3",listOf(40,48,52,55) to "C:maj/3",listOf(45,48,52,55) to "C:maj/6",listOf(36,57,60,64,67) to "A:min7/b3",listOf(38,48,52,55) to "C:maj/2",listOf(43,48,50,55) to "C:sus2/5",listOf(48,50,55) to "C:sus2",listOf(55,60,62) to "G:sus4",listOf(48,51,54,57) to "C:dim7",listOf(36,60,63,66,69) to "C:dim7",listOf(42,48,51,54,57) to "C:dim7/b5",listOf(44,48,52,56) to "C:aug/#5",listOf(48,50,52,55) to null,listOf(48,55) to null,listOf(60) to null)
        cases.forEach { (p,e) -> assertEquals(p.toString(),e,MidiChords.spelled(p)) }
    }
    @Test fun commonToneAndUnknownChords() {
        val n=listOf(N(0,16,48),N(0,16,52),N(0,8,55),N(8,17,45))+chord(listOf(43,47,50,53),16,24)+chord(listOf(48,50,52,55),24,32)+chord(listOf(48,52,55),32,40)
        val (rows,unnamed)=MidiChords.read(timed(n),Fraction(48))
        assertEquals(listOf(Harmony(Fraction(0),Fraction(8),"C:maj"),Harmony(Fraction(8),Fraction(16),"A:min"),Harmony(Fraction(16),Fraction(24),"G:7"),Harmony(Fraction(24),Fraction(48),"C:maj")),rows); assertEquals(listOf(Fraction(24)),unnamed)
        assertEquals(listOf(Harmony(Fraction(0),Fraction(8),"N")) to listOf(Fraction(0)),MidiChords.read(timed(listOf(N(0,8,60))),Fraction(8)))
    }
}
