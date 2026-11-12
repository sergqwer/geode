/** JUnit cases from YuE2-ComfyUI's tests/test_phrasing.py (Apache-2.0). */
package xyz.tifu.geode.music.midi

import org.junit.Assert.*
import org.junit.Test
import xyz.tifu.geode.music.AbcScore
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

class PhrasingTest {
    private val la="la la la la la la la"; private val na="na na na na na na na"; private val lo="lo lo lo lo lo lo lo"
    private val phrase="z2C2D2E2F2G2A2B2|"; private val answer="z2c2B2A2G2F2E2D2|"; private val riff="z4c4G8|"
    private fun pitches(abc: String)=sung(abc).map { it.second }
    @Test fun vowelCountsAcrossScripts() { for ((line,n) in listOf("Hold the tune and let it ring" to 7,"played wanted fades boxes table whole" to 9,"Привет, как дела" to 5,"你好世界" to 4,"café naïve" to 4,"-- 42 --" to 0)) assertEquals(line,n,Phrasing.syllables(line)) }
    @Test fun wordsAsSung() {
        for ((word,n) in mapOf("every" to 2,"everything" to 3,"everyone" to 3,"different" to 2,"going" to 2,"trying" to 2,"sing" to 1,"piano" to 3,"radio" to 3,"million" to 2,"nation" to 2,"special" to 2,"serious" to 3,"people" to 2)) assertEquals(word,n,Phrasing.syllables(word))
        assertEquals(7,Phrasing.syllables("Every street remembers night"))
    }
    @Test fun tagLabels() { for ((tag,label) in listOf("Verse 2" to "verse","Pre-Chorus" to "pre-chorus","prechorus" to "pre-chorus","Hook" to "chorus","Outro:" to "outro","Something else" to "verse")) assertEquals(label,Phrasing.labelOf(tag)) }
    @Test fun lyricSections() {
        val s=Phrasing.lyricSections("first line\n\n[Intro]\n\n[Chorus 2]\nsing this\n...\nand this\n")
        assertEquals(listOf("verse" to listOf("first line"),"intro" to emptyList(),"chorus" to listOf("sing this","and this")),s.map { it.label to it.lines }); assertEquals("Chorus 2",s[2].tag)
    }
    @Test fun lineEndsBeforeRiff() {
        val score=tune("z2e2g2e2d3d2cd2|z2B2B2B2c2G2c2G2|d4G8G4|z4G,8z4|")
        val laid=Underlay.lay(score,"[Verse]\nNeon fades along the lane\nEvery street remembers night")!!
        assertEquals(pitches(tune("z2e2g2e2d3d2cd2|z2B2B2B2c2G2c2G2|")),pitches(laid.abc)); assertEquals(listOf("Verse","1,2"),laid.notices[0].args.take(2))
    }
    @Test fun breathHeldNoteAndBars() {
        val n=listOf(AbcScore.Note(Fraction(0),60,Fraction(1,2)),AbcScore.Note(Fraction(1,2),62,Fraction(1,4)),AbcScore.Note(Fraction(1),64,Fraction(1,2)),AbcScore.Note(Fraction(2),65,Fraction(2)),AbcScore.Note(Fraction(4),67,Fraction(1)))
        assertEquals(listOf(Phrasing.Piece(Fraction(0),Fraction(3,2),3,true),Phrasing.Piece(Fraction(2),Fraction(4),1,true),Phrasing.Piece(Fraction(4),Fraction(5),1,true)),Phrasing.pieces(n,listOf(Fraction(0),Fraction(4))))
        assertEquals(listOf(8 to true,8 to false,8 to false),Phrasing.pieces((0 until 24).map { AbcScore.Note(Fraction(it.toLong(),2),60,Fraction(1,2)) },listOf(Fraction(0),Fraction(4),Fraction(8))).map { it.notes to it.breath })
    }
    @Test fun unchangedScores() {
        val words="[Verse]\n$la"
        assertNull(Underlay.lay(tune(Group(phrase,comments=listOf("verse"))),words)); assertNull(Underlay.lay("X:1\nK:C\nCDEF|",words)); assertNull(Underlay.lay(tune("Z|"),words)); assertNull(Underlay.lay(tune(phrase),"[Intro]\n\n[Instrumental]"))
    }
    @Test fun fitsPhrasesSkipsRiffAndLoops() {
        val laid=Underlay.lay(tune("Z4|",phrase+answer+phrase+answer,riff.repeat(4)),"[Verse]\n"+List(4) { la }.joinToString("\n")+"\n\n[Chorus]\n"+List(4) { na }.joinToString("\n"))!!
        val parsed=AbcScore.parse(laid.abc); val v=parsed.voices.getValue("Vocal"); val verse=pitches(tune(phrase+answer+phrase+answer))
        assertEquals(listOf("intro","verse","chorus"),comments(laid.abc)); assertEquals(verse+verse,pitches(laid.abc)); assertEquals(13,v.bars.size)
        assertEquals(v.time-Fraction(4),v.notes.last().let { it.start+it.duration }); assertEquals(31.2,v.time.value()*60/parsed.bpm,1e-8)
        assertEquals(listOf("intro","1,2,3,4","Verse","5,6,7,8","Chorus","5,6,7,8"),laid.notices[0].args.dropLast(2))
        assertEquals(Notice(Code.UNSUNG_BARS,listOf("9","10","11","12")),laid.notices[1]); assertEquals(36.3,Phrasing.ceiling(31.2),1e-8); assertEquals(37,laid.maxSeconds)
    }
    @Test fun repeatedChorusUsesSameBars() {
        val source=listOf("z2C2D2E2F2G2A2B2|","z2D2E2F2G2A2B2c2|","z2E2F2G2A2B2c2d2|","z2F2G2A2B2c2d2e2|")
        val laid=Underlay.lay(tune(source.joinToString("")),"[Verse]\n$la\n[Chorus]\n$na\n[Verse]\n$lo\n[Chorus]\n$na")!!
        assertEquals(listOf(0,1,2,1).flatMap { pitches(tune(source[it])) },pitches(laid.abc)); assertEquals(listOf("verse","chorus","verse","chorus"),comments(laid.abc))
        assertEquals(listOf("Verse","1","Chorus","2","Verse","3","Chorus","2"),laid.notices[0].args.dropLast(2))
    }
    @Test fun unsungNotesAreRested() {
        val laid=Underlay.lay(tune("C2D2E2F2z8|G2A2B2c2z4d4|"),"[Verse]\nla la la la\nla la la la\nla la la la")!!
        assertEquals(listOf(60,62,64,65,67,69,71,72,60,62,64,65),pitches(laid.abc)); assertEquals(4,AbcScore.parse(laid.abc).voices.getValue("Vocal").bars.size)
    }
    @Test fun chordsKeysAndMetersFollowBars() {
        val score=tune(Group("\"C\"z2C2D2E2F2G2A2B2|\"G\"z2B2A2G2F2E2D2C2|"),Group("\"D\"z2D2E2F2G2A2|",fields=listOf("M:3/4","K:G")))
        val laid=Underlay.lay(score,"[Verse]\n$la\n$la\nla la la la la")!!; val v=AbcScore.parse(laid.abc).voices.getValue("Vocal")
        assertEquals(listOf("C","G","D"),v.chords.map { it.name }); assertEquals(listOf(4L to 4L,4L to 4L,3L to 4L,3L to 4L),v.bars.map { it.meter.numerator to it.meter.denominator }); assertEquals(listOf("C","G"),v.keys.map { it.name }); assertEquals(pitches(score),pitches(laid.abc))
    }
    @Test fun crowdedNotesWarning() { val laid=Underlay.lay(tune("CDEFGABcdcBAGFz2|CDEFGABcdcBAGFz2|",bpm=60),"[Verse]\nla la la la\nla la la la")!!; assertEquals(Notice(Code.CROWDED_NOTES,listOf("3.5","14","4")),laid.notices.last()) }
    @Test fun crowdedWordsWarning() { val laid=Underlay.lay(tune("z4C4z4D4|z4E4z4F4|z4G4z4A4|z4B4z4c4|"),"[Verse]\n"+"la ".repeat(30))!!; assertEquals(Notice(Code.CROWDED_WORDS,listOf("1","30")),laid.notices.last()) }
    @Test fun unmatchedLineStillLands() { assertTrue(AbcScore.parse(Underlay.lay(tune(riff.repeat(4)),"[Verse]\n$la")!!.abc).voices.getValue("Vocal").notes.isNotEmpty()) }
    @Test fun bareAndLabelledHelpers() {
        val score=tune(Group(phrase,comments=listOf("chorus"))); assertTrue(Phrasing.named(score)); assertFalse(Phrasing.named(Phrasing.bare(score))); assertEquals(tune(phrase),Phrasing.bare(score))
        assertEquals(tune(Group(phrase,comments=listOf("verse"))),Phrasing.labelled(tune(phrase))); assertEquals(score,Phrasing.labelled(score)); assertEquals("X:1\nnot a score",Phrasing.labelled("X:1\nnot a score"))
    }
    @Test fun chordSymbolsAndLabels() { for ((symbol,label) in listOf("C" to "C:maj","Bbm7/F" to "Bb:min7/F","F#m7b5" to "F#:hdim7","Ab6" to "Ab:maj6","G7sus4" to "G:sus4(b7)","Dm(maj7)" to "D:minmaj7")) { assertEquals(label,Phrasing.chordLabel(symbol)); assertEquals(symbol,MidiChords.symbol(label)) } }
}
