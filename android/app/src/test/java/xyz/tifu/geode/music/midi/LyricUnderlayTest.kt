/** JUnit cases from YuE2-ComfyUI's tests/test_underlay.py, excluding HTTP routes (Apache-2.0). */
package xyz.tifu.geode.music.midi

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
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

class LyricUnderlayTest {
    private val seven="C2D2E2F2G2A2B2z2|"; private val eight="C2D2E2F2G2A2B1c1z2|"; private val three="c4B4A4z4|"; private val rest="z16|"
    private val verse="[Verse]\nla la la la la la la\nna na na na na na na\n"; private val chorus="[Chorus]\noh oh oh\nhey hey hey\n"
    private fun score(vararg groups: Pair<String?,String>)=tune(*groups.map { Group(it.second,comments=it.first?.let { listOf(it) }.orEmpty()) }.toTypedArray())
    private val song get()=score("intro" to rest,"verse" to seven+seven,"chorus" to three+three)
    private fun lay(abc: String=song,lyrics: String=verse+chorus)=LyricUnderlay.lay(abc,lyrics)
    private fun spans(l: LyricUnderlay.Laid)=l.lines.map { it.first to it.stop }
    private fun kinds(l: LyricUnderlay.Laid)=l.lines.joinToString("") { it.kind.name.lowercase().take(1) }
    private fun texts(l: LyricUnderlay.Laid)=l.notes.map { it.text }
    @Test fun syllablePerNote() {
        val l=lay(); assertTrue(l.paired); assertFalse(l.bare); assertEquals(listOf(0 to 7,7 to 14,14 to 17,17 to 20),spans(l)); assertEquals("ffff",kinds(l)); assertEquals(mapOf(LyricUnderlay.Kind.FIT to 4,LyricUnderlay.Kind.SPARE to 0,LyricUnderlay.Kind.CROWD to 0),l.counts)
        assertEquals(List(7) { "la" }+"na",texts(l).take(8)); assertEquals(listOf("oh","oh","oh","hey","hey","hey"),texts(l).drop(14)); assertTrue(l.spare.isEmpty()); assertEquals(20,l.sung)
    }
    @Test fun writtenLineNumbers() { val l=lay(lyrics="\n$verse\n$chorus"); assertEquals(listOf(2,3,6,7),l.lines.map { it.line }); assertEquals("la la la la la la la",l.lines[0].text); assertEquals(7,l.lines[0].syllables); assertEquals(7,l.lines[0].notes) }
    @Test fun hyphenatedWords() { val l=lay(score("verse" to eight),"[Verse]\nMorning breaks above the river\n"); assertEquals(listOf("Mor-","ning","breaks","a-","bove","the","ri-","ver"),texts(l)); assertEquals("f",kinds(l)) }
    @Test fun excessSyllablesShareLastNote() {
        val l=lay(lyrics="[Verse]\nla la la la la la morning light\nna na na na na na na\n$chorus")
        assertEquals("cfff",kinds(l)); assertEquals(listOf(0 to 7,7 to 14),spans(l).take(2)); assertEquals(listOf(LyricUnderlay.Drawn(5,0,"la"),LyricUnderlay.Drawn(6,0,"morning light"),LyricUnderlay.Drawn(7,1,"na")),l.notes.subList(5,8)); assertEquals(9,l.lines[0].syllables); assertEquals(7,l.lines[0].notes)
    }
    @Test fun cutWordShownWhole() { val l=lay(score("verse" to three),"[Verse]\nsing tomorrow\n"); assertEquals(listOf("sing","to-","morrow"),texts(l)); assertEquals("c",kinds(l)) }
    @Test fun spareNotesAndNextPhrase() { val l=lay(lyrics="[Verse]\nla la la la la\nna na na na na na na\n$chorus"); assertEquals("sfff",kinds(l)); assertEquals(listOf(0 to 7,7 to 14),spans(l).take(2)); assertEquals(listOf(5,6),l.spare); assertEquals((0..4).toList(),l.notes.filter { it.line==0 }.map { it.note }) }
    @Test fun russianPrepositions() {
        assertEquals(listOf("в до","ме","ти","хо"),LyricUnderlay.partsOf("в доме тихо").map { it.text }); assertEquals(listOf("ну","и ж"),LyricUnderlay.partsOf("ну и ж").map { it.text })
        assertEquals(listOf("в до-","ме","ти-","хо"),texts(lay(score("verse" to "C4D4E4F4|"),"[Verse]\nв доме тихо\n")))
    }
    @Test fun linesWithoutWordsAreIgnored() {
        val b=LyricUnderlay.blocksOf("[Verse]\n...\nla la\n(1, 2, 3)\n[Bridge]\n\n[Chorus]\noh\n")
        assertEquals(listOf("verse" to listOf(2),"chorus" to listOf(7)),b.map { it.label to it.rows.map { r -> r.line } }); assertEquals("verse",LyricUnderlay.blocksOf("la la\n[Chorus]\noh\n")[0].label)
    }
    @Test fun namesSkipHummedIntro() { val l=lay(score("intro" to "c4c4z8|","verse" to seven+seven,"interlude" to "c8z8|","chorus" to three+three)); assertTrue(l.paired); assertEquals(listOf(2 to 9,9 to 16,17 to 20,20 to 23),spans(l)); assertEquals("ffff",kinds(l)) }
    @Test fun pickupBelongsToNextSection() { val l=lay(score("intro" to "z12C2D2|","verse" to "E2F2G2A2B2z6|","chorus" to three),"[Verse]\nla la la la la la la\n[Chorus]\noh oh oh\n"); assertEquals(listOf(0 to 7,7 to 10),spans(l)); assertEquals("ff",kinds(l)) }
    @Test fun swappedNamesPairBySize() { val l=lay(score("intro" to "c4c4z8|","chorus" to seven+seven,"verse" to three+three)); assertTrue(l.paired); assertEquals(listOf(2 to 9,9 to 16,16 to 19,19 to 22),spans(l)); assertEquals("ffff",kinds(l)) }
    @Test fun pairByNameAndSize() {
        val s=listOf("intro" to 2,"verse" to 14,"chorus" to 6,"verse" to 14,"chorus" to 6,"outro" to 0); val labels=listOf("verse","chorus","verse","chorus")
        assertEquals(listOf(1,2,3,4),LyricUnderlay.byName(labels,s)); assertEquals(listOf(1,2,3,4),LyricUnderlay.bySize(labels,listOf(14,6,14,6),s)); assertNull(LyricUnderlay.byName(listOf("verse","bridge"),s)); assertEquals(listOf(1,2),LyricUnderlay.bySize(listOf("verse","bridge"),listOf(14,6),s)); assertEquals(listOf(0,1),LyricUnderlay.byName(listOf("verse","verse"),listOf("verse" to 7,"chorus" to 7))); assertNull(LyricUnderlay.bySize(listOf("verse","chorus","bridge"),listOf(7,7,7),listOf("verse" to 7,"chorus" to 0)))
    }
    @Test fun excessSectionsUseWholeSong() { val l=lay(score("verse" to seven+seven),verse+chorus+"[Bridge]\nla la\n"); assertFalse(l.paired); assertEquals(5,l.lines.size); assertEquals(0,l.lines[0].first); assertEquals(14,l.lines.last().stop) }
    @Test fun noNotesLeft() { val l=lay(score("verse" to three),"[Verse]\noh\nhey\nla\nna\nda\n"); assertEquals(listOf(1,1,1,0,0),l.lines.map { it.notes }); assertEquals("fffcc",kinds(l)); assertEquals(listOf(LyricUnderlay.Drawn(0,0,"oh"),LyricUnderlay.Drawn(1,1,"hey"),LyricUnderlay.Drawn(2,2,"la")),l.notes) }
    @Test fun bareScoreIsNotDrawn() { val l=lay(score(null to seven+seven),verse); assertTrue(l.bare); assertTrue(l.lines.isEmpty()); assertTrue(l.notes.isEmpty()) }
    @Test fun noWords() { for (words in listOf("","[Verse]\n\n[Chorus]\n","...\n")) { val l=lay(lyrics=words); assertFalse(l.bare); assertTrue(l.lines.isEmpty()); assertTrue(l.notes.isEmpty()); assertTrue(l.spare.isEmpty()) } }
    @Test fun silentVoiceCrowdsEveryLine() { val l=lay(score("verse" to rest+rest),verse); assertEquals(0,l.sung); assertEquals("cc",kinds(l)); assertTrue(l.notes.isEmpty()) }
    private fun songs()=JSONObject(javaClass.classLoader!!.getResource("midi/underlay_songs.json")!!.readText())
    @Test fun modelsOwnWords() {
        val songs=songs()
        for ((name,counts) in listOf("en_pop" to listOf(12,4,0),"ru_pop" to listOf(13,3,0))) {
            val s=songs.getJSONObject(name); val l=lay(s.getString("score"),s.getJSONObject("lyrics").getString("own")); assertTrue(l.paired)
            assertEquals(counts,listOf(LyricUnderlay.Kind.FIT,LyricUnderlay.Kind.SPARE,LyricUnderlay.Kind.CROWD).map { l.counts.getValue(it) }); assertEquals(16,l.lines.size); assertEquals(l.lines.map { it.first }.sorted(),l.lines.map { it.first })
        }
    }
    @Test fun rewrittenWordsKeepCuts() {
        val songs=songs()
        for (name in listOf("en_pop","ru_pop")) {
            val s=songs.getJSONObject(name); val lyrics=s.getJSONObject("lyrics"); val own=lay(s.getString("score"),lyrics.getString("own"))
            for (kind in listOf("same","more","less")) assertEquals("$name $kind",spans(own),spans(lay(s.getString("score"),lyrics.getString(kind))))
            assertEquals(kinds(own),kinds(lay(s.getString("score"),lyrics.getString("same"))))
            val expected=if (name=="en_pop") "ccccfffscccffffs" to "ssssfffsssssfffs" else "ccccfsffccccfssf" to "ssssfsffssssfssf"
            assertEquals(expected.first,kinds(lay(s.getString("score"),lyrics.getString("more")))); assertEquals(expected.second,kinds(lay(s.getString("score"),lyrics.getString("less"))))
        }
    }
    @Test fun everySyllableDrawnInOrder() {
        val songs=songs()
        for (name in listOf("en_pop","ru_pop")) { val s=songs.getJSONObject(name); val lyrics=s.getJSONObject("lyrics")
            for (kind in lyrics.keys()) {
                val words=lyrics.getString(kind); val l=lay(s.getString("score"),words); val notes=l.notes.map { it.note }
                assertEquals(notes.distinct().sorted(),notes); assertTrue(notes.toSet().intersect(l.spare.toSet()).isEmpty())
                val drawn=l.notes.joinToString("") { it.text.trimEnd('-') }.replace(" ","")
                val written=LyricUnderlay.blocksOf(words).flatMap { it.rows }.flatMap { it.parts }.joinToString("") { it.text }.replace(" ","")
                assertEquals("$name $kind",written,drawn)
            }
        }
    }
}
