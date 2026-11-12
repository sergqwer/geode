package xyz.tifu.geode.music

import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import xyz.tifu.geode.tokenizer.Tokenizer
import xyz.tifu.geode.ui.music.LyricsEdit

/** The music's pure parts: the prompt's tokens against the reference's, the writer's parsing, the editor's tools. */
class MusicTest {
    /**
     * The heads MusicJob.request builds against the ones tools/yue2_request.py made with the reference tokenizer
     * (work/tok-yue2/song_request.json: the 10 s Ukrainian song sung on the Redmi); skipped without them.
     */
    @Test
    fun promptMatchesReference() {
        val dir = File(System.getProperty("geode.yue2tokdir") ?: "")
        val bin = File(dir, Tokenizer.FILE)
        val req = File(dir, "song_request.json")
        assumeTrue("work/tok-yue2/tokenizer.bin and song_request.json", bin.isFile && req.isFile)
        val tok = Tokenizer.load(bin)
        val o = JSONObject(req.readText())
        val mode = YuE2Prompt.Mode.of(o.getString("cot"))
        val pos = intArrayOf(YuE2Prompt.EOD) + tok.encode(YuE2Prompt.text(mode, o.getString("style"), o.getString("lyrics")))
        val neg = intArrayOf(YuE2Prompt.EOD) + tok.encode(mode.instruction)
        fun ints(k: String) = o.getJSONArray(k).let { a -> IntArray(a.length()) { a.getInt(it) } }
        assertArrayEquals(ints("positive_head"), pos)
        assertArrayEquals(ints("negative_head"), neg)
    }

    /** A sung song asks the engine for its editable state beside its audio; a plan (nothing sung) does not. */
    @Test
    fun aSungSongKeepsItsStateAPlanDoesNot() {
        val dir = File(System.getProperty("geode.yue2tokdir") ?: "")
        val bin = File(dir, Tokenizer.FILE)
        assumeTrue("work/tok-yue2/tokenizer.bin", bin.isFile)
        val tok = Tokenizer.load(bin)
        val songs = File(System.getProperty("java.io.tmpdir"), "geode-state-test-" + System.nanoTime())
        try {
            val store = SongStore(File(songs, "songs"), File(songs, "shared"))
            val sung = store.create("One", "pop", "[verse]\nla la", JSONObject().put("mode", "full").put("seed", 3L))
            val q = JSONObject(MusicJob.request(sung, tok))
            assertEquals(File(sung.dir, MusicJob.STATE_FILE).path, q.getString("state_out"))
            val plan = store.create("Two", "pop", "[verse]\nla la",
                JSONObject().put("mode", "full").put("seed", 4L).put("score_only", true))
            assertTrue(!JSONObject(MusicJob.request(plan, tok)).has("state_out"))
            // a retake's request: the edit, and what it sings with - its seed, the AR's weights, the song's own sampling
            // only when the song had its own
            val edit = JSONObject().put("op", "retake").put("parent", sung.id).put("first", 100).put("stop", 200)
                .put("state_in", "s.state").put("audio_in", "s.flac")
            val sampling = JSONObject().put("temperature", 0.9)
            val retake = store.create("One (retake)", "pop", "[verse]\nla la", JSONObject().put("mode", "full").put("seed", 77L)
                .put("ar", "q4").put("default_sampling", false).put("semantic_sampling", sampling).put("edit", edit))
            val r = JSONObject(MusicJob.request(retake, tok))
            assertEquals("retake", r.getJSONObject("edit").getString("op"))
            assertEquals(100, r.getJSONObject("edit").getInt("first"))
            assertEquals(77L, r.getLong("seed"))
            assertEquals("q4", r.getString("ar"))
            assertEquals(0.9, r.getJSONObject("semantic_sampling").getDouble("temperature"), 1e-9)
            assertEquals(File(retake.dir, MusicJob.STATE_FILE).path, r.getString("state_out"))
            val plain = store.create("One (retake)", "pop", "[verse]\nla la", JSONObject().put("mode", "full").put("seed", 78L)
                .put("default_sampling", true).put("semantic_sampling", sampling).put("edit", edit))
            assertTrue(!JSONObject(MusicJob.request(plain, tok)).has("semantic_sampling"))
            // a cut's sings nothing: no seed of its own
            val cut = store.create("One (cut)", "pop", "[verse]\nla la", JSONObject().put("mode", "full").put("seed", 79L)
                .put("edit", JSONObject(edit.toString()).put("op", "cut")))
            assertTrue(!JSONObject(MusicJob.request(cut, tok)).has("seed"))
        } finally {
            songs.deleteRecursively()
        }
    }

    /**
     * A score the model wrote on the Redmi (work/tok-yue2/redmi_abc.json: a song's "abc" ids) read back to text, read
     * by the score format's parser, moved and tokenized again - the path a new take in another key takes.
     */
    @Test
    fun songScoreMovesAndTokenizes() {
        val dir = File(System.getProperty("geode.yue2tokdir") ?: "")
        val bin = File(dir, Tokenizer.FILE)
        val src = File(dir, "redmi_abc.json")
        assumeTrue("work/tok-yue2/tokenizer.bin and redmi_abc.json", bin.isFile && src.isFile)
        val tok = Tokenizer.load(bin)
        val a = JSONObject(src.readText()).getJSONArray("abc")
        val text = tok.decode(IntArray(a.length()) { a.getInt(it) })
        AbcScore.parse(text.trim())
        assertEquals(text.trim(), AbcScore.move(text, 0).text)
        val key = AbcScore.key(text)
        assertTrue("a key", key != null)
        for (s in intArrayOf(-5, -1, 2, 7)) {
            val moved = AbcScore.move(text, s)
            assertEquals(key, moved.before)
            assertEquals(moved.text, tok.decode(tok.encode(moved.text)))   // (the engine gets these ids)
            assertEquals(moved.after, AbcScore.key(moved.text))
        }
    }

    @Test
    fun writerSplitsTheHeadedLayout() {
        val answer = """
            <think>planning</think>
            STYLE
            "Ukrainian, indie pop, warm female voice, acoustic guitar, 96 BPM"

            LYRICS:
            **Verse 1**
            1. Перший рядок
            (softly)
            Другий рядок
            chorus
            "Приспів іде"
            ```
        """.trimIndent()
        val (style, lyrics) = LyricsWriter.split(answer)
        assertEquals("Ukrainian, indie pop, warm female voice, acoustic guitar, 96 BPM", style)
        assertEquals("[Verse 1]\nПерший рядок\nДругий рядок\n\n[Chorus]\nПриспів іде", lyrics)
        assertTrue(LyricsWriter.complete(style, lyrics))
    }

    @Test
    fun writerFallsBackToTheFirstTag() {
        val (style, lyrics) = LyricsWriter.split("Rock, male voice, 120 BPM\n[Verse]\nLine one\n[Interlude]\n[Chorus]\nLa la")
        assertEquals("Rock, male voice, 120 BPM", style)
        assertEquals("[Verse]\nLine one\n\n[Interlude]\n\n[Chorus]\nLa la", lyrics)
    }

    @Test
    fun writerDropsNotesAboutASection() {
        // what Gemma 4 E2B wrote on the Redmi: a note under the [Intro] instead of nothing
        val (_, lyrics) = LyricsWriter.split("Pop, 90 BPM\n[Intro]\nNo tags\n[Verse]\nNone of us\n(instrumental only)\nInstrumental only.")
        assertEquals("[Intro]\n\n[Verse]\nNone of us", lyrics)
    }

    @Test
    fun writerUnclosedReasoningIsNoAnswer() {
        assertEquals("" to "", LyricsWriter.split("<think>still thinking about the verse"))
    }

    @Test
    fun writerNotes() {
        val notes = LyricsWriter.findings("Pop, female voice", "line\nline", 16)
        assertEquals(listOf(WriterNote.SHORT_OR_LONG, WriterNote.NO_TEMPO, WriterNote.NO_TAGS), notes)
    }

    @Test
    fun sectionPlanScalesWithTheBudget() {
        assertEquals(2 to 4, LyricsWriter.sectionPlan(8))
        assertEquals(4 to 4, LyricsWriter.sectionPlan(16))
        assertEquals(8 to 4, LyricsWriter.sectionPlan(32))
    }

    @Test
    fun systemPromptHasTheRulesAskedFor() {
        val on = LyricsWriter.systemPrompt(MusicPrefs(writerLanguage = "Ukrainian"))
        assertTrue("Sing it in Ukrainian" in on && "ФраголІно" in on && "[Interlude]" in on)
        val off = LyricsWriter.systemPrompt(MusicPrefs(writerStress = false, writerBreaks = false, writerLanguage = ""))
        assertTrue("ФраголІно" !in off && "Interlude" !in off && "Sing it in" !in off && "{" !in off)
        // English by default
        assertTrue("Sing it in English" in LyricsWriter.systemPrompt(MusicPrefs()))
    }

    @Test
    fun writerLanguageEnglishUnlessPicked() {
        // the older key's "" was its default (the writer decides), not a choice: English now
        assertEquals("English", MusicPrefs.parse("""{"writer_language":""}""").writerLanguage)
        // a language picked then stays
        assertEquals("Ukrainian", MusicPrefs.parse("""{"writer_language":"Ukrainian"}""").writerLanguage)
        // "the writer decides" picked under the new key stays so
        assertEquals("", MusicPrefs.parse("""{"writer_lang":""}""").writerLanguage)
        assertEquals("", MusicPrefs.parse(MusicPrefs(writerLanguage = "").json()).writerLanguage)
        assertEquals("English", MusicPrefs.parse(null).writerLanguage)
    }

    @Test
    fun lengthCeiling() {
        assertEquals(180, YuE2Prompt.autoSeconds("[Intro]\n\n[Outro]"))
        assertEquals(40, YuE2Prompt.autoSeconds("[Verse]\nOne line"))
        assertEquals(12 + 12 * 4, YuE2Prompt.autoSeconds("[Verse]\na\nb\nc\nd"))
        // a long line counts one line for every eight words; a direction in brackets is not a word
        assertEquals(2, YuE2Prompt.sungLines("one two three four five six seven eight nine ten eleven twelve [Guitar stab]"))
        assertEquals(360, YuE2Prompt.autoSeconds((1..40).joinToString("\n") { "line $it" }))
    }

    @Test
    fun insertTagOnItsOwnLine() {
        val v = LyricsEdit.insertTag(TextFieldValue("[Verse]\nFirst line", TextRange(10)), "Chorus")
        assertEquals("[Verse]\nFirst line\n\n[Chorus]\n", v.text)
        assertEquals(v.text.length, v.selection.end)
        // on an empty line: there, with the words after it under it
        val w = LyricsEdit.insertTag(TextFieldValue("[Verse]\nOne\n\n\nTwo", TextRange(12)), "Bridge")
        assertEquals("[Verse]\nOne\n\n[Bridge]\nTwo", w.text)
        // a break: no words under it
        val p = LyricsEdit.insertTag(TextFieldValue("", TextRange(0)), "Interlude")
        assertEquals("[Interlude]\n\n", p.text)
    }

    @Test
    fun instrumentalPlanOneTagALine() {
        val a = LyricsEdit.insertLine(TextFieldValue("", TextRange(0)), "[intro]")
        assertEquals("[intro]", a.text)
        assertEquals(7, a.selection.end)
        val b = LyricsEdit.insertLine(a, "[verse]")
        assertEquals("[intro]\n[verse]", b.text)
        assertEquals(b.text.length, b.selection.end)
        // between two lines: after the cursor's, the cursor at its end, no blank lines
        val c = LyricsEdit.insertLine(TextFieldValue("[intro]\n[chorus]", TextRange(3)), "[verse]")
        assertEquals("[intro]\n[verse]\n[chorus]", c.text)
        assertEquals(15, c.selection.end)
        // blank -> the adapter's own "choose for me"
        assertEquals(MusicAdapter.BARE, MusicAdapter.lyricsOf("  \n"))
        assertEquals("[intro]\n[outro]", MusicAdapter.lyricsOf("\n[intro]\n[outro]\n"))
    }

    @Test
    fun stressFlipsTheLetterBeforeTheCursor() {
        val v = LyricsEdit.toggleStress(TextFieldValue("Фраголіно", TextRange(7)))
        assertEquals("ФраголІно", v.text)
        assertEquals("Фраголіно", LyricsEdit.toggleStress(v).text)
        // a selected letter
        assertEquals("зАмок", LyricsEdit.toggleStress(TextFieldValue("замок", TextRange(1, 2))).text)
    }

    @Test
    fun presetKeepsTheLanguage() {
        assertEquals("Ukrainian, City Pop, synth bass", LyricsEdit.applyPreset("Ukrainian, rock", "City Pop, synth bass", ""))
        assertEquals("Polish, Jazz", LyricsEdit.applyPreset("", "Jazz", "Polish"))
        assertEquals("Jazz", LyricsEdit.applyPreset("", "Jazz", ""))
        assertEquals("Pop, Funk", LyricsEdit.addGenre("Pop,", "Funk"))
    }

    @Test
    fun againKeepsTheSongsRequest() {
        val job = JSONObject().put("pack_dir", "/old/pack").put("npu_dir", "/old/lib").put("mode", "full")
            .put("score", "generate").put("abc", JSONArray()).put("cfg", 1.3).put("seed", 42L).put("seconds", 90)
            .put("max_frames", 2250).put("ode_steps", 48).put("ar", "q4").put("nar_precision", "best")
            .put("nar", "cpu").put("threads", 2).put("format", "wav").put("output", "/old/song.wav")
        val result = JSONObject().put("ok", true).put("abc", JSONArray().put(5).put(6).put(7))
        val song = Song("s1", File("/songs/s1"), "Night", "pop", "[Verse]\nA line", 1L, job, result)
        val p = MusicPrefs().copy(threads = 6)
        val again = MusicJob.again(song, File("/new/pack"), "/new/lib", p, melody = false)
        assertEquals(File("/new/pack").path, again.getString("pack_dir"))
        assertEquals("/new/lib", again.getString("npu_dir"))
        assertTrue(!again.has("output"))
        assertTrue(again.getLong("seed") != 42L)
        // the song's own settings
        assertEquals(1.3, again.getDouble("cfg"), 0.0)
        assertEquals(48, again.getInt("ode_steps"))
        assertEquals("q4", again.getString("ar"))
        assertEquals("best", again.getString("nar_precision"))
        assertEquals("wav", again.getString("format"))
        assertEquals(2250, again.getInt("max_frames"))
        // this phone's unit and threads, as the settings say now
        assertEquals(p.narUnit.key, again.getString("nar"))
        assertEquals(6, again.getInt("threads"))
        assertEquals("generate", again.getString("score"))
        assertEquals(0, again.getJSONArray("abc").length())
        // the same tune: the score the song was sung from
        val tune = MusicJob.again(song, File("/new/pack"), "/new/lib", p, melody = true)
        assertEquals("given", tune.getString("score"))
        assertEquals(3, tune.getJSONArray("abc").length())
        // the song's job itself is not changed
        assertEquals(42L, job.getLong("seed"))
        assertEquals("/old/song.wav", job.getString("output"))
    }

    @Test
    fun aPlanIsAScoreAloneThenSung() {
        val dir = kotlin.io.path.createTempDirectory("plan").toFile()
        try {
            val p = MusicPrefs().copy(randomSeed = false, seed = 100)
            val job = MusicJob.create(File("/pack"), "/lib", YuE2Prompt.Mode.FULL, "[Verse]\nA line", p, take = 1, scoreOnly = true)
            assertTrue(job.getBoolean("score_only"))
            assertEquals(101L, job.getLong("seed"))
            // not with a borrowed score, nor in the mode without one
            assertTrue(!MusicJob.create(File("/pack"), "/lib", YuE2Prompt.Mode.FULL, "", p, intArrayOf(1, 2), scoreOnly = true).has("score_only"))
            assertTrue(!MusicJob.create(File("/pack"), "/lib", YuE2Prompt.Mode.OFF, "", p, scoreOnly = true).has("score_only"))
            val plan = Song("p1", dir, "Night", "pop", "[Verse]\nA line", 1L, job, null)
            assertTrue(plan.plan)
            // a companion's run wrote its score: done without a run of its own - another seed's is not its own
            File(dir, MusicJob.AR_FILE).writeText("""{"seed":7,"ar":"q8","abc":[4,5],"abc_eos":true,"codes":[],"codes_eos":true}""")
            assertEquals(null, MusicJob.planOf(plan))
            File(dir, MusicJob.AR_FILE).writeText("""{"seed":101,"ar":"q8","abc":[4,5],"abc_eos":false,"codes":[],"codes_eos":true,"made":41.5}""")
            val r = MusicJob.planOf(plan)!!
            assertEquals(41.5, plan.copy(result = r).makeSeconds, 0.0)
            assertTrue(r.getBoolean("ok"))
            assertEquals(2, r.getJSONArray("abc").length())
            assertTrue(!r.getBoolean("abc_eos"))
            // sung: the plan's score given, its seed, no longer a plan
            val done = plan.copy(result = r)
            val song = MusicJob.fromPlan(done, File("/new"), "/lib2", p)
            assertTrue(!song.has("score_only"))
            assertEquals("given", song.getString("score"))
            assertEquals(101L, song.getLong("seed"))
            assertEquals(2, song.getJSONArray("abc").length())
            // a plan again: another plan; again on its score: sung
            assertTrue(MusicJob.again(done, File("/new"), "/lib2", p, melody = false).getBoolean("score_only"))
            assertTrue(!MusicJob.again(done, File("/new"), "/lib2", p, melody = true).has("score_only"))
            // a song is never a plan's result: no ar.json of a plan for it
            assertEquals(null, MusicJob.planOf(Song("s", dir, "", "", "", 1L, JSONObject().put("seed", 101L), null)))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun prefsRoundTrip() {
        val p = MusicPrefs(mode = YuE2Prompt.Mode.MELODY, cfg = 1.2f, randomSeed = false, seed = 42, odeSteps = 32,
            defaultSampling = false, abc = StageSampling.ABC.copy(topK = 50), narUnit = NarUnit.CPU, bits = 16,
            format = "wav", writerLength = LyricsLength.LONG, writerStress = false)
        assertEquals(p, MusicPrefs.parse(p.json()))
        assertEquals(MusicPrefs(), MusicPrefs.parse("not json"))
    }
}
