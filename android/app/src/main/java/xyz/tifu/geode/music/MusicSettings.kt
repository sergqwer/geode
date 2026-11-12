package xyz.tifu.geode.music

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStoreFile
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import org.json.JSONObject

/** One AR stage's sampling (yue2 protocol.py's Sampling); the pack's defaults are [ABC] and [SEMANTIC]. */
data class StageSampling(
    val temperature: Float,
    val topP: Float,
    val topK: Int,
    val repetitionPenalty: Float,
    val penaltyWindow: Int,
    val minTokens: Int,
) {
    fun json(): JSONObject = JSONObject().put("temperature", temperature.toDouble()).put("top_p", topP.toDouble())
        .put("top_k", topK).put("repetition_penalty", repetitionPenalty.toDouble()).put("penalty_window", penaltyWindow)
        .put("min_tokens", minTokens)

    companion object {
        /** The score's: protocol.py GenerationConfig.abc (its max_tokens, 4096, stays the pack's). */
        val ABC = StageSampling(0.7f, 0.9f, 30, 1.005f, 100, 32)
        /** The codes': protocol.py's Sampling() defaults. */
        val SEMANTIC = StageSampling(1.0f, 0.95f, 100, 1.2f, 50, 200)

        fun of(o: JSONObject?, d: StageSampling): StageSampling = if (o == null) d else StageSampling(
            o.optDouble("temperature", d.temperature.toDouble()).toFloat(),
            o.optDouble("top_p", d.topP.toDouble()).toFloat(),
            o.optInt("top_k", d.topK),
            o.optDouble("repetition_penalty", d.repetitionPenalty.toDouble()).toFloat(),
            o.optInt("penalty_window", d.penaltyWindow),
            o.optInt("min_tokens", d.minTokens),
        )
    }
}

/** Which processor computes the acoustic stage (the NAR): the engine's pick, the NPU (HMX), or the CPU. */
enum class NarUnit(val key: String) { AUTO("auto"), DSP("dsp"), GPU("gpu"), CPU("cpu") }

/** The sound stage's precision (yue2_song.hpp NarPrecision): BEST as the model is; BALANCED only what costs at most a
 *  third of the bf16 run's own error (the processor's attention scores in int8); FAST also 8-bit weights (the NPU's
 *  layers, the processor's products) - noticeably less precise, made once from the installed model (1.5 GB). */
enum class SoundPrecision(val key: String) {
    BEST("best"), BALANCED("balanced"), FAST("fast");

    companion object {
        fun of(key: String?, d: SoundPrecision) = entries.firstOrNull { it.key == key } ?: d
    }
}

/** How long a song the writer asks for: its sung lines (YuE2-ComfyUI's WRITER_LENGTH_LINES). */
enum class LyricsLength(val lines: Int) { SHORT(8), NORMAL(16), LONG(24), VERY_LONG(32) }

/**
 * The music's settings, apart from the chat's (Settings.kt): the song model's inference, the mastering, the lyrics
 * writer. The defaults are the workstation's settings the user called the best (32 midpoint steps, cot "full",
 * -12.5 LUFS with at most +4 dB) - never a quicker, worse preset in their place.
 */
data class MusicPrefs(
    // ---- the song model ----
    /** The installed song model (MusicModels: files/music/models/<id>); "": the only one there is. */
    val modelId: String = "",
    val mode: YuE2Prompt.Mode = YuE2Prompt.Mode.FULL,
    /** Guidance; 0: the mode's own (1.0, "off" 1.01). 1.2 makes words about the arrangement in the style count. */
    val cfg: Float = 0f,
    /** A new seed for every song; false: [seed] each time. */
    val randomSeed: Boolean = true,
    val seed: Long = 831001,
    /** The song's ceiling in seconds; 0: worked out from the lyrics (YuE2Prompt.autoSeconds). */
    val maxSeconds: Int = 0,
    val odeSteps: Int = 32,
    /** The score and the codes from 4-bit weights (made once from the installed ones): quicker, less precise. */
    val arQ4: Boolean = false,
    /** The sound stage's precision: balanced by default (its losses a fraction of the model's own bf16 noise). */
    val soundPrecision: SoundPrecision = SoundPrecision.BALANCED,
    /** The pack's sampling for both stages; false: [abc] and [semantic]. */
    val defaultSampling: Boolean = true,
    val abc: StageSampling = StageSampling.ABC,
    val semantic: StageSampling = StageSampling.SEMANTIC,
    val narUnit: NarUnit = NarUnit.AUTO,
    /** Compute threads; 0: every core. */
    val threads: Int = 0,
    /** The chat's model is closed while a song is made: its memory is the song's (the song model is large). */
    val unloadChat: Boolean = true,
    // ---- the mastering ----
    val master: Boolean = true,
    val targetLufs: Float = -12.5f,
    val maxGainDb: Float = 4f,
    val bits: Int = 24,
    /** "flac" or "wav". */
    val format: String = "flac",
    // ---- the lyrics writer ----
    /** The chat model that writes the lyrics (ModelStore's id); "": the chat's selected one. */
    val writerModel: String = "",
    /** The language the song is sung in ("" = the writer's choice from the idea); a name in English. English unless
     *  the user picks another (the user, 2026-11-12: "дефолтна мова хай також стоїть англійською"). */
    val writerLanguage: String = "English",
    val writerLength: LyricsLength = LyricsLength.NORMAL,
    /** The writer marks the stressed vowel of names and rare words with a capital letter ("ФраголІно"). */
    val writerStress: Boolean = true,
    /** The writer may put instrumental breaks ([Interlude] alone) where the voice should rest. */
    val writerBreaks: Boolean = true,
    val writerTemperature: Float = 0.9f,
    val writerTopP: Float = 0.95f,
    val writerTopK: Int = 40,
    val writerMaxTokens: Int = 900,
    val writerThinking: Boolean = false,
) {
    /** The guidance the engine is given. */
    fun guidance(): Float = if (cfg > 0f) cfg else YuE2Prompt.defaultCfg(mode)

    fun json(): String = JSONObject()
        .put("model", modelId).put("mode", mode.key).put("cfg", cfg.toDouble()).put("random_seed", randomSeed)
        .put("seed", seed).put("max_seconds", maxSeconds).put("ode_steps", odeSteps).put("ar_q4", arQ4).put("sound_precision", soundPrecision.key)
        .put("default_sampling", defaultSampling).put("abc", abc.json()).put("semantic", semantic.json())
        .put("nar", narUnit.key).put("threads", threads).put("unload_chat", unloadChat)
        .put("master", master).put("target_lufs", targetLufs.toDouble()).put("max_gain_db", maxGainDb.toDouble())
        .put("bits", bits).put("format", format)
        .put("writer_model", writerModel).put("writer_lang", writerLanguage).put("writer_length", writerLength.name)
        .put("writer_stress", writerStress).put("writer_breaks", writerBreaks)
        .put("writer_temperature", writerTemperature.toDouble()).put("writer_top_p", writerTopP.toDouble())
        .put("writer_top_k", writerTopK).put("writer_max_tokens", writerMaxTokens).put("writer_thinking", writerThinking)
        .toString()

    companion object {
        /** Takes of one song asked for at once, the Music screen's and the server's (each is a whole song's time: a
         *  quarter of an hour on a mid-range phone). */
        const val MAX_TAKES = 4
        /** The sound steps offered (32: what the model was made for). */
        val STEPS = listOf(8, 16, 24, 32, 48, 64)
        /** The song lengths offered, in seconds (0: from the lyrics). */
        val LENGTHS = listOf(0, 30, 60, 90, 120, 180, 240, 300, 360)
        /** The files offered: format and bits. */
        val FORMATS = listOf("flac" to 24, "flac" to 16, "wav" to 24, "wav" to 16)

        /** A value written by an older or newer build: what it has, the defaults for the rest. */
        fun parse(text: String?): MusicPrefs {
            val d = MusicPrefs()
            val o = runCatching { JSONObject(text ?: return d) }.getOrNull() ?: return d
            fun f(k: String, v: Float) = o.optDouble(k, v.toDouble()).toFloat()
            return MusicPrefs(
                modelId = o.optString("model", d.modelId),
                mode = YuE2Prompt.Mode.of(o.optString("mode")),
                cfg = f("cfg", d.cfg).coerceIn(0f, 3f),
                randomSeed = o.optBoolean("random_seed", d.randomSeed),
                seed = o.optLong("seed", d.seed).coerceAtLeast(0),
                maxSeconds = o.optInt("max_seconds", d.maxSeconds).coerceIn(0, YuE2Prompt.MAX_SECONDS),
                odeSteps = o.optInt("ode_steps", d.odeSteps).coerceIn(1, 128),
                arQ4 = o.optBoolean("ar_q4", d.arQ4),
                soundPrecision = SoundPrecision.of(o.optString("sound_precision").ifEmpty { null }
                    ?: if (o.optBoolean("nar_q8", false)) "fast" else null, d.soundPrecision),
                defaultSampling = o.optBoolean("default_sampling", d.defaultSampling),
                abc = StageSampling.of(o.optJSONObject("abc"), d.abc),
                semantic = StageSampling.of(o.optJSONObject("semantic"), d.semantic),
                narUnit = NarUnit.entries.firstOrNull { it.key == o.optString("nar") } ?: d.narUnit,
                threads = o.optInt("threads", d.threads).coerceIn(0, 32),
                unloadChat = o.optBoolean("unload_chat", d.unloadChat),
                master = o.optBoolean("master", d.master),
                targetLufs = f("target_lufs", d.targetLufs).coerceIn(-30f, -6f),
                maxGainDb = f("max_gain_db", d.maxGainDb).coerceIn(0f, 24f),
                bits = o.optInt("bits", d.bits).takeIf { it == 16 || it == 24 } ?: d.bits,
                format = o.optString("format", d.format).takeIf { it == "flac" || it == "wav" } ?: d.format,
                writerModel = o.optString("writer_model", d.writerModel),
                // (writer_lang since English became the default; the older key's "" was its default "the writer
                // decides", not a choice - English now; a language named there was picked, and stays)
                writerLanguage = if (o.has("writer_lang")) o.optString("writer_lang")
                    else o.optString("writer_language").takeIf { it.isNotEmpty() } ?: d.writerLanguage,
                writerLength = runCatching { LyricsLength.valueOf(o.optString("writer_length")) }.getOrDefault(d.writerLength),
                writerStress = o.optBoolean("writer_stress", d.writerStress),
                writerBreaks = o.optBoolean("writer_breaks", d.writerBreaks),
                writerTemperature = f("writer_temperature", d.writerTemperature).coerceIn(0f, 2f),
                writerTopP = f("writer_top_p", d.writerTopP).coerceIn(0.05f, 1f),
                writerTopK = o.optInt("writer_top_k", d.writerTopK).coerceIn(1, 200),
                writerMaxTokens = o.optInt("writer_max_tokens", d.writerMaxTokens).coerceIn(200, 4000),
                writerThinking = o.optBoolean("writer_thinking", d.writerThinking),
            )
        }
    }
}

/** The music's settings in a DataStore of their own (one per process: the UI's, GeodeApp.music). */
class MusicSettings(context: Context) {
    private val store: DataStore<Preferences> = Store.of(context.applicationContext)
    private val key = stringPreferencesKey("music")

    val flow: Flow<MusicPrefs> = store.data.map { MusicPrefs.parse(it[key]) }

    suspend fun current(): MusicPrefs = flow.first()

    suspend fun update(change: (MusicPrefs) -> MusicPrefs) {
        store.edit { p -> p[key] = change(MusicPrefs.parse(p[key])).json() }
    }

    private object Store {
        @Volatile private var s: DataStore<Preferences>? = null

        fun of(context: Context): DataStore<Preferences> = s ?: synchronized(this) {
            s ?: PreferenceDataStoreFactory.create(
                corruptionHandler = ReplaceFileCorruptionHandler { emptyPreferences() },
                produceFile = { context.preferencesDataStoreFile("music") },
            ).also { s = it }
        }
    }
}
