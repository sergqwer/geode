package xyz.tifu.geode.data

import android.content.Context
import android.util.Base64
import androidx.datastore.core.DataStore
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.preferences.core.MutablePreferences
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.doublePreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.mutablePreferencesOf
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStoreFile
import java.io.File
import java.io.FileOutputStream
import java.util.zip.CRC32
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.probe.MemoryTest
import xyz.tifu.geode.probe.ProbeService

/**
 * A model family's sampling without reasoning and with it, as its model card recommends: Qwen3.6-35B-A3B wants
 * different values for each mode and a presence penalty against a long reasoning going round in circles; Qwen3.5-2B
 * others again (no top-p and a stronger penalty without reasoning: its card's text tasks); Gemma 4 one set for every
 * use and no penalty (it behaves oddly at low temperatures).
 */
data class SamplingPrefs(
    val temperature: Float,
    val topP: Float,
    val topK: Int,
    val thinkTemperature: Float,
    val thinkTopP: Float,
    val thinkTopK: Int,
    /** Keeps a long answer (or, [thinkPresencePenalty], a long reasoning) from going round in circles. */
    val presencePenalty: Float,
    val thinkPresencePenalty: Float,
) {
    fun temperature(thinking: Boolean) = if (thinking) thinkTemperature else temperature
    fun topP(thinking: Boolean) = if (thinking) thinkTopP else topP
    fun topK(thinking: Boolean) = if (thinking) thinkTopK else topK
    fun presence(thinking: Boolean) = if (thinking) thinkPresencePenalty else presencePenalty

    companion object {
        val QWEN = SamplingPrefs(0.7f, 0.8f, 20, 1.0f, 0.95f, 20, 1.5f, 1.5f)
        val QWEN_SMALL = SamplingPrefs(1.0f, 1.0f, 20, 1.0f, 0.95f, 20, 2.0f, 1.5f)
        val GEMMA = SamplingPrefs(1.0f, 0.95f, 64, 1.0f, 0.95f, 64, 0f, 0f)
    }
}

/** Which model card's sampling a model follows: Qwen3.6's MoE, the small dense Qwen3.5, Gemma 4. */
enum class ModelFamily(val defaults: SamplingPrefs) {
    QWEN(SamplingPrefs.QWEN), QWEN_SMALL(SamplingPrefs.QWEN_SMALL), GEMMA(SamplingPrefs.GEMMA);

    companion object {
        /** The model's family (Qwen3.6's when none is selected). */
        fun of(m: xyz.tifu.geode.models.InstalledModel?): ModelFamily = when {
            m == null -> QWEN
            m.manifest.gemma -> GEMMA
            m.manifest.arch == "qwen35" -> QWEN_SMALL
            else -> QWEN
        }
    }
}

/** A family's sampling in the settings: Qwen's under the keys from before there were two, Gemma's under its own. */
private class SamplingKeys(prefix: String, val default: SamplingPrefs) {
    val temperature = floatPreferencesKey(prefix + "temperature")
    val topP = floatPreferencesKey(prefix + "top_p")
    val topK = intPreferencesKey(prefix + "top_k")
    val thinkTemperature = floatPreferencesKey(prefix + "think_temperature")
    val thinkTopP = floatPreferencesKey(prefix + "think_top_p")
    val thinkTopK = intPreferencesKey(prefix + "think_top_k")
    val presence = floatPreferencesKey(prefix + "presence_penalty")
    val thinkPresence = floatPreferencesKey(prefix + "think_presence_penalty")

    fun read(p: Preferences) = SamplingPrefs(
        temperature = p[temperature] ?: default.temperature,
        topP = p[topP] ?: default.topP,
        topK = p[topK] ?: default.topK,
        thinkTemperature = p[thinkTemperature] ?: default.thinkTemperature,
        thinkTopP = p[thinkTopP] ?: default.thinkTopP,
        thinkTopK = p[thinkTopK] ?: default.thinkTopK,
        presencePenalty = p[presence] ?: default.presencePenalty,
        // (one penalty for both modes before: a value set then stands for reasoning too)
        thinkPresencePenalty = p[thinkPresence] ?: p[presence] ?: default.thinkPresencePenalty,
    )

    fun write(p: MutablePreferences, v: SamplingPrefs) {
        p[temperature] = v.temperature
        p[topP] = v.topP
        p[topK] = v.topK
        p[thinkTemperature] = v.thinkTemperature
        p[thinkTopP] = v.thinkTopP
        p[thinkTopK] = v.thinkTopK
        p[presence] = v.presencePenalty
        p[thinkPresence] = v.thinkPresencePenalty
    }
}

data class AppSettings(
    /** Memory the engine may hold in all; 0 = not set (use the recommendation or a default). */
    val memoryBudgetMb: Int = 0,
    /** From the memory test: the last level before the phone pushed memory into swap, and where it killed. */
    val recommendedMb: Int = 0,
    val limitMb: Int = 0,
    val thinking: Boolean = false,
    /** The sampling of Qwen3.6-35B-A3B, of Qwen3.5-2B and of Gemma 4: each family its own (model cards). */
    val qwenSampling: SamplingPrefs = SamplingPrefs.QWEN,
    val qwenSmallSampling: SamplingPrefs = SamplingPrefs.QWEN_SMALL,
    val gemmaSampling: SamplingPrefs = SamplingPrefs.GEMMA,
    /** What the user turned off of the models' inputs and drafting (OFF_* bits, models/ModelStore.kt): images,
     *  video, audio, speculative decoding - each where a model has it (InstalledModel.off). */
    val featuresOff: Int = 0,
    /** Reasoning tokens before the model is told to answer (Qwen's way: a closing sentence and </think>); 0 = no
     *  limit. */
    val thinkBudget: Int = 0,
    /** The longest answer in tokens (reasoning included); 0 = until the context is full. */
    val maxAnswer: Int = 0,
    /** The longest context; the engine allocates it chunk by chunk, as the chat grows. */
    val contextLength: Int = 32768,
    /** The storage the saved conversations (their KV cache and states) may take, MB; 0: none kept. */
    val stateCacheMb: Int = 4096,
    /** How finely the model sees an image: ImageStore.DETAIL_PIXELS's index (more pixels, more tokens and time). */
    val imageDetail: Int = 0,
    /** 0 = automatic. */
    val computeThreads: Int = 0,
    /** Expert readers on the little cores; 0 = automatic. */
    val ioThreads: Int = 0,
    /** Little-core threads that join a prompt's products; 0 = none. */
    val helperThreads: Int = 0,
    /** Read the next layer's predicted experts ahead. */
    val prefetch: Boolean = true,
    /** A prompt's big dense products shared with the GPU (the Adreno 618, packs with q2 / q4 dense weights). */
    val gpu: Boolean = true,
    val gpuShare: Float = 0.3f,
    /** Cache-conditioned routing (EngineOptions.routePrior): off by default - it changes answers a little. */
    val routePrior: Boolean = false,
    /** A prompt read with the NPU where the phone has one Geode's module runs on (EngineOptions.npu): on by default. */
    val npu: Boolean = true,
    /** Which of the phone's units run the model (RunModes; NPU + GPU by default), in place of the GPU and NPU
     *  switches (a user who had turned one off starts from the units left on). */
    val runMode: Int = RunModes.DEFAULT,
    /** The engine's "units" probe (RunModes.Units: the GPU and the NPU the phone has), and the app version it ran in. */
    val unitsJson: String = "",
    val unitsVersion: String = "",
    /**
     * Experts read for each word: 8 = every expert the router picks (the default, the model's full quality); 6 / 4 =
     * the last 2 / 4 of them are left out when not in memory (EngineOptions.tailSkip) - faster, less accurate.
     */
    val wordExperts: Int = 8,
    /** 0 = system, 1 = light, 2 = dark. */
    val theme: Int = 0,
    /** Material You: the colours from the wallpaper (Android 12+); off: Geode's violet. */
    val dynamicColor: Boolean = true,
    /** The chat's text (its messages and its input) at this many times the system's size. */
    val chatTextScale: Float = 1f,
    /** The screen kept on: [KEEP_OFF] (the phone's own timeout), [KEEP_ANSWERING], [KEEP_ALWAYS] (while Geode is open). */
    val keepScreen: Int = KEEP_OFF,
    /** While Geode keeps the screen on: black after this many minutes without a touch (OLED burn-in); 0 = never. */
    val blackScreenMin: Int = 5,
    /** Answers read aloud as they are written (the phone's text-to-speech). */
    val readAloud: Boolean = false,
    /** Its speed: 1 = the engine's normal. */
    val speechRate: Float = 1f,
    /** Its pitch: 1 = the voice's own. */
    val speechPitch: Float = 1f,
    /** The voice picked (the engine's name for it) for its language; "" = each sentence's language's default. */
    val speechVoice: String = "",
    /** What reads aloud: 0 the phone's text-to-speech, 1 Geode's own voice (NeuralVoice, once downloaded). */
    val speechEngine: Int = 0,
    /** Geode's voice's style: 0-4 female, 5-9 male. */
    val neuralVoice: Int = 0,
    val selectedModel: String = "",
    /** A model of an outside service the chats use instead of the phone's (Providers: its id); "": the phone's. */
    val selectedRemote: String = "",
    val hfToken: String = "",
    /** The share of expert picks the cache served in the last answer (0..1; 0 = not measured yet). */
    val lastHitRate: Float = 0f,
    /** The local server (OpenAI and Anthropic APIs on the local network) is on: it starts with the app. */
    val serverOn: Boolean = false,
    val serverPort: Int = 8080,
    /** Requests need [serverKey]; off: anyone on the network may use the model. */
    val serverKeyOn: Boolean = true,
    val serverKey: String = "",
    /** The web chat (the app's chats in a browser on the local network, behind [webPassword]) is on. */
    val webOn: Boolean = false,
    val webPassword: String = "",
) {
    /** The local server runs: the API or the web chat is on. */
    val serverRuns: Boolean get() = serverOn || webOn

    /** The sampling a model of this family generates with. */
    fun samplingOf(f: ModelFamily): SamplingPrefs = when (f) {
        ModelFamily.QWEN -> qwenSampling
        ModelFamily.QWEN_SMALL -> qwenSmallSampling
        ModelFamily.GEMMA -> gemmaSampling
    }

    /** The context a model opens with: the setting, at most what the model was trained for. */
    fun contextFor(m: xyz.tifu.geode.models.InstalledModel?): Int =
        m?.manifest?.maxContext?.takeIf { it > 0 }?.let { minOf(contextLength, it) } ?: contextLength

    /** What the engine opens the model with (the model: its longest context, what of it the user turned off). */
    fun engineOptions(m: xyz.tifu.geode.models.InstalledModel?): xyz.tifu.geode.engine.EngineOptions =
        xyz.tifu.geode.engine.EngineOptions(
            budgetBytes = effectiveBudgetMb.toLong() shl 20, threads = computeThreads, ioThreads = ioThreads,
            helpers = helperThreads, contextLength = contextFor(m), prefetch = prefetch, gpu = gpu, gpuShare = gpuShare,
            // upstream Strata 0.2.7's lambda: 2.0 showed copy and code defects there
            routePrior = if (routePrior) 1.5f else 0f,
            tailSkip = if (wordExperts in 1..7) wordExperts else 0,
            npu = npu,
            mode = runMode,
            off = m?.copy(off = featuresOff)?.engineOff ?: 0,
        )

    /** The budget actually used: what the user set, else the test's recommendation, else a cautious default. */
    val effectiveBudgetMb: Int get() = when {
        memoryBudgetMb > 0 -> memoryBudgetMb
        recommendedMb > 0 -> recommendedMb
        else -> DEFAULT_BUDGET_MB
    }

    companion object {
        const val DEFAULT_BUDGET_MB = 2500
        /** The shortest context the agent works in: its system prompt and tools take ~7k tokens with their margin. */
        const val MIN_CONTEXT = 16384
        const val KEEP_OFF = 0
        const val KEEP_ANSWERING = 1
        const val KEEP_ALWAYS = 2
    }
}

private object K {
    val budget = intPreferencesKey("memory_budget_mb")
    val recommended = intPreferencesKey("recommended_mb")
    val limit = intPreferencesKey("limit_mb")
    val thinking = booleanPreferencesKey("thinking")
    val qwenSampling = SamplingKeys("", SamplingPrefs.QWEN)
    val qwenSmallSampling = SamplingKeys("qwen_small_", SamplingPrefs.QWEN_SMALL)
    val gemmaSampling = SamplingKeys("gemma_", SamplingPrefs.GEMMA)
    val featuresOff = intPreferencesKey("features_off")
    val context = intPreferencesKey("context_length")
    val threads = intPreferencesKey("compute_threads")
    val theme = intPreferencesKey("theme")
    val dynamicColor = booleanPreferencesKey("dynamic_color")
    val chatTextScale = floatPreferencesKey("chat_text_scale")
    val keepScreen = intPreferencesKey("keep_screen")
    val blackScreenMin = intPreferencesKey("black_screen_min")
    val readAloud = booleanPreferencesKey("read_aloud")
    val speechRate = floatPreferencesKey("speech_rate")
    val speechPitch = floatPreferencesKey("speech_pitch")
    val speechVoice = stringPreferencesKey("speech_voice")
    val speechEngine = intPreferencesKey("speech_engine")
    val neuralVoice = intPreferencesKey("neural_voice")
    val model = stringPreferencesKey("selected_model")
    val remote = stringPreferencesKey("selected_remote")
    val hfToken = stringPreferencesKey("hf_token")
    val hitRate = floatPreferencesKey("last_hit_rate")
    val thinkBudget = intPreferencesKey("think_budget")
    val maxAnswer = intPreferencesKey("max_answer")
    val ioThreads = intPreferencesKey("io_threads")
    val helpers = intPreferencesKey("helper_threads")
    val prefetch = booleanPreferencesKey("prefetch")
    val gpu = booleanPreferencesKey("gpu")
    val gpuShare = floatPreferencesKey("gpu_share")
    val routePrior = booleanPreferencesKey("route_prior")
    val npu = booleanPreferencesKey("npu")
    val runMode = intPreferencesKey("run_mode")
    val units = stringPreferencesKey("units")
    val unitsVersion = stringPreferencesKey("units_version")
    val wordExperts = intPreferencesKey("word_experts")
    val stateCache = intPreferencesKey("state_cache_mb")
    val imageDetail = intPreferencesKey("image_detail")
    val serverOn = booleanPreferencesKey("server_on")
    val serverPort = intPreferencesKey("server_port")
    val serverKeyOn = booleanPreferencesKey("server_key_on")
    val serverKey = stringPreferencesKey("server_key")
    val webOn = booleanPreferencesKey("web_on")
    val webPassword = stringPreferencesKey("web_password")
}

/**
 * The settings' last good values outside DataStore, for when its file is found corrupt: a phone restarted while the
 * settings were being written left settings.preferences_pb as 173 zero bytes once, and DataStore threw at every
 * start - the app died before its first frame. Two generations of a small text file (a sequence number first, a
 * CRC32 last), each written whole, synced and renamed into place; the newer one that checks out is read back.
 */
internal class SettingsBackup(private val dir: File) {
    private fun file(i: Int) = File(dir, "settings_backup_$i.txt")

    private fun load(i: Int): Pair<Long, Preferences>? = runCatching {
        val text = file(i).takeIf { it.isFile }?.readText() ?: return null
        val cut = text.lastIndexOf(CRC_TAG)
        if (cut < 0) return null
        val body = text.substring(0, cut)
        val crc = CRC32().apply { update(body.toByteArray()) }.value
        if (text.substring(cut + CRC_TAG.length).trim().toLong(16) != crc) return null
        var seq = -1L
        val p = mutablePreferencesOf()
        for (line in body.lines()) {
            val f = line.split('\t')
            if (f.size == 2 && f[0] == SEQ_TAG) seq = f[1].toLong()
            if (f.size != 3) continue
            val (type, name, v) = f
            when (type) {
                "i" -> p[intPreferencesKey(name)] = v.toInt()
                "l" -> p[longPreferencesKey(name)] = v.toLong()
                "f" -> p[floatPreferencesKey(name)] = v.toFloat()
                "d" -> p[doublePreferencesKey(name)] = v.toDouble()
                "b" -> p[booleanPreferencesKey(name)] = v.toBoolean()
                "s" -> p[stringPreferencesKey(name)] = String(Base64.decode(v, Base64.NO_WRAP))
            }
        }
        if (seq < 0) null else seq to p.toPreferences()
    }.getOrNull()

    /** The newest backup that checks out; null when there is none. */
    fun read(): Preferences? = listOfNotNull(load(0), load(1)).maxByOrNull { it.first }?.second

    @Synchronized
    fun write(p: Preferences) {
        val seq = maxOf(load(0)?.first ?: -1L, load(1)?.first ?: -1L) + 1
        val body = StringBuilder("$SEQ_TAG\t$seq\n")
        for ((k, v) in p.asMap()) {
            val (type, text) = when (v) {
                is Int -> "i" to v.toString()
                is Long -> "l" to v.toString()
                is Float -> "f" to v.toString()
                is Double -> "d" to v.toString()
                is Boolean -> "b" to v.toString()
                is String -> "s" to Base64.encodeToString(v.toByteArray(), Base64.NO_WRAP)
                else -> continue
            }
            body.append(type).append('\t').append(k.name).append('\t').append(text).append('\n')
        }
        val crc = CRC32().apply { update(body.toString().toByteArray()) }.value
        body.append(CRC_TAG).append(java.lang.Long.toHexString(crc)).append('\n')
        // the older generation is written over; the newer one stays whole whatever happens to this write
        val target = file((seq % 2).toInt())
        val tmp = File(dir, target.name + ".tmp")
        FileOutputStream(tmp).use {
            it.write(body.toString().toByteArray())
            it.fd.sync()
        }
        tmp.renameTo(target)
    }

    companion object {
        private const val SEQ_TAG = "#seq"
        private const val CRC_TAG = "#crc\t"
    }
}

/**
 * The settings file corrupt and no backup either (the first such start after an update): what can be worked out
 * again - the memory test's result from its history (the probe's file, the test's own arithmetic) and the model when
 * exactly one is installed. The rest comes back as the defaults.
 */
private fun rebuilt(context: Context): Preferences {
    val p = mutablePreferencesOf()
    runCatching {
        val f = ProbeService.statusFile(context)
        if (f.isFile) {
            val steps = MemoryTest.parseSteps(f.readLines())
            val clean = MemoryTest.lastCleanStep(steps)?.held ?: 0L
            val safe = (clean - MemoryTest.SAFETY_MARGIN).coerceAtLeast(0L)
            if (safe > 0 && steps.isNotEmpty()) {
                p[K.recommended] = (safe shr 20).toInt()
                p[K.limit] = (steps.last().held shr 20).toInt()
            }
        }
    }
    runCatching { ModelStore(context).installed().singleOrNull()?.let { p[K.model] = it.id } }
    return p.toPreferences()
}

private object Stores {
    @Volatile
    private var settings: DataStore<Preferences>? = null

    /** One DataStore per file and process (the UI process's; GeodeApp). */
    fun settings(context: Context): DataStore<Preferences> = settings ?: synchronized(this) {
        settings ?: PreferenceDataStoreFactory.create(
            corruptionHandler = ReplaceFileCorruptionHandler {
                SettingsBackup(context.filesDir).read() ?: rebuilt(context)
            },
            produceFile = { context.preferencesDataStoreFile("settings") },
        ).also { settings = it }
    }
}

class SettingsRepository(context: Context) {
    private val app = context.applicationContext
    private val store = Stores.settings(app)
    private val backup = SettingsBackup(app.filesDir)

    val flow: Flow<AppSettings> = store.data.map { p ->
        val d = AppSettings()
        AppSettings(
            memoryBudgetMb = p[K.budget] ?: d.memoryBudgetMb,
            recommendedMb = p[K.recommended] ?: d.recommendedMb,
            limitMb = p[K.limit] ?: d.limitMb,
            thinking = p[K.thinking] ?: d.thinking,
            qwenSampling = K.qwenSampling.read(p),
            qwenSmallSampling = K.qwenSmallSampling.read(p),
            gemmaSampling = K.gemmaSampling.read(p),
            featuresOff = p[K.featuresOff] ?: d.featuresOff,
            // every chat is the agent's, whose prompt and tools take ~7k tokens: never less than MIN_CONTEXT (a
            // shorter setting from before is raised)
            contextLength = (p[K.context] ?: d.contextLength).coerceAtLeast(AppSettings.MIN_CONTEXT),
            computeThreads = p[K.threads] ?: d.computeThreads,
            theme = p[K.theme] ?: d.theme,
            dynamicColor = p[K.dynamicColor] ?: d.dynamicColor,
            chatTextScale = p[K.chatTextScale] ?: d.chatTextScale,
            keepScreen = p[K.keepScreen] ?: d.keepScreen,
            blackScreenMin = p[K.blackScreenMin] ?: d.blackScreenMin,
            readAloud = p[K.readAloud] ?: d.readAloud,
            speechRate = p[K.speechRate] ?: d.speechRate,
            speechPitch = p[K.speechPitch] ?: d.speechPitch,
            speechVoice = p[K.speechVoice] ?: d.speechVoice,
            speechEngine = p[K.speechEngine] ?: d.speechEngine,
            neuralVoice = p[K.neuralVoice] ?: d.neuralVoice,
            selectedModel = p[K.model] ?: d.selectedModel,
            selectedRemote = p[K.remote] ?: d.selectedRemote,
            hfToken = p[K.hfToken] ?: d.hfToken,
            lastHitRate = p[K.hitRate] ?: d.lastHitRate,
            thinkBudget = p[K.thinkBudget] ?: d.thinkBudget,
            maxAnswer = p[K.maxAnswer] ?: d.maxAnswer,
            ioThreads = p[K.ioThreads] ?: d.ioThreads,
            helperThreads = p[K.helpers] ?: d.helperThreads,
            prefetch = p[K.prefetch] ?: d.prefetch,
            gpu = p[K.gpu] ?: d.gpu,
            gpuShare = p[K.gpuShare] ?: d.gpuShare,
            routePrior = p[K.routePrior] ?: d.routePrior,
            npu = p[K.npu] ?: d.npu,
            runMode = p[K.runMode] ?: RunModes.fromSwitches(p[K.gpu] ?: d.gpu, p[K.npu] ?: d.npu),
            unitsJson = p[K.units] ?: d.unitsJson,
            unitsVersion = p[K.unitsVersion] ?: d.unitsVersion,
            wordExperts = p[K.wordExperts] ?: d.wordExperts,
            stateCacheMb = p[K.stateCache] ?: d.stateCacheMb,
            imageDetail = p[K.imageDetail] ?: d.imageDetail,
            serverOn = p[K.serverOn] ?: d.serverOn,
            serverPort = p[K.serverPort] ?: d.serverPort,
            serverKeyOn = p[K.serverKeyOn] ?: d.serverKeyOn,
            serverKey = p[K.serverKey] ?: d.serverKey,
            webOn = p[K.webOn] ?: d.webOn,
            webPassword = p[K.webPassword] ?: d.webPassword,
        )
    }

    /** A change, then the backup of what the settings are now (but for the hit rate, rewritten after every answer). */
    private suspend fun edit(keep: Boolean = true, block: suspend (MutablePreferences) -> Unit) {
        store.edit(block)
        // one backup at a time, each of the settings as they are when it is written: two quick changes (a pinch's
        // scale) cannot leave the older one in the backup last
        if (keep) withContext(Dispatchers.IO) { backupLock.withLock { runCatching { backup.write(store.data.first()) } } }
    }

    private val backupLock = kotlinx.coroutines.sync.Mutex()

    /** At start: a backup of the settings as they are, when there is none yet (an install from before backups). */
    suspend fun ensureBackup() = withContext(Dispatchers.IO) {
        if (backup.read() == null) runCatching { backup.write(store.data.first()) }
    }

    suspend fun setMemoryBudget(mb: Int) = edit { it[K.budget] = mb }
    suspend fun setMemoryTest(recommendedMb: Int, limitMb: Int) = edit {
        it[K.recommended] = recommendedMb
        it[K.limit] = limitMb
    }
    suspend fun setThinking(on: Boolean) = edit { it[K.thinking] = on }
    /** A change of one family's sampling, made on the values stored (two quick changes never undo each other). */
    suspend fun updateSampling(f: ModelFamily, change: (SamplingPrefs) -> SamplingPrefs) = edit {
        val k = keysOf(f)
        k.write(it, change(k.read(it)))
    }

    private fun keysOf(f: ModelFamily) = when (f) {
        ModelFamily.QWEN -> K.qwenSampling
        ModelFamily.QWEN_SMALL -> K.qwenSmallSampling
        ModelFamily.GEMMA -> K.gemmaSampling
    }

    /** One of the models' inputs or their drafting (an OFF_* bit) on or off. */
    suspend fun setFeature(bit: Int, on: Boolean) = edit {
        val v = it[K.featuresOff] ?: 0
        it[K.featuresOff] = if (on) v and bit.inv() else v or bit
    }
    suspend fun setContextLength(v: Int) = edit { it[K.context] = v }
    suspend fun setComputeThreads(v: Int) = edit { it[K.threads] = v }
    suspend fun setTheme(v: Int) = edit { it[K.theme] = v }
    suspend fun setDynamicColor(v: Boolean) = edit { it[K.dynamicColor] = v }
    suspend fun setChatTextScale(v: Float) = edit { it[K.chatTextScale] = v }
    suspend fun setKeepScreen(v: Int) = edit { it[K.keepScreen] = v }
    suspend fun setBlackScreenMin(v: Int) = edit { it[K.blackScreenMin] = v }
    suspend fun setReadAloud(v: Boolean) = edit { it[K.readAloud] = v }
    suspend fun setSpeechRate(v: Float) = edit { it[K.speechRate] = v }
    suspend fun setSpeechPitch(v: Float) = edit { it[K.speechPitch] = v }
    suspend fun setSpeechVoice(v: String) = edit { it[K.speechVoice] = v }
    suspend fun setSpeechEngine(v: Int) = edit { it[K.speechEngine] = v }
    suspend fun setNeuralVoice(v: Int) = edit { it[K.neuralVoice] = v }
    suspend fun setSelectedModel(id: String) = edit { it[K.model] = id }
    /** A service's model for the chats ("": the phone's own again). */
    suspend fun setSelectedRemote(id: String) = edit { it[K.remote] = id }
    suspend fun setHfToken(v: String) = edit { it[K.hfToken] = v }
    suspend fun setLastHitRate(v: Float) = edit(keep = false) { it[K.hitRate] = v }
    suspend fun setThinkBudget(v: Int) = edit { it[K.thinkBudget] = v }
    suspend fun setMaxAnswer(v: Int) = edit { it[K.maxAnswer] = v }
    suspend fun setIoThreads(v: Int) = edit { it[K.ioThreads] = v }
    suspend fun setHelperThreads(v: Int) = edit { it[K.helpers] = v }
    suspend fun setPrefetch(v: Boolean) = edit { it[K.prefetch] = v }
    suspend fun setGpu(v: Boolean) = edit { it[K.gpu] = v }
    suspend fun setGpuShare(v: Float) = edit { it[K.gpuShare] = v }
    suspend fun setRoutePrior(v: Boolean) = edit { it[K.routePrior] = v }
    suspend fun setNpu(v: Boolean) = edit { it[K.npu] = v }
    suspend fun setRunMode(v: Int) = edit { it[K.runMode] = v }
    suspend fun setUnits(json: String, version: String) = edit {
        it[K.units] = json
        it[K.unitsVersion] = version
    }
    suspend fun setWordExperts(v: Int) = edit { it[K.wordExperts] = v }
    suspend fun setStateCacheMb(v: Int) = edit { it[K.stateCache] = v }
    suspend fun setImageDetail(v: Int) = edit { it[K.imageDetail] = v }
    suspend fun setServerOn(v: Boolean) = edit { it[K.serverOn] = v }
    suspend fun setServerPort(v: Int) = edit { it[K.serverPort] = v }
    suspend fun setServerKeyOn(v: Boolean) = edit { it[K.serverKeyOn] = v }
    suspend fun setServerKey(v: String) = edit { it[K.serverKey] = v }
    suspend fun setWebOn(v: Boolean) = edit { it[K.webOn] = v }
    suspend fun setWebPassword(v: String) = edit { it[K.webPassword] = v }

    /** The family's recommended sampling for both modes (its model card). */
    suspend fun resetSampling(f: ModelFamily) = updateSampling(f) { f.defaults }
}
