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
import kotlinx.coroutines.withContext
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.probe.MemoryTest
import xyz.tifu.geode.probe.ProbeService

data class AppSettings(
    /** Memory the engine may hold in all; 0 = not set (use the recommendation or a default). */
    val memoryBudgetMb: Int = 0,
    /** From the memory test: the last level before the phone pushed memory into swap, and where it killed. */
    val recommendedMb: Int = 0,
    val limitMb: Int = 0,
    val thinking: Boolean = false,
    // Qwen3.6's recommended sampling without reasoning
    val temperature: Float = 0.7f,
    val topP: Float = 0.8f,
    val topK: Int = 20,
    // ... and with it (the model card: thinking mode, general tasks)
    val thinkTemperature: Float = 1.0f,
    val thinkTopP: Float = 0.95f,
    val thinkTopK: Int = 20,
    /** Both modes: keeps a long answer or a long reasoning from going round in circles. */
    val presencePenalty: Float = 1.5f,
    /** Reasoning tokens before the model is told to answer (Qwen's way: a closing sentence and </think>); 0 = no
     *  limit. */
    val thinkBudget: Int = 0,
    /** The longest answer in tokens (reasoning included); 0 = until the context is full. */
    val maxAnswer: Int = 0,
    /** The longest context; the engine allocates it chunk by chunk, as the chat grows. */
    val contextLength: Int = 32768,
    /** At 90% of the context the model summarizes the conversation and goes on from the summary. */
    val autoCompress: Boolean = true,
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
    /**
     * Experts read for each word: 8 = every expert the router picks (the default, the model's full quality); 6 / 4 =
     * the last 2 / 4 of them are left out when not in memory (EngineOptions.tailSkip) - faster, less accurate.
     */
    val wordExperts: Int = 8,
    /** 0 = system, 1 = light, 2 = dark. */
    val theme: Int = 0,
    /** Material You: the colours from the wallpaper (Android 12+); off: Geode's violet. */
    val dynamicColor: Boolean = true,
    val selectedModel: String = "",
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

    /** The sampling of the mode the chat is in. */
    fun sampling(maxTokens: Int) = xyz.tifu.geode.engine.Sampling(
        temperature = if (thinking) thinkTemperature else temperature,
        topP = if (thinking) thinkTopP else topP,
        topK = if (thinking) thinkTopK else topK,
        maxTokens = maxTokens,
        presencePenalty = presencePenalty,
    )

    /** What the engine opens the model with. */
    val engineOptions: xyz.tifu.geode.engine.EngineOptions
        get() = xyz.tifu.geode.engine.EngineOptions(
            budgetBytes = effectiveBudgetMb.toLong() shl 20, threads = computeThreads, ioThreads = ioThreads,
            helpers = helperThreads, contextLength = contextLength, prefetch = prefetch, gpu = gpu, gpuShare = gpuShare,
            // upstream Strata 0.2.7's lambda: 2.0 showed copy and code defects there
            routePrior = if (routePrior) 1.5f else 0f,
            tailSkip = if (wordExperts in 1..7) wordExperts else 0,
            npu = npu,
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
    }
}

private object K {
    val budget = intPreferencesKey("memory_budget_mb")
    val recommended = intPreferencesKey("recommended_mb")
    val limit = intPreferencesKey("limit_mb")
    val thinking = booleanPreferencesKey("thinking")
    val temperature = floatPreferencesKey("temperature")
    val topP = floatPreferencesKey("top_p")
    val topK = intPreferencesKey("top_k")
    val context = intPreferencesKey("context_length")
    val threads = intPreferencesKey("compute_threads")
    val theme = intPreferencesKey("theme")
    val dynamicColor = booleanPreferencesKey("dynamic_color")
    val model = stringPreferencesKey("selected_model")
    val hfToken = stringPreferencesKey("hf_token")
    val hitRate = floatPreferencesKey("last_hit_rate")
    val thinkTemperature = floatPreferencesKey("think_temperature")
    val thinkTopP = floatPreferencesKey("think_top_p")
    val thinkTopK = intPreferencesKey("think_top_k")
    val presence = floatPreferencesKey("presence_penalty")
    val thinkBudget = intPreferencesKey("think_budget")
    val maxAnswer = intPreferencesKey("max_answer")
    val ioThreads = intPreferencesKey("io_threads")
    val helpers = intPreferencesKey("helper_threads")
    val prefetch = booleanPreferencesKey("prefetch")
    val gpu = booleanPreferencesKey("gpu")
    val gpuShare = floatPreferencesKey("gpu_share")
    val routePrior = booleanPreferencesKey("route_prior")
    val npu = booleanPreferencesKey("npu")
    val wordExperts = intPreferencesKey("word_experts")
    val autoCompress = booleanPreferencesKey("auto_compress")
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
            temperature = p[K.temperature] ?: d.temperature,
            topP = p[K.topP] ?: d.topP,
            topK = p[K.topK] ?: d.topK,
            // every chat is the agent's, whose prompt and tools take ~7k tokens: never less than MIN_CONTEXT (a
            // shorter setting from before is raised)
            contextLength = (p[K.context] ?: d.contextLength).coerceAtLeast(AppSettings.MIN_CONTEXT),
            computeThreads = p[K.threads] ?: d.computeThreads,
            theme = p[K.theme] ?: d.theme,
            dynamicColor = p[K.dynamicColor] ?: d.dynamicColor,
            selectedModel = p[K.model] ?: d.selectedModel,
            hfToken = p[K.hfToken] ?: d.hfToken,
            lastHitRate = p[K.hitRate] ?: d.lastHitRate,
            thinkTemperature = p[K.thinkTemperature] ?: d.thinkTemperature,
            thinkTopP = p[K.thinkTopP] ?: d.thinkTopP,
            thinkTopK = p[K.thinkTopK] ?: d.thinkTopK,
            presencePenalty = p[K.presence] ?: d.presencePenalty,
            thinkBudget = p[K.thinkBudget] ?: d.thinkBudget,
            maxAnswer = p[K.maxAnswer] ?: d.maxAnswer,
            ioThreads = p[K.ioThreads] ?: d.ioThreads,
            helperThreads = p[K.helpers] ?: d.helperThreads,
            prefetch = p[K.prefetch] ?: d.prefetch,
            gpu = p[K.gpu] ?: d.gpu,
            gpuShare = p[K.gpuShare] ?: d.gpuShare,
            routePrior = p[K.routePrior] ?: d.routePrior,
            npu = p[K.npu] ?: d.npu,
            wordExperts = p[K.wordExperts] ?: d.wordExperts,
            autoCompress = p[K.autoCompress] ?: d.autoCompress,
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
        val p = store.edit(block)
        if (keep) withContext(Dispatchers.IO) { runCatching { backup.write(p) } }
    }

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
    suspend fun setTemperature(v: Float) = edit { it[K.temperature] = v }
    suspend fun setTopP(v: Float) = edit { it[K.topP] = v }
    suspend fun setTopK(v: Int) = edit { it[K.topK] = v }
    suspend fun setContextLength(v: Int) = edit { it[K.context] = v }
    suspend fun setComputeThreads(v: Int) = edit { it[K.threads] = v }
    suspend fun setTheme(v: Int) = edit { it[K.theme] = v }
    suspend fun setDynamicColor(v: Boolean) = edit { it[K.dynamicColor] = v }
    suspend fun setSelectedModel(id: String) = edit { it[K.model] = id }
    suspend fun setHfToken(v: String) = edit { it[K.hfToken] = v }
    suspend fun setLastHitRate(v: Float) = edit(keep = false) { it[K.hitRate] = v }
    suspend fun setThinkTemperature(v: Float) = edit { it[K.thinkTemperature] = v }
    suspend fun setThinkTopP(v: Float) = edit { it[K.thinkTopP] = v }
    suspend fun setThinkTopK(v: Int) = edit { it[K.thinkTopK] = v }
    suspend fun setPresencePenalty(v: Float) = edit { it[K.presence] = v }
    suspend fun setThinkBudget(v: Int) = edit { it[K.thinkBudget] = v }
    suspend fun setMaxAnswer(v: Int) = edit { it[K.maxAnswer] = v }
    suspend fun setIoThreads(v: Int) = edit { it[K.ioThreads] = v }
    suspend fun setHelperThreads(v: Int) = edit { it[K.helpers] = v }
    suspend fun setPrefetch(v: Boolean) = edit { it[K.prefetch] = v }
    suspend fun setGpu(v: Boolean) = edit { it[K.gpu] = v }
    suspend fun setGpuShare(v: Float) = edit { it[K.gpuShare] = v }
    suspend fun setRoutePrior(v: Boolean) = edit { it[K.routePrior] = v }
    suspend fun setNpu(v: Boolean) = edit { it[K.npu] = v }
    suspend fun setWordExperts(v: Int) = edit { it[K.wordExperts] = v }
    suspend fun setAutoCompress(v: Boolean) = edit { it[K.autoCompress] = v }
    suspend fun setStateCacheMb(v: Int) = edit { it[K.stateCache] = v }
    suspend fun setImageDetail(v: Int) = edit { it[K.imageDetail] = v }
    suspend fun setServerOn(v: Boolean) = edit { it[K.serverOn] = v }
    suspend fun setServerPort(v: Int) = edit { it[K.serverPort] = v }
    suspend fun setServerKeyOn(v: Boolean) = edit { it[K.serverKeyOn] = v }
    suspend fun setServerKey(v: String) = edit { it[K.serverKey] = v }
    suspend fun setWebOn(v: Boolean) = edit { it[K.webOn] = v }
    suspend fun setWebPassword(v: String) = edit { it[K.webPassword] = v }

    /** Qwen3.6's recommended sampling for both modes (the model card). */
    suspend fun resetSampling() = edit {
        val d = AppSettings()
        it[K.temperature] = d.temperature
        it[K.topP] = d.topP
        it[K.topK] = d.topK
        it[K.thinkTemperature] = d.thinkTemperature
        it[K.thinkTopP] = d.thinkTopP
        it[K.thinkTopK] = d.thinkTopK
        it[K.presence] = d.presencePenalty
    }
}
