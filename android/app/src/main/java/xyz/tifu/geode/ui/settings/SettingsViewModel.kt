package xyz.tifu.geode.ui.settings

import android.app.ActivityManager
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.bench.ModelShape
import xyz.tifu.geode.bench.SpeedModel
import xyz.tifu.geode.data.AppSettings
import xyz.tifu.geode.data.RunModes
import xyz.tifu.geode.data.SamplingPrefs
import xyz.tifu.geode.engine.GeodeNative
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.probe.MemoryTest

data class MemoryTestUi(val running: Boolean = false, val heldBytes: Long = 0, val error: String? = null)

/**
 * How a memory budget splits (PLAN.md, P6): the dense weights, the expert cache in a new chat, and what the context
 * takes from the cache as it grows (ContextPlan).
 */
data class MemorySplit(val weights: Long, val cacheStart: Long, val chunkBytes: Long, val maxContext: Int,
                       val cacheAtMax: Long,
                       /** The KV cache at the longest context the budget allows, and a thousand tokens of it. */
                       val kvAtMax: Long, val kvPerThousand: Long)

class SettingsViewModel(private val app: GeodeApp) : ViewModel() {
    private val store = ModelStore(app)

    /** The app's lock (its section here, its screen in place of the app). */
    val lock = app.lock

    val settings: StateFlow<AppSettings> = app.settings.flow.stateIn(viewModelScope, SharingStarted.Eagerly,
        AppSettings())

    /** The selected model's shape (its dense weights stay in memory, its experts go through the cache), or null. */
    val shape: StateFlow<ModelShape?> = app.settings.flow.map { it.selectedModel }.distinctUntilChanged().map { id ->
        withContext(Dispatchers.IO) { id.takeIf { it.isNotEmpty() }?.let(store::find)?.let { ModelShape.read(it.dir) } }
    }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** The selected model as installed (its name, what its pack can, what of that the user left on), or null: none
     *  picked or gone. */
    val model: StateFlow<xyz.tifu.geode.models.InstalledModel?> = app.settings.flow
        .map { it.selectedModel to it.featuresOff }
        .distinctUntilChanged().map { (id, off) ->
            withContext(Dispatchers.IO) { id.takeIf { it.isNotEmpty() }?.let(store::find)?.copy(off = off) }
        }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** The GPU path as the engine last reported it (null before a model opened). */
    val gpuStatus: StateFlow<String?> = app.engine.gpuStatus

    private val _test = MutableStateFlow(MemoryTestUi())
    val test: StateFlow<MemoryTestUi> = _test.asStateFlow()

    val totalMemBytes: Long = app.getSystemService(ActivityManager::class.java).let { am ->
        ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem
    }

    val appVersion: String = runCatching {
        app.packageManager.getPackageInfo(app.packageName, 0).versionName
    }.getOrNull() ?: "?"

    val engineVersion: String = runCatching { GeodeNative.version() }.getOrElse { "-" }

    fun split(budgetMb: Int, contextLength: Int, shape: ModelShape): MemorySplit {
        val plan = SpeedModel.plan(shape, budgetMb.toLong() shl 20, contextLength)
        return MemorySplit(shape.denseFileBytes, plan.startSlots * plan.slotBytes, plan.chunkBytes, plan.maxContext,
            plan.slotsAt(plan.maxContext) * plan.slotBytes,
            // whole chunks: the engine allocates the context chunk by chunk
            (plan.maxContext.toLong() + plan.chunkTokens - 1) / plan.chunkTokens * plan.chunkBytes,
            shape.kvBytesPerToken * 1000)
    }

    fun runMemoryTest() {
        if (_test.value.running) return
        _test.value = MemoryTestUi(running = true)
        viewModelScope.launch {
            // the engine's own memory would be counted against the probe: let it go first
            app.gate.close(app.engine)
            try {
                val r = MemoryTest(app).run { held -> _test.update { it.copy(heldBytes = held) } }
                app.settings.setMemoryTest((r.safeBytes shr 20).toInt(), (r.limitBytes shr 20).toInt())
                _test.value = MemoryTestUi()
            } catch (e: Exception) {
                _test.value = MemoryTestUi(error = e.message ?: e.javaClass.simpleName)
            }
        }
    }

    fun setMemoryBudget(mb: Int) = launch {
        app.settings.setMemoryBudget(mb)
        // the engine takes its budget when it opens the model: reopen it with the new one
        app.gate.close(app.engine)
    }
    fun setThinking(on: Boolean) = launch { app.settings.setThinking(on) }

    private val _cacheBytes = kotlinx.coroutines.flow.MutableStateFlow(0L)
    /** What the saved conversations take on the storage now. */
    val cacheBytes: kotlinx.coroutines.flow.StateFlow<Long> = _cacheBytes

    fun refreshCache() = launch {
        _cacheBytes.value = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { app.stateCache.bytes() }
    }

    fun setStateCacheMb(mb: Int) = launch {
        app.settings.setStateCacheMb(mb)
        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            if (mb <= 0) app.stateCache.clear() else app.stateCache.trim(mb.toLong() shl 20)
        }
        refreshCache()
    }

    /** Every saved conversation gone (each is read again next time). */
    fun clearCache() = launch {
        app.gate.use(this) { kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { app.stateCache.clear() } }
        refreshCache()
    }
    fun setImageDetail(v: Int) = launch { app.settings.setImageDetail(v) }
    /** The sampling of the selected model's family (Qwen3.6's when none is selected): each keeps its own. */
    private val family get() = xyz.tifu.geode.data.ModelFamily.of(model.value)
    private fun sampling(change: (SamplingPrefs) -> SamplingPrefs) {
        val f = family
        launch { app.settings.updateSampling(f, change) }
    }
    fun setTemperature(v: Float) = sampling { it.copy(temperature = v) }
    fun setTopP(v: Float) = sampling { it.copy(topP = v) }
    fun setTopK(v: Int) = sampling { it.copy(topK = v) }
    fun setContextLength(v: Int) = launch {
        app.settings.setContextLength(v)
        // the engine sets the context's limit when it opens the model: reopen it with the new one
        app.gate.close(app.engine)
    }
    fun setComputeThreads(v: Int) = launch {
        app.settings.setComputeThreads(v)
        app.gate.close(app.engine)
    }
    fun setThinkTemperature(v: Float) = sampling { it.copy(thinkTemperature = v) }
    fun setThinkTopP(v: Float) = sampling { it.copy(thinkTopP = v) }
    fun setThinkTopK(v: Int) = sampling { it.copy(thinkTopK = v) }
    fun setPresencePenalty(v: Float) = sampling { it.copy(presencePenalty = v) }
    fun setThinkPresencePenalty(v: Float) = sampling { it.copy(thinkPresencePenalty = v) }
    /** One of the model's inputs or its drafting (models/ModelStore.kt OFF_*) on or off; the engine opens the model
     *  again when what it loads changes (an encoder, the drafting). */
    fun setFeature(bit: Int, on: Boolean) = launch {
        val m = model.value
        val before = m?.engineOff
        app.settings.setFeature(bit, on)
        val after = m?.copy(off = if (on) m.off and bit.inv() else m.off or bit)?.engineOff
        if (before != after) app.gate.close(app.engine)
    }
    fun setThinkBudget(v: Int) = launch { app.settings.setThinkBudget(v) }
    fun setMaxAnswer(v: Int) = launch { app.settings.setMaxAnswer(v) }
    fun resetSampling() {
        val f = family
        launch { app.settings.resetSampling(f) }
    }
    // the engine's options: each opens the model again with the new value (the next question does)
    fun setIoThreads(v: Int) = launch {
        app.settings.setIoThreads(v)
        app.gate.close(app.engine)
    }
    fun setHelperThreads(v: Int) = launch {
        app.settings.setHelperThreads(v)
        app.gate.close(app.engine)
    }
    fun setPrefetch(v: Boolean) = launch {
        app.settings.setPrefetch(v)
        app.gate.close(app.engine)
    }
    fun setGpu(v: Boolean) = launch {
        app.settings.setGpu(v)
        app.gate.close(app.engine)
    }
    /** Cache-conditioned routing: the engine reopens with it */
    fun setRoutePrior(v: Boolean) = launch {
        app.settings.setRoutePrior(v)
        app.gate.close(app.engine)
    }
    /** The NPU for prompts: the engine reopens with it */
    fun setNpu(v: Boolean) = launch {
        app.settings.setNpu(v)
        app.gate.close(app.engine)
    }
    /** Which units run the model: the engine reopens with it */
    fun setRunMode(v: Int) = launch {
        app.settings.setRunMode(v)
        app.gate.close(app.engine)
    }

    /** The GPU and the NPU the phone has (the engine's "units" probe), null until it ran: the run modes offered. */
    val units: StateFlow<RunModes.Units?> = app.settings.flow.map { RunModes.parseUnits(it.unitsJson) }
        .distinctUntilChanged().stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** The probe once an app version, while no model is open (a benchmark closes the model; an answer or the local
     *  server would be cut off). Half a second: OpenCL's device and a session on the DSP opened and closed. */
    private fun probeUnits() = viewModelScope.launch {
        val s = app.settings.flow.first()
        if (s.unitsJson.isNotEmpty() && s.unitsVersion == appVersion) return@launch
        if (app.engine.openModel.value != null || app.engine.running.value) return@launch
        val json = runCatching { app.gate.use(this@SettingsViewModel) { app.engine.bench("units", "") } }.getOrNull()
        if (json != null && RunModes.parseUnits(json) != null) app.settings.setUnits(json, appVersion)
    }

    init {
        probeUnits()
    }
    fun setWordExperts(v: Int) = launch {
        app.settings.setWordExperts(v)
        app.gate.close(app.engine)
    }
    fun setGpuShare(v: Float) = launch {
        app.settings.setGpuShare(v)
        app.gate.close(app.engine)
    }
    fun setTheme(v: Int) = launch { app.settings.setTheme(v) }
    fun setDynamicColor(on: Boolean) = launch { app.settings.setDynamicColor(on) }
    fun setChatTextScale(v: Float) = launch { app.settings.setChatTextScale(v) }
    fun setKeepScreen(v: Int) = launch { app.settings.setKeepScreen(v) }
    fun setBlackScreenMin(v: Int) = launch { app.settings.setBlackScreenMin(v) }
    fun setReadAloud(v: Boolean) = launch { app.settings.setReadAloud(v) }
    fun setSpeechRate(v: Float) = launch {
        app.settings.setSpeechRate(v)
        app.speaker.setRate(v)
    }
    fun setSpeechPitch(v: Float) = launch {
        app.settings.setSpeechPitch(v)
        app.speaker.setPitch(v)
    }
    fun setSpeechVoice(name: String) = launch {
        app.settings.setSpeechVoice(name)
        app.speaker.setVoice(name)
    }
    /** Geode's own voice ([setSpeechEngine] 1) or the phone's (0). */
    fun setSpeechEngine(v: Int) = launch {
        app.settings.setSpeechEngine(v)
        app.speaker.setEngine(v == 1)
    }
    fun setNeuralVoice(v: Int) = launch {
        app.settings.setNeuralVoice(v)
        app.speaker.setNeuralVoice(v)
    }
    /** Geode's voice: on the storage or not, its download. */
    val neuralInstalled = app.speaker.neural.installed
    val neuralDownload = app.speaker.neural.download
    fun downloadNeural() = app.speaker.neural.startDownload(app.appScope)
    fun cancelNeural() = app.speaker.neural.cancelDownload()
    fun deleteNeural() = app.speaker.neural.delete()
    /** Its licence's text (on the storage with it). */
    suspend fun neuralLicence(): String? = withContext(Dispatchers.IO) { app.speaker.neural.licence() }

    /** The engine's voices (it is started for them when the choice opens). */
    val voices = app.speaker.voices
    fun prepareVoices() = app.speaker.prepare()
    /** An example read with the voice, rate and pitch set. */
    fun sampleVoice(text: String) = app.speaker.sample(text)
    /** Whether the phone's text-to-speech started (null: not used yet). */
    val speakerAvailable = app.speaker.available

    /** "" = the system's language. */
    fun language(): String = AppCompatDelegate.getApplicationLocales().toLanguageTags()

    fun setLanguage(tag: String) {
        AppCompatDelegate.setApplicationLocales(
            if (tag.isEmpty()) LocaleListCompat.getEmptyLocaleList() else LocaleListCompat.forLanguageTags(tag))
    }

    private fun launch(block: suspend () -> Unit) {
        viewModelScope.launch { block() }
    }

    companion object {
        // from the agent's least (its prompt and tools take ~7k tokens): every chat is the agent's
        val CONTEXT_CHOICES = listOf(16384, 32768, 65536, 131072, 262144)
        /** Reasoning budgets and answer lengths in tokens; 0 = no limit. */
        val THINK_BUDGETS = listOf(0, 256, 512, 1024, 2048, 4096, 8192)
        val ANSWER_LIMITS = listOf(0, 256, 512, 1024, 2048, 4096, 8192)
    }
}
