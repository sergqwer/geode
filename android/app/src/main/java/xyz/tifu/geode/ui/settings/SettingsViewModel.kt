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
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.bench.ModelShape
import xyz.tifu.geode.bench.SpeedModel
import xyz.tifu.geode.data.AppSettings
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

    val settings: StateFlow<AppSettings> = app.settings.flow.stateIn(viewModelScope, SharingStarted.Eagerly,
        AppSettings())

    /** The selected model's shape (its dense weights stay in memory, its experts go through the cache), or null. */
    val shape: StateFlow<ModelShape?> = app.settings.flow.map { it.selectedModel }.distinctUntilChanged().map { id ->
        withContext(Dispatchers.IO) { id.takeIf { it.isNotEmpty() }?.let(store::find)?.let { ModelShape.read(it.dir) } }
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
    fun setAutoCompress(on: Boolean) = launch { app.settings.setAutoCompress(on) }

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
    fun setTemperature(v: Float) = launch { app.settings.setTemperature(v) }
    fun setTopP(v: Float) = launch { app.settings.setTopP(v) }
    fun setTopK(v: Int) = launch { app.settings.setTopK(v) }
    fun setContextLength(v: Int) = launch {
        app.settings.setContextLength(v)
        // the engine sets the context's limit when it opens the model: reopen it with the new one
        app.gate.close(app.engine)
    }
    fun setComputeThreads(v: Int) = launch {
        app.settings.setComputeThreads(v)
        app.gate.close(app.engine)
    }
    fun setThinkTemperature(v: Float) = launch { app.settings.setThinkTemperature(v) }
    fun setThinkTopP(v: Float) = launch { app.settings.setThinkTopP(v) }
    fun setThinkTopK(v: Int) = launch { app.settings.setThinkTopK(v) }
    fun setPresencePenalty(v: Float) = launch { app.settings.setPresencePenalty(v) }
    fun setThinkBudget(v: Int) = launch { app.settings.setThinkBudget(v) }
    fun setMaxAnswer(v: Int) = launch { app.settings.setMaxAnswer(v) }
    fun resetSampling() = launch { app.settings.resetSampling() }
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
