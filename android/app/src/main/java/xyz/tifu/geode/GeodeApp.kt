package xyz.tifu.geode

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import xyz.tifu.geode.bench.BenchStore
import xyz.tifu.geode.data.Reports
import xyz.tifu.geode.data.SettingsRepository
import xyz.tifu.geode.engine.EngineClient
import xyz.tifu.geode.engine.EngineGate
import xyz.tifu.geode.server.ApiEngine
import xyz.tifu.geode.tokenizer.Tokenizer
import java.io.File

class GeodeApp : Application() {
    lateinit var settings: SettingsRepository
        private set
    lateinit var engine: EngineClient
        private set
    /** The chat and the local server take turns with the engine through it. */
    lateinit var gate: EngineGate
        private set

    /** The API's way to the engine, one for the app: the local server, the web chat and the agent share the
     *  conversation the engine holds. */
    val apiEngine: ApiEngine by lazy { ApiEngine(this) }

    /** The agent (DeepSeek Harness in the app's Linux). */
    val agent: xyz.tifu.geode.agent.Agent by lazy { xyz.tifu.geode.agent.Agent(this) }

    /** What the user gives every chat's agent: MCP servers and skills of their own. */
    val agentConfig: xyz.tifu.geode.agent.AgentConfig by lazy { xyz.tifu.geode.agent.AgentConfig(this) }

    /** The phone's folders the user gives every chat's agent (/phone in its Linux). */
    val phoneFolders: xyz.tifu.geode.agent.PhoneFolders by lazy { xyz.tifu.geode.agent.PhoneFolders(this) }

    /** Work that outlives a screen (a conversation's save after a quiet while). */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    /** The engine's conversations kept on the storage (KV cache and states), to go on without reading them again. */
    val stateCache: xyz.tifu.geode.cache.StateCache by lazy { xyz.tifu.geode.cache.StateCache(this) }

    /** The app's lock: a PIN or a password (or a fingerprint) before the chats show. */
    val lock: xyz.tifu.geode.data.AppLock by lazy { xyz.tifu.geode.data.AppLock(this) }
    /** Answers read aloud (the phone's text-to-speech, started at its first use). */
    val speaker: xyz.tifu.geode.media.Speaker by lazy { xyz.tifu.geode.media.Speaker(this) }

    /** The outside services the user added (their models answer the chats instead of the phone's, when chosen). */
    val providers: xyz.tifu.geode.models.Providers by lazy { xyz.tifu.geode.models.Providers(this) }

    /** The service's model the chats use now; null: the phone's own (none chosen, or the chosen one deleted). */
    suspend fun selectedProvider(): xyz.tifu.geode.models.Provider? =
        settings.flow.first().selectedRemote.takeIf { it.isNotEmpty() }?.let { providers.find(it) }

    /**
     * The chosen service's model with its key, for a request; null: the phone's own model answers. A service chosen
     * whose key cannot be read is an ApiError - never the phone's model answering in its place unasked.
     */
    suspend fun remoteModel(): Pair<xyz.tifu.geode.models.Provider, String>? {
        val p = selectedProvider() ?: return null
        val key = kotlinx.coroutines.withContext(Dispatchers.IO) { providers.key(p.id) }
            ?: throw xyz.tifu.geode.server.ApiError(401, "${p.name}: its API key is missing - add it again on the Models screen")
        return p to key
    }

    /** The web search (geode-search, run for each search). */
    val webSearch: xyz.tifu.geode.search.WebSearch by lazy { xyz.tifu.geode.search.WebSearch(this) }

    /**
     * What the engine opens the model with: the settings' (the whole budget: every chat is the agent's, which itself
     * takes a few MB; its Linux borrows [AGENT_RESERVE] from the model only while it runs something - Agent.kt; when
     * the model cannot lend it, the command runs without the loan).
     */
    fun engineOptions(s: xyz.tifu.geode.data.AppSettings, m: xyz.tifu.geode.models.InstalledModel?):
        xyz.tifu.geode.engine.EngineOptions = s.engineOptions(m)

    /** The benchmarks' last results: read when the Benchmarks screen first wants them, not at every start. */
    val bench: BenchStore by lazy { BenchStore(this) }
    lateinit var reports: Reports
        private set

    override fun onCreate() {
        super.onCreate()
        // The :engine and :probe processes run this too; only the UI process needs the rest.
        if (!isMainProcess()) return
        // the lock's file read from now on, on a thread of SharedPreferences' own: MainActivity's onCreate asks the lock
        // before the first frame and would otherwise wait there for the read (AppLock.kt)
        xyz.tifu.geode.data.AppLock.preload(this)
        // debug builds with files/debug-strict (adb: run-as ... touch files/debug-strict): the main thread's disk and
        // network use said in the log (StrictMode), each with its stack - off by default, it slows what it watches
        if (BuildConfig.DEBUG && File(filesDir, "debug-strict").exists()) android.os.StrictMode.setThreadPolicy(
            android.os.StrictMode.ThreadPolicy.Builder().detectDiskReads().detectDiskWrites().detectNetwork()
                .penaltyLog().build())
        settings = SettingsRepository(this)
        // a backup of the settings from the start (Settings.kt, SettingsBackup): a corrupt settings file comes back
        // from it instead of killing the app at every start
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch { settings.ensureBackup() }
        engine = EngineClient(this)
        gate = EngineGate()
        reports = Reports(this)
        // answer reports that could not go before (offline)
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            // (a full storage makes its rewrite throw: not a reason for the app to stop at every start)
            runCatching { reports.flush() }
            runCatching { xyz.tifu.geode.bench.DeviceReport.flush(this@GeodeApp) }
        }
        // media no chat holds (attached, never sent) and the local server's copies not used for a day - the agent's
        // requests make those too, also when the network server is never on
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            runCatching { xyz.tifu.geode.data.ChatStore(this@GeodeApp).dropUnheldMedia() }
            runCatching { apiEngine.tidy() }
            runCatching { xyz.tifu.geode.engine.PromptFiles.tidy(cacheDir) }
        }
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_TRANSFERS, getString(R.string.notif_channel_transfers),
                NotificationManager.IMPORTANCE_LOW)
        )
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ANSWERS, getString(R.string.notif_channel_engine),
                NotificationManager.IMPORTANCE_LOW)
        )
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_SERVER, getString(R.string.notif_channel_server),
                NotificationManager.IMPORTANCE_LOW)
        )
    }

    private var tokenizerDir: String? = null
    private var tokenizerCached: Tokenizer? = null

    /**
     * The model out of memory now (the Models screen's Unload, the chat's details): an answer running - the chat's, the
     * web chat's, the local server's - is stopped first, not waited for; nothing in the background opens it again.
     */
    suspend fun unloadNow() {
        agent.cancelTurn()
        engine.stop()
        gate.unload(engine)
    }

    /** The model's tokenizer, one kept (the chat and the local server share it: ~10 MB); off the main thread. */
    @Synchronized
    fun tokenizer(modelDir: File): Tokenizer {
        tokenizerCached?.takeIf { tokenizerDir == modelDir.path }?.let { return it }
        return Tokenizer.load(File(modelDir, Tokenizer.FILE)).also {
            tokenizerCached = it
            tokenizerDir = modelDir.path
        }
    }

    private fun isMainProcess(): Boolean = getProcessName() == packageName

    companion object {
        const val CHANNEL_TRANSFERS = "transfers"
        const val CHANNEL_ANSWERS = "answers"
        const val CHANNEL_SERVER = "server"
        /** The agent's RAM beside the model's: its Linux, the harness (Node.js, its heap capped), its commands. */
        /** What the model gives back while the agent's Linux runs a command or an MCP server (proot, bash, python). */
        const val AGENT_RESERVE = 256L shl 20
    }
}
