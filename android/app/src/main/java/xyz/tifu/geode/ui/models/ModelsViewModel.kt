package xyz.tifu.geode.ui.models

import android.content.Intent
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.engine.EngineRun
import xyz.tifu.geode.models.Catalog
import xyz.tifu.geode.models.CatalogPack
import xyz.tifu.geode.models.Device
import xyz.tifu.geode.models.HfApi
import xyz.tifu.geode.models.HfPack
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.models.PartialModel
import xyz.tifu.geode.models.TransferService
import xyz.tifu.geode.models.TransferSource
import xyz.tifu.geode.models.Transfers

data class ModelsUiState(
    val installed: List<InstalledModel> = emptyList(),
    val partials: List<PartialModel> = emptyList(),
    val selected: String = "",
    /** The service whose model the chats use ("": the phone's, [selected]). */
    val remote: String = "",
    val freeBytes: Long = 0,
    val token: String = "",
    val searching: Boolean = false,
    /** The packs found in the repository last searched (null: no search yet). */
    val packs: List<HfPack>? = null,
    val searchError: String? = null,
    /** Geode's own packs (Catalog), null while they load or after an error. */
    val catalog: List<CatalogPack>? = null,
    val catalogLoading: Boolean = false,
    val catalogError: String? = null,
    val ramBytes: Long = 0,
    /** The engine's GPU path runs on this chip (Device). */
    val gpuPath: Boolean = false,
    /** The model in memory and how it runs (null: none). */
    val run: EngineRun? = null,
    /** Unloading (it waits for an answer running to end) or loading now. */
    val unloading: Boolean = false,
    val loading: Boolean = false,
    /** Why the last Load failed. */
    val loadError: String? = null,
)

class ModelsViewModel(private val app: GeodeApp) : ViewModel() {
    private val store = ModelStore(app)
    private val _state = MutableStateFlow(ModelsUiState())
    val state: StateFlow<ModelsUiState> = _state.asStateFlow()
    val transfer = Transfers.state

    init {
        refresh()
        _state.update { it.copy(ramBytes = Device.ramBytes(app), gpuPath = Device.gpuPath()) }
        loadCatalog()
        viewModelScope.launch {
            app.settings.flow.collect { s ->
                _state.update { it.copy(selected = s.selectedModel, remote = s.selectedRemote, token = s.hfToken) }
            }
        }
        viewModelScope.launch { app.engine.run.collect { r -> _state.update { it.copy(run = r) } } }
        viewModelScope.launch {
            // a finished transfer adds a model (or leaves a partial one)
            Transfers.state.collect { t -> if (t != null && !t.active) refresh() }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            val (installed, partials, free) = withContext(Dispatchers.IO) {
                Triple(store.installed(), store.partials(), store.freeBytes())
            }
            // a partial that belongs to the transfer running now is shown as the transfer, not as a leftover
            val running = Transfers.state.value?.takeIf { it.active }?.id
            _state.update { it.copy(installed = installed, partials = partials.filter { p -> p.id != running },
                freeBytes = free) }
        }
    }

    /** Geode's own packs, read from Hugging Face (again after an error). */
    fun loadCatalog() {
        if (_state.value.catalogLoading) return
        _state.update { it.copy(catalogLoading = true, catalogError = null) }
        viewModelScope.launch {
            val token = app.settings.flow.first().hfToken
            val r = withContext(Dispatchers.IO) { runCatching { Catalog.load(token) } }
            _state.update {
                it.copy(catalogLoading = false, catalog = r.getOrNull() ?: it.catalog,
                    catalogError = r.exceptionOrNull()?.let { e -> e.message ?: e.javaClass.simpleName })
            }
        }
    }

    fun setToken(token: String) {
        _state.update { it.copy(token = token) }
        viewModelScope.launch { app.settings.setHfToken(token.trim()) }
    }

    fun search(input: String) {
        val (repo, revision) = parseRepo(input) ?: run {
            _state.update { it.copy(packs = null, searchError = app.getString(R.string.models_bad_repo)) }
            return
        }
        _state.update { it.copy(searching = true, packs = null, searchError = null) }
        viewModelScope.launch {
            val token = app.settings.flow.first().hfToken
            val r = withContext(Dispatchers.IO) { runCatching { HfApi.findPacks(repo, revision, token) } }
            _state.update {
                it.copy(searching = false, packs = r.getOrNull(), searchError = r.exceptionOrNull()?.let { e ->
                    e.message ?: e.javaClass.simpleName
                })
            }
        }
    }

    fun download(pack: HfPack) {
        TransferService.start(app, TransferSource.Hf(pack.repo, pack.revision, pack.path),
            if (pack.path.isEmpty()) pack.repo else "${pack.repo} · ${pack.path}")
        _state.update { it.copy(packs = null) }
    }

    fun import(uri: Uri, move: Boolean) {
        val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or (if (move) Intent.FLAG_GRANT_WRITE_URI_PERMISSION else 0)
        // persisted, so the copy can resume after the app is killed (TransferService releases it at the end)
        runCatching { app.contentResolver.takePersistableUriPermission(uri, flags) }
        val name = DocumentFile.fromTreeUri(app, uri)?.name ?: uri.lastPathSegment.orEmpty()
        TransferService.start(app, TransferSource.Folder(uri.toString(), move), name)
    }

    fun resume(p: PartialModel) {
        val src = p.source ?: return
        val title = when (src) {
            is TransferSource.Hf -> if (src.packPath.isEmpty()) src.repo else "${src.repo} · ${src.packPath}"
            is TransferSource.Folder -> p.id
        }
        TransferService.start(app, src, title)
        _state.update { it.copy(partials = it.partials - p) }
    }

    fun cancelTransfer() = TransferService.cancel(app)

    fun dismissTransfer() = Transfers.dismiss()

    fun select(m: InstalledModel) {
        viewModelScope.launch {
            app.settings.setSelectedModel(m.id)
            // the phone's own model answers again
            app.settings.setSelectedRemote("")
        }
    }

    // ---- outside services' models

    val providers: StateFlow<List<xyz.tifu.geode.models.Provider>> = app.providers.list

    fun useProvider(p: xyz.tifu.geode.models.Provider) {
        viewModelScope.launch { app.settings.setSelectedRemote(p.id) }
    }

    fun hasKey(p: xyz.tifu.geode.models.Provider): Boolean = app.providers.hasKey(p.id)

    /** [p] saved with [key] (null: the kept one stays); [use]: its model answers the chats from now. */
    fun saveProvider(p: xyz.tifu.geode.models.Provider, key: String?, use: Boolean) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { app.providers.save(p, key) }
            if (use) app.settings.setSelectedRemote(p.id)
        }
    }

    fun deleteProvider(p: xyz.tifu.geode.models.Provider) {
        viewModelScope.launch {
            if (_state.value.remote == p.id) app.settings.setSelectedRemote("")
            withContext(Dispatchers.IO) { app.providers.delete(p.id) }
        }
    }

    suspend fun listModels(p: xyz.tifu.geode.models.Provider, key: String?): Result<List<String>> =
        runCatching { app.providers.models(p, key) }

    /** The model out of memory now (an answer running is stopped, not waited for); nothing in the background loads it
     *  again. */
    fun unload() {
        if (_state.value.unloading) return
        _state.update { it.copy(unloading = true, loadError = null) }
        viewModelScope.launch {
            try {
                app.unloadNow()
            } finally {
                _state.update { it.copy(unloading = false) }
            }
        }
    }

    /** The selected model into memory now. */
    fun load() {
        if (_state.value.loading) return
        _state.update { it.copy(loading = true, loadError = null) }
        viewModelScope.launch {
            val r = runCatching { app.apiEngine.load() }
            _state.update { it.copy(loading = false, loadError = r.exceptionOrNull()?.let { e -> e.message ?: e.javaClass.simpleName }) }
        }
    }

    fun delete(m: InstalledModel) {
        viewModelScope.launch {
            // after a generation that reads its files, if one runs
            if (app.engine.openModel.value == m.dir.path) app.gate.close(app.engine)
            if (_state.value.selected == m.id) app.settings.setSelectedModel("")
            withContext(Dispatchers.IO) { store.delete(m.id) }
            refresh()
        }
    }

    fun deletePartial(p: PartialModel) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { store.deletePartial(p.id) }
            refresh()
        }
    }

    val transferRunning: Boolean get() = Transfers.state.value?.active == true

    companion object {
        /**
         * "owner/name", "owner/name@revision", or a huggingface.co link (…/owner/name, …/tree/revision/…).
         * Returns the repository and the revision.
         */
        fun parseRepo(input: String): Pair<String, String>? {
            var s = input.trim().removeSuffix("/")
            s = s.removePrefix("https://").removePrefix("http://").removePrefix("www.")
                .removePrefix("huggingface.co/").removePrefix("hf.co/")
            var revision = "main"
            val parts = s.split('/')
            if (parts.size >= 4 && parts[2] == "tree") revision = parts[3]
            var repo = parts.take(2).joinToString("/")
            if ('@' in repo) {
                revision = repo.substringAfter('@')
                repo = repo.substringBefore('@')
            }
            val ok = Regex("[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*").matches(repo)
            return if (ok && revision.isNotEmpty()) repo to revision else null
        }
    }
}
