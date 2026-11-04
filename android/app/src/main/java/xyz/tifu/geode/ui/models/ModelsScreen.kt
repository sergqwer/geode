package xyz.tifu.geode.ui.models

import androidx.compose.material.icons.outlined.Menu
import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.FolderOpen
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.GraphicEq
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Movie
import androidx.compose.material.icons.outlined.Psychology
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Surface
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import xyz.tifu.geode.R
import xyz.tifu.geode.models.Catalog
import xyz.tifu.geode.ui.runModeText
import xyz.tifu.geode.engine.EngineRun
import xyz.tifu.geode.models.CatalogPack
import xyz.tifu.geode.models.HfPack
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.PartialModel
import xyz.tifu.geode.models.TransferPhase
import xyz.tifu.geode.models.TransferState
import xyz.tifu.geode.util.formatBytes

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(vm: ModelsViewModel, openDrawer: () -> Unit) {
    val st by vm.state.collectAsStateWithLifecycle()
    val transfer by vm.transfer.collectAsStateWithLifecycle()
    val busy = transfer?.active == true
    val context = LocalContext.current

    // the transfer's progress notification; the transfer itself runs either way
    val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    fun askNotifications() {
        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(context,
                Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
    var move by rememberSaveable { mutableStateOf(false) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) vm.import(uri, move)
    }

    // the list narrowed: words of the name or the description, and what the model must take in
    var query by rememberSaveable { mutableStateOf("") }
    var want by rememberSaveable { mutableIntStateOf(0) }
    fun shows(name: String, desc: String, arch: String, c: Caps): Boolean {
        if (want and NEED_PHOTOS != 0 && !c.photos) return false
        if (want and NEED_VIDEO != 0 && !c.video) return false
        if (want and NEED_SOUND != 0 && !c.sound) return false
        return query.split(' ').filter { it.isNotBlank() }.all { w ->
            name.contains(w, ignoreCase = true) || desc.contains(w, ignoreCase = true) || arch.contains(w, ignoreCase = true)
        }
    }
    val narrowed = query.isNotBlank() || want != 0
    val installed = st.installed.filter { shows(it.title, it.manifest.description, it.manifest.arch, Caps.of(it)) }
    // the catalog's packs this phone has are under Installed already
    val catalog = st.catalog?.filter { cp ->
        st.installed.none { it.manifest.name == cp.manifest.name && it.manifest.bytes == cp.manifest.bytes } &&
            shows(cp.manifest.name.ifEmpty { cp.pack.label }, cp.manifest.description, cp.manifest.arch, Caps.of(cp.manifest))
    }

    val providers by vm.providers.collectAsStateWithLifecycle()
    var editing by remember { mutableStateOf<xyz.tifu.geode.models.Provider?>(null) }
    var adding by remember { mutableStateOf(false) }
    if (adding || editing != null) {
        val e = editing
        ServiceDialog(e, hasKey = e != null && vm.hasKey(e), listModels = vm::listModels,
            onSave = { p, key, use ->
                vm.saveProvider(p, key, use)
                editing = null
                adding = false
            },
            onDelete = { p ->
                vm.deleteProvider(p)
                editing = null
            },
            onDismiss = {
                editing = null
                adding = false
            })
    }

    Column(Modifier.fillMaxSize().imePadding()) {
        TopAppBar(title = { Text(stringResource(R.string.nav_models)) }, navigationIcon = {
            IconButton(onClick = openDrawer) {
                Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
            }
        })
        SearchBar(query, { query = it }, want, { want = want xor it })
        LazyColumn(
            Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            transfer?.let { t -> item(key = "transfer") { TransferCard(t, vm::cancelTransfer, vm::dismissTransfer) } }

            // the phone's own models first: the one in use among them
            item(key = "installed-title") { Section(stringResource(R.string.models_installed)) }
            if (st.installed.isEmpty() && st.partials.isEmpty()) {
                item(key = "none") {
                    Text(stringResource(R.string.models_none), color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            } else if (narrowed && installed.isEmpty()) {
                item(key = "none-match") { NoMatch() }
            }
            items(installed, key = { "m-" + it.id }) { m ->
                InstalledCard(m, selected = m.id == st.selected && st.remote.isEmpty(), run = st.run?.takeIf { r -> r.dir == m.dir.path }, st = st,
                    onUse = { vm.select(m) }, onDelete = { vm.delete(m) }, onLoad = vm::load, onUnload = vm::unload)
            }
            if (!narrowed) items(st.partials, key = { "p-" + it.id }) { p ->
                PartialCard(p, canResume = !busy && p.source != null, onResume = {
                    askNotifications()
                    vm.resume(p)
                }, onDelete = { vm.deletePartial(p) })
            }

            // the models of outside services the user has keys for, beside the phone's own
            if (!narrowed) services(providers, st.remote, vm::useProvider, { editing = it; adding = false },
                { editing = null; adding = true })

            // Geode's own packs, one tap each (Catalog), with what this phone has for them
            // (no title when every pack of it is installed already)
            if (catalog == null || catalog.isNotEmpty() || narrowed) {
                item(key = "catalog-title") { Section(stringResource(R.string.models_catalog)) }
            }
            when {
                catalog != null -> {
                    if (narrowed && catalog.isEmpty()) item(key = "catalog-none-match") { NoMatch() }
                    items(catalog, key = { "c-" + it.pack.repo + "/" + it.pack.path }) { cp ->
                        CatalogCard(cp, st, installed = false, busy = busy, onDownload = {
                            askNotifications()
                            vm.download(cp.pack)
                        })
                    }
                }
                st.catalogLoading -> item(key = "catalog-loading") {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                        Text(stringResource(R.string.models_catalog_loading), Modifier.padding(start = 10.dp),
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
                else -> item(key = "catalog-error") {
                    Column {
                        Text(stringResource(R.string.models_catalog_error, st.catalogError.orEmpty()),
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                        TextButton(onClick = vm::loadCatalog) { Text(stringResource(R.string.models_catalog_retry)) }
                    }
                }
            }

            item(key = "download") {
                DownloadCard(st, busy, onToken = vm::setToken, onSearch = vm::search, onDownload = { pack ->
                    askNotifications()
                    vm.download(pack)
                })
            }
            item(key = "import") {
                ImportCard(busy, move, onMove = { move = it }, onPick = {
                    askNotifications()
                    picker.launch(null)
                })
            }
            item(key = "free") {
                Text(stringResource(R.string.models_free_space, formatBytes(st.freeBytes)),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

/** One of Geode's own packs: its name and description, and what this phone has for it - room, memory, GPU. */
@Composable
private fun CatalogCard(cp: CatalogPack, st: ModelsUiState, installed: Boolean, busy: Boolean, onDownload: () -> Unit) {
    val need = Catalog.spaceNeeded(cp)
    val room = st.freeBytes >= need
    val ramGb = st.ramBytes / 1e9
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
        Column(Modifier.padding(16.dp)) {
            Text(cp.manifest.name.ifEmpty { cp.pack.label }, style = MaterialTheme.typography.titleMedium)
            Capabilities(Caps.of(cp.manifest))
            if (cp.manifest.description.isNotEmpty()) {
                Text(cp.manifest.description, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 4.dp))
            }
            Spacer(Modifier.height(10.dp))
            if (!installed) {
                FitLine(room, if (room) stringResource(R.string.models_catalog_space, formatBytes(need))
                    else stringResource(R.string.models_catalog_space_low, formatBytes(need), formatBytes(st.freeBytes)))
                if (Catalog.wantsEightGb(cp)) {
                    // a dense model on the processor, but one a 6 GB phone has no room for (an 8 GB one reports ~7.5)
                    if (ramGb >= 7.0) FitLine(true, stringResource(R.string.models_catalog_mid))
                    else FitLine(false, stringResource(R.string.models_catalog_ram_short, formatBytes(st.ramBytes)))
                } else if (Catalog.light(cp)) {
                    // a dense small model: the processor (and the NPU for prompts), any phone's memory
                    FitLine(true, stringResource(R.string.models_catalog_light))
                } else {
                    FitLine(st.gpuPath, stringResource(if (st.gpuPath) R.string.models_catalog_gpu else R.string.models_catalog_cpu))
                    if (ramGb < 8.0) FitLine(false, stringResource(R.string.models_catalog_ram_low, formatBytes(st.ramBytes)))
                }
                Spacer(Modifier.height(10.dp))
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (installed) {
                    Icon(Icons.Outlined.CheckCircle, null, tint = MaterialTheme.colorScheme.primary)
                    Text(stringResource(R.string.models_catalog_installed), Modifier.padding(start = 8.dp),
                        style = MaterialTheme.typography.titleSmall)
                } else {
                    Button(onClick = onDownload, enabled = room && !busy) {
                        Icon(Icons.Outlined.Download, null, Modifier.size(18.dp))
                        Text(stringResource(R.string.models_catalog_download, formatBytes(cp.pack.bytes)),
                            Modifier.padding(start = 8.dp))
                    }
                }
            }
        }
    }
}

/** What a model takes in and does beyond text: photos, video, sound, reasoning. */
private data class Caps(val photos: Boolean, val video: Boolean, val sound: Boolean, val reasoning: Boolean) {
    companion object {
        /** A pack as its geode.json describes it (before it is downloaded). */
        fun of(m: xyz.tifu.geode.models.PackManifest) =
            Caps(m.vision, m.gemma && m.vision, m.audio, m.arch in xyz.tifu.geode.models.PackManifest.ARCHS)

        /** An installed model as its files are: an encoder not downloaded (an optional file) is not there. */
        fun of(m: InstalledModel) =
            Caps(m.vision, m.video, m.audio, m.manifest.arch in xyz.tifu.geode.models.PackManifest.ARCHS)
    }
}

private const val NEED_PHOTOS = 1
private const val NEED_VIDEO = 2
private const val NEED_SOUND = 4

/** The search over the models (their names and descriptions), and what they must take in. */
@Composable
private fun SearchBar(query: String, onQuery: (String) -> Unit, want: Int, onToggle: (Int) -> Unit) {
    val keyboard = LocalSoftwareKeyboardController.current
    Column(Modifier.padding(horizontal = 16.dp)) {
        OutlinedTextField(
            value = query,
            onValueChange = onQuery,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            placeholder = { Text(stringResource(R.string.models_search)) },
            leadingIcon = { Icon(Icons.Outlined.Search, contentDescription = null) },
            trailingIcon = {
                if (query.isNotEmpty()) IconButton(onClick = { onQuery("") }) {
                    Icon(Icons.Outlined.Close, contentDescription = stringResource(R.string.models_cancel))
                }
            },
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
            keyboardActions = KeyboardActions(onSearch = { keyboard?.hide() }),
        )
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(top = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for ((bit, icon, label) in listOf(
                Triple(NEED_PHOTOS, Icons.Outlined.Image, R.string.models_cap_photos),
                Triple(NEED_VIDEO, Icons.Outlined.Movie, R.string.models_cap_video),
                Triple(NEED_SOUND, Icons.Outlined.GraphicEq, R.string.models_cap_sound),
            )) {
                FilterChip(
                    selected = want and bit != 0,
                    onClick = { onToggle(bit) },
                    label = { Text(stringResource(label)) },
                    leadingIcon = { Icon(icon, contentDescription = null, Modifier.size(FilterChipDefaults.IconSize)) },
                )
            }
        }
    }
}

/** A model's capabilities as small labels under its name. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Capabilities(c: Caps) {
    val shown = listOfNotNull(
        (Icons.Outlined.Image to R.string.models_cap_photos).takeIf { c.photos },
        (Icons.Outlined.Movie to R.string.models_cap_video).takeIf { c.video },
        (Icons.Outlined.GraphicEq to R.string.models_cap_sound).takeIf { c.sound },
        (Icons.Outlined.Psychology to R.string.models_cap_reasoning).takeIf { c.reasoning },
    )
    if (shown.isEmpty()) return
    FlowRow(Modifier.padding(top = 6.dp, bottom = 2.dp), horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)) {
        for ((icon, label) in shown) {
            Surface(shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.surfaceVariant,
                contentColor = MaterialTheme.colorScheme.onSurfaceVariant) {
                Row(Modifier.padding(horizontal = 8.dp, vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(icon, contentDescription = null, Modifier.size(14.dp))
                    Text(stringResource(label), Modifier.padding(start = 4.dp), style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }
}

@Composable
private fun NoMatch() {
    Text(stringResource(R.string.models_no_match), style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
}

/** A requirement met (a check) or a warning */
@Composable
private fun FitLine(ok: Boolean, text: String) {
    Row(Modifier.padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(if (ok) Icons.Outlined.CheckCircle else Icons.Outlined.WarningAmber, null, Modifier.size(16.dp),
            tint = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
        Text(text, Modifier.padding(start = 8.dp), style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun Section(title: String) {
    Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 4.dp))
}

@Composable
private fun TransferCard(t: TransferState, onCancel: () -> Unit, onDismiss: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
        Column(Modifier.padding(16.dp)) {
            val title = when (t.phase) {
                TransferPhase.PREPARE, TransferPhase.DOWNLOAD -> stringResource(R.string.models_transfer_download, t.title)
                TransferPhase.COPY -> stringResource(R.string.models_transfer_import, t.title)
                TransferPhase.VERIFY -> stringResource(R.string.models_transfer_verify, t.title)
                TransferPhase.DONE -> stringResource(R.string.models_transfer_done, t.title)
                TransferPhase.FAILED -> stringResource(R.string.models_transfer_failed, t.error.orEmpty())
                TransferPhase.CANCELLED -> stringResource(R.string.models_transfer_cancelled, t.title)
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                when (t.phase) {
                    TransferPhase.DONE -> Icon(Icons.Outlined.CheckCircle, null, tint = MaterialTheme.colorScheme.primary)
                    TransferPhase.FAILED -> Icon(Icons.Outlined.ErrorOutline, null, tint = MaterialTheme.colorScheme.error)
                    else -> Unit
                }
                Text(title, Modifier.weight(1f).padding(start = if (t.active) 0.dp else 8.dp),
                    style = MaterialTheme.typography.titleSmall)
            }
            if (t.active) {
                Spacer(Modifier.height(10.dp))
                if (t.total > 0) LinearProgressIndicator(progress = { t.fraction }, modifier = Modifier.fillMaxWidth())
                else LinearProgressIndicator(Modifier.fillMaxWidth())
                Spacer(Modifier.height(6.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        if (t.total > 0) Text(stringResource(R.string.models_transfer_progress, formatBytes(t.done),
                            formatBytes(t.total)), style = MaterialTheme.typography.bodySmall)
                        if (t.bytesPerSecond > 0) {
                            val left = if (t.total > t.done) (t.total - t.done) / t.bytesPerSecond else 0
                            Text(stringResource(R.string.models_transfer_speed, formatBytes(t.bytesPerSecond),
                                formatDuration(left)), style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    TextButton(onClick = onCancel) { Text(stringResource(R.string.models_transfer_cancel)) }
                }
            } else {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = onDismiss) { Text(stringResource(R.string.models_ok)) }
                }
            }
        }
    }
}

@Composable
private fun InstalledCard(m: InstalledModel, selected: Boolean, run: EngineRun?, st: ModelsUiState, onUse: () -> Unit,
                          onDelete: () -> Unit, onLoad: () -> Unit, onUnload: () -> Unit) {
    var confirm by remember { mutableStateOf(false) }
    Card {
        Column(Modifier.padding(16.dp)) {
            Text(m.title, style = MaterialTheme.typography.titleMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Capabilities(Caps.of(m))
            if (m.manifest.description.isNotEmpty()) {
                Text(m.manifest.description, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text(formatBytes(m.manifest.bytes), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            // in memory or not, and how it runs (the selected model: the one the chat and the server use)
            if (selected) {
                Spacer(Modifier.height(6.dp))
                if (run != null) {
                    Text(stringResource(R.string.models_in_memory, formatBytes(run.memoryBytes)),
                        style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
                    Text(runModeText(run), style = MaterialTheme.typography.bodySmall)
                } else {
                    Text(stringResource(R.string.models_not_in_memory), style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (st.unloading) Text(stringResource(R.string.models_unloading), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                st.loadError?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
            }
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (selected) {
                    FilledTonalButton(onClick = {}, enabled = false) { Text(stringResource(R.string.models_in_use)) }
                    Spacer(Modifier.width(8.dp))
                    if (run != null) {
                        OutlinedButton(onClick = onUnload, enabled = !st.unloading) { Text(stringResource(R.string.models_unload)) }
                    } else if (st.loading) {
                        CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                    } else {
                        OutlinedButton(onClick = onLoad, enabled = !st.unloading) { Text(stringResource(R.string.models_load)) }
                    }
                } else {
                    Button(onClick = onUse) { Text(stringResource(R.string.models_use)) }
                }
                Spacer(Modifier.weight(1f))
                IconButton(onClick = { confirm = true }) {
                    Icon(Icons.Outlined.Delete, contentDescription = stringResource(R.string.models_delete))
                }
            }
        }
    }
    if (confirm) {
        AlertDialog(
            onDismissRequest = { confirm = false },
            text = { Text(stringResource(R.string.models_delete_confirm, m.title, formatBytes(m.manifest.bytes))) },
            confirmButton = {
                TextButton(onClick = {
                    confirm = false
                    onDelete()
                }) { Text(stringResource(R.string.models_delete)) }
            },
            dismissButton = { TextButton(onClick = { confirm = false }) { Text(stringResource(R.string.models_cancel)) } },
        )
    }
}

@Composable
private fun PartialCard(p: PartialModel, canResume: Boolean, onResume: () -> Unit, onDelete: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHighest)) {
        Column(Modifier.padding(16.dp)) {
            Text(stringResource(R.string.models_partial, p.id), style = MaterialTheme.typography.titleSmall)
            Text(formatBytes(p.bytes), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onResume, enabled = canResume) { Text(stringResource(R.string.models_resume)) }
                Spacer(Modifier.weight(1f))
                IconButton(onClick = onDelete) {
                    Icon(Icons.Outlined.Delete, contentDescription = stringResource(R.string.models_delete))
                }
            }
        }
    }
}

@Composable
private fun DownloadCard(
    st: ModelsUiState,
    busy: Boolean,
    onToken: (String) -> Unit,
    onSearch: (String) -> Unit,
    onDownload: (HfPack) -> Unit,
) {
    var repo by rememberSaveable { mutableStateOf("") }
    var showToken by rememberSaveable { mutableStateOf(st.token.isNotEmpty()) }
    Card {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.Download, null)
                Text(stringResource(R.string.models_download_title), Modifier.padding(start = 8.dp),
                    style = MaterialTheme.typography.titleMedium)
            }
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = repo,
                onValueChange = { repo = it },
                label = { Text(stringResource(R.string.models_repo_hint)) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Search),
                keyboardActions = KeyboardActions(onSearch = { if (repo.isNotBlank()) onSearch(repo) }),
                trailingIcon = {
                    if (st.searching) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                    else IconButton(onClick = { onSearch(repo) }, enabled = repo.isNotBlank()) {
                        Icon(Icons.Outlined.Search, contentDescription = stringResource(R.string.models_find))
                    }
                },
                modifier = Modifier.fillMaxWidth(),
            )
            if (showToken) {
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = st.token,
                    onValueChange = onToken,
                    label = { Text(stringResource(R.string.models_token_hint)) },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                )
            } else {
                TextButton(onClick = { showToken = true }) { Text(stringResource(R.string.models_token_add)) }
            }
            st.searchError?.let {
                Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 8.dp))
            }
            st.packs?.let { packs ->
                Spacer(Modifier.height(8.dp))
                if (packs.isEmpty()) {
                    Text(stringResource(R.string.models_found_none), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error)
                }
                packs.forEach { p ->
                    val fits = st.freeBytes > p.bytes
                    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(p.label, style = MaterialTheme.typography.bodyLarge)
                            Text(
                                if (fits) formatBytes(p.bytes)
                                else stringResource(R.string.models_too_big, formatBytes(p.bytes), formatBytes(st.freeBytes)),
                                style = MaterialTheme.typography.bodySmall,
                                color = if (fits) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error,
                            )
                        }
                        Spacer(Modifier.width(8.dp))
                        Button(onClick = { onDownload(p) }, enabled = !busy && fits) {
                            Text(stringResource(R.string.models_download))
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ImportCard(busy: Boolean, move: Boolean, onMove: (Boolean) -> Unit, onPick: () -> Unit) {
    Card {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.FolderOpen, null)
                Text(stringResource(R.string.models_import_title), Modifier.padding(start = 8.dp),
                    style = MaterialTheme.typography.titleMedium)
            }
            Text(stringResource(R.string.models_import_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
            Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.models_move), style = MaterialTheme.typography.bodyMedium)
                    Text(stringResource(R.string.models_move_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = move, onCheckedChange = onMove)
            }
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = onPick, enabled = !busy) { Text(stringResource(R.string.models_import)) }
        }
    }
}

@Composable
private fun formatDuration(seconds: Long): String = when {
    seconds >= 3600 -> stringResource(R.string.duration_hm, seconds / 3600, seconds % 3600 / 60)
    seconds >= 60 -> stringResource(R.string.duration_m, seconds / 60)
    else -> stringResource(R.string.duration_s, seconds)
}
