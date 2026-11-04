package xyz.tifu.geode.ui.chat

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.interaction.DragInteraction
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.outlined.VolumeUp
import androidx.compose.material.icons.outlined.StopCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.material.icons.outlined.AddPhotoAlternate
import androidx.compose.material.icons.outlined.AttachFile
import androidx.compose.material.icons.outlined.AudioFile
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.automirrored.outlined.VolumeOff
import androidx.compose.material.icons.automirrored.outlined.VolumeUp
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.VideoCall
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.AddComment
import androidx.compose.material.icons.outlined.Compress
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.FolderOpen
import androidx.compose.material.icons.outlined.ForkRight
import androidx.compose.material.icons.outlined.Forum
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.outlined.Summarize
import androidx.compose.material.icons.outlined.ExpandLess
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.Flag
import androidx.compose.material.icons.outlined.Psychology
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Button
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Checkbox
import androidx.compose.material3.RadioButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.TextButton
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import kotlin.math.roundToInt
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.unit.Density
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.draw.alpha
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import xyz.tifu.geode.R
import xyz.tifu.geode.data.ChatInfo
import xyz.tifu.geode.data.Compactions
import xyz.tifu.geode.data.Reports
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel, openModels: () -> Unit, openDrawer: () -> Unit) {
    val st by vm.state.collectAsStateWithLifecycle()
    val chats by vm.chats.collectAsStateWithLifecycle()
    val speaking by vm.speaking.collectAsStateWithLifecycle()
    val commands by vm.agentCommands.collectAsStateWithLifecycle()
    val engineRun by vm.engineRun.collectAsStateWithLifecycle()
    val unloading by vm.unloading.collectAsStateWithLifecycle()
    var showHistory by rememberSaveable { mutableStateOf(false) }
    var showInfo by rememberSaveable { mutableStateOf(false) }
    var showFiles by rememberSaveable { mutableStateOf(false) }
    var openFile by rememberSaveable { mutableStateOf<String?>(null) }
    openFile?.let { path ->
        vm.agentFolder()?.let { AgentFileView(java.io.File(it, path), onDismiss = { openFile = null }) }
    }
    if (showFiles) vm.agentFolder()?.let { AgentFiles(it, onDismiss = { showFiles = false }) }
    var reporting by remember { mutableStateOf<ChatMessage?>(null) }
    val context = LocalContext.current
    LaunchedEffect(Unit) { vm.refreshModel() }
    // the system's photo picker: no storage permission, the user picks one image
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(vm::attach)
    }
    // a video (the photo picker's videos)
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(vm::attachVideo)
    }
    // a sound file (the system's document picker: no storage permission either)
    val soundPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let(vm::attachAudio)
    }
    // the microphone's permission, asked on the first press of the record button (that press records nothing)
    val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        Toast.makeText(context, if (granted) R.string.chat_record_hold else R.string.chat_mic_denied, Toast.LENGTH_SHORT).show()
    }
    LaunchedEffect(Unit) { vm.notices.collect { Toast.makeText(context, it, Toast.LENGTH_SHORT).show() } }
    // the phone's camera app, without leaving the chat: a photo or a video into the app's cache (the FileProvider's
    // camera/), attached as a picked one would be and deleted once read. The manifest asks for the camera (the web
    // chat's sign-in codes), so the camera app's intent needs that permission granted first.
    val hasCamera = remember { context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY) }
    var shot by rememberSaveable { mutableStateOf<String?>(null) }
    var shotVideo by rememberSaveable { mutableStateOf(false) }
    val takePhoto = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        shot?.let { vm.attachShot(java.io.File(it), video = false, taken = ok) }
        shot = null
    }
    val recordVideo = rememberLauncherForActivityResult(CaptureVideoFor(VIDEO_SECONDS)) { ok ->
        shot?.let { vm.attachShot(java.io.File(it), video = true, taken = ok) }
        shot = null
    }
    fun openCamera(video: Boolean) {
        val f = runCatching {
            val dir = java.io.File(context.cacheDir, "camera").apply { mkdirs() }
            java.io.File(dir, (if (video) "video_" else "photo_") + System.currentTimeMillis() + if (video) ".mp4" else ".jpg")
        }.getOrNull() ?: return
        val uri = androidx.core.content.FileProvider.getUriForFile(context, context.packageName + ".files", f)
        shot = f.path
        try {
            if (video) recordVideo.launch(uri) else takePhoto.launch(uri)
        } catch (e: ActivityNotFoundException) {
            shot = null
            f.delete()
            Toast.makeText(context, R.string.chat_camera_none, Toast.LENGTH_SHORT).show()
        }
    }
    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) openCamera(shotVideo)
        else Toast.makeText(context, R.string.chat_camera_denied, Toast.LENGTH_LONG).show()
    }
    val camera: ((Boolean) -> Unit)? = if (!hasCamera) null else { video ->
        shotVideo = video
        if (androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED) openCamera(video)
        else cameraPermission.launch(android.Manifest.permission.CAMERA)
    }

    if (showHistory) {
        History(chats, st.chatId, enabled = !st.generating,
            onOpen = {
                vm.openChat(it)
                showHistory = false
            },
            onNew = {
                vm.newChat()
                showHistory = false
            },
            onDelete = vm::deleteChat,
            onDismiss = { showHistory = false })
    }
    if (showInfo) {
        ChatInfo(st, engineRun, unloading, onUnload = vm::unloadModel, onCompress = {
            showInfo = false
            vm.compress()
        }, onFork = {
            showInfo = false
            vm.forkChat()
        }, onDismiss = { showInfo = false })
    }
    reporting?.let { m ->
        ReportDialog(
            onDismiss = { reporting = null },
            onSend = { reason, note, include ->
                reporting = null
                vm.report(m, reason, note, include) { sent ->
                    Toast.makeText(context, if (sent) R.string.report_sent else R.string.report_queued,
                        Toast.LENGTH_SHORT).show()
                }
            },
        )
    }

    Column(Modifier.fillMaxSize().imePadding()) {
        TopAppBar(
            navigationIcon = {
                IconButton(onClick = openDrawer) {
                    Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
                }
            },
            title = {
                // the chat's short name; a tap shows its details (the model, the speeds, the context)
                Row(
                    Modifier.clip(RoundedCornerShape(10.dp)).clickable { showInfo = true }
                        .padding(horizontal = 6.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(st.title.ifEmpty { stringResource(R.string.chat_untitled) }, maxLines = 1,
                        overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.weight(1f, fill = false))
                    Icon(Icons.Outlined.Info, contentDescription = stringResource(R.string.chat_info),
                        modifier = Modifier.padding(start = 6.dp).size(18.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            },
            actions = {
                // every chat is the agent's: its folder (the files it made, the user's own put in)
                if (st.chatId != null && st.messages.isNotEmpty()) {
                    IconButton(onClick = { showFiles = true }) {
                        Icon(Icons.Outlined.FolderOpen, contentDescription = stringResource(R.string.agent_files))
                    }
                }
                // answers read aloud as they are written: on and off here as in the settings
                val reading by vm.readAloud.collectAsStateWithLifecycle()
                IconButton(onClick = {
                    vm.setReadAloud(!reading)
                    Toast.makeText(context, if (reading) R.string.chat_read_aloud_off else R.string.chat_read_aloud_on,
                        Toast.LENGTH_SHORT).show()
                }) {
                    Icon(if (reading) Icons.AutoMirrored.Outlined.VolumeUp else Icons.AutoMirrored.Outlined.VolumeOff,
                        contentDescription = stringResource(if (reading) R.string.chat_read_aloud_on else R.string.chat_read_aloud_off))
                }
                IconButton(onClick = vm::newChat, enabled = st.messages.isNotEmpty()) {
                    Icon(Icons.Outlined.AddComment, contentDescription = stringResource(R.string.chat_new))
                }
                IconButton(onClick = { showHistory = true }) {
                    Icon(Icons.Outlined.Forum, contentDescription = stringResource(R.string.chat_history))
                }
            },
        )
        if (st.loadingModel) LinearProgressIndicator(Modifier.fillMaxWidth())
        // the agent's Linux being unpacked (the first start, or an app update brought a new image)
        st.agentInstalling?.let { p ->
            LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth())
            Text(stringResource(R.string.agent_installing, (p * 100).toInt()), Modifier.padding(horizontal = 16.dp),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
        }

        // the chat's text at its own size (the setting; two fingers on the conversation change it): sp scaled, dp not
        val savedScale by vm.textScale.collectAsStateWithLifecycle()
        var pinch by remember { mutableStateOf<Float?>(null) }
        var pinching by remember { mutableStateOf(false) }
        // the pinch's value stays until the setting has it (no frame of the old size between)
        LaunchedEffect(savedScale) { pinch = null }
        val scale = pinch ?: savedScale
        val density = LocalDensity.current
        val chatDensity = remember(density, scale) { Density(density.density, density.fontScale * scale) }
        val current by rememberUpdatedState(scale)
        Box(Modifier.weight(1f).fillMaxWidth().pointerInput(Unit) {
            awaitEachGesture {
                awaitFirstDown(requireUnconsumed = false)
                var zoomed = false
                do {
                    // before the list sees it: two fingers zoom, one still scrolls
                    val event = awaitPointerEvent(PointerEventPass.Initial)
                    if (event.changes.count { it.pressed } >= 2) {
                        val z = event.calculateZoom()
                        if (z != 1f) {
                            zoomed = true
                            pinching = true
                            pinch = (current * z).coerceIn(xyz.tifu.geode.ui.settings.CHAT_TEXT_MIN,
                                xyz.tifu.geode.ui.settings.CHAT_TEXT_MAX)
                        }
                        event.changes.forEach { it.consume() }
                    }
                } while (event.changes.any { it.pressed })
                pinching = false
                if (zoomed) pinch?.let { vm.setTextScale((it * 20).roundToInt() / 20f) }
            }
        }) {
            CompositionLocalProvider(LocalDensity provides chatDensity) {
                when {
                    !st.modelChecked -> Unit
                    !st.ready -> NoModel(openModels)
                    // a service's model: said plainly that what is sent leaves the phone
                    st.messages.isEmpty() -> Hint(st.remote?.let { stringResource(R.string.chat_empty_remote, it.name, it.host) }
                        ?: stringResource(R.string.chat_empty))
                    else -> Messages(st, vm::thumbnail, vm::clipSeconds, vm::togglePlay, onReport = { reporting = it },
                        onOpenFile = { openFile = it }, speaking = speaking, onSpeak = vm::speak)
                }
            }
            if (pinching) {
                Surface(Modifier.align(Alignment.TopCenter).padding(top = 12.dp), shape = RoundedCornerShape(50),
                    color = MaterialTheme.colorScheme.inverseSurface, contentColor = MaterialTheme.colorScheme.inverseOnSurface) {
                    Text("${(scale * 100).roundToInt()}%", Modifier.padding(horizontal = 14.dp, vertical = 6.dp),
                        style = MaterialTheme.typography.labelLarge)
                }
            }
        }
        if (st.loadingModel) {
            Text(stringResource(R.string.chat_loading_model), Modifier.padding(horizontal = 16.dp),
                style = MaterialTheme.typography.bodySmall)
        }
        if (st.answeringElsewhere) {
            Text(stringResource(R.string.chat_answering_elsewhere), Modifier.padding(horizontal = 16.dp),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
        }
        if (st.waiting) {
            Text(stringResource(R.string.chat_waiting), Modifier.padding(horizontal = 16.dp),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
        }
        st.agentStatus?.let {
            Text(it, Modifier.padding(horizontal = 16.dp), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary)
        }
        // what the model takes looked up once per model (each is a look at its files), and the input's parts kept
        // as they are while an answer streams in: the input is skipped then, not composed again at every token
        val takes = remember(st.model, st.remote) { Triple(st.sees, st.hears, st.watches) }
        val onCamera = remember(camera) { camera?.let { c -> { c(false) } } }
        val sound = remember(takes.second, st.clips, st.recording, st.playing) {
            Sound(
                hears = takes.second,
                clips = st.clips,
                recording = st.recording,
                playing = st.playing,
                seconds = vm::clipSeconds,
                onAttach = { soundPicker.launch("audio/*") },
                onRecord = {
                    if (androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) ==
                        android.content.pm.PackageManager.PERMISSION_GRANTED) {
                        vm.startRecording()
                        true
                    } else {
                        microphone.launch(android.Manifest.permission.RECORD_AUDIO)
                        false
                    }
                },
                onRecorded = vm::stopRecording,
                onPlay = vm::togglePlay,
                onRemove = vm::removeClip,
            )
        }
        val films = remember(takes.third, st.videos, st.preparingVideo, camera) {
            Films(
                watches = takes.third,
                videos = st.videos,
                preparing = st.preparingVideo,
                onAttach = { videoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) },
                onRecord = camera?.let { c -> { c(true) } },
                onRemove = vm::removeVideo,
            )
        }
        CompositionLocalProvider(LocalDensity provides chatDensity) { Input(
            agent = true,
            commands = commands,
            enabled = st.ready && !st.answeringElsewhere,
            generating = st.generating,
            vision = takes.first,
            attachments = st.attachments,
            thumb = vm::thumbnail,
            onAttach = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            onCamera = onCamera,
            onRemove = vm::removeAttachment,
            sound = sound,
            films = films,
            onSend = vm::send,
            onStop = vm::stop,
        ) }
    }
}

/**
 * The chat's details, behind its title: the model, the last answer's speeds, how full the context is, where the
 * experts came from while it was written, and compressing the context by hand.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatInfo(st: ChatUiState, run: xyz.tifu.geode.engine.EngineRun?, unloading: Boolean, onUnload: () -> Unit,
                     onCompress: () -> Unit, onFork: () -> Unit, onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.padding(start = 24.dp, end = 24.dp, bottom = 24.dp)) {
            Text(st.title.ifEmpty { stringResource(R.string.chat_untitled) }, style = MaterialTheme.typography.titleLarge,
                maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(st.modelTitle ?: stringResource(R.string.chat_info_no_model), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 2.dp))
            st.remote?.let {
                Text(stringResource(R.string.chat_info_service, it.host), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 8.dp))
            }
            // in memory or not, how it runs, and letting it go (the next message loads it again)
            if (st.model != null && st.remote == null) {
                if (run != null) {
                    Text(stringResource(R.string.models_in_memory, xyz.tifu.geode.util.formatBytes(run.memoryBytes)),
                        style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(top = 8.dp))
                    Text(xyz.tifu.geode.ui.runModeText(run), style = MaterialTheme.typography.bodySmall)
                    OutlinedButton(onClick = onUnload, enabled = !unloading,
                        modifier = Modifier.padding(top = 8.dp)) {
                        Text(stringResource(if (unloading) R.string.models_unloading else R.string.models_unload))
                    }
                } else {
                    Text(stringResource(R.string.models_not_in_memory), style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp))
                }
            }
            Spacer(Modifier.height(16.dp))
            if (st.ready) {
                Text(stringResource(R.string.chat_info_last), style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(bottom = 8.dp))
                StatsGrid(st)
            }
            Text(stringResource(R.string.chat_info_this_chat), style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 20.dp, bottom = 8.dp))
            ChatCounts(st)
            // compressing needs a question after the last summary
            val sinceSummary = st.messages.drop(st.messages.indexOfLast { it.summary } + 1)
            Spacer(Modifier.height(16.dp))
            OutlinedButton(onClick = onCompress, enabled = !st.generating && sinceSummary.any { it.user }) {
                Icon(Icons.Outlined.Compress, null, Modifier.size(18.dp))
                Text(stringResource(R.string.chat_compress), Modifier.padding(start = 8.dp))
            }
            Text(stringResource(R.string.chat_compress_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
            // a copy to go on from apart: the messages, and the agent's Linux with its files and session
            OutlinedButton(onClick = onFork, enabled = !st.generating && st.chatId != null && st.messages.isNotEmpty(),
                modifier = Modifier.padding(top = 12.dp)) {
                Icon(Icons.Outlined.ForkRight, null, Modifier.size(18.dp))
                Text(stringResource(R.string.chat_fork), Modifier.padding(start = 8.dp))
            }
            Text(stringResource(R.string.chat_fork_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
        }
    }
}

/**
 * The chat's counts: compressions, prompt tokens read, tokens found in the context already (the cache) and their
 * share, tokens generated.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ChatCounts(st: ChatUiState) {
    val locale = java.util.Locale.getDefault()
    fun n(v: Long) = String.format(locale, "%,d", v)
    val s = st.stats
    FlowRow(horizontalArrangement = Arrangement.spacedBy(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Stat(stringResource(R.string.chat_stat_compressions), (s.compactions + st.messages.count { it.summary }).toString())
        Stat(stringResource(R.string.chat_stat_read), n(s.read))
        Stat(stringResource(R.string.chat_stat_cached), n(s.cached))
        Stat(stringResource(R.string.chat_stat_cache_hits),
            s.hitRate?.let { String.format(locale, "%.0f%%", it * 100) } ?: "—")
        Stat(stringResource(R.string.chat_stat_generated), n(s.generated))
    }
}

/**
 * The prompt's and the generation's speed in the last (or current) answer, how full the context is, and while the
 * answer is generated where its experts came from: the share already in RAM, and how often a read ahead was right.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun StatsGrid(st: ChatUiState) {
    val locale = java.util.Locale.getDefault()
    fun one(v: Double?) = v?.let { String.format(locale, "%.1f", it) } ?: "—"
    // whole thousands of tokens as "32K" (the engine's sizes are multiples of 1024)
    fun size(n: Int) = if (n >= 1024 && n % 1024 == 0) "${n / 1024}K" else n.toString()
    fun percent(v: Double?) = v?.let { String.format(locale, "%.0f%%", it * 100) } ?: "—"
    run {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Stat(stringResource(R.string.chat_stat_prefill), stringResource(R.string.chat_stat_tps, one(st.prefillTps)))
            Stat(stringResource(R.string.chat_stat_decode), stringResource(R.string.chat_stat_tps, one(st.decodeTps)))
            if (st.contextLength > 0) {
                Stat(
                    if (st.contextAllocated > 0) stringResource(R.string.chat_stat_context_memory, size(st.contextAllocated))
                    else stringResource(R.string.chat_stat_context),
                    stringResource(R.string.chat_stat_context_value, String.format(locale, "%,d", st.contextUsed),
                        size(st.contextLength)),
                )
            }
            Stat(stringResource(R.string.chat_stat_ram), percent(st.ramHitRate))
            Stat(stringResource(R.string.chat_stat_predict), percent(st.predictRate))
        }
    }
}

@Composable
private fun Stat(label: String, value: String) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
private fun Messages(st: ChatUiState, thumb: (String, Int) -> Bitmap?, seconds: (String) -> Float,
                     onPlay: (String) -> Unit, onReport: (ChatMessage) -> Unit, onOpenFile: (String) -> Unit,
                     speaking: Long?, onSpeak: (ChatMessage) -> Unit) {
    val list = rememberLazyListState()
    // The answer followed as it grows - while the list is at its end: the user's drag up stops it (what is read
    // earlier in the answer stays put), reaching the end again or a new message starts it again. A jump per token,
    // not an animation: an animation started at every token kept the screen drawing at full rate the whole answer
    // long (on the Redmi, the UI's thread then took a third of a core from the model's).
    var follow by remember { mutableStateOf(true) }
    LaunchedEffect(list) {
        list.interactionSource.interactions.collect { i ->
            when (i) {
                is DragInteraction.Start -> follow = false
                is DragInteraction.Stop, is DragInteraction.Cancel -> if (!list.canScrollForward) follow = true
            }
        }
    }
    LaunchedEffect(list) {
        // at the end and still (a fling that came to rest there included)
        snapshotFlow { !list.canScrollForward && !list.isScrollInProgress }.collect { end -> if (end) follow = true }
    }
    LaunchedEffect(st.messages.size) { follow = true }
    val last = st.messages.lastOrNull()
    LaunchedEffect(st.messages.size, last?.text?.length, last?.thinking?.length, last?.steps?.size) {
        if (follow && st.messages.isNotEmpty()) list.scrollToItem(st.messages.size - 1, Int.MAX_VALUE / 2)
    }
    LazyColumn(
        state = list,
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // the chat's start went after its compactions (Compactions): the model goes on from its summary of it
        if (Compactions.pruned(st.messages)) item(key = "pruned") {
            Text(stringResource(R.string.chat_pruned), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 4.dp))
        }
        items(st.messages, key = { it.id }) { m ->
            when {
                m.summary -> SummaryBubble(m)
                m.user -> UserBubble(m, thumb, st.playing, seconds, onPlay)
                else -> AnswerBubble(m, encoding = st.encodingImage && m.streaming, onReport = onReport,
                    onOpenFile = onOpenFile, speaking = speaking == m.id, onSpeak = onSpeak)
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun UserBubble(m: ChatMessage, thumb: (String, Int) -> Bitmap?, playing: String?, seconds: (String) -> Float,
                       onPlay: (String) -> Unit) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
        if (m.images.isNotEmpty()) {
            FlowRow(Modifier.widthIn(max = 320.dp).padding(bottom = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.End),
                verticalArrangement = Arrangement.spacedBy(6.dp)) {
                for (n in m.images) Thumb(n, 140.dp, thumb)
            }
        }
        if (m.videos.isNotEmpty()) {
            FlowRow(Modifier.widthIn(max = 320.dp).padding(bottom = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.End),
                verticalArrangement = Arrangement.spacedBy(6.dp)) {
                for (v in m.videos) Poster(v, 140.dp, thumb)
            }
        }
        if (m.audio.isNotEmpty()) {
            FlowRow(Modifier.widthIn(max = 320.dp).padding(bottom = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.End),
                verticalArrangement = Arrangement.spacedBy(6.dp)) {
                for (n in m.audio) Clip(n, playing == n, seconds, onPlay, onRemove = null)
            }
        }
        if (m.text.isNotEmpty()) {
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer,
                shape = RoundedCornerShape(18.dp, 18.dp, 4.dp, 18.dp),
                modifier = Modifier.widthIn(max = 320.dp),
            ) {
                SelectionContainer {
                    Text(m.text, Modifier.padding(horizontal = 14.dp, vertical = 10.dp))
                }
            }
        }
    }
}

/** What the input does with sound, for a model that hears (Gemma 4): its clips, the recording, the player. */
private class Sound(
    val hears: Boolean,
    val clips: List<String>,
    val recording: Float?,
    val playing: String?,
    val seconds: (String) -> Float,
    val onAttach: () -> Unit,
    /** The record button pressed: true when it records (false: the permission is asked for first). */
    val onRecord: () -> Boolean,
    val onRecorded: () -> Unit,
    val onPlay: (String) -> Unit,
    val onRemove: (String) -> Unit,
)

private fun clock(seconds: Float): String {
    val s = seconds.toInt()
    return "%d:%02d".format(s / 60, s % 60)
}

/** A sound clip: a tap plays it (or stops it), its length beside; [onRemove] puts a button that takes it off. */
@Composable
private fun Clip(name: String, playing: Boolean, seconds: (String) -> Float, onPlay: (String) -> Unit,
                 onRemove: (() -> Unit)?, enabled: Boolean = true) {
    val length = remember(name) { seconds(name) }
    Surface(shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.secondaryContainer) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 4.dp, end = 8.dp)) {
            IconButton(onClick = { onPlay(name) }, modifier = Modifier.size(36.dp)) {
                Icon(if (playing) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                    contentDescription = stringResource(if (playing) R.string.chat_clip_stop else R.string.chat_clip_play))
            }
            Icon(Icons.Outlined.Mic, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.onSecondaryContainer)
            Text(clock(length), style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(start = 4.dp))
            if (onRemove != null) {
                IconButton(onClick = onRemove, enabled = enabled, modifier = Modifier.padding(start = 2.dp).size(28.dp)) {
                    Icon(Icons.Filled.Close, contentDescription = stringResource(R.string.chat_remove_clip),
                        modifier = Modifier.size(18.dp))
                }
            }
        }
    }
}

/** What the input does with videos, for a model that watches them (Gemma 4 with vision). */
private class Films(
    val watches: Boolean,
    val videos: List<ChatVideo>,
    val preparing: Boolean,
    val onAttach: () -> Unit,
    /** The camera app records one (null: the phone has no camera). */
    val onRecord: (() -> Unit)?,
    val onRemove: (ChatVideo) -> Unit,
)

/** A video the camera app records: at most [seconds] long (what Gemma 4 takes of one: its sound's first 30 s). */
private class CaptureVideoFor(private val seconds: Int) : ActivityResultContracts.CaptureVideo() {
    override fun createIntent(context: Context, input: Uri): Intent =
        super.createIntent(context, input).putExtra(MediaStore.EXTRA_DURATION_LIMIT, seconds)
}

private const val VIDEO_SECONDS = 30

/** One thing the attach button offers: its label, icon, whether it can now, what it does. */
private class AttachItem(val label: Int, val icon: androidx.compose.ui.graphics.vector.ImageVector,
                         val enabled: Boolean, val onClick: () -> Unit)

/** A video as its first frame, a camera and its length over it. */
@Composable
private fun Poster(v: ChatVideo, size: Dp, thumb: (String, Int) -> Bitmap?) {
    Box {
        v.frames.firstOrNull()?.let { Thumb(it, size, thumb) }
        Surface(shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.scrim.copy(alpha = 0.6f),
            modifier = Modifier.align(Alignment.BottomStart).padding(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)) {
                Icon(Icons.Filled.Videocam, null, Modifier.size(14.dp), tint = androidx.compose.ui.graphics.Color.White)
                Text(clock(v.ms / 1000f), style = MaterialTheme.typography.labelSmall,
                    color = androidx.compose.ui.graphics.Color.White, modifier = Modifier.padding(start = 4.dp))
            }
        }
    }
}

/** A chat image, decoded small off the main thread. */
@Composable
private fun Thumb(name: String, size: Dp, thumb: (String, Int) -> Bitmap?, modifier: Modifier = Modifier) {
    val px = with(LocalDensity.current) { size.roundToPx() }
    val bmp by produceState<ImageBitmap?>(null, name, px) {
        value = withContext(Dispatchers.IO) { thumb(name, px)?.asImageBitmap() }
    }
    Box(modifier.size(size).clip(RoundedCornerShape(12.dp)).background(MaterialTheme.colorScheme.surfaceVariant)) {
        bmp?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
    }
}

@Composable
private fun AnswerBubble(m: ChatMessage, encoding: Boolean = false, onReport: (ChatMessage) -> Unit,
                         onOpenFile: (String) -> Unit = {}, speaking: Boolean = false, onSpeak: (ChatMessage) -> Unit = {}) {
    Column(Modifier.fillMaxWidth()) {
        // the agent condensed its context before this answer: what came before, it knows from its summary
        if (m.compacted) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 4.dp, bottom = 6.dp)) {
                Icon(Icons.Outlined.Summarize, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.tertiary)
                Text(stringResource(R.string.chat_summary_title), style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.tertiary, modifier = Modifier.padding(start = 6.dp))
            }
        }
        if (m.steps.isNotEmpty()) AgentSteps(m.steps)
        if (m.thinking.isNotEmpty()) Thinking(m)
        if (m.text.isNotEmpty()) {
            Surface(
                color = MaterialTheme.colorScheme.surfaceContainerHigh,
                shape = RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp),
            ) {
                SelectionContainer {
                    MarkdownText(m.text, Modifier.padding(horizontal = 14.dp, vertical = 10.dp))
                }
            }
        } else if (m.streaming && m.thinking.isEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Dots(Modifier.padding(horizontal = 12.dp, vertical = 16.dp))
                if (encoding) {
                    Text(stringResource(R.string.chat_encoding_image), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        // what the agent made for this answer: a tap shows it
        if (m.files.isNotEmpty()) AgentFileChips(m.files, onOpenFile)
        if (m.endedThinking) {
            Text(stringResource(R.string.chat_ended_thinking), color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp, start = 4.dp))
        }
        m.error?.let {
            Text(stringResource(R.string.chat_error, it), color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp, start = 4.dp))
        }
        // the speed, reading it aloud, and the report action of a finished answer (Play's AI-generated content policy)
        if (m.streaming && speaking) {
            // read as it is written: its reading can be stopped meanwhile
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) { SpeakButton(true) { onSpeak(m) } }
        }
        if (!m.streaming && (m.tokensPerSecond > 0f || m.text.isNotEmpty())) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                if (m.tokensPerSecond > 0f) {
                    Text(stringResource(R.string.chat_speed, m.tokensPerSecond),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(start = 4.dp))
                }
                Spacer(Modifier.weight(1f))
                if (m.text.isNotEmpty()) SpeakButton(speaking) { onSpeak(m) }
                if (m.text.isNotEmpty()) {
                    IconButton(onClick = { onReport(m) }, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.Outlined.Flag, contentDescription = stringResource(R.string.report_action),
                            modifier = Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

/** An answer read aloud (the speaker), or its reading stopped. */
@Composable
private fun SpeakButton(speaking: Boolean, onClick: () -> Unit) {
    IconButton(onClick = onClick, modifier = Modifier.size(36.dp)) {
        Icon(if (speaking) Icons.Outlined.StopCircle else Icons.AutoMirrored.Outlined.VolumeUp,
            contentDescription = stringResource(if (speaking) R.string.chat_speak_stop else R.string.chat_speak),
            modifier = Modifier.size(18.dp),
            tint = if (speaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Reporting an answer: a reason, a note, and whether the answer and its question go with it (Reports). */
@Composable
private fun ReportDialog(onDismiss: () -> Unit, onSend: (reason: String, note: String, include: Boolean) -> Unit) {
    var reason by rememberSaveable { mutableStateOf(Reports.REASONS.first()) }
    var note by rememberSaveable { mutableStateOf("") }
    var include by rememberSaveable { mutableStateOf(true) }
    val labels = mapOf(
        "offensive" to R.string.report_offensive,
        "harmful" to R.string.report_harmful,
        "sexual" to R.string.report_sexual,
        "false" to R.string.report_false,
        "other" to R.string.report_other,
    )
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = { Icon(Icons.Outlined.Flag, contentDescription = null) },
        title = { Text(stringResource(R.string.report_title)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text(stringResource(R.string.report_intro), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                for (r in Reports.REASONS) {
                    Row(
                        Modifier.fillMaxWidth()
                            .selectable(selected = reason == r, onClick = { reason = r }, role = Role.RadioButton)
                            .padding(vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(selected = reason == r, onClick = null)
                        Text(stringResource(labels.getValue(r)), Modifier.padding(start = 8.dp))
                    }
                }
                OutlinedTextField(
                    value = note,
                    onValueChange = { note = it.take(Reports.MAX_NOTE) },
                    label = { Text(stringResource(R.string.report_note)) },
                    minLines = 2,
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                )
                Row(
                    Modifier.fillMaxWidth()
                        .toggleable(value = include, onValueChange = { include = it }, role = Role.Checkbox)
                        .padding(top = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Checkbox(checked = include, onCheckedChange = null)
                    Text(stringResource(R.string.report_include), Modifier.padding(start = 8.dp),
                        style = MaterialTheme.typography.bodySmall)
                }
            }
        },
        confirmButton = { TextButton(onClick = { onSend(reason, note, include) }) { Text(stringResource(R.string.report_send)) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.chat_cancel)) } },
    )
}

/** A context compression: the summary the model goes on from, marked as the point the context starts again. */
@Composable
private fun SummaryBubble(m: ChatMessage) {
    var open by rememberSaveable(m.id) { mutableStateOf(false) }
    Surface(color = MaterialTheme.colorScheme.tertiaryContainer, shape = RoundedCornerShape(14.dp),
        modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
            Row(Modifier.clickable { open = !open }, verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.Summarize, null, Modifier.size(18.dp),
                    tint = MaterialTheme.colorScheme.onTertiaryContainer)
                Text(stringResource(R.string.chat_summary_title), Modifier.weight(1f).padding(horizontal = 8.dp),
                    style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onTertiaryContainer)
                if (m.streaming) Dots(dot = 4.dp, color = MaterialTheme.colorScheme.onTertiaryContainer)
                else Icon(if (open) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore, null, Modifier.size(18.dp),
                    tint = MaterialTheme.colorScheme.onTertiaryContainer)
            }
            when {
                m.streaming && m.text.isEmpty() -> Text(stringResource(R.string.chat_compressing),
                    style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 6.dp))
                open || m.streaming -> SelectionContainer {
                    MarkdownText(m.text, Modifier.padding(top = 6.dp))
                }
            }
            if (!m.streaming && m.error == null) {
                Text(stringResource(R.string.chat_summary_note), style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer, modifier = Modifier.padding(top = 6.dp))
            }
            m.error?.let {
                Text(stringResource(R.string.chat_error, it), color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp))
            }
        }
    }
}

/** The saved chats: open one, start a new one, delete one (asked first). */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun History(chats: List<ChatInfo>, current: String?, enabled: Boolean, onOpen: (String) -> Unit,
                    onNew: () -> Unit, onDelete: (String) -> Unit, onDismiss: () -> Unit) {
    var deleting by remember { mutableStateOf<ChatInfo?>(null) }
    val fmt = remember { java.text.DateFormat.getDateTimeInstance(java.text.DateFormat.SHORT, java.text.DateFormat.SHORT) }
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.padding(bottom = 24.dp)) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.chat_history), style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.weight(1f))
                TextButton(onClick = onNew, enabled = enabled) {
                    Icon(Icons.Outlined.AddComment, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.chat_new), Modifier.padding(start = 6.dp))
                }
            }
            if (chats.isEmpty()) {
                Text(stringResource(R.string.chat_history_empty), Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            LazyColumn(Modifier.fillMaxWidth()) {
                items(chats, key = { it.id }) { c ->
                    Row(
                        Modifier.fillMaxWidth().clickable(enabled = enabled) { onOpen(c.id) }
                            .padding(start = 16.dp, top = 8.dp, bottom = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(c.title.ifBlank { "…" }, maxLines = 1, style = MaterialTheme.typography.bodyLarge,
                                color = if (c.id == current) MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.onSurface)
                            Text(fmt.format(java.util.Date(c.updated)), style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        IconButton(onClick = { deleting = c }, enabled = enabled) {
                            Icon(Icons.Outlined.Delete, contentDescription = stringResource(R.string.chat_delete))
                        }
                    }
                    HorizontalDivider()
                }
            }
        }
    }
    deleting?.let { c ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            text = { Text(stringResource(R.string.chat_delete_confirm, c.title)) },
            confirmButton = {
                TextButton(onClick = {
                    onDelete(c.id)
                    deleting = null
                }) { Text(stringResource(R.string.chat_delete)) }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text(stringResource(R.string.chat_cancel)) } },
        )
    }
}

@Composable
private fun Thinking(m: ChatMessage) {
    // open while the model is still reasoning, folded once the answer starts
    var open by rememberSaveable(m.id) { mutableStateOf<Boolean?>(null) }
    val shown = open ?: (m.streaming && m.text.isEmpty())
    Column(Modifier.padding(bottom = 6.dp)) {
        Row(
            Modifier.clickable { open = !shown }.padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Outlined.Psychology, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(stringResource(R.string.chat_thinking), Modifier.padding(horizontal = 6.dp),
                style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Icon(if (shown) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore, null, Modifier.size(18.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        AnimatedVisibility(shown) {
            Text(m.thinking.trim(), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(start = 24.dp))
        }
        if (m.thinkingCut) {
            Text(stringResource(R.string.chat_thinking_cut), style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(start = 24.dp, top = 2.dp))
        }
    }
}

@Composable
private fun Input(agent: Boolean, commands: List<xyz.tifu.geode.agent.AgentCommand>, enabled: Boolean,
                  generating: Boolean, vision: Boolean, attachments: List<String>,
                  thumb: (String, Int) -> Bitmap?, onAttach: () -> Unit, onCamera: (() -> Unit)?,
                  onRemove: (String) -> Unit,
                  sound: Sound, films: Films, onSend: (String) -> Unit, onStop: () -> Unit) {
    var text by rememberSaveable { mutableStateOf("") }
    // a "/" typed to the agent: the commands it takes that begin so, a tap puts one in
    val typed = text.takeIf { it.startsWith("/") && ' ' !in it }?.removePrefix("/")
    val matching = typed?.let { t -> commands.filter { it.name.startsWith(t, ignoreCase = true) } }.orEmpty()
    if (matching.isNotEmpty() && !generating) {
        Surface(tonalElevation = 3.dp, shape = RoundedCornerShape(16.dp),
            modifier = Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 8.dp)) {
            Column(Modifier.heightIn(max = 240.dp).verticalScroll(rememberScrollState()).padding(vertical = 4.dp)) {
                for (c in matching) {
                    Column(Modifier.fillMaxWidth().clickable { text = "/${c.name} " }
                        .padding(horizontal = 16.dp, vertical = 8.dp)) {
                        Text("/${c.name}" + (c.hint?.let { " $it" } ?: ""), style = MaterialTheme.typography.bodyMedium,
                            fontFamily = FontFamily.Monospace)
                        if (c.description.isNotEmpty()) {
                            Text(c.description, style = MaterialTheme.typography.bodySmall, maxLines = 2,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
    }
    // the images for the next question, each with a button that takes it off
    if (vision && attachments.isNotEmpty()) {
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, top = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (n in attachments) {
                Box {
                    Thumb(n, 72.dp, thumb)
                    Surface(shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                        modifier = Modifier.align(Alignment.TopEnd).padding(2.dp).size(24.dp)
                            .clickable(enabled = !generating) { onRemove(n) }) {
                        Icon(Icons.Filled.Close, contentDescription = stringResource(R.string.chat_remove_image),
                            modifier = Modifier.padding(4.dp))
                    }
                }
            }
        }
    }
    // the videos for the next question (and one being prepared), each with a button that takes it off
    if (films.watches && (films.videos.isNotEmpty() || films.preparing)) {
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, top = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            for (v in films.videos) {
                Box {
                    Poster(v, 72.dp, thumb)
                    Surface(shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                        modifier = Modifier.align(Alignment.TopEnd).padding(2.dp).size(24.dp)
                            .clickable(enabled = !generating) { films.onRemove(v) }) {
                        Icon(Icons.Filled.Close, contentDescription = stringResource(R.string.chat_remove_video),
                            modifier = Modifier.padding(4.dp))
                    }
                }
            }
            if (films.preparing) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                    Text(stringResource(R.string.chat_video_preparing), style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(start = 8.dp))
                }
            }
        }
    }
    // the clips for the next question: each plays, each comes off
    if (sound.hears && sound.clips.isNotEmpty()) {
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, top = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (n in sound.clips) Clip(n, sound.playing == n, sound.seconds, sound.onPlay,
                onRemove = { sound.onRemove(n) }, enabled = !generating)
        }
    }
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        // what may go with the question, by what the model takes: images (picked, or taken with the camera), videos
        // (picked or recorded), sound files - nothing a model cannot use (Qwen: no video, no sound; a text model: none)
        val items = buildList {
            if (vision) {
                add(AttachItem(R.string.chat_attach_image, Icons.Outlined.Image, true, onAttach))
                onCamera?.let { add(AttachItem(R.string.chat_take_photo, Icons.Outlined.PhotoCamera, true, it)) }
            }
            if (films.watches) {
                add(AttachItem(R.string.chat_attach_video, Icons.Outlined.Videocam, !films.preparing, films.onAttach))
                films.onRecord?.let { add(AttachItem(R.string.chat_record_video, Icons.Outlined.VideoCall, !films.preparing, it)) }
            }
            if (sound.hears) add(AttachItem(R.string.chat_attach_audio, Icons.Outlined.AudioFile, true, sound.onAttach))
        }
        if (items.size > 1) {
            var menu by remember { mutableStateOf(false) }
            Box {
                IconButton(onClick = { menu = true }, enabled = enabled && !generating && sound.recording == null,
                    modifier = Modifier.padding(bottom = 4.dp)) {
                    Icon(Icons.Outlined.AttachFile, contentDescription = stringResource(R.string.chat_attach))
                }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    for (it in items) DropdownMenuItem(text = { Text(stringResource(it.label)) },
                        leadingIcon = { Icon(it.icon, null) },
                        enabled = it.enabled,
                        onClick = {
                            menu = false
                            it.onClick()
                        })
                }
            }
        } else if (items.size == 1) {
            val it = items[0]
            IconButton(onClick = it.onClick, enabled = enabled && !generating && it.enabled,
                modifier = Modifier.padding(bottom = 4.dp)) {
                Icon(if (it.label == R.string.chat_attach_image) Icons.Outlined.AddPhotoAlternate else it.icon,
                    contentDescription = stringResource(it.label))
            }
        }
        val recording = sound.recording
        Box(Modifier.weight(1f)) {
            // the words stay (hidden while a voice message records): their focus and the keyboard with them, so the
            // row does not move under the finger that holds the button
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                enabled = enabled,
                placeholder = { Text(stringResource(if (agent) R.string.agent_input_hint else R.string.chat_input_hint)) },
                maxLines = 6,
                shape = RoundedCornerShape(24.dp),
                modifier = Modifier.fillMaxWidth().alpha(if (recording != null) 0f else 1f),
            )
            if (recording != null) {
                // the recording over them: its length, the limit, what letting go does
                Surface(shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.errorContainer,
                    modifier = Modifier.matchParentSize()) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 16.dp)) {
                        Box(Modifier.size(10.dp).clip(CircleShape).background(MaterialTheme.colorScheme.error))
                        Text(stringResource(R.string.chat_recording, clock(recording), clock(30f)),
                            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onErrorContainer,
                            maxLines = 2, modifier = Modifier.padding(start = 10.dp))
                    }
                }
            }
        }
        Spacer(Modifier.size(8.dp))
        if (sound.hears && !generating) {
            // held: it records; let go: the clip is attached
            val on = recording != null
            val desc = stringResource(R.string.chat_record)
            val canRecord by rememberUpdatedState(enabled)
            val record by rememberUpdatedState(sound.onRecord)
            val recorded by rememberUpdatedState(sound.onRecorded)
            Surface(shape = CircleShape,
                color = if (on) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.secondaryContainer,
                modifier = Modifier.padding(bottom = 4.dp, end = 8.dp).size(48.dp)
                    .semantics { contentDescription = desc; role = Role.Button }
                    .pointerInput(Unit) {
                        awaitEachGesture {
                            awaitFirstDown()
                            if (canRecord && record()) {
                                try {
                                    // until every finger is up, wherever it went
                                    do {
                                        val e = awaitPointerEvent()
                                    } while (e.changes.any { it.pressed })
                                } finally {
                                    // let go - or the gesture cut short (the button gone): the recording ends either way
                                    recorded()
                                }
                            }
                        }
                    }) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(Icons.Filled.Mic, contentDescription = null,
                        tint = if (on) MaterialTheme.colorScheme.onError else MaterialTheme.colorScheme.onSecondaryContainer)
                }
            }
        }
        if (generating) {
            FilledIconButton(onClick = onStop, modifier = Modifier.padding(bottom = 4.dp)) {
                Icon(Icons.Filled.Stop, contentDescription = stringResource(R.string.chat_stop))
            }
        } else {
            FilledIconButton(
                onClick = {
                    onSend(text)
                    text = ""
                },
                enabled = enabled && sound.recording == null && !films.preparing &&
                    (text.isNotBlank() || (vision && attachments.isNotEmpty()) || (sound.hears && sound.clips.isNotEmpty()) ||
                        (films.watches && films.videos.isNotEmpty())),
                modifier = Modifier.padding(bottom = 4.dp),
            ) {
                Icon(Icons.AutoMirrored.Filled.Send, contentDescription = stringResource(R.string.chat_send))
            }
        }
    }
}

@Composable
private fun NoModel(openModels: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(stringResource(R.string.chat_no_model), textAlign = TextAlign.Center)
        Spacer(Modifier.height(16.dp))
        Button(onClick = openModels) { Text(stringResource(R.string.chat_open_models)) }
    }
}

@Composable
private fun Hint(text: String) {
    Box(Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
        Text(text, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
