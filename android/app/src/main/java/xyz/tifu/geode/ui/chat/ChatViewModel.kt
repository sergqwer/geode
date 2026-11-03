package xyz.tifu.geode.ui.chat

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.data.ChatInfo
import xyz.tifu.geode.data.ChatStats
import xyz.tifu.geode.data.ChatStore
import xyz.tifu.geode.data.Reports
import xyz.tifu.geode.agent.AgentEvent
import xyz.tifu.geode.agent.AgentTurns
import xyz.tifu.geode.agent.finished
import xyz.tifu.geode.agent.withAgent
import xyz.tifu.geode.engine.AnswerService
import xyz.tifu.geode.engine.EngineEvent
import xyz.tifu.geode.media.AudioDecoder
import xyz.tifu.geode.media.AudioStore
import xyz.tifu.geode.media.ClipPlayer
import xyz.tifu.geode.media.Microphone
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.server.GenResult
import xyz.tifu.geode.server.WebChat
import xyz.tifu.geode.vision.ImageStore

data class ChatMessage(
    val id: Long,
    val user: Boolean,
    val text: String,
    val thinking: String = "",
    val streaming: Boolean = false,
    val error: String? = null,
    val tokensPerSecond: Float = 0f,
    /** The reasoning reached its budget and was closed for the answer. */
    val thinkingCut: Boolean = false,
    /** The answer ended while the model was still reasoning: there is no answer text. */
    val endedThinking: Boolean = false,
    /** A context compression's summary (chats from before every chat was the agent's): the model went on from it. */
    val summary: Boolean = false,
    /** A question's images (ImageStore's names), before its text. */
    val images: List<String> = emptyList(),
    /** A question's sound clips (AudioStore's names: a voice message, a sound file), after its images. */
    val audio: List<String> = emptyList(),
    /** A question's videos, as the model watches them (after its images, before its clips). */
    val videos: List<ChatVideo> = emptyList(),
    /** The agent's answer: what it did on the way (its words between the tools, the tools' calls and results). */
    val steps: List<AgentStep> = emptyList(),
    /** The files the agent made or changed for this answer (paths in its /workspace): shown under it, opened here. */
    val files: List<String> = emptyList(),
)

/**
 * A video as the model watches it (VideoDecoder): its frames (ImageStore's names) with the second each is at, its
 * sound (AudioStore's name; null: none), its length. The file itself is not kept.
 */
data class ChatVideo(val frames: List<String>, val seconds: List<Int>, val audio: String?, val ms: Long)

/**
 * One thing the agent did: a tool's call - [tool] its name, [title] what it was asked (a command, a path, a pattern)
 * - and its result ([ok]: null while it runs), or with no [tool] its words between the calls ([title]).
 */
data class AgentStep(
    val tool: String,
    val title: String,
    val output: String = "",
    val ok: Boolean? = null,
    val id: String = "",
    /** The call's arguments as JSON. */
    val input: String = "",
)

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val model: InstalledModel? = null,
    /** The selected model was looked up (so "no model" is not shown while it loads). */
    val modelChecked: Boolean = false,
    val loadingModel: Boolean = false,
    val generating: Boolean = false,
    /** The model is writing a summary of the conversation (a context compression). */
    val compressing: Boolean = false,
    /** Tokens in the conversation, the engine's limit, and the part of it in memory now (chunk by chunk). */
    val contextUsed: Int = 0,
    val contextLength: Int = 0,
    val contextAllocated: Int = 0,
    /** The last (or current) answer's speeds, tokens/s: reading the prompt, then generating; null until known. */
    val prefillTps: Double? = null,
    val decodeTps: Double? = null,
    /** While the answer is generated: the share of expert picks already in RAM, and how often a read ahead (the
     *  prediction of the next layer's experts) was right; null until known. */
    val ramHitRate: Double? = null,
    val predictRate: Double? = null,
    /** The open chat (null: a new one, saved with its first question). */
    val chatId: String? = null,
    /** Its short name (the first line of its first question; empty for a new chat). */
    val title: String = "",
    /** Its token counts over all its answers (saved with it). */
    val stats: ChatStats = ChatStats(),
    /** Images attached to the next question (ImageStore's names). */
    val attachments: List<String> = emptyList(),
    /** Sound clips attached to the next question (AudioStore's names). */
    val clips: List<String> = emptyList(),
    /** Videos attached to the next question. */
    val videos: List<ChatVideo> = emptyList(),
    /** A picked video being turned into its frames and sound. */
    val preparingVideo: Boolean = false,
    /** A voice message being recorded: its seconds so far (null: none). */
    val recording: Float? = null,
    /** The clip playing back (null: none). */
    val playing: String? = null,
    /** The engine is encoding the question's images (before it reads the prompt). */
    val encodingImage: Boolean = false,
    /** The question waits: the local server is answering a request. */
    val waiting: Boolean = false,
    /** The agent's Linux being unpacked (the first start, or an update brought a new image): the share done. */
    val agentInstalling: Float? = null,
    /** The agent answers this chat for the web chat now (the phone's question waits for it). */
    val answeringElsewhere: Boolean = false,
    /** What the agent does right now, in a line (reading the prompt, a tool running); null when nothing. */
    val agentStatus: String? = null,
)

/**
 * The conversation with the agent. Every chat is the agent's (geode-agent, native in the app: its tools, the chat's
 * own Linux and folder, web search, the user's skills and MCP servers - the model is always told what it has); its
 * model is Geode's own, served on the phone's loopback (ApiEngine), which keeps the conversation and its shared start
 * on the storage (StateCache).
 *
 * Chats are saved (ChatStore) after each question and answer. The agent compacts its context itself when it fills up
 * (and on /compact); a chat from before every chat was the agent's goes on with the agent, its earlier messages told
 * to it with the first new one.
 */
class ChatViewModel(private val app: GeodeApp) : ViewModel() {
    private val store = ModelStore(app)
    private val chatStore = ChatStore(app)
    private val images = ImageStore(app)
    private val clips = AudioStore(app)
    private val player = ClipPlayer()
    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()
    private val _notices = kotlinx.coroutines.flow.MutableSharedFlow<String>(extraBufferCapacity = 4)

    /** Short messages for the user (a toast): a recording too short, a sound file not read. */
    val notices: kotlinx.coroutines.flow.SharedFlow<String> = _notices
    private val _chats = MutableStateFlow<List<ChatInfo>>(emptyList())
    val chats: StateFlow<List<ChatInfo>> = _chats.asStateFlow()

    /** The model in memory and how it runs (null: none), for the chat's details. */
    val engineRun: StateFlow<xyz.tifu.geode.engine.EngineRun?> = app.engine.run

    /** Unloading the model now (an answer running is stopped first). */
    private val _unloading = MutableStateFlow(false)
    val unloading: StateFlow<Boolean> = _unloading.asStateFlow()

    /** The model out of memory now (an answer running is stopped, not waited for); the next message loads it again. */
    fun unloadModel() {
        if (_unloading.value) return
        _unloading.value = true
        if (_state.value.generating) stop()
        viewModelScope.launch {
            try {
                app.unloadNow()
            } finally {
                _unloading.value = false
            }
        }
    }

    /** The commands the open chat's agent takes ("/name"), as it lists them. */
    val agentCommands: StateFlow<List<xyz.tifu.geode.agent.AgentCommand>> = app.agent.commands

    private var nextId = 1L
    private var chatTitle = ""
    private var chatCreated = 0L
    /** The question running. */
    private var answerJob: Job? = null
    /** The answer the chat shows as running (a stopped one's own end - the harness's, seconds later - is not it). */
    private var answering = 0L
    /** The agent's session for this chat (its next message goes on with it). */
    private var agentSession: String? = null

    init {
        viewModelScope.launch {
            app.settings.flow.collect { s ->
                val m = withContext(Dispatchers.IO) { s.selectedModel.takeIf { it.isNotEmpty() }?.let(store::find) }
                val changed = m?.dir?.path != _state.value.model?.dir?.path
                // with a model open the engine's own limit stands (the budget may allow less than the setting)
                _state.update {
                    it.copy(model = m, modelChecked = true,
                        contextLength = if (app.engine.openModel.value != null) it.contextLength else s.contextLength)
                }
                // the agent's shared start read ahead for it, once the engine is idle (Agent.warmAhead)
                if (changed && m != null) app.agent.warmAhead()
            }
        }
        viewModelScope.launch {
            // a generation keeps the app alive in the background (AnswerService), until a moment after it ends
            _state.map { it.generating }.distinctUntilChanged().collectLatest { on ->
                if (on) AnswerService.start(app) else {
                    delay(3000)
                    AnswerService.stop(app)
                }
            }
        }
        viewModelScope.launch {
            // the web chat changed a chat: the list follows, and the open chat when it is that one (not while this
            // chat writes an answer: its own save comes after)
            WebChat.changes.collect { id ->
                refreshChats()
                val st = _state.value
                if (id == st.chatId && !st.generating) {
                    if (withContext(Dispatchers.IO) { chatStore.load(id) } == null) newChat() else openChat(id)
                }
            }
        }
        refreshChats()
        // the agent's Linux onto the phone when it is not there or an update of the app brought a new image (the
        // chats' folders stay), its progress in the chat
        app.agent.installIfNeeded()
        viewModelScope.launch {
            app.agent.installing.collect { p ->
                val was = _state.value.agentInstalling
                _state.update { it.copy(agentInstalling = p) }
                // unpacked: the open chat's harness (the unpacking stopped it)
                if (was != null && p == null) warmAgent()
            }
        }
        viewModelScope.launch {
            app.agent.installError.collect { e -> if (e != null) _state.update { it.copy(agentStatus = e) } }
        }
        viewModelScope.launch {
            // the shared start read with a harness of its own: the open chat's again after it
            app.agent.warming.collect { on -> if (!on) warmAgent() }
        }
        viewModelScope.launch {
            // the web chat's turn on the chat open here: the phone waits for it (and shows its answer, saved, after)
            kotlinx.coroutines.flow.combine(app.agent.turn, _state.map { it.chatId to it.generating }) { t, (c, g) ->
                t != null && t == c && !g
            }.distinctUntilChanged().collect { e -> _state.update { it.copy(answeringElsewhere = e) } }
        }
        if (xyz.tifu.geode.BuildConfig.DEBUG) {
            // debug builds: an image or a clip attached from adb (LinuxDebugReceiver's attach), as the pickers would
            viewModelScope.launch {
                xyz.tifu.geode.util.DebugHooks.attach.collect { name -> _state.update { it.copy(attachments = it.attachments + name) } }
            }
            viewModelScope.launch {
                xyz.tifu.geode.util.DebugHooks.attachAudio.collect { name -> _state.update { it.copy(clips = it.clips + name) } }
            }
            viewModelScope.launch { xyz.tifu.geode.util.DebugHooks.attachVideo.collect { attachVideo(it) } }
        }
    }

    /** Looks the selected model up again (after a download or a delete on the Models tab). */
    fun refreshModel() {
        viewModelScope.launch {
            val s = app.settings.flow.first()
            val m = withContext(Dispatchers.IO) { s.selectedModel.takeIf { it.isNotEmpty() }?.let(store::find) }
            _state.update { it.copy(model = m, modelChecked = true) }
        }
    }

    // ---- the saved chats ----

    private fun refreshChats() {
        viewModelScope.launch { _chats.value = withContext(Dispatchers.IO) { chatStore.list() } }
    }

    /** The open chat to disk (a new one gets its id and title from its first question). */
    private fun persist() {
        val msgs = _state.value.messages
        if (msgs.none { it.user }) return
        val id = _state.value.chatId ?: System.currentTimeMillis().toString().also { id ->
            chatCreated = System.currentTimeMillis()
            _state.update { it.copy(chatId = id) }
        }
        // the title too (a chat's first question names it)
        if (chatTitle.isEmpty()) {
            val first = msgs.first { it.user }
            chatTitle = first.text.lineSequence().first().take(60)
                .ifBlank {
                    app.getString(when {
                        first.images.isNotEmpty() -> R.string.chat_image_title
                        first.videos.isNotEmpty() -> R.string.chat_video_title
                        first.audio.isNotEmpty() -> R.string.chat_voice_title
                        else -> R.string.chat_image_title
                    })
                }
            _state.update { it.copy(title = chatTitle) }
        }
        val title = chatTitle
        val created = chatCreated
        val stats = _state.value.stats
        val session = agentSession
        viewModelScope.launch {
            withContext(Dispatchers.IO) { runCatching { chatStore.save(id, title, created, msgs, stats, true, session) } }
            refreshChats()
        }
    }

    fun openChat(id: String) {
        if (_state.value.generating) return
        viewModelScope.launch {
            val loaded = withContext(Dispatchers.IO) { chatStore.load(id) } ?: return@launch
            chatTitle = loaded.title
            agentSession = loaded.agentSession
            chatCreated = id.toLongOrNull() ?: System.currentTimeMillis()
            nextId = (loaded.messages.maxOfOrNull { it.id } ?: 0L) + 1
            _state.update {
                it.copy(title = chatTitle, stats = loaded.stats, messages = loaded.messages, chatId = id, contextUsed = 0,
                    contextAllocated = allocated(0, it.contextLength))
            }
            warmAgent()
        }
    }

    /** A copy of the open chat (its agent's Linux too) opened to go on from apart. */
    fun forkChat() {
        val id = _state.value.chatId ?: return
        if (_state.value.generating) return
        viewModelScope.launch {
            val copy = withContext(Dispatchers.IO) { chatStore.fork(id, app.getString(R.string.chat_fork_suffix)) } ?: return@launch
            refreshChats()
            openChat(copy)
        }
    }

    fun deleteChat(id: String) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                chatStore.delete(id)
                // a chat from before every chat was the agent's kept its conversation's state too
                app.stateCache.delete("chat-$id")
            }
            if (_state.value.chatId == id) newChat()
            refreshChats()
        }
    }

    fun newChat() {
        if (_state.value.generating) stop()
        chatTitle = ""
        agentSession = null
        _state.update {
            it.copy(title = chatTitle, stats = ChatStats(), chatId = null, messages = emptyList(), contextUsed = 0,
                contextAllocated = allocated(0, it.contextLength))
        }
        app.agent.stop()
        warmAgent()
    }

    // ---- images ----

    /** A picked image into the chat's store, attached to the next question. */
    fun attach(uri: Uri) {
        viewModelScope.launch {
            val name = withContext(Dispatchers.IO) { runCatching { images.import(uri) }.getOrNull() } ?: return@launch
            _state.update { it.copy(attachments = it.attachments + name) }
        }
    }

    fun removeAttachment(name: String) {
        _state.update { it.copy(attachments = it.attachments - name) }
        viewModelScope.launch(Dispatchers.IO) { images.delete(listOf(name)) }
    }

    /** A chat image's small copy (on a background thread). */
    fun thumbnail(name: String, px: Int) = images.thumbnail(name, px)

    // ---- sound ----

    private var recordJob: Job? = null
    private var recordStop: java.util.concurrent.atomic.AtomicBoolean? = null

    /**
     * A voice message recorded while the button is held ([stopRecording] on its release): attached to the next
     * question as a clip, a too short one (a tap) dropped. The caller has the microphone's permission.
     */
    @android.annotation.SuppressLint("MissingPermission")
    fun startRecording() {
        if (recordJob?.isActive == true || _state.value.model?.audio != true) return
        player.stop()
        val stop = java.util.concurrent.atomic.AtomicBoolean(false)
        recordStop = stop
        _state.update { it.copy(recording = 0f, playing = null) }
        recordJob = viewModelScope.launch {
            try {
                val pcm = Microphone.record(stop) { n -> _state.update { it.copy(recording = n.toFloat() / AudioStore.RATE) } }
                if (pcm.size >= MIN_CLIP) {
                    val name = withContext(Dispatchers.IO) { clips.save(pcm) }
                    _state.update { it.copy(clips = it.clips + name) }
                } else {
                    _notices.tryEmit(app.getString(R.string.chat_record_hold))
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _notices.tryEmit(app.getString(R.string.chat_record_failed, e.message.orEmpty()))
            } finally {
                _state.update { it.copy(recording = null) }
            }
        }
    }

    /** The button let go: the recording so far becomes the clip. */
    fun stopRecording() {
        recordStop?.set(true)
    }

    /** A picked sound file into the chat's store (its first 30 s), attached to the next question. */
    fun attachAudio(uri: Uri) {
        viewModelScope.launch {
            val name = withContext(Dispatchers.IO) {
                runCatching { clips.save(AudioDecoder.decode(app, uri)) }.getOrNull()
            }
            if (name == null) _notices.tryEmit(app.getString(R.string.chat_audio_unreadable))
            else _state.update { it.copy(clips = it.clips + name) }
        }
    }

    fun removeClip(name: String) {
        if (_state.value.playing == name) stopPlaying()
        _state.update { it.copy(clips = it.clips - name) }
        viewModelScope.launch(Dispatchers.IO) { clips.delete(listOf(name)) }
    }

    /**
     * A picked video into the chat's stores - its frames (about one a second, at most 32) and its sound - attached to
     * the next question.
     */
    fun attachVideo(uri: Uri) {
        if (_state.value.model?.video != true || _state.value.preparingVideo) return
        _state.update { it.copy(preparingVideo = true) }
        viewModelScope.launch {
            val v = withContext(Dispatchers.IO) { runCatching { importVideo(uri) }.getOrNull() }
            _state.update { st -> if (v == null) st.copy(preparingVideo = false) else st.copy(preparingVideo = false, videos = st.videos + v) }
            if (v == null) _notices.tryEmit(app.getString(R.string.chat_video_unreadable))
        }
    }

    private fun importVideo(uri: Uri): ChatVideo {
        val video = xyz.tifu.geode.media.VideoDecoder.decode(app, uri)
        val names = ArrayList<String>()
        try {
            for (f in video.frames) {
                val bytes = java.io.ByteArrayOutputStream().use { o ->
                    f.bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, o)
                    o.toByteArray()
                }
                f.bitmap.recycle()
                names += images.importChatBytes(bytes)
            }
            val sound = video.audio?.takeIf { AudioStore.tokens(it.size.toLong()) > 0 }?.let { clips.save(it) }
            return ChatVideo(names, video.frames.map { it.second }, sound, video.durationMs)
        } catch (e: Exception) {
            for (f in video.frames) if (!f.bitmap.isRecycled) f.bitmap.recycle()
            images.delete(names)
            throw e
        }
    }

    fun removeVideo(v: ChatVideo) {
        _state.update { it.copy(videos = it.videos - v) }
        viewModelScope.launch(Dispatchers.IO) {
            images.delete(v.frames)
            v.audio?.let { clips.delete(listOf(it)) }
        }
    }

    /** A clip's length in seconds (its file's). */
    fun clipSeconds(name: String): Float = clips.seconds(name)

    /** Plays a clip, or stops it when it is the one playing. */
    fun togglePlay(name: String) {
        if (_state.value.playing == name) return stopPlaying()
        viewModelScope.launch {
            val pcm = withContext(Dispatchers.IO) { runCatching { AudioStore.read(clips.file(name)) }.getOrNull() } ?: return@launch
            _state.update { it.copy(playing = name) }
            val playing = player.play(pcm) { _state.update { if (it.playing == name) it.copy(playing = null) else it } }
            if (!playing) {
                _state.update { if (it.playing == name) it.copy(playing = null) else it }
                _notices.tryEmit(app.getString(R.string.chat_clip_unplayable))
            }
        }
    }

    private fun stopPlaying() {
        player.stop()
        _state.update { it.copy(playing = null) }
    }

    /**
     * Reports an answer to the developer (Reports): the reason, the note and, with [include], the answer and the
     * question it answered. [done] gets true when it went, false when it waits for the next start.
     */
    fun report(answer: ChatMessage, reason: String, note: String, include: Boolean, done: (Boolean) -> Unit) {
        val msgs = _state.value.messages
        val question = msgs.subList(0, msgs.indexOfFirst { it.id == answer.id }.coerceAtLeast(0))
            .lastOrNull { it.user }?.text.orEmpty()
        val r = Reports.Report(
            reason = reason,
            note = note.trim(),
            question = if (include) question else "",
            answer = if (include) answer.text else "",
            model = _state.value.model?.title.orEmpty(),
        )
        viewModelScope.launch { done(app.reports.send(r)) }
    }

    // ---- questions and answers ----

    fun send(text: String) {
        val question = text.trim()
        val st = _state.value
        val attached = if (st.model?.vision == true) st.attachments else emptyList()
        val heard = if (st.model?.audio == true) st.clips else emptyList()
        val watched = if (st.model?.video == true) st.videos else emptyList()
        if ((question.isEmpty() && attached.isEmpty() && heard.isEmpty() && watched.isEmpty()) || st.generating ||
            st.model == null || st.answeringElsewhere || st.recording != null || st.preparingVideo) return
        val before = st.messages
        val user = ChatMessage(nextId++, user = true, text = question, images = attached, audio = heard, videos = watched)
        val answer = ChatMessage(nextId++, user = false, text = "", streaming = true)
        _state.update {
            it.copy(messages = it.messages + user + answer, generating = true, attachments = emptyList(),
                clips = emptyList(), videos = emptyList())
        }
        persist()
        val turn = ++answering
        answerJob = viewModelScope.launch { agentAnswer(before, user, answer.id, turn) }
    }

    /**
     * Stops the answer: the agent's run stops whole (its console and whatever it started). At once for the chat: the
     * answer ends as it is now and the next message can go (it waits for the harness to end the stopped turn, if that
     * is still under way); the engine stops what it runs or is about to run for this chat.
     */
    fun stop() {
        val st = _state.value
        if (!st.generating) return
        answering++
        answerJob?.cancel()
        // the engine is this chat's turn's (or about to be): not the web chat's this one may still wait for
        if (!st.waiting) app.engine.stop()
        st.messages.lastOrNull { !it.user && it.streaming }?.let { m ->
            edit(m.id) { it.copy(streaming = false, error = app.getString(R.string.chat_stopped)) }
        }
        _state.update { it.copy(generating = false, agentStatus = null, encodingImage = false, waiting = false) }
        persist()
    }

    /** The agent compacts the conversation (its /compact): a summary it goes on from, its context free again. */
    fun compress() {
        val st = _state.value
        if (st.generating || st.model == null || st.messages.none { it.user }) return
        send(COMPACT)
    }

    // ---- the agent ----

    /** The chat's folder (the agent's /workspace), once the chat is saved. */
    fun agentFolder(): java.io.File? = _state.value.chatId?.let { app.agent.workspace(it) }

    /**
     * The agent answers [user]'s message in message [id]: the model's words appear as Geode writes them, each tool's
     * call becomes a card (its result filled in when it comes), the words between the calls stay above the answer.
     * [before]: the chat's messages before it - told to the agent when its session has not seen them (a chat from
     * before every chat was the agent's).
     */
    private suspend fun agentAnswer(before: List<ChatMessage>, user: ChatMessage, id: Long, turn: Long) {
        val chatId = _state.value.chatId ?: return fail(id, "the chat is not saved")
        val started = System.currentTimeMillis()
        // the speeds, the context and the expert cache's figures of each of the agent's requests, as a chat's
        var base: LongArray? = null
        var last: LongArray? = null
        val figures = viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            launch(start = CoroutineStart.UNDISPATCHED) { app.apiEngine.results.collect { r -> onResult(id, r) } }
            launch(start = CoroutineStart.UNDISPATCHED) {
                app.engine.events.collect { e ->
                    when (e) {
                        is EngineEvent.Stats -> {
                            val b = base ?: e.stats.also { base = it }
                            last = e.stats
                            cacheFigures(b, e.stats)
                        }
                        is EngineEvent.Encoding -> if (turn == answering) _state.update { it.copy(encodingImage = e.on) }
                        else -> Unit
                    }
                }
            }
        }
        _state.update { it.copy(prefillTps = null, decodeTps = null, ramHitRate = null, predictRate = null) }
        try {
            // the session the web chat may have begun meanwhile (it saves it with the chat)
            if (agentSession == null) agentSession = withContext(Dispatchers.IO) { chatStore.load(chatId)?.agentSession }
            val task = (if (agentSession == null) AgentTurns.earlier(before) else "") + user.text
            val files = withContext(Dispatchers.IO) { user.images.map { images.file(it) }.filter { it.isFile } }
            val sounds = withContext(Dispatchers.IO) { user.audio.map { clips.file(it) }.filter { it.isFile } }
            val films = withContext(Dispatchers.IO) {
                user.videos.map { v ->
                    xyz.tifu.geode.agent.VideoInput(v.frames.map { images.file(it) }.zip(v.seconds),
                        v.audio?.let { clips.file(it) }?.takeIf { it.isFile }, v.ms)
                }.filter { v -> v.frames.isNotEmpty() && v.frames.all { it.first.isFile } }
            }
            // (a stopped answer's last updates, while the harness ends its turn, change nothing)
            val stop = app.agent.prompt(chatId, agentSession, task, files, sounds, films) { e ->
                if (turn == answering || e is AgentEvent.Session) onAgent(id, e)
            }
            if (stop == "max_tokens") edit(id) { it.copy(error = app.getString(R.string.agent_max_tokens)) }
            // cancelled from elsewhere (an unload)
            if (stop == "cancelled") edit(id) { it.copy(error = app.getString(R.string.chat_stopped)) }
        } catch (e: CancellationException) {
            withContext(NonCancellable) { edit(id) { it.copy(error = app.getString(R.string.chat_stopped)) } }
            throw e
        } catch (e: Exception) {
            edit(id) { it.copy(error = e.message ?: e.javaClass.simpleName) }
        } finally {
            figures.cancel()
            // what it made or changed meanwhile, however it did (its file tools, a script), for the user to open
            val made = withContext(NonCancellable + Dispatchers.IO) {
                AgentTurns.changedSince(app.agent.workspace(chatId), started)
            }
            edit(id) { it.finished(made) }
            // (a stopped answer freed the chat already: a newer one may run now)
            if (turn == answering)
                _state.update { it.copy(generating = false, agentStatus = null, encodingImage = false, waiting = false) }
            persist()
            // the share of expert picks the cache served: the Benchmark tab's speed estimate uses it
            val b = base
            val l = last
            if (b != null && l != null) {
                val hits = (l.getOrNull(2) ?: 0L) - (b.getOrNull(2) ?: 0L)
                val misses = (l.getOrNull(3) ?: 0L) - (b.getOrNull(3) ?: 0L)
                if (hits + misses > 0) withContext(NonCancellable) { app.settings.setLastHitRate(hits.toFloat() / (hits + misses)) }
            }
        }
    }

    /** One of the agent's requests ended: its speeds, the context it fills, the chat's counts. */
    private fun onResult(id: Long, r: GenResult) {
        val limit = app.engine.openStats?.getOrNull(6)?.toInt()
        _state.update {
            val length = limit?.takeIf { l -> l > 0 } ?: it.contextLength
            it.copy(
                prefillTps = if (r.prefillTps > 0) r.prefillTps.toDouble() else it.prefillTps,
                decodeTps = if (r.decodeTps > 0) r.decodeTps.toDouble() else it.decodeTps,
                contextLength = length,
                contextUsed = r.contextUsed,
                contextAllocated = allocated(r.contextUsed, length),
                stats = it.stats.copy(read = it.stats.read + (r.promptTokens - r.cachedTokens),
                    cached = it.stats.cached + r.cachedTokens, generated = it.stats.generated + r.outputTokens),
            )
        }
        if (r.decodeTps > 0) edit(id) { it.copy(tokensPerSecond = r.decodeTps) }
    }

    /** The harness up for the open chat ahead of its next message (a new chat's starts with its first: its folder too). */
    private fun warmAgent() {
        if (_state.value.generating || _state.value.messages.none { it.user }) return
        val id = _state.value.chatId ?: return
        viewModelScope.launch {
            _state.update { it.copy(agentStatus = app.getString(R.string.agent_starting)) }
            runCatching { app.agent.warm(id, agentSession)?.let { agentSession = it } }
                .onFailure { e -> _state.update { it.copy(agentStatus = e.message) } }
            _state.update { if (it.agentStatus == app.getString(R.string.agent_starting)) it.copy(agentStatus = null) else it }
        }
    }

    private fun onAgent(id: Long, e: AgentEvent) {
        // the turn waits for another (the web chat's) or for the Linux; anything else: it runs
        _state.update { it.copy(waiting = e is AgentEvent.Waiting) }
        when (e) {
            is AgentEvent.Session -> agentSession = e.id
            is AgentEvent.Reading -> _state.update { it.copy(agentStatus = app.getString(R.string.agent_reading, e.tokens)) }
            is AgentEvent.Installing -> _state.update { it.copy(agentStatus = app.getString(R.string.agent_starting)) }
            is AgentEvent.ToolCall -> _state.update { it.copy(agentStatus = app.getString(R.string.agent_running_tool, e.tool)) }
            is AgentEvent.Waiting -> Unit
            else -> _state.update { it.copy(agentStatus = null) }
        }
        edit(id) { it.withAgent(e) }
    }

    /** The answer's expert picks so far (the engine's counters now against those after the prompt). */
    private fun cacheFigures(base: LongArray, now: LongArray) {
        fun d(i: Int) = (now.getOrNull(i) ?: 0L) - (base.getOrNull(i) ?: 0L)
        val hits = d(2)
        val picks = hits + d(3)
        val ahead = d(10)
        val reads = d(11)
        _state.update {
            it.copy(
                ramHitRate = if (picks > 0) (hits - ahead).toDouble() / picks else it.ramHitRate,
                predictRate = if (reads > 0) minOf(1.0, ahead.toDouble() / reads) else it.predictRate,
            )
        }
    }

    /** What the engine holds for n tokens: whole chunks, at least one, up to the limit (its own figure at the end). */
    private fun allocated(n: Int, limit: Int): Int {
        val chunk = KV_CHUNK
        return minOf(limit, maxOf(1, (n + chunk - 1) / chunk) * chunk)
    }

    private fun fail(id: Long, error: String) {
        edit(id) { it.copy(streaming = false, error = error) }
        _state.update { it.copy(generating = false, loadingModel = false, waiting = false) }
    }

    private fun edit(id: Long, f: (ChatMessage) -> ChatMessage) {
        _state.update { st -> st.copy(messages = st.messages.map { if (it.id == id) f(it) else it }) }
    }

    /** The chat is gone (its activity finished): nothing keeps the app alive for it any more. */
    override fun onCleared() {
        player.stop()
        AnswerService.stop(app)
        app.agent.stop()
    }

    companion object {
        /** The engine's KV chunk on the Adreno 7xx+ path (Model::kv_chunk_); the engine's figures replace it. */
        private const val KV_CHUNK = 4096
        /** A voice message shorter than this (samples: a quarter of a second) was a tap, not a message. */
        private const val MIN_CLIP = AudioStore.RATE / 4
        /** The agent's command that compacts the conversation (geode-agent's /compact). */
        private const val COMPACT = "/compact"
    }
}
