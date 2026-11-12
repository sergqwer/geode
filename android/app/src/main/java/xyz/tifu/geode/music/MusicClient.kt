package xyz.tifu.geode.music

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import xyz.tifu.geode.R

/** Where the running song is: its stage (MusicStage's ordinal) and the stage's units. */
data class MusicProgress(
    val song: String,
    val stage: Int,
    val chunk: Int,
    val chunks: Int,
    val done: Long,
    val total: Long,
    val elapsed: Double,
    /** SystemClock.elapsedRealtime() when [elapsed] was measured (one clock for both processes); 0 unknown. */
    val at: Long = 0L,
) {
    /** The time spent by [nowMs]: a long stage (the model's loading) reports nothing for minutes. */
    fun elapsedAt(nowMs: Long): Double = if (at > 0L) elapsed + (nowMs - at).coerceAtLeast(0L) / 1000.0 else elapsed

    val fraction: Float get() = if (total > 0) (done.toDouble() / total).toFloat().coerceIn(0f, 1f) else 0f
    val musicStage: MusicStage get() = MusicStage.entries.getOrElse(stage) { MusicStage.LOAD }

    companion object {
        fun stageLabel(stage: MusicStage): Int = when (stage) {
            MusicStage.LOAD -> R.string.music_stage_load
            MusicStage.SCORE -> R.string.music_stage_score
            MusicStage.CODES -> R.string.music_stage_codes
            MusicStage.PREFIX -> R.string.music_stage_prefix
            MusicStage.ACOUSTIC -> R.string.music_stage_acoustic
            MusicStage.DECODE -> R.string.music_stage_decode
            MusicStage.MASTER -> R.string.music_stage_master
            MusicStage.WRITE -> R.string.music_stage_write
            MusicStage.JOIN -> R.string.music_stage_join
        }

        /** "Composing - 120 / 750" (with the part of the song when it is made in several). */
        fun stageText(context: Context, p: MusicProgress): String {
            val name = context.getString(stageLabel(p.musicStage))
            val part = if (p.chunks > 1) " " + context.getString(R.string.music_stage_part, p.chunk + 1, p.chunks) else ""
            val units = if (p.total > 1) " - ${p.done} / ${p.total}" else ""
            return name + part + units
        }
    }
}

/** What MusicService does now: the song it makes ("" none), those queued after it, its progress. */
data class MusicState(
    val running: String = "",
    val queued: List<String> = emptyList(),
    val progress: MusicProgress? = null,
    /** The service is bound: [running] and [queued] are its own (else not known yet). */
    val connected: Boolean = false,
) {
    fun busy(id: String) = id == running || id in queued
}

/**
 * The app's side of MusicService (the :music process): it is bound while a screen wants its progress (the Music
 * screen's view model attaches), its state and the songs' ends come over a Messenger. Starting and cancelling a song
 * go through the service's intents, so they work bound or not.
 */
class MusicClient(private val context: Context) {
    private val _state = MutableStateFlow(MusicState())
    val state: StateFlow<MusicState> = _state.asStateFlow()

    private val _finished = MutableSharedFlow<String>(extraBufferCapacity = 16)
    /** A song's id when its result is written (done, failed or cancelled). */
    val finished: SharedFlow<String> = _finished.asSharedFlow()

    private val incoming = Messenger(Handler(Looper.getMainLooper()) { msg -> onMessage(msg); true })
    private var service: Messenger? = null
    private var users = 0
    private var bound = false

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val s = Messenger(binder ?: return)
            service = s
            try {
                s.send(Message.obtain(null, MusicProtocol.REGISTER).apply { replyTo = incoming })
            } catch (e: RemoteException) {
                service = null
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            // the :music process went (the system's, or a crash): the song it ran has no result - the screen shows it
            // as interrupted; the binding brings the service back for the next one
            service = null
            _state.value = MusicState()
        }
    }

    /** A screen wants the progress (main thread). */
    fun attach() {
        users++
        if (bound) return
        bound = context.bindService(Intent(context, MusicService::class.java), connection, Context.BIND_AUTO_CREATE)
    }

    fun detach() {
        users = (users - 1).coerceAtLeast(0)
        if (users > 0 || !bound) return
        runCatching { service?.send(Message.obtain(null, MusicProtocol.UNREGISTER).apply { replyTo = incoming }) }
        runCatching { context.unbindService(connection) }
        bound = false
        service = null
        _state.value = MusicState()
    }

    /** The song (SongStore.create wrote it) queued in the service; call it with the app in front. */
    fun make(id: String) {
        MusicService.make(context, id)
        // shown as queued at once (the service's state follows)
        val s = _state.value
        if (!s.busy(id)) _state.value = s.copy(queued = s.queued + id)
    }

    fun cancel(id: String = "") = MusicService.cancel(context, id)

    private fun onMessage(msg: Message) {
        val d: Bundle = msg.data ?: Bundle()
        when (msg.what) {
            MusicProtocol.STATE -> {
                val song = d.getString(MusicProtocol.K_SONG).orEmpty()
                val p = if (song.isNotEmpty() && d.containsKey(MusicProtocol.K_STAGE)) MusicProgress(song,
                    d.getInt(MusicProtocol.K_STAGE), d.getInt(MusicProtocol.K_CHUNK), d.getInt(MusicProtocol.K_CHUNKS, 1),
                    d.getLong(MusicProtocol.K_DONE), d.getLong(MusicProtocol.K_TOTAL), d.getDouble(MusicProtocol.K_ELAPSED),
                    d.getLong(MusicProtocol.K_AT, 0L))
                else null
                _state.value = MusicState(song, d.getStringArray(MusicProtocol.K_QUEUED)?.toList().orEmpty(), p, true)
            }
            MusicProtocol.DONE -> d.getString(MusicProtocol.K_SONG)?.let { _finished.tryEmit(it) }
        }
    }

    companion object {
        /** A song's notification asked for the Music screen (MainActivity; GeodeRoot opens it). */
        val openRequested = MutableStateFlow(false)

        /** The song the song-ready notification's Share asked for (MainActivity; the Music screen shares it). */
        val shareRequested = MutableStateFlow<String?>(null)
    }
}
