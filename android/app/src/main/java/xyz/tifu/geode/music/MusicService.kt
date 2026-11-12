package xyz.tifu.geode.music

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.PowerManager
import android.os.RemoteException
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.MainActivity
import xyz.tifu.geode.R
import xyz.tifu.geode.tokenizer.Tokenizer

/** The Messenger protocol between the app (MusicClient) and the :music process (MusicService). */
object MusicProtocol {
    // app -> music
    const val REGISTER = 1     // replyTo: the client; answered with STATE
    const val UNREGISTER = 2
    // music -> app
    const val STATE = 101      // data: song (the running one, "" none), queued (ids), and the progress keys
    const val DONE = 102       // data: song, result (the engine's JSON)

    const val K_SONG = "song"
    const val K_QUEUED = "queued"
    const val K_STAGE = "stage"
    const val K_CHUNK = "chunk"
    const val K_CHUNKS = "chunks"
    const val K_DONE = "done"
    const val K_TOTAL = "total"
    const val K_ELAPSED = "elapsed"
    const val K_AT = "at"
    const val K_RESULT = "result"
}

/**
 * Makes the songs, one at a time, in a process of its own (:music): the song model's memory - its weights mapped from
 * the storage, its caches, the NPU's buffers - is never the app's, and a song the system stops takes nothing else down.
 * A special-use foreground service while it works (the user asked for the song; it takes minutes): the notification
 * shows the stage and has Cancel. The songs' folders (SongStore) are what it reads and writes; the app learns of the
 * progress over a Messenger and of the end from the song's result.json as well.
 */
class MusicService : Service() {
    private val ipc = HandlerThread("music-ipc").apply { start() }
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "music-worker") }
    private lateinit var handler: Handler
    private lateinit var messenger: Messenger
    /** On the ipc thread only. */
    private val clients = mutableListOf<Messenger>()
    private val queue = java.util.concurrent.ConcurrentLinkedDeque<String>()
    @Volatile private var running = ""
    @Volatile private var last: Bundle = Bundle()
    /**
     * Songs cancelled before they started (or as they ran). Its lock also covers the queue's changes and [running]'s
     * (a song taken off the queue is running in the same step: a cancel never finds it in neither).
     */
    private val cancelled = mutableSetOf<String>()
    private val main = Handler(android.os.Looper.getMainLooper())
    /** The last start's id (main thread): the service stops only if no start came after the check. */
    private var lastStartId = 0
    private var lastNotify = 0L
    private var lastBroadcast = 0L
    private var lastStage = -1
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        handler = Handler(ipc.looper) { msg -> onMessage(msg); true }
        messenger = Messenger(handler)
    }

    override fun onBind(intent: Intent): IBinder = messenger.binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        lastStartId = startId
        val id = intent?.getStringExtra(EXTRA_SONG).orEmpty()
        when (intent?.action) {
            ACTION_MAKE -> {
                ServiceCompat.startForeground(this, NOTIFICATION_ID, progressNotification(null),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
                val added = synchronized(cancelled) { id.isNotEmpty() && id != running && id !in queue && queue.add(id) }
                if (added) worker.execute { make() }
                handler.post { broadcastState() }
            }
            ACTION_CANCEL -> cancel(id)
            else -> idle()   // (a restart with nothing asked: nothing to keep it for)
        }
        return START_NOT_STICKY
    }

    /**
     * [id] cancelled; "": every one. A queued song is taken out (its result says cancelled); the running one stops at
     * its next check. [make] sets [running] before it clears the engine's flag and only then looks at [cancelled], so a
     * cancel that comes in between is seen by one of the two: never lost.
     */
    private fun cancel(id: String) {
        val (r, dropped) = synchronized(cancelled) {
            val r = running
            id.ifEmpty { r }.takeIf { it.isNotEmpty() }?.let { cancelled.add(it) }
            r to (if (id.isEmpty()) generateSequence { queue.pollFirst() }.toList() else listOf(id).filter { queue.remove(it) })
        }
        if (r.isNotEmpty() && (id.isEmpty() || id == r)) MusicNative.cancel()
        // the work on the worker, in order with the songs: a queued song's result, then the end when nothing is left
        worker.execute {
            val store = SongStore(this)
            for (d in dropped) if (store.find(d)?.result == null) {
                runCatching { store.cancelled(d) }
                done(d, store)
            }
            idle()
        }
        handler.post { broadcastState() }
    }

    /**
     * Nothing running or queued: the service ends (its process stays for the system to reclaim - the song's memory is
     * given back already). Decided on the main thread, where the starts come: a song asked for after the worker found
     * the queue empty is either in the queue by then or a start the stop by id does not cover.
     */
    private fun idle() {
        main.post {
            if (synchronized(cancelled) { running.isNotEmpty() || queue.isNotEmpty() }) return@post
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf(lastStartId)
        }
    }

    override fun onDestroy() {
        worker.shutdown()
        ipc.quitSafely()
        releaseWakeLock()
        super.onDestroy()
    }

    private fun onMessage(msg: Message) {
        when (msg.what) {
            MusicProtocol.REGISTER -> msg.replyTo?.let { c ->
                if (clients.none { it.binder == c.binder }) clients += c
                send(c, state())
            }
            MusicProtocol.UNREGISTER -> msg.replyTo?.let { c -> clients.removeAll { it.binder == c.binder } }
        }
    }

    private fun state(): Message = Message.obtain(null, MusicProtocol.STATE).apply {
        data = Bundle(last).apply {
            putString(MusicProtocol.K_SONG, running)
            putStringArray(MusicProtocol.K_QUEUED, queue.toTypedArray())
        }
    }

    private fun broadcastState() {
        for (c in clients.toList()) send(c, state())
    }

    private fun send(c: Messenger, m: Message) {
        try {
            c.send(m)
        } catch (e: RemoteException) {
            clients.removeAll { it.binder == c.binder }
        }
    }

    /** The next queued song, on the worker. */
    private fun make() {
        val id = synchronized(cancelled) { queue.pollFirst()?.also { running = it } } ?: return idle()
        val store = SongStore(this)
        val song = store.find(id)
        if (song == null || song.result != null) {
            running = ""
            return idle()
        }
        last = Bundle()
        MusicNative.reset()
        if (synchronized(cancelled) { cancelled.remove(id) }) {
            runCatching { store.cancelled(id) }
            running = ""
            done(id, store)
            return idle()
        }
        acquireWakeLock()
        var result = JSONObject().put("ok", false).put("error", "interrupted")
        try {
            result = MusicJob.planOf(song) ?: try {
                val pack = File(song.job.optString("pack_dir"))
                val tok = Tokenizer.load(File(pack, Tokenizer.FILE))
                if (MusicJob.goingOn(song)) goOn(song, tok) else if (MusicJob.breaking(song)) breakIn(song, tok) else {
                    val request = MusicJob.request(song, tok, companionsFor(song, store))
                    val started = SystemClock.elapsedRealtime()
                    JSONObject(MusicNative.makeSong(request) { stage, chunk, chunks, done, total, elapsed ->
                        progress(song, stage, chunk, chunks, done, total, elapsed)
                    }).put("wall", (SystemClock.elapsedRealtime() - started) / 1000.0)
                }
            } catch (e: Throwable) {
                JSONObject().put("ok", false).put("error", e.message ?: e.javaClass.simpleName)
            }
            // a cancel that came as it finished: the song is kept (it is whole)
            synchronized(cancelled) { cancelled.remove(id) }
            // (the storage full, the song deleted while it was made: nothing to write it into - not the worker's end)
            runCatching { store.finish(id, result) }
        } finally {
            running = ""
            last = Bundle()
            releaseWakeLock()
        }
        runCatching { notifyEnd(song, result) }
        done(id, store)
        idle()
    }

    /**
     * A song going on (MusicJob.goOn): its score written on from its head - up to three tries, a seed apart, while the
     * score the model writes cannot be read (a cancel or an engine's failure ends it at once) - then its length from
     * the written bars at the song's tempo, then the song sung on and drawn. One clock for the progress and the result;
     * the result also holds what the score took ("score_seconds", "score_tries") and the new song's bars ("grid").
     */
    private fun goOn(song: Song, tok: Tokenizer): JSONObject {
        val started = SystemClock.elapsedRealtime()
        val since = { (SystemClock.elapsedRealtime() - started) / 1000.0 }
        val (plan, clock, old) = MusicJob.goOnParts(song)
        val frames = song.job.getJSONObject("edit").getInt("frames")
        var abc: IntArray? = null
        var abcEos = true
        var written = ""
        var problem = ""
        var tries = 0
        while (abc == null && tries < GoOn.SCORE_TRIES) {
            val base = since()
            val r = JSONObject(MusicNative.makeSong(MusicJob.goOnScore(song, tok, tries)) { stage, chunk, chunks, done, total, elapsed ->
                progress(song, stage, chunk, chunks, done, total, base + elapsed)
            })
            ++tries
            if (!r.optBoolean("ok")) return r.put("wall", since())
            val a = r.optJSONArray("abc") ?: return r.put("ok", false).put("error", "the score came back empty").put("wall", since())
            val ids = IntArray(a.length()) { a.getInt(it) }
            val text = tok.decode(ids)
            try {
                AbcScore.parse(text.trim())
                abc = ids
                abcEos = r.optBoolean("abc_eos", true)
                written = text.trim()
            } catch (e: IllegalArgumentException) {
                problem = e.message ?: e.javaClass.simpleName
            }
        }
        val ids = abc ?: return JSONObject().put("ok", false)
            .put("error", "The model wrote ${GoOn.SCORE_TRIES} scores for the new part and none of them can be read: $problem")
            .put("wall", since())
        val length = GoOn.extensionFrames(old, clock, frames, plan, written)
        val scored = since()
        val r = JSONObject(MusicNative.makeSong(MusicJob.goOnSong(song, tok, ids, abcEos, length)) { stage, chunk, chunks, done, total, elapsed ->
            progress(song, stage, chunk, chunks, done, total, scored + elapsed)
        })
        r.put("score_seconds", scored).put("score_tries", tries).put("wall", since())
        r.optJSONObject("times")?.let { t -> t.put("score", t.optDouble("score", 0.0) + scored).put("all", since()) }
        if (r.optBoolean("ok")) runCatching {
            val (lines, sections) = GoOn.linesAfter(clock, plan, written)
            r.put("grid", BarGrid.fixed(lines, sections, clock.speed).json())
        }
        return r
    }

    /**
     * A break put in (MusicJob.breakIn): the model writes on from the score's head - up to three tries, a seed apart,
     * while what it writes does not give the break's bars (ScoreEdit.interluded refuses it; a cancel or an engine's
     * failure ends it at once) - then the break's length from those bars at the song's tempo, then the break sung in
     * and drawn. One clock; the result also holds what the score took and the new song's bars ("grid").
     */
    private fun breakIn(song: Song, tok: Tokenizer): JSONObject {
        val started = SystemClock.elapsedRealtime()
        val since = { (SystemClock.elapsedRealtime() - started) / 1000.0 }
        val (plan, old, grid) = MusicJob.breakParts(song)
        var full: String? = null
        var problem = ""
        var tries = 0
        while (full == null && tries < GoOn.SCORE_TRIES) {
            val base = since()
            val r = JSONObject(MusicNative.makeSong(MusicJob.breakScore(song, tok, tries)) { stage, chunk, chunks, done, total, elapsed ->
                progress(song, stage, chunk, chunks, done, total, base + elapsed)
            })
            ++tries
            if (!r.optBoolean("ok")) return r.put("wall", since())
            val a = r.optJSONArray("abc") ?: return r.put("ok", false).put("error", "the score came back empty").put("wall", since())
            val written = tok.decode(IntArray(a.length()) { a.getInt(it) }).trim()
            try {
                full = ScoreEdit.interluded(old, written, plan.bar, plan.bars, plan.seam)
            } catch (e: IllegalArgumentException) {
                problem = e.message ?: e.javaClass.simpleName
            }
        }
        val score = full ?: return JSONObject().put("ok", false)
            .put("error", "The model wrote ${GoOn.SCORE_TRIES} scores for the break and none of them gives its bars: $problem")
            .put("wall", since())
        val length = TrackEdit.breakFrames(plan, score, grid)
        val scored = since()
        val r = JSONObject(MusicNative.makeSong(MusicJob.breakSongRequest(song, tok, tok.encode(score), length)) { stage, chunk, chunks, done, total, elapsed ->
            progress(song, stage, chunk, chunks, done, total, scored + elapsed)
        })
        r.put("score_seconds", scored).put("score_tries", tries).put("wall", since())
        r.optJSONObject("times")?.let { t -> t.put("score", t.optDouble("score", 0.0) + scored).put("all", since()) }
        if (r.optBoolean("ok")) runCatching {
            val count = r.getJSONObject("insert").getInt("count")
            r.put("grid", grid.afterBreak(score, plan.bar, plan.bars, count).json())
        }
        return r
    }

    /** The queued takes [song]'s run makes the score and codes of too (MusicJob.companionOf; theirs not made yet):
     *  at most three, the engine taking as many as the free memory holds - each then skips its own AR. */
    private fun companionsFor(song: Song, store: SongStore): List<Song> {
        if (MusicJob.arOf(song) != null) return emptyList()   // (this one's own made already: its run has no AR)
        val ids = synchronized(cancelled) { queue.toList().filter { it !in cancelled } }
        return ids.asSequence().mapNotNull { store.find(it) }
            .filter { it.result == null && MusicJob.arOf(it) == null && MusicJob.planOf(it) == null && MusicJob.companionOf(song, it) }
            .take(3).toList()
    }

    /** The song's end to the clients. */
    private fun done(id: String, store: SongStore) {
        val r = store.find(id)?.result?.toString().orEmpty()
        handler.post {
            for (c in clients.toList()) send(c, Message.obtain(null, MusicProtocol.DONE).apply {
                data = Bundle().apply {
                    putString(MusicProtocol.K_SONG, id)
                    putString(MusicProtocol.K_RESULT, r)
                }
            })
            broadcastState()
        }
    }

    private fun progress(song: Song, stage: Int, chunk: Int, chunks: Int, done: Long, total: Long, elapsed: Double) {
        val now = SystemClock.elapsedRealtime()
        last = Bundle().apply {
            putInt(MusicProtocol.K_STAGE, stage)
            putInt(MusicProtocol.K_CHUNK, chunk)
            putInt(MusicProtocol.K_CHUNKS, chunks)
            putLong(MusicProtocol.K_DONE, done)
            putLong(MusicProtocol.K_TOTAL, total)
            putDouble(MusicProtocol.K_ELAPSED, elapsed)
            putLong(MusicProtocol.K_AT, now)
        }
        // to the clients a few times a second and at every stage's change (the engine calls this at every token)
        if (stage != lastStage || now - lastBroadcast >= 250) {
            lastStage = stage
            lastBroadcast = now
            handler.post { broadcastState() }
        }
        if (now - lastNotify < 1000) return
        lastNotify = now
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, progressNotification(MusicProgress(song.id, stage, chunk, chunks, done, total, elapsed), song.title))
    }

    private fun progressNotification(p: MusicProgress?, title: String = ""): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java)
            .setAction(MainActivity.ACTION_OPEN_MUSIC), PendingIntent.FLAG_IMMUTABLE)
        val cancel = PendingIntent.getService(this, 1, Intent(this, MusicService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE)
        val b = NotificationCompat.Builder(this, GeodeApp.CHANNEL_MUSIC)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(title.ifEmpty { getString(R.string.music_notif_title) })
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(0, getString(R.string.music_cancel), cancel)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
        if (p == null) b.setProgress(0, 0, true)
        else {
            b.setContentText(MusicProgress.stageText(this, p))
            if (p.total > 0) b.setProgress(1000, (p.fraction * 1000).toInt(), false) else b.setProgress(0, 0, true)
        }
        return b.build()
    }

    private fun notifyEnd(song: Song, result: JSONObject) {
        if (result.optBoolean("cancelled")) return
        val open = PendingIntent.getActivity(this, 2, Intent(this, MainActivity::class.java)
            .setAction(MainActivity.ACTION_OPEN_MUSIC), PendingIntent.FLAG_IMMUTABLE)
        val ok = result.optBoolean("ok")
        val b = NotificationCompat.Builder(this, GeodeApp.CHANNEL_MUSIC)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(getString(when {
                !ok -> R.string.music_notif_failed
                song.plan -> R.string.music_notif_score_ready
                else -> R.string.music_notif_ready
            }))
            .setContentText(if (ok) song.title else result.optString("error"))
            .setContentIntent(open)
            .setAutoCancel(true)
        // straight to the share sheet (Viber, Telegram, WhatsApp...): the app opens on the song and shares it
        if (ok && !song.plan) b.addAction(0, getString(R.string.music_share), PendingIntent.getActivity(this, 3,
            Intent(this, MainActivity::class.java).setAction(MainActivity.ACTION_SHARE_SONG)
                .putExtra(MainActivity.EXTRA_SONG, song.id),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT))
        val n = b.build()
        getSystemService(NotificationManager::class.java).notify(END_NOTIFICATION_ID, n)
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        // a song takes minutes, its acoustic stage hours on a slow phone: held for it alone, given back at its end
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "geode:music").apply { acquire(6 * 3600_000L) }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    companion object {
        private const val NOTIFICATION_ID = 4
        const val END_NOTIFICATION_ID = 5
        private const val ACTION_MAKE = "xyz.tifu.geode.music.MAKE"
        private const val ACTION_CANCEL = "xyz.tifu.geode.music.CANCEL"
        private const val EXTRA_SONG = "song"

        /** The song [id] (its folder written by SongStore.create) queued; call it while the app is in front. */
        fun make(context: Context, id: String) {
            ContextCompat.startForegroundService(context,
                Intent(context, MusicService::class.java).setAction(ACTION_MAKE).putExtra(EXTRA_SONG, id))
        }

        /** [id] cancelled; "": every one. */
        fun cancel(context: Context, id: String = "") {
            runCatching {
                context.startService(Intent(context, MusicService::class.java).setAction(ACTION_CANCEL).putExtra(EXTRA_SONG, id))
            }
        }
    }
}
