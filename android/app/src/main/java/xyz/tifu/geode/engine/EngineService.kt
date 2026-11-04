package xyz.tifu.geode.engine

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import java.io.File
import java.util.concurrent.Executors
import xyz.tifu.geode.engine.EngineProtocol as P

/**
 * Runs the engine in the :engine process. The UI binds it with BIND_IMPORTANT, so while the chat is on screen this
 * process has the UI's priority (PLAN.md P6.1). IPC messages arrive on one thread; open/generate/close run on a
 * worker, so STOP is handled at once while a generation is running.
 */
class EngineService : Service() {
    private val ipc = HandlerThread("geode-ipc").apply { start() }
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "geode-worker") }
    @Volatile private var handle = 0L
    /**
     * The GENERATE and PREFILL requests are numbered as they come (on the ipc thread, in order): a STOP stops every one
     * that came before it - the one running, its media still to encode, those still queued behind it - and none that
     * comes after. (A flag the next request cleared lost a STOP that came before that request started: the answer
     * then ran to its end.)
     */
    private var queued = 0L
    /** The last request a STOP stopped (ipc thread writes, the worker and the engine's thread read). */
    @Volatile private var stopUpTo = 0L
    /** The request the worker runs now (0: none); changed and looked at under [runLock]: a late re-stop that saw a
     *  stopped request cannot land on the next one starting meanwhile. */
    @Volatile private var running = 0L
    private val runLock = Any()
    /** Taken by a stop from the ipc thread and by the worker letting the model go (stopEngine, closeModel). */
    private val handleLock = Any()
    private lateinit var ipcHandler: Handler
    private lateinit var messenger: Messenger

    override fun onCreate() {
        super.onCreate()
        debugEnv()
        ipcHandler = Handler(ipc.looper) { msg -> onMessage(msg); true }
        messenger = Messenger(ipcHandler)
    }

    /**
     * A debuggable build only: the engine's measuring switches (GEODE_*, read when a model opens) from
     * files/engine.env, a KEY=VALUE a line - set from adb (run-as) for an A/B on a phone, then this process killed so
     * the next one reads them. A release build never looks.
     */
    private fun debugEnv() {
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val f = File(filesDir, "engine.env")
        if (!f.isFile) return
        for (line in runCatching { f.readLines() }.getOrDefault(emptyList())) {
            val (k, v) = line.trim().split('=', limit = 2).takeIf { it.size == 2 } ?: continue
            if (k.matches(Regex("GEODE_[A-Z0-9_]+"))) runCatching { android.system.Os.setenv(k, v, true) }
        }
    }

    override fun onBind(intent: Intent): IBinder = messenger.binder

    override fun onDestroy() {
        worker.execute { closeModel() }
        worker.shutdown()
        ipc.quitSafely()
        super.onDestroy()
    }

    private fun onMessage(msg: Message) {
        val reply = msg.replyTo
        val data = msg.data
        when (msg.what) {
            P.OPEN -> worker.execute { open(data, reply) }
            P.GENERATE -> {
                val seq = ++queued
                worker.execute { generate(data, reply, seq) }
            }
            P.STOP -> {
                stopUpTo = queued
                stopRunning()
            }
            P.RESET -> worker.execute { handle.takeIf { it != 0L }?.let { GeodeNative.reset(it) } }
            P.CLOSE -> worker.execute { closeModel() }
            P.PREPARE -> worker.execute { prepare(data) }
            // on the worker like GENERATE: never while a generation runs
            P.SAVE -> worker.execute {
                state(reply, P.SAVED) { h, err ->
                    GeodeNative.stateSave(h, data.getString(P.K_PATH).orEmpty(), data.getBoolean(P.K_APPEND), err)
                }
            }
            P.LOAD -> worker.execute {
                state(reply, P.LOADED) { h, err -> GeodeNative.stateLoad(h, data.getString(P.K_PATH).orEmpty(), err) }
            }
            P.PREFILL -> {
                val seq = ++queued
                worker.execute {
                    synchronized(runLock) { running = seq }
                    try {
                        state(reply, P.PREFILLED) { h, err ->
                            val prompt = PromptFiles.take(data)
                            val rc = if (stopped(seq)) GeodeNative.E_STOPPED
                            else if (prompt == null) GeodeNative.E_IO.also { err[0] = "the prompt's file could not be read" }
                            else GeodeNative.prefill(h, prompt, err)
                            if (rc == GeodeNative.OK) GeodeNative.stats(h)[5].toInt() else -rc
                        }
                    } finally {
                        synchronized(runLock) { running = 0L }
                    }
                }
            }
            P.RESERVE -> worker.execute { reserve(data.getLong(P.K_BYTES), reply) }
            P.PROFILE -> worker.execute {
                val h = handle
                send(reply, Message.obtain(null, P.PROFILED, if (h != 0L) 1 else 0, 0).apply {
                    this.data = Bundle().apply {
                        putString(P.K_JSON, if (h != 0L) GeodeNative.profile(h) else "")
                        putString(P.K_ERROR, if (h != 0L) null else "no model is open")
                        if (h != 0L) putLongArray(P.K_STATS, GeodeNative.stats(h))
                    }
                })
            }
            P.EXPERTS -> worker.execute {
                val h = handle
                val err = arrayOfNulls<String>(1)
                val rc = if (h == 0L) {
                    err[0] = "no model is open"
                    GeodeNative.E_ARG
                } else GeodeNative.setExperts(h, data.getInt(P.K_EXPERTS), err)
                send(reply, Message.obtain(null, P.EXPERTS_SET, if (rc == GeodeNative.OK) 1 else 0, 0).apply {
                    this.data = Bundle().apply {
                        putString(P.K_ERROR, err[0])
                        if (h != 0L) putLongArray(P.K_STATS, GeodeNative.stats(h))
                    }
                })
            }
            P.BENCH -> worker.execute {
                closeModel()
                val json = GeodeNative.bench(data.getString(P.K_WHAT).orEmpty(), data.getString(P.K_ARG).orEmpty())
                send(reply, Message.obtain(null, P.BENCH_DONE).apply {
                    this.data = Bundle().apply { putString(P.K_JSON, json) }
                })
            }
        }
    }

    private fun open(data: Bundle, reply: Messenger?) {
        closeModel()
        val err = arrayOfNulls<String>(1)
        handle = GeodeNative.open(
            data.getString(P.K_DIR).orEmpty(), data.getLong(P.K_BUDGET), data.getInt(P.K_THREADS),
            data.getInt(P.K_IO_THREADS), data.getInt(P.K_HELPERS), data.getInt(P.K_CONTEXT),
            data.getBoolean(P.K_PREFETCH, true), data.getBoolean(P.K_GPU), data.getFloat(P.K_GPU_SHARE),
            data.getInt(P.K_EXPERTS), data.getFloat(P.K_ROUTE_PRIOR), data.getInt(P.K_TAIL_SKIP),
            data.getBoolean(P.K_NPU, true), data.getInt(P.K_MODE), data.getInt(P.K_OFF), err
        )
        send(reply, Message.obtain(null, P.OPENED, if (handle != 0L) 1 else 0, 0).apply {
            this.data = Bundle().apply {
                putString(P.K_ERROR, err[0])
                // the context's limit as the budget allows it, among them
                if (handle != 0L) {
                    putLongArray(P.K_STATS, GeodeNative.stats(handle))
                    // how it runs: the GPU path's and the NPU's state, the threads and the speculation (geode_profile)
                    val lines = GeodeNative.profile(handle).lines()
                    fun line(prefix: String) = lines.firstOrNull { it.startsWith(prefix) }?.removePrefix(prefix)
                    putString(P.K_GPU_STATUS, line("GPU: "))
                    putString(P.K_NPU_STATUS, line("NPU: "))
                    putString(P.K_RUN, line("run: "))
                }
            }
        })
    }

    private fun stopped(seq: Long) = seq <= stopUpTo

    /**
     * The engine told to stop the request running now. geode_generate clears the engine's flag as it begins, so a STOP
     * that comes in the moment between the worker's last look and that would be lost: told again a little later while
     * a stopped request still runs.
     */
    private fun stopRunning() {
        stopEngine()
        for (ms in RESTOP_MS) ipcHandler.postDelayed({
            synchronized(runLock) {
                val r = running
                if (r != 0L && stopped(r)) stopEngine()
            }
        }, ms)
    }

    /**
     * geode_stop on the open model, from the ipc thread while the worker may be closing it: under [handleLock], which
     * closeModel takes to let the handle go - a stop never reaches a model already freed.
     */
    private fun stopEngine() = synchronized(handleLock) {
        handle.takeIf { it != 0L }?.let { GeodeNative.stop(it) }
    }

    private fun generate(data: Bundle, reply: Messenger?, seq: Long) {
        synchronized(runLock) { running = seq }
        try {
            generateNow(data, reply, seq)
        } finally {
            synchronized(runLock) { running = 0L }
        }
    }

    private fun generateNow(data: Bundle, reply: Messenger?, seq: Long) {
        val prompt = PromptFiles.take(data)
        val h = handle
        if (prompt == null) {
            send(reply, Message.obtain(null, P.DONE, GeodeNative.E_IO, 0).apply {
                this.data = Bundle().apply { putString(P.K_ERROR, "the prompt's file could not be read") }
            })
            return
        }
        if (h == 0L) {
            send(reply, Message.obtain(null, P.DONE, -1, 0).apply {
                this.data = Bundle().apply { putString(P.K_ERROR, "no model is open") }
            })
            return
        }
        val err = arrayOfNulls<String>(1)
        // stopped before it began: nothing read, nothing changed
        if (stopped(seq)) {
            send(reply, Message.obtain(null, P.DONE, GeodeNative.E_STOPPED, 0).apply {
                this.data = Bundle().apply {
                    putString(P.K_ERROR, "stopped")
                    putLongArray(P.K_STATS, GeodeNative.stats(h))
                }
            })
            return
        }
        // the prompt's media first, in order: each read from the UI's file and encoded (seconds each)
        val images = data.getStringArray(P.K_IMAGES) ?: emptyArray()
        if (images.isNotEmpty()) {
            val ws = data.getIntArray(P.K_IMAGE_W) ?: IntArray(0)
            val hs = data.getIntArray(P.K_IMAGE_H) ?: IntArray(0)
            val kinds = data.getIntArray(P.K_MEDIA_KIND) ?: IntArray(0)
            send(reply, Message.obtain(null, P.ENCODING, 1, 0))
            var code = GeodeNative.OK
            for (i in images.indices) {
                if (stopped(seq)) {
                    code = GeodeNative.E_STOPPED
                    break
                }
                val r = when (kinds.getOrElse(i) { EngineMedia.IMAGE }) {
                    EngineMedia.AUDIO -> {
                        val pcm = runCatching { xyz.tifu.geode.media.AudioStore.read(File(images[i])) }.getOrNull()
                        if (pcm == null) -GeodeNative.E_IO else GeodeNative.addAudio(h, pcm, err)
                    }
                    else -> {
                        val rgb = runCatching { File(images[i]).readBytes() }.getOrNull()
                        if (rgb == null || i >= ws.size || i >= hs.size) -GeodeNative.E_IO
                        // the features beside the RGB file (ImageStore's cache): a conversation sent again skips the
                        // encoder
                        else if (kinds.getOrElse(i) { EngineMedia.IMAGE } == EngineMedia.FRAME)
                            GeodeNative.addVideoFrame(h, rgb, ws[i], hs[i], featureFile(images[i]), err)
                        else GeodeNative.addImage(h, rgb, ws[i], hs[i], featureFile(images[i]), err)
                    }
                }
                if (r < 0) {
                    code = -r
                    if (err[0] == null) err[0] = "cannot read the image or the sound"
                    break
                }
            }
            if (code == GeodeNative.OK && stopped(seq)) code = GeodeNative.E_STOPPED
            send(reply, Message.obtain(null, P.ENCODING, 0, 0))
            if (code != GeodeNative.OK) {
                // the images added so far would wait for pads that are not coming: the conversation goes
                GeodeNative.reset(h)
                send(reply, Message.obtain(null, P.DONE, code, 0).apply {
                    this.data = Bundle().apply {
                        putString(P.K_ERROR, err[0])
                        putLongArray(P.K_STATS, GeodeNative.stats(h))
                    }
                })
                return
            }
        }
        var n = 0
        val rc = GeodeNative.generate(
            h, prompt, data.getFloat(P.K_TEMPERATURE), data.getFloat(P.K_TOP_P),
            data.getInt(P.K_TOP_K), data.getFloat(P.K_PRESENCE), data.getLong(P.K_SEED), data.getInt(P.K_MAX_TOKENS),
            data.getIntArray(P.K_STOPS) ?: IntArray(0),
            TokenListener { token ->
                // the cache's figures as the answer goes (the token callback runs on the engine's own thread)
                if (n++ % STATS_EVERY == 0) {
                    send(reply, Message.obtain(null, P.STATS).apply {
                        this.data = Bundle().apply { putLongArray(P.K_STATS, GeodeNative.stats(h)) }
                    })
                }
                // true: stop (the UI is gone, or a STOP came for this request - the engine's own flag may have missed it)
                !send(reply, Message.obtain(null, P.TOKEN, token, 0)) || stopped(seq)
            },
            err,
        )
        send(reply, Message.obtain(null, P.DONE, rc, 0).apply {
            this.data = Bundle().apply {
                putString(P.K_ERROR, err[0])
                putLongArray(P.K_STATS, GeodeNative.stats(h))
            }
        })
    }

    /** A conversation's state saved, loaded or a beginning read ([op]: its length in tokens, or -(a geode.h code)),
     *  answered with [what]: arg1 that result, the error and the stats. */
    private fun state(reply: Messenger?, what: Int, op: (Long, Array<String?>) -> Int) {
        val h = handle
        val err = arrayOfNulls<String>(1)
        val r = if (h == 0L) {
            err[0] = "no model is open"
            -GeodeNative.E_ARG
        } else op(h, err)
        send(reply, Message.obtain(null, what, r, 0).apply {
            this.data = Bundle().apply {
                putString(P.K_ERROR, err[0])
                if (h != 0L) putLongArray(P.K_STATS, GeodeNative.stats(h))
            }
        })
    }

    /** The expert cache gives [bytes] back (0: takes it again): RESERVED with what it holds now (and why less). */
    private fun reserve(bytes: Long, reply: Messenger?) {
        val h = handle
        val err = arrayOfNulls<String>(1)
        val held = if (h == 0L) {
            err[0] = "no model is open"
            -GeodeNative.E_ARG.toLong()
        } else GeodeNative.reserveMemory(h, bytes, err)
        send(reply, Message.obtain(null, P.RESERVED, if (held >= 0) 1 else 0, 0).apply {
            this.data = Bundle().apply {
                putLong(P.K_BYTES, if (held >= 0) held else 0)
                putString(P.K_ERROR, err[0])
                if (h != 0L) putLongArray(P.K_STATS, GeodeNative.stats(h))
            }
        })
    }

    /** An attached image encoded ahead into its feature file (a failure only costs the encode at GENERATE). */
    private fun prepare(data: Bundle) {
        val h = handle
        val path = data.getStringArray(P.K_IMAGES)?.firstOrNull() ?: return
        val w = data.getIntArray(P.K_IMAGE_W)?.firstOrNull() ?: return
        val hh = data.getIntArray(P.K_IMAGE_H)?.firstOrNull() ?: return
        if (h == 0L) return
        val rgb = runCatching { File(path).readBytes() }.getOrNull() ?: return
        GeodeNative.cacheImage(h, rgb, w, hh, featureFile(path))
    }

    private fun closeModel() {
        val h = synchronized(handleLock) {
            handle.also { handle = 0L }
        }
        if (h != 0L) GeodeNative.close(h)
    }

    private companion object {
        /** When a stopped request still runs, the engine is told again (ms after the STOP). */
        private val RESTOP_MS = longArrayOf(50, 250, 1000)
        const val STATS_EVERY = 8

        /** An image's feature file: beside its raw RGB in ImageStore's cache. */
        fun featureFile(rgbPath: String) = rgbPath.removeSuffix(".rgb") + ".feat"
    }

    /** False when the UI is gone (then generation stops: nobody reads the tokens). */
    private fun send(to: Messenger?, msg: Message): Boolean = try {
        to?.send(msg)
        to != null
    } catch (e: RemoteException) {
        false
    }
}
