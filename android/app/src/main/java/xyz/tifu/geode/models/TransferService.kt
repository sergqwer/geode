package xyz.tifu.geode.models

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.MainActivity
import xyz.tifu.geode.R
import xyz.tifu.geode.util.formatBytes

/**
 * Downloads a model from Hugging Face or copies one from a folder the user picked, as a dataSync foreground service
 * with a progress notification. One transfer at a time; its state is in [Transfers].
 *
 * Downloads resume (HTTP Range) after a dropped connection, after the app is killed, and after a reboot (the partial
 * directory remembers its source). Every LFS file is checked against its SHA-256 from Hugging Face, files kept in git
 * against their git blob hash; a model directory appears only when every file is complete and checked.
 */
class TransferService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var job: Job? = null
    private lateinit var store: ModelStore
    private var wakeLock: PowerManager.WakeLock? = null
    private var lastNotify = 0L
    /** The model the running transfer writes, once known. */
    @Volatile private var currentId = ""
    /** The folder the running import reads (its persisted permission), or null. */
    @Volatile private var importUri: String? = null

    override fun onCreate() {
        super.onCreate()
        store = ModelStore(this)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CANCEL -> {
                job?.cancel()
                return START_NOT_STICKY
            }
            ACTION_START -> {
                if (job?.isActive == true) return START_NOT_STICKY
                val source = sourceOf(intent) ?: return START_NOT_STICKY
                val title = intent.getStringExtra(EXTRA_TITLE).orEmpty()
                startForeground(NOTIFICATION_ID, notification(TransferState("", title, TransferPhase.PREPARE)),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
                acquireWakeLock()
                job = scope.launch { run(source, title) }
            }
        }
        return START_NOT_STICKY
    }

    // Android 15 limits dataSync services to 6 h a day; the partial files stay, so the user can resume later
    override fun onTimeout(startId: Int, fgsType: Int) {
        job?.cancel(CancellationException("timeout"))
        // stopped now: the system gives a few seconds, and a read waiting on the network (up to its 60 s timeout) or a
        // big file being hashed would not see the cancel in time (the app would be stopped for it)
        stopForeground(STOP_FOREGROUND_DETACH)
        stopSelf()
    }

    override fun onDestroy() {
        scope.cancel()
        releaseWakeLock()
        super.onDestroy()
    }

    private suspend fun run(source: TransferSource, title: String) {
        currentId = ""
        try {
            val id = when (source) {
                is TransferSource.Hf -> download(source, title)
                is TransferSource.Folder -> import(source, title)
            }
            val app = application as GeodeApp
            if (app.settings.flow.first().selectedModel.isEmpty()) app.settings.setSelectedModel(id)
            publish(TransferState(id, title, TransferPhase.DONE), force = true)
        } catch (e: CancellationException) {
            val timedOut = e.message == "timeout"
            // an explicit cancel throws the partial files away; the system's time limit keeps them
            if (!timedOut && currentId.isNotEmpty()) {
                store.deletePartial(currentId)
                releaseUriPermissions()
            }
            publish(TransferState(currentId, title, if (timedOut) TransferPhase.FAILED else TransferPhase.CANCELLED,
                error = if (timedOut) getString(R.string.models_transfer_timeout) else null), force = true)
        } catch (e: Exception) {
            publish(TransferState(currentId, title, TransferPhase.FAILED, error = e.message ?: e.javaClass.simpleName),
                force = true)
        } finally {
            releaseWakeLock()
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }
    }

    // ---- Hugging Face ----

    private suspend fun download(src: TransferSource.Hf, title: String): String {
        val id = ModelStore.idFor(src.repo, src.packPath)
        currentId = id
        check(!store.dir(id).exists()) { getString(R.string.models_already_installed, id) }
        publish(TransferState(id, title, TransferPhase.PREPARE))
        val token = (application as GeodeApp).settings.flow.first().hfToken
        val pack = HfApi.findPacks(src.repo, src.revision, token).firstOrNull { it.path == src.packPath }
            ?: throw IOException(getString(R.string.models_no_pack, src.repo))
        val part = store.partialDir(id).apply { mkdirs() }
        src.write(File(part, ModelStore.SOURCE))
        val verified = verifiedSet(part)

        // the manifest first: a repository that is not a Geode model fails before any big download
        val files = pack.files.sortedWith(compareBy({ it.path != PackManifest.FILE }, { it.size }))
        val total = pack.bytes
        val have = files.sumOf { f -> File(part, f.path).length().coerceAtMost(f.size) }
        val free = store.freeBytes()
        if (free < total - have + SPACE_MARGIN)
            throw IOException(getString(R.string.models_no_space, formatBytes(total - have + SPACE_MARGIN),
                formatBytes(free)))

        val progress = Progress(id, title, TransferPhase.DOWNLOAD, total)
        for (f in files) {
            val dest = File(part, f.path)
            if (f.path in verified && dest.length() == f.size) {
                progress.add(f.size)
                continue
            }
            downloadFile(src, token, f, dest, progress)
            if (f.path == PackManifest.FILE) {
                val m = PackManifest.parse(dest.readText())
                    ?: throw IOException(getString(R.string.models_not_geode))
                if (!m.runnable) throw IOException(getString(R.string.models_not_geode))
            }
            File(part, VERIFIED).appendText(f.path + "\n")
        }
        return finish(id, part)
    }

    private suspend fun downloadFile(src: TransferSource.Hf, token: String, f: HfFile, dest: File, p: Progress) {
        dest.parentFile?.mkdirs()
        val url = HfApi.resolveUrl(src.repo, src.revision, if (src.packPath.isEmpty()) f.path else "${src.packPath}/${f.path}")
        var attempt = 0
        var badHash = 0
        while (true) {
            currentCoroutineContext().ensureActive()
            val base = p.done
            try {
                if (dest.length() > f.size) dest.delete()
                val digest = newDigest(f)
                // resuming: the bytes already on disk go into the hash first
                var offset = dest.length()
                if (offset > 0) {
                    p.phase(TransferPhase.VERIFY)
                    dest.inputStream().use { hashInto(digest, it, offset) { n -> p.add(n) } }
                    p.phase(TransferPhase.DOWNLOAD)
                }
                if (offset < f.size) {
                    val (conn, ranged) = HfApi.download(url, token, offset)
                    try {
                        if (!ranged && offset > 0) {
                            // the server sent the whole file: start over
                            p.add(-offset)
                            offset = 0
                            digest.reset()
                            if (f.sha256.isEmpty()) digest.update("blob ${f.size}\u0000".toByteArray())
                        }
                        FileOutputStream(dest, offset > 0).use { out ->
                            conn.inputStream.use { input -> copy(input, out, digest, f.size - offset, p) }
                        }
                    } finally {
                        conn.disconnect()
                    }
                }
                if (dest.length() != f.size) throw IOException("${f.path}: incomplete")
                p.phase(TransferPhase.DOWNLOAD)
                val got = digest.digest().toHex()
                val want = f.sha256.ifEmpty { f.gitOid }
                if (want.isNotEmpty() && !got.equals(want, ignoreCase = true)) {
                    dest.delete()
                    p.add(-(p.done - base))
                    if (++badHash >= 2) throw IOException(getString(R.string.models_bad_hash, f.path))
                    continue
                }
                return
            } catch (e: HfException) {
                // access denied or not found will not change by retrying
                if (e.httpCode in 400..499 && e.httpCode != 408 && e.httpCode != 429 && e.httpCode != 416) throw e
                if (e.httpCode == 416) dest.delete()
                if (++attempt >= MAX_ATTEMPTS) throw e
            } catch (e: IOException) {
                if (++attempt >= MAX_ATTEMPTS) throw e
            }
            p.add(-(p.done - base))
            delay(minOf(30_000L, 2_000L shl attempt.coerceAtMost(4)))
        }
    }

    // ---- a folder on the phone ----

    private suspend fun import(src: TransferSource.Folder, title: String): String {
        importUri = src.uri
        val tree = DocumentFile.fromTreeUri(this, Uri.parse(src.uri))
            ?: throw IOException(getString(R.string.models_folder_unreadable))
        // geode.json in the picked folder itself, or in one of its subfolders
        val packDir = listOf(tree).plus(tree.listFiles().filter { it.isDirectory })
            .firstOrNull { it.findFile(PackManifest.FILE)?.isFile == true }
            ?: throw IOException(getString(R.string.models_no_manifest))
        val manifestDoc = packDir.findFile(PackManifest.FILE)!!
        val manifestText = contentResolver.openInputStream(manifestDoc.uri)!!.use { it.readBytes().decodeToString() }
        val manifest = PackManifest.parse(manifestText) ?: throw IOException(getString(R.string.models_not_geode))
        if (!manifest.runnable) throw IOException(getString(R.string.models_not_geode))

        val id = ModelStore.idFor(manifest.name.ifEmpty { packDir.name ?: "model" }, "")
        currentId = id
        check(!store.dir(id).exists()) { getString(R.string.models_already_installed, id) }
        val part = store.partialDir(id).apply { mkdirs() }
        src.write(File(part, ModelStore.SOURCE))
        val verified = verifiedSet(part)

        val docs = manifest.files.map { f ->
            val d = packDir.findFile(f.name) ?: throw IOException(getString(R.string.models_missing_file, f.name))
            if (d.length() != f.size) throw IOException(getString(R.string.models_wrong_size, f.name))
            f to d
        }
        val todo = docs.filter { (f, _) -> f.name !in verified }
        val need = if (src.move) todo.maxOfOrNull { it.first.size } ?: 0 else todo.sumOf { it.first.size }
        val free = store.freeBytes()
        if (free < need + SPACE_MARGIN)
            throw IOException(getString(R.string.models_no_space, formatBytes(need + SPACE_MARGIN), formatBytes(free)))

        File(part, PackManifest.FILE).writeText(manifestText)
        val progress = Progress(id, title, TransferPhase.COPY, manifest.bytes)
        for ((f, doc) in docs) {
            if (f.name in verified) {
                progress.add(f.size)
                continue
            }
            val dest = File(part, f.name)
            dest.parentFile?.mkdirs()
            val digest = MessageDigest.getInstance("SHA-256")
            val base = progress.done
            contentResolver.openInputStream(doc.uri)!!.use { input ->
                FileOutputStream(dest).use { out -> copy(input, out, digest, f.size, progress) }
            }
            // a source that ended early (a file without a hash in geode.json would pass, and a move delete it)
            if (dest.length() != f.size) {
                dest.delete()
                progress.add(-(progress.done - base))
                throw IOException(getString(R.string.models_wrong_size, f.name))
            }
            if (f.sha256.isNotEmpty() && !digest.digest().toHex().equals(f.sha256, ignoreCase = true)) {
                dest.delete()
                progress.add(-(progress.done - base))
                throw IOException(getString(R.string.models_bad_hash, f.name))
            }
            File(part, VERIFIED).appendText(f.name + "\n")
            if (src.move) doc.delete()
        }
        if (src.move) manifestDoc.delete()
        return finish(id, part)
    }

    // ---- shared ----

    private fun finish(id: String, part: File): String {
        File(part, ModelStore.SOURCE).delete()
        File(part, VERIFIED).delete()
        val dir = store.dir(id)
        if (!part.renameTo(dir)) throw IOException("cannot move the model into place")
        releaseUriPermissions()
        return id
    }

    /**
     * A folder import keeps a persisted permission (to resume after the app is killed) only while it runs: that one
     * given up - not the others the app holds (the agent's phone folders: PhoneFolders), and not this one either when
     * the same folder is also given to the agent (one grant serves both).
     */
    private fun releaseUriPermissions() = runCatching {
        val uri = importUri ?: return@runCatching
        importUri = null
        if ((application as GeodeApp).phoneFolders.folders.value.any { it.uri == uri }) return@runCatching
        val grant = contentResolver.persistedUriPermissions.firstOrNull { it.uri.toString() == uri } ?: return@runCatching
        contentResolver.releasePersistableUriPermission(grant.uri,
            (if (grant.isReadPermission) Intent.FLAG_GRANT_READ_URI_PERMISSION else 0) or
                (if (grant.isWritePermission) Intent.FLAG_GRANT_WRITE_URI_PERMISSION else 0))
    }

    private suspend fun copy(input: InputStream, out: FileOutputStream, digest: MessageDigest, limit: Long,
                             p: Progress) {
        val buf = ByteArray(1 shl 20)
        var left = limit
        while (left > 0) {
            currentCoroutineContext().ensureActive()
            val n = input.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
            if (n < 0) break
            out.write(buf, 0, n)
            digest.update(buf, 0, n)
            left -= n
            p.add(n.toLong())
        }
    }

    private fun hashInto(digest: MessageDigest, input: InputStream, limit: Long, onBytes: (Long) -> Unit) {
        val buf = ByteArray(1 shl 20)
        var left = limit
        while (left > 0) {
            val n = input.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
            if (n < 0) break
            digest.update(buf, 0, n)
            left -= n
            onBytes(n.toLong())
        }
    }

    /** SHA-256 for LFS files; git's blob hash (SHA-1 of "blob <size>\0" + content) for files kept in git. */
    private fun newDigest(f: HfFile): MessageDigest =
        if (f.sha256.isNotEmpty()) MessageDigest.getInstance("SHA-256")
        else MessageDigest.getInstance("SHA-1").apply { update("blob ${f.size}\u0000".toByteArray()) }

    private fun verifiedSet(part: File): Set<String> =
        File(part, VERIFIED).takeIf { it.isFile }?.readLines()?.filter { it.isNotBlank() }?.toSet() ?: emptySet()

    /** Byte counting with a speed estimate, published at most every half second. */
    private inner class Progress(val id: String, val title: String, var phase: TransferPhase, val total: Long) {
        var done = 0L
        private var windowStart = SystemClock.elapsedRealtime()
        private var windowBytes = 0L
        private var speed = 0L

        fun phase(p: TransferPhase) {
            phase = p
        }

        fun add(n: Long) {
            done += n
            if (n > 0 && phase != TransferPhase.VERIFY) windowBytes += n
            val now = SystemClock.elapsedRealtime()
            if (now - windowStart >= 1000) {
                speed = windowBytes * 1000 / (now - windowStart)
                windowStart = now
                windowBytes = 0
            }
            publish(TransferState(id, title, phase, done, total, speed))
        }
    }

    private fun publish(s: TransferState, force: Boolean = false) {
        Transfers.set(s)
        val now = SystemClock.elapsedRealtime()
        if (!force && now - lastNotify < 500) return
        lastNotify = now
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(s))
    }

    private fun notification(s: TransferState): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        val b = NotificationCompat.Builder(this, GeodeApp.CHANNEL_TRANSFERS)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
        when (s.phase) {
            TransferPhase.DONE -> b.setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setContentTitle(getString(R.string.models_transfer_done, s.title)).setAutoCancel(true)
            TransferPhase.FAILED -> b.setSmallIcon(android.R.drawable.stat_notify_error)
                .setContentTitle(getString(R.string.models_transfer_failed, s.error.orEmpty()))
                .setContentText(s.title).setAutoCancel(true)
            TransferPhase.CANCELLED -> b.setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setContentTitle(s.title).setAutoCancel(true)
            else -> {
                val cancel = PendingIntent.getService(this, 1,
                    Intent(this, TransferService::class.java).setAction(ACTION_CANCEL), PendingIntent.FLAG_IMMUTABLE)
                b.setSmallIcon(android.R.drawable.stat_sys_download)
                    .setContentTitle(getString(when (s.phase) {
                        TransferPhase.COPY -> R.string.models_transfer_import
                        TransferPhase.VERIFY -> R.string.models_transfer_verify
                        else -> R.string.models_transfer_download
                    }, s.title))
                    .setContentText(if (s.total > 0) getString(R.string.models_transfer_progress,
                        formatBytes(s.done), formatBytes(s.total)) else null)
                    .setProgress(1000, (s.fraction * 1000).toInt(), s.total <= 0)
                    .setOngoing(true)
                    .addAction(0, getString(R.string.models_transfer_cancel), cancel)
            }
        }
        return b.build()
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "geode:transfer")
            .apply { acquire(6 * 3600_000L) }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    companion object {
        private const val ACTION_START = "xyz.tifu.geode.transfer.START"
        private const val ACTION_CANCEL = "xyz.tifu.geode.transfer.CANCEL"
        private const val EXTRA_KIND = "kind"
        private const val EXTRA_REPO = "repo"
        private const val EXTRA_REVISION = "revision"
        private const val EXTRA_PACK = "pack"
        private const val EXTRA_URI = "uri"
        private const val EXTRA_MOVE = "move"
        private const val EXTRA_TITLE = "title"
        private const val NOTIFICATION_ID = 1
        private const val MAX_ATTEMPTS = 8
        private const val SPACE_MARGIN = 512L shl 20
        private const val VERIFIED = ".verified"

        fun start(context: Context, source: TransferSource, title: String) {
            val i = Intent(context, TransferService::class.java).setAction(ACTION_START).putExtra(EXTRA_TITLE, title)
            when (source) {
                is TransferSource.Hf -> i.putExtra(EXTRA_KIND, "hf").putExtra(EXTRA_REPO, source.repo)
                    .putExtra(EXTRA_REVISION, source.revision).putExtra(EXTRA_PACK, source.packPath)
                is TransferSource.Folder -> i.putExtra(EXTRA_KIND, "folder").putExtra(EXTRA_URI, source.uri)
                    .putExtra(EXTRA_MOVE, source.move)
            }
            context.startForegroundService(i)
        }

        fun cancel(context: Context) {
            context.startService(Intent(context, TransferService::class.java).setAction(ACTION_CANCEL))
        }

        private fun sourceOf(i: Intent): TransferSource? {
            return when (i.getStringExtra(EXTRA_KIND)) {
                "hf" -> TransferSource.Hf(i.getStringExtra(EXTRA_REPO) ?: return null,
                    i.getStringExtra(EXTRA_REVISION) ?: "main", i.getStringExtra(EXTRA_PACK).orEmpty())
                "folder" -> TransferSource.Folder(i.getStringExtra(EXTRA_URI) ?: return null,
                    i.getBooleanExtra(EXTRA_MOVE, false))
                else -> null
            }
        }
    }
}

private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
