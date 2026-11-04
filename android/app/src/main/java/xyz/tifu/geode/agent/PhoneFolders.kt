package xyz.tifu.geode.agent

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.DocumentsContract.Document
import android.webkit.MimeTypeMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.bool
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.num
import xyz.tifu.geode.util.obj
import xyz.tifu.geode.util.str
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * A folder of the phone the user gave the agent: the document tree ([uri]) the app holds a persisted permission for,
 * seen by every chat's agent at /phone/[name]; [write]: the agent's changes go back to the phone, else read-only.
 */
data class PhoneFolder(val name: String, val uri: String, val label: String, val write: Boolean, val canWrite: Boolean)

/** A folder's copy as the agent screen shows it. */
data class FolderState(val files: Int = 0, val bytes: Long = 0, val syncing: Boolean = false, val skipped: Int = 0,
                       val error: String? = null)

/**
 * The phone's folders the user gives the agent (the tester: "Общая папка только на чтение"; the user: give and take
 * back the agent's folders, change what it may do in them).
 *
 * The agent's Linux cannot open a document tree (Android hands one out as content URIs, not paths), so each folder
 * is kept as a copy in the app's files (files/agent/phone/<name>), bound into every chat's Linux at /phone/<name>.
 * The copy is brought up to date before every turn; a read-and-write folder gets the agent's changes back when the
 * turn ends - written, created, deleted - each checked against what the phone held at the last copy: a file the user
 * changed on the phone meanwhile is kept and the agent's version written beside it ("name (Geode).ext"), a file the
 * user changed is never deleted. A read-only folder is never written to: its copy is made read-only, the agent's file
 * tools refuse it, and whatever is changed in it anyway is put back as the phone has it at the next turn.
 *
 * What was copied is remembered per file (files/agent/phone/.<name>.json): the phone's document, its time and size,
 * and the copy's time and size then - the change on either side is told by them.
 */
class PhoneFolders(private val context: Context) {
    private val resolver get() = context.contentResolver
    private val file = File(context.filesDir, "agent/folders.json")
    private val root = File(context.filesDir, "agent/phone")
    private val lock = Mutex()

    private val _folders = MutableStateFlow(load())
    val folders: StateFlow<List<PhoneFolder>> = _folders.asStateFlow()
    private val _states = MutableStateFlow<Map<String, FolderState>>(emptyMap())
    val states: StateFlow<Map<String, FolderState>> = _states.asStateFlow()

    /** Counts the changes: a harness started before the last one starts again (it is told its folders at start). */
    @Volatile
    var version = 0
        private set

    /** The folder's copy (its /phone/<name>). */
    fun copyOf(name: String) = File(root, name)

    /** The folders as the harness is told them (GEODE_FOLDERS). */
    fun forHarness(): String = Json.write(_folders.value.map { f ->
        linkedMapOf("guest" to "$GUEST/${f.name}", "host" to copyOf(f.name).path, "write" to f.write)
    })

    /** Its copies bound into a Linux started by the app (the harness in Node, a build without geode-agent). */
    fun binds(): List<Pair<File, String>> = _folders.value.map { f -> copyOf(f.name).apply { mkdirs() } to "$GUEST/${f.name}" }

    // ---- given, changed, taken back

    /**
     * A tree the user picked (OpenDocumentTree): its permission kept, read-and-write when the phone gives it;
     * [write]: what the agent may do in it. Copied in at once. Null: it could not be kept.
     */
    suspend fun add(uri: Uri, write: Boolean): PhoneFolder? = withContext(Dispatchers.IO) {
        val rw = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
        runCatching { resolver.takePersistableUriPermission(uri, rw) }.recoverCatching {
            resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }.getOrElse { return@withContext null }
        val canWrite = resolver.persistedUriPermissions.any { it.uri == uri && it.isWritePermission }
        val treeId = runCatching { DocumentsContract.getTreeDocumentId(uri) }.getOrElse { return@withContext null }
        val shown = runCatching { androidx.documentfile.provider.DocumentFile.fromTreeUri(context, uri)?.name }.getOrNull()
        // the phone's storage: "primary:Documents/Notes" -> Documents/Notes; another provider's: the folder's name
        val label = if (uri.authority == "com.android.externalstorage.documents")
            treeId.substringAfter(':').trimEnd('/').ifEmpty { shown ?: "/" } else shown ?: treeId
        val existing = _folders.value.find { it.uri == uri.toString() }
        if (existing != null) return@withContext existing
        val f = PhoneFolder(uniqueName(shown ?: label.substringAfterLast('/')), uri.toString(), label, write && canWrite,
            canWrite)
        lock.withLock {
            store(_folders.value + f)
            copyOf(f.name).mkdirs()
        }
        sync(f, out = false)
        f
    }

    /** What the agent may do in [name]: read only, or read and write. */
    suspend fun setWrite(name: String, write: Boolean) = withContext(Dispatchers.IO) {
        val f = _folders.value.find { it.name == name } ?: return@withContext
        if (f.write == write || (write && !f.canWrite)) return@withContext
        // now read-only: what the agent changed is copied back at its turn's end, so the copy is the phone's;
        // now writable: the copy put back as the phone has it first - nothing changed while it was read-only goes out
        if (write) sync(f, out = false)
        lock.withLock { store(_folders.value.map { if (it.name == name) it.copy(write = write) else it }) }
        lock.withLock { writable(copyOf(name), true) }
        if (!write) sync(f.copy(write = false), out = false)
    }

    /** [name] taken back: its permission given up, its copy deleted (the phone's folder stays as it is). */
    suspend fun remove(name: String) = withContext(Dispatchers.IO) {
        val f = _folders.value.find { it.name == name } ?: return@withContext
        lock.withLock {
            store(_folders.value.filter { it.name != name })
            runCatching {
                resolver.releasePersistableUriPermission(Uri.parse(f.uri),
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or (if (f.canWrite) Intent.FLAG_GRANT_WRITE_URI_PERMISSION else 0))
            }
            val copy = copyOf(name)
            writable(copy, true)
            copy.deleteRecursively()
            manifest(name).delete()
            _states.update { it - name }
        }
    }

    // ---- a turn's syncs

    /** Before a turn: the agent's changes not yet back on the phone (a turn that ended without) go out, then every
     *  copy brought up to date. */
    suspend fun beforeTurn() = withContext(Dispatchers.IO) {
        for (f in _folders.value) {
            if (f.write) sync(f, out = true)
            sync(f, out = false)
        }
    }

    /** After a turn: what the agent changed in the read-and-write folders goes back to the phone. */
    suspend fun afterTurn() = withContext(Dispatchers.IO) {
        for (f in _folders.value) if (f.write) sync(f, out = true)
    }

    /** The states shown for every folder (the screen opened): their copies as they are now. */
    suspend fun refresh() = withContext(Dispatchers.IO) {
        for (f in _folders.value) if (_states.value[f.name] == null) {
            val m = loadManifest(f.name)
            set(f.name) { it.copy(files = m.files.size, bytes = m.files.values.sumOf { k -> k.size }) }
        }
    }

    private suspend fun sync(f: PhoneFolder, out: Boolean) = lock.withLock {
        if (_folders.value.none { it.name == f.name }) return@withLock
        set(f.name) { it.copy(syncing = true) }
        val t0 = android.os.SystemClock.elapsedRealtime()
        val m = loadManifest(f.name)
        try {
            val tree = Uri.parse(f.uri)
            if (resolver.persistedUriPermissions.none { it.uri == tree && it.isReadPermission }) throw IOException("no permission")
            val local = if (out) local(f.name, m) else null
            // the agent changed nothing in the copy since it was last in step with the phone: nothing to send, and the
            // phone's tree is not walked for it (most turns leave most folders as they were)
            if (out && (local == null || !changed(local, m))) {
                set(f.name) { it.copy(syncing = false) }
                return@withLock
            }
            val phone = list(tree)
            val failed = try {
                if (out) pushOut(tree, f.name, m, phone, local!!) else copyIn(f, m, phone)
            } finally {
                // what was done is remembered even when the rest failed (else a file copied in would look the agent's)
                saveManifest(f.name, m)
            }
            set(f.name) {
                FolderState(m.files.size, m.files.values.sumOf { k -> k.size }, false,
                    phone.skipped + (if (out) 0 else failed.size), if (out) failed.firstOrNull() else null)
            }
            android.util.Log.d(TAG, "${f.name}: ${if (out) "out" else "in"}: ${m.files.size} files, " +
                "${failed.size} failed, ${android.os.SystemClock.elapsedRealtime() - t0} ms")
        } catch (e: Exception) {
            android.util.Log.w(TAG, "${f.name}: ${if (out) "out" else "in"}: $e")
            set(f.name) { it.copy(syncing = false, error = e.message ?: e.javaClass.simpleName) }
        }
    }

    // ---- the phone's tree

    private class Doc(val id: String, val time: Long, val size: Long)

    /** A tree as it is on the phone: its files and folders by their path in it, and how many were left out. */
    private class Listing(val files: MutableMap<String, Doc>, val dirs: MutableMap<String, String>, var skipped: Int)

    /**
     * The tree walked folder by folder (one query each: name, type, size, time), each folder's entries by name - the
     * same files kept whenever the tree is bigger than a copy holds -, at most [MAX_FILES] files and [MAX_BYTES] in
     * all; the rest left out, and once a copy is full the walk ends (a tree of tens of thousands of folders would take
     * minutes at every turn).
     */
    private fun list(tree: Uri): Listing {
        val out = Listing(LinkedHashMap(), LinkedHashMap(), 0)
        val rootId = DocumentsContract.getTreeDocumentId(tree)
        out.dirs[""] = rootId
        val queue = ArrayDeque<Pair<String, String>>().apply { add("" to rootId) }
        var bytes = 0L
        var full = false
        while (queue.isNotEmpty() && !full) {
            val (rel, id) = queue.removeFirst()
            val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, id)
            class Row(val id: String, val name: String, val dir: Boolean, val size: Long, val time: Long)
            val rows = ArrayList<Row>()
            resolver.query(children, COLUMNS, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(1)?.takeIf { fits(it) } ?: continue
                    rows += Row(c.getString(0), name, c.getString(2) == Document.MIME_TYPE_DIR,
                        if (c.isNull(3)) 0L else c.getLong(3), if (c.isNull(4)) 0L else c.getLong(4))
                }
            } ?: throw IOException("the folder cannot be read")
            rows.sortBy { it.name }
            for (r in rows) {
                val path = if (rel.isEmpty()) r.name else "$rel/${r.name}"
                if (r.dir) {
                    out.dirs[path] = r.id
                    queue.add(path to r.id)
                } else if (full || out.files.size >= MAX_FILES || bytes + r.size > MAX_BYTES) {
                    out.skipped++
                    full = full || out.files.size >= MAX_FILES || bytes >= MAX_BYTES
                } else {
                    bytes += r.size
                    out.files[path] = Doc(r.id, r.time, r.size)
                }
            }
        }
        return out
    }

    /** A name the copy can hold as it is. */
    private fun fits(name: String) = name.isNotEmpty() && name != "." && name != ".." && '/' in name == false &&
        '\u0000' !in name

    // ---- phone -> copy

    /** The phone's tree into the copy; the paths that could not be copied (left for the next time). */
    private fun copyIn(f: PhoneFolder, m: Manifest, phone: Listing): List<String> {
        val copy = copyOf(f.name)
        val failed = ArrayList<String>()
        writable(copy, true)
        try {
            copy.mkdirs()
            for (d in phone.dirs.keys) if (d.isNotEmpty()) File(copy, d).mkdirs()
            // gone from the phone: gone from the copy (a read-and-write folder's change not yet out stays)
            for ((p, k) in m.files.entries.toList()) {
                if (p in phone.files) continue
                val c = File(copy, p)
                if (!f.write || !c.isFile || k.unchanged(c)) c.delete()
                if (!c.exists()) m.files.remove(p)
            }
            for ((p, d) in phone.files) {
                val k = m.files[p]
                val c = File(copy, p)
                if (k != null && k.doc == d.id && k.docTime == d.time && k.docSize == d.size && c.isFile && k.unchanged(c)) continue
                // in a read-and-write folder, what the agent changed or made and has not gone out yet (its out failed)
                // is never written over - the next out sends it, or writes it beside the phone's when that changed too
                if (f.write && c.isFile && (k == null || !k.unchanged(c))) continue
                // one file that cannot be read (a cloud file not on the phone, a provider's error) stops no other
                try {
                    copyFile(DocumentsContract.buildDocumentUriUsingTree(Uri.parse(f.uri), d.id), c, d.time)
                    m.files[p] = Known(d.id, d.time, d.size, c.lastModified(), c.length())
                } catch (e: Exception) {
                    android.util.Log.w(TAG, "${f.name}: $p not copied: $e")
                    failed += p
                }
            }
            // what the phone does not have: in a read-only copy, put back as the phone has it (nothing else in it)
            if (!f.write) {
                copy.walkBottomUp().filter { it != copy }.forEach { c ->
                    val p = c.relativeTo(copy).invariantSeparatorsPath
                    if (if (c.isDirectory) p !in phone.dirs else p !in m.files) c.deleteRecursively()
                }
            } else {
                for (d in m.dirs.toList()) if (d !in phone.dirs) File(copy, d).let { if (it.list()?.isEmpty() == true) it.delete() }
            }
            m.dirs.clear()
            m.dirs.addAll(phone.dirs.keys.filter { it.isNotEmpty() })
        } finally {
            if (!f.write) writable(copy, false)
        }
        return failed
    }

    private fun copyFile(from: Uri, to: File, time: Long) {
        to.parentFile?.mkdirs()
        val tmp = File(to.parentFile, ".${to.name}.geode-copy")
        try {
            resolver.openInputStream(from)?.use { i -> FileOutputStream(tmp).use { o -> i.copyTo(o, BUFFER) } }
                ?: throw IOException("cannot read ${to.name}")
            if (time > 0) tmp.setLastModified(time)
            if (!tmp.renameTo(to)) throw IOException("cannot write ${to.name}")
        } finally {
            tmp.delete()
        }
    }

    // ---- copy -> phone (read-and-write folders)

    /**
     * A copy as it is now: its files, each with the file that holds its bytes ([files]: a link stands for the file of
     * the copy it points at - proot makes a hard link two links to a hidden file, ".l2s.<name>0001"); the path of
     * everything in it, links and the Linux's own files too ([present]: what is not there was deleted); its folders.
     */
    private class Local(val files: LinkedHashMap<String, File>, val present: HashSet<String>, val dirs: ArrayList<String>)

    private fun local(name: String, m: Manifest): Local? {
        val copy = copyOf(name)
        // no copy at all: nothing is taken for deleted
        if (!copy.isDirectory) return null
        val base = runCatching { copy.canonicalFile }.getOrNull() ?: return null
        val l = Local(LinkedHashMap(), HashSet(), ArrayList())
        copy.walkTopDown().onEnter { it == copy || !link(it) }.forEach { c ->
            if (c == copy) return@forEach
            val p = c.relativeTo(copy).invariantSeparatorsPath
            val isLink = link(c)
            if (!isLink && c.isDirectory) {
                if (!own(c.name) || p in m.dirs) l.dirs += p
                return@forEach
            }
            l.present += p
            // the app's or the Linux's own (a copy being written, a hard link's data): not the user's - unless the
            // phone has a file of that name (it was copied in)
            if (own(c.name) && p !in m.files) return@forEach
            val data = if (isLink) target(base, name, c) else c.takeIf { it.isFile }
            if (data != null) l.files[p] = data
        }
        return l
    }

    private fun link(f: File) = java.nio.file.Files.isSymbolicLink(f.toPath())

    /**
     * The file of the copy a link in it points at, or null (a link out of the copy, to a folder, or to nothing: the
     * link stays the agent's own - nothing of the app's or its Linux's ever goes to the phone through one). A target
     * written as the Linux sees it (/phone/<name>/...) is the copy's.
     */
    private fun target(base: File, name: String, link: File): File? = runCatching {
        val t = java.nio.file.Files.readSymbolicLink(link.toPath()).toString()
        val guest = "$GUEST/$name/"
        val f = when {
            t.startsWith(guest) -> File(base, t.removePrefix(guest))
            t.startsWith("/") -> File(t)
            else -> File(link.parentFile, t)
        }.canonicalFile
        f.takeIf { it.isFile && it.path.startsWith(base.path + File.separator) }
    }.getOrNull()

    /** Whether the copy differs from what was last in step with the phone: a file changed, made or deleted, a folder
     *  made or deleted. */
    private fun changed(l: Local, m: Manifest): Boolean {
        for ((p, data) in l.files) {
            val k = m.files[p] ?: return true
            if (!k.unchanged(data)) return true
        }
        if (m.files.keys.any { it !in l.present }) return true
        if (l.dirs.any { it !in m.dirs }) return true
        return m.dirs.any { it !in l.dirs && it !in l.present }
    }

    /** The agent's changes in the copy to the phone; what could not go (left for the next out). */
    private fun pushOut(tree: Uri, name: String, m: Manifest, phone: Listing, l: Local): List<String> {
        val failed = ArrayList<String>()
        fun docUri(id: String) = DocumentsContract.buildDocumentUriUsingTree(tree, id)
        // the folders the agent made, outermost first
        for (d in l.dirs) {
            if (d in phone.dirs) continue
            val parent = phone.dirs[d.substringBeforeLast('/', "")] ?: continue
            val made = runCatching {
                DocumentsContract.createDocument(resolver, docUri(parent), Document.MIME_TYPE_DIR, d.substringAfterLast('/'))
            }.getOrNull() ?: continue
            phone.dirs[d] = DocumentsContract.getDocumentId(made)
        }
        // the files it wrote or made
        for ((p, c) in l.files) {
            val k = m.files[p]
            if (k != null && k.unchanged(c)) continue
            val doc = phone.files[p]
            val parent = phone.dirs[p.substringBeforeLast('/', "")] ?: continue
            val fileName = p.substringAfterLast('/')
            try {
                if (doc != null && k == null && doc.size == c.length() && same(docUri(doc.id), c)) {
                    // the phone's file as it is (its record lost): nothing to send
                    m.files[p] = Known(doc.id, doc.time, doc.size, c.lastModified(), c.length())
                    continue
                }
                val theirs = doc != null && (k == null || doc.id != k.doc || doc.time != k.docTime || doc.size != k.docSize)
                if (theirs) {
                    // the user changed it on the phone meanwhile (or made one of that name): theirs stays, the agent's
                    // goes beside it - once: the copy's version taken as sent, the phone's (an unknown one) copied in next
                    create(docUri(parent), beside(fileName), c) ?: continue
                    m.files[p] = Known(doc!!.id, -1, -1, c.lastModified(), c.length())
                    continue
                }
                val id = if (doc != null) {
                    write(docUri(doc.id), c)
                    doc.id
                } else {
                    create(docUri(parent), fileName, c) ?: continue
                }
                val (time, size) = stat(docUri(id)) ?: continue
                m.files[p] = Known(id, time, size, c.lastModified(), c.length())
            } catch (e: Exception) {
                // one file that cannot go stops no other (it stays changed: the next out tries again)
                android.util.Log.w(TAG, "$name: $p not sent: $e")
                failed += "$p: ${e.message ?: e.javaClass.simpleName}"
            }
        }
        // the files it deleted: deleted on the phone unless the user changed them there since (anything still in the
        // copy under its name - a link, a file of the Linux's own - is not deleted)
        for ((p, k) in m.files.entries.toList()) {
            if (p in l.present) continue
            val doc = phone.files[p]
            if (doc == null) {
                m.files.remove(p)
            } else if (doc.id == k.doc && doc.time == k.docTime && doc.size == k.docSize) {
                if (runCatching { DocumentsContract.deleteDocument(resolver, docUri(doc.id)) }.getOrDefault(false)) {
                    m.files.remove(p)
                    phone.files.remove(p)
                }
            }
        }
        // the folders it deleted, innermost first, when nothing is left in them on the phone
        for (d in m.dirs.sortedByDescending { it.count { ch -> ch == '/' } }) {
            if (d in l.dirs || d in l.present) continue
            val id = phone.dirs[d] ?: continue
            val empty = resolver.query(DocumentsContract.buildChildDocumentsUriUsingTree(tree, id),
                arrayOf(Document.COLUMN_DOCUMENT_ID), null, null, null)?.use { it.count == 0 } ?: false
            if (empty && runCatching { DocumentsContract.deleteDocument(resolver, docUri(id)) }.getOrDefault(false)) {
                phone.dirs.remove(d)
            }
        }
        m.dirs.clear()
        m.dirs.addAll(l.dirs.filter { it in phone.dirs })
        return failed
    }

    /** A new document [name] in [parent] holding [from]'s bytes; its id. */
    private fun create(parent: Uri, name: String, from: File): String? {
        val mime = MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase())
            ?: "application/octet-stream"
        val doc = DocumentsContract.createDocument(resolver, parent, mime, name) ?: return null
        write(doc, from)
        return DocumentsContract.getDocumentId(doc)
    }

    private fun write(doc: Uri, from: File) {
        val o = runCatching { resolver.openOutputStream(doc, "wt") }.getOrNull() ?: resolver.openOutputStream(doc, "w")
            ?: throw IOException("cannot write ${from.name}")
        o.use { out -> from.inputStream().use { it.copyTo(out, BUFFER) } }
    }

    /** Whether a document holds exactly [file]'s bytes. */
    private fun same(doc: Uri, file: File): Boolean = runCatching {
        val a = resolver.openInputStream(doc) ?: return false
        a.use { file.inputStream().use { b -> sameBytes(a, b) } }
    }.getOrDefault(false)

    private fun sameBytes(a: java.io.InputStream, b: java.io.InputStream): Boolean {
        val x = ByteArray(BUFFER)
        val y = ByteArray(BUFFER)
        while (true) {
            val n = fill(a, x)
            if (fill(b, y) != n) return false
            for (i in 0 until n) if (x[i] != y[i]) return false
            if (n < BUFFER) return true
        }
    }

    /** [buf] filled from [s] as far as it goes (less only at its end); the bytes read. */
    private fun fill(s: java.io.InputStream, buf: ByteArray): Int {
        var n = 0
        while (n < buf.size) {
            val r = s.read(buf, n, buf.size - n)
            if (r < 0) break
            n += r
        }
        return n
    }

    private fun stat(doc: Uri): Pair<Long, Long>? =
        resolver.query(doc, arrayOf(Document.COLUMN_LAST_MODIFIED, Document.COLUMN_SIZE), null, null, null)?.use { c ->
            if (!c.moveToFirst()) null
            else (if (c.isNull(0)) 0L else c.getLong(0)) to (if (c.isNull(1)) 0L else c.getLong(1))
        }

    /** "report.txt" -> "report (Geode).txt" */
    private fun beside(name: String): String {
        val dot = name.lastIndexOf('.').takeIf { it > 0 } ?: return "$name (Geode)"
        return name.substring(0, dot) + " (Geode)" + name.substring(dot)
    }

    /** What is the app's or the Linux's own in a copy, not the user's: a copy being written, proot's hard links. */
    private fun own(name: String) = name.endsWith(".geode-copy") || name.startsWith(".l2s.") ||
        TMP.matches(name)

    // ---- what was copied

    /** A file as it was copied: the phone's document, its time and size, the copy's time and size then. */
    private class Known(val doc: String, val docTime: Long, val docSize: Long, val time: Long, val size: Long) {
        fun unchanged(c: File) = c.lastModified() == time && c.length() == size
    }

    private class Manifest(val files: MutableMap<String, Known>, val dirs: MutableSet<String>)

    private fun manifest(name: String) = File(root, ".$name.json")

    @Suppress("UNCHECKED_CAST")
    private fun loadManifest(name: String): Manifest = runCatching {
        val o = Json.parse(manifest(name).readText()) as Map<String, Any?>
        val files = LinkedHashMap<String, Known>()
        o.obj("files")?.forEach { (p, v) ->
            val a = v as? List<Any?> ?: return@forEach
            if (a.size == 5) files[p] = Known(a[0].toString(), (a[1] as Number).toLong(), (a[2] as Number).toLong(),
                (a[3] as Number).toLong(), (a[4] as Number).toLong())
        }
        Manifest(files, o.list("dirs").orEmpty().map { it.toString() }.toMutableSet())
    }.getOrElse { Manifest(LinkedHashMap(), LinkedHashSet()) }

    private fun saveManifest(name: String, m: Manifest) {
        root.mkdirs()
        val tmp = File(root, ".$name.json.tmp")
        tmp.writeText(Json.write(linkedMapOf(
            "files" to m.files.mapValues { (_, k) -> listOf(k.doc, k.docTime, k.docSize, k.time, k.size) },
            "dirs" to m.dirs.toList())))
        tmp.renameTo(manifest(name))
    }

    // ---- the list

    private fun store(list: List<PhoneFolder>) {
        file.parentFile?.mkdirs()
        val tmp = File(file.path + ".tmp")
        FileOutputStream(tmp).use {
            it.write(Json.write(list.map { f ->
                linkedMapOf("name" to f.name, "uri" to f.uri, "label" to f.label, "write" to f.write, "canWrite" to f.canWrite)
            }).toByteArray())
            it.fd.sync()
        }
        tmp.renameTo(file)
        _folders.value = list
        version++
    }

    @Suppress("UNCHECKED_CAST")
    private fun load(): List<PhoneFolder> = runCatching {
        (Json.parse(file.readText()) as List<Map<String, Any?>>).mapNotNull { o ->
            val name = o.str("name")?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
            PhoneFolder(name, o.str("uri") ?: return@mapNotNull null, o.str("label").orEmpty(), o.bool("write") == true,
                o.bool("canWrite") == true)
        }
    }.getOrDefault(emptyList())

    /** A folder name for /phone: letters, digits, `.`, `-`, `_` (any other character `_`), not one already given. */
    private fun uniqueName(label: String): String {
        val base = label.map { if (it.isLetterOrDigit() || it in "._-") it else '_' }.joinToString("")
            .trim('.', '_').take(48).ifEmpty { "folder" }
        val taken = _folders.value.map { it.name }.toSet()
        var name = base
        var n = 2
        while (name in taken) name = "$base-${n++}"
        return name
    }

    private fun set(name: String, change: (FolderState) -> FolderState) =
        _states.update { it + (name to change(it[name] ?: FolderState())) }

    /** A copy's files and folders made read-only (or writable again): nothing in it written, made or deleted. */
    private fun writable(dir: File, on: Boolean) {
        if (!dir.exists()) return
        // writable: outermost first (a folder opened before its contents); read-only: innermost first
        val all = if (on) dir.walkTopDown() else dir.walkBottomUp()
        // writable for the app alone; read-only for everyone
        all.forEach { it.setWritable(on, on) }
    }

    companion object {
        private const val TAG = "GeodeFolders"
        /** Where the folders are in the agent's Linux. */
        const val GUEST = "/phone"
        /** A folder's copy at most this big (the rest left out, and said so). */
        const val MAX_FILES = 10_000
        const val MAX_BYTES = 1L shl 30
        private const val BUFFER = 64 * 1024
        private val COLUMNS = arrayOf(Document.COLUMN_DOCUMENT_ID, Document.COLUMN_DISPLAY_NAME, Document.COLUMN_MIME_TYPE,
            Document.COLUMN_SIZE, Document.COLUMN_LAST_MODIFIED)
        /** geode-agent's write_atomic: ".name.<8 hex>.tmp" while a file is written. */
        private val TMP = Regex("^\\..+\\.[0-9a-f]{8}\\.tmp$")
    }
}
