package xyz.tifu.geode.media

import java.io.File
import java.util.UUID

/**
 * A folder of the chats' media - their images (vision/ImageStore), their clips (AudioStore): the files by name, and
 * the rules both keep. The local server's copies sit among them, named by their content ([SERVER_PREFIX]): a client
 * sends a conversation's media again with every request, the same bytes keep one file.
 */
open class MediaFolder(protected val dir: File, private val defaultExt: String) {
    init {
        dir.mkdirs()
    }

    fun file(name: String) = File(dir, name)

    fun delete(names: Collection<String>) {
        for (n in names) file(n).delete()
    }

    /** A copy under a new name (a forked chat's own), its extension kept; null when it is gone. */
    fun copy(name: String): String? {
        val src = file(name).takeIf { it.isFile } ?: return null
        val n = UUID.randomUUID().toString() + "." + name.substringAfterLast('.', defaultExt)
        src.copyTo(file(n))
        return n
    }

    /**
     * The files no chat holds ([held]: every saved chat's) - picked and never sent, a video's frames whose reading was
     * cut off - older than [ageMs] (one being attached now is newer). The server's copies have their own rule
     * ([dropServerCopies]).
     */
    fun dropUnheld(held: Set<String>, ageMs: Long) {
        val old = System.currentTimeMillis() - ageMs
        (dir.listFiles() ?: emptyArray())
            .filter { it.isFile && !it.name.startsWith(SERVER_PREFIX) && it.name !in held && it.lastModified() < old }
            .forEach { it.delete() }
    }

    /** The local server's copies not used for a day. */
    fun dropServerCopies() {
        val old = System.currentTimeMillis() - SERVER_AGE_MS
        (dir.listFiles() ?: emptyArray()).filter { it.name.startsWith(SERVER_PREFIX) && it.lastModified() < old }
            .forEach { it.delete() }
    }

    /** The server's copy named [name] (the content's name): kept fresh when it is there already, else [make]d. */
    protected fun serverCopy(name: String, make: (String) -> Unit): String {
        val f = file(name)
        if (f.isFile) f.setLastModified(System.currentTimeMillis()) else make(name)
        return name
    }

    companion object {
        const val SERVER_PREFIX = "srv_"
        private const val SERVER_AGE_MS = 24 * 3600_000L

        /** A content name: the first 16 bytes of [bytes]' SHA-256 in hex, after [SERVER_PREFIX], with [ext]. */
        fun contentName(bytes: ByteArray, ext: String): String {
            val digest = java.security.MessageDigest.getInstance("SHA-256").digest(bytes)
            return SERVER_PREFIX + digest.take(16).joinToString("") { "%02x".format(it) } + "." + ext
        }
    }
}
