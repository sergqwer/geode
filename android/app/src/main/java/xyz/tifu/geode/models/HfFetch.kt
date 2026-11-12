package xyz.tifu.geode.models

import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest

/** A file that came with another hash than the one expected, twice: fetching it again will not help. */
class BadHashException(message: String) : IOException(message)

/**
 * [url]'s file into [dest], [size] bytes, checked as it comes: SHA-256 [sha256] (an LFS file), else git's blob hash
 * [gitOid] (a file kept in git); neither given: unchecked. Goes on from what [dest] holds already (those bytes hashed
 * first, [verifying] while it is done); a dropped connection or a server's error is retried after a growing pause, up
 * to [MAX_FETCH_ATTEMPTS] times. A file that comes with another hash is fetched once more, then [badHash]'s message
 * is thrown ([BadHashException]). [progress]: the bytes counted - negative when a failed attempt's are taken back.
 * Model packs (TransferService) and the reading voice (NeuralVoice) come this way.
 */
suspend fun HfApi.fetchVerified(url: String, token: String, dest: File, size: Long, sha256: String, gitOid: String = "",
                                badHash: () -> String, progress: (Long) -> Unit, verifying: (Boolean) -> Unit = {}) {
    dest.parentFile?.mkdirs()
    var attempt = 0
    var bad = 0
    while (true) {
        currentCoroutineContext().ensureActive()
        var counted = 0L
        val count = { n: Long ->
            counted += n
            progress(n)
        }
        try {
            if (dest.length() > size) dest.delete()
            val digest = newDigest(sha256, size)
            // resuming: the bytes already on disk go into the hash first
            var offset = dest.length()
            if (offset > 0) {
                verifying(true)
                dest.inputStream().use { hashInto(digest, it, offset, count) }
                verifying(false)
            }
            if (offset < size) {
                val (conn, ranged) = download(url, token, offset)
                try {
                    if (!ranged && offset > 0) {
                        // the server sent the whole file: start over
                        count(-offset)
                        offset = 0
                        digest.reset()
                        if (sha256.isEmpty()) digest.update(blobHead(size))
                    }
                    FileOutputStream(dest, offset > 0).use { out ->
                        conn.inputStream.use { input -> copyHashed(input, out, digest, size - offset, count) }
                    }
                } finally {
                    conn.disconnect()
                }
            }
            if (dest.length() != size) throw IOException("${dest.name}: incomplete")
            verifying(false)
            val want = sha256.ifEmpty { gitOid }
            if (want.isNotEmpty() && !digest.digest().toHex().equals(want, ignoreCase = true)) {
                dest.delete()
                progress(-counted)
                if (++bad >= 2) throw BadHashException(badHash())
                continue
            }
            return
        } catch (e: BadHashException) {
            throw e
        } catch (e: HfException) {
            // access denied or not found will not change by retrying
            if (e.httpCode in 400..499 && e.httpCode != 408 && e.httpCode != 429 && e.httpCode != 416) throw e
            if (e.httpCode == 416) dest.delete()
            if (++attempt >= MAX_FETCH_ATTEMPTS) throw e
        } catch (e: IOException) {
            if (++attempt >= MAX_FETCH_ATTEMPTS) throw e
        }
        progress(-counted)
        delay(minOf(30_000L, 2_000L shl attempt.coerceAtMost(4)))
    }
}

const val MAX_FETCH_ATTEMPTS = 8

/** [limit] bytes of [input] into [out] and [digest], each read counted ([onBytes]). */
internal suspend fun copyHashed(input: InputStream, out: FileOutputStream, digest: MessageDigest, limit: Long,
                                onBytes: (Long) -> Unit) {
    val buf = ByteArray(1 shl 20)
    var left = limit
    while (left > 0) {
        currentCoroutineContext().ensureActive()
        val n = input.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
        if (n < 0) break
        out.write(buf, 0, n)
        digest.update(buf, 0, n)
        left -= n
        onBytes(n.toLong())
    }
}

/** [limit] bytes of [input] into [digest], each read counted ([onBytes]). */
internal fun hashInto(digest: MessageDigest, input: InputStream, limit: Long, onBytes: (Long) -> Unit) {
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
private fun newDigest(sha256: String, size: Long): MessageDigest =
    if (sha256.isNotEmpty()) MessageDigest.getInstance("SHA-256")
    else MessageDigest.getInstance("SHA-1").apply { update(blobHead(size)) }

private fun blobHead(size: Long) = "blob $size\u0000".toByteArray()

internal fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
