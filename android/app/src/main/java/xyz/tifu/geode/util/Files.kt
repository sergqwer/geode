package xyz.tifu.geode.util

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.ThreadLocalRandom

/**
 * [target] written whole or not at all: [write] fills a temporary file of this write's own beside it - ".name.<8
 * hex>.tmp", geode-agent's write_atomic's name, so two writers never share one and the phone folders' sync skips it
 * (PhoneFolders.own) -, [sync]ed to the storage, then moved over [target] in one step: the old file stays until the new
 * one replaces it. Throws when it cannot be written; the temporary file is gone either way.
 */
fun writeAtomic(target: File, sync: Boolean = true, write: (OutputStream) -> Unit) {
    val parent = target.absoluteFile.parentFile ?: throw IOException("no folder for ${target.name}")
    parent.mkdirs()
    var tmp: File
    do {
        tmp = File(parent, ".%s.%08x.tmp".format(target.name, ThreadLocalRandom.current().nextInt()))
    } while (!tmp.createNewFile())
    try {
        FileOutputStream(tmp).use { o ->
            write(o)
            o.flush()
            if (sync) o.fd.sync()
        }
        Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    } finally {
        tmp.delete()   // (gone already when it was moved)
    }
}

fun writeAtomic(target: File, bytes: ByteArray) = writeAtomic(target) { it.write(bytes) }

fun writeAtomic(target: File, text: String) = writeAtomic(target, text.toByteArray())
