package xyz.tifu.geode.agent

import android.content.Context
import android.system.Os
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream

/**
 * The agent's Linux: a Debian root filesystem in the app's files (files/linux/rootfs), run under proot - the app may
 * not exec files from its own storage (Android 10+), but proot (shipped in the APK's native library directory, where
 * exec is allowed, tools/linux_tools.py) runs them through its loader. Each chat's folder is mounted as /workspace,
 * the command's working directory; inside, the agent is root of its own system (proot's fake root: the files belong
 * to the app as ever).
 */
class Linux(context: Context) {
    private val base = File(context.filesDir, "linux")
    val root = File(base, "rootfs")
    /** proot's own temporary folder (PROOT_TMP_DIR). */
    val tmp = File(base, "tmp")
    private val libDir = context.applicationInfo.nativeLibraryDir

    private val fresh = File(base, "rootfs.new")
    private val old = File(base, "rootfs.old")

    /** The root filesystem is unpacked whole (its marker is written last). */
    val installed: Boolean get() = File(root, MARKER).isFile

    /** The version of the image unpacked (its marker; "ok": one from before images had versions), null: none. */
    val version: String? get() = runCatching { File(root, MARKER).readText().trim() }.getOrNull()

    /**
     * An update cut short finished or undone (at the app's start, before anything runs in the Linux): a whole new
     * root filesystem whose swap was interrupted goes into place, a half-unpacked one goes, the old one goes.
     */
    fun settle() {
        if (!installed && File(fresh, MARKER).isFile) fresh.renameTo(root)
        if (fresh.exists()) wipe(fresh)
        if (old.exists() && installed) wipe(old)
        if (!installed && File(old, MARKER).isFile) old.renameTo(root)
    }

    /**
     * [f] gone whole, as the app may: each folder opened to its owner before it is read (the guest may have taken
     * every right off one, a plain delete then stops there and the swap after it fails), symbolic links removed
     * themselves, never followed (an absolute one points out of the image).
     */
    fun wipe(f: File) {
        val st = runCatching { Os.lstat(f.path) }.getOrNull() ?: return
        if (android.system.OsConstants.S_ISDIR(st.st_mode)) {
            runCatching { Os.chmod(f.path, 0x1c0) }
            f.listFiles()?.forEach(::wipe)
        }
        f.delete()
    }

    /**
     * Unpacks a root filesystem (.tar.xz or .tar.gz) beside the old one and swaps it in, its marker [version]: modes
     * kept (the owner may always write: the app is the real owner), symbolic links as they are, hard links (apps may
     * not make them) as copies, devices and pipes left out (/dev comes from the phone). [progress] gets the bytes
     * unpacked. The chats' own folders (files/agents) are not in it: an update leaves them as they are. The old one
     * stays until the new one is whole, unless the storage has no room for both ([need]: the bytes it takes).
     */
    fun install(input: InputStream, version: String, need: Long = 0, progress: (Long) -> Unit = {}) {
        wipe(fresh)
        if (need > 0 && base.usableSpace < need) wipe(root)
        fresh.mkdirs()
        // .tar.xz (the image Geode ships) or .tar.gz, by their magic
        val buffered = BufferedInputStream(input, 1 shl 16)
        buffered.mark(8)
        val magic = ByteArray(6).also { buffered.read(it) }
        buffered.reset()
        val xz = magic.contentEquals(byteArrayOf(0xFD.toByte(), '7'.code.toByte(), 'z'.code.toByte(), 'X'.code.toByte(),
            'Z'.code.toByte(), 0))
        val tar = Tar(if (xz) BufferedInputStream(org.tukaani.xz.XZInputStream(buffered), 1 shl 16)
            else GZIPInputStream(buffered, 1 shl 16))
        val hard = ArrayList<Pair<File, String>>()
        val dirs = ArrayList<Pair<File, Int>>()
        var done = 0L
        while (true) {
            val e = tar.next() ?: break
            val rel = e.name.removePrefix("./").trimEnd('/')
            if (rel.isEmpty() || rel == ".." || rel.startsWith("../") || "/../" in rel) continue
            val f = File(fresh, rel)
            when (e.type) {
                '5' -> {
                    f.mkdirs()
                    dirs += f to e.mode
                }
                '0', '\u0000', '7' -> {
                    f.parentFile?.mkdirs()
                    FileOutputStream(f).use { tar.copyData(it) }
                    Os.chmod(f.path, (e.mode and 0x1ff) or 0x180)
                    done += e.size
                    progress(done)
                }
                '2' -> {
                    f.parentFile?.mkdirs()
                    f.delete()
                    Os.symlink(e.link, f.path)
                }
                '1' -> hard += f to e.link.removePrefix("./")
                else -> Unit   // devices, pipes
            }
        }
        for ((f, target) in hard) {
            val src = File(fresh, target)
            if (!src.isFile) continue
            f.parentFile?.mkdirs()
            src.copyTo(f, overwrite = true)
            Os.chmod(f.path, Os.stat(src.path).st_mode and 0x1ff)
        }
        // directories last: a read-only one would have refused its files; the owner keeps rwx
        for ((d, mode) in dirs) runCatching { Os.chmod(d.path, (mode and 0x1ff) or 0x1c0) }
        // the phone's resolver is not a file: the guest asks public DNS servers
        File(fresh, "etc").mkdirs()
        File(fresh, "etc/resolv.conf").apply { delete() }.writeText("nameserver 1.1.1.1\nnameserver 8.8.8.8\n")
        File(fresh, "etc/hosts").writeText("127.0.0.1 localhost\n::1 localhost\n")
        File(fresh, MARKER).writeText(version + "\n")
        // the swap: two renames, settle() finishes it if the app dies between them
        wipe(old)
        if (root.exists() && !root.renameTo(old)) error("the old root filesystem could not be moved aside")
        if (!fresh.renameTo(root)) {
            old.renameTo(root)
            error("the root filesystem could not be moved into place")
        }
        wipe(old)
    }

    /**
     * A command in [workspace] (the guest's /workspace): bash -lc, as root, in a clean environment with [env] added,
     * [binds] (a folder of the app's, its path in the guest) mounted too. Its errors go with its output unless
     * [mergeErrors] is false (a program whose output is data: the agent's events).
     */
    fun start(workspace: File, command: String, env: Map<String, String> = emptyMap(),
              binds: List<Pair<File, String>> = emptyList(), mergeErrors: Boolean = true): Process {
        workspace.mkdirs()
        tmp.mkdirs()
        val args = ArrayList<String>()
        args += listOf(
            "$libDir/libproot.so",
            "--kill-on-exit", "--link2symlink", "--sysvipc", "-0",
            "-r", root.path,
            "-b", "/dev", "-b", "/proc", "-b", "/sys", "-b", "/proc/self/fd:/dev/fd",
            "-b", "${workspace.path}:/workspace",
        )
        for ((dir, at) in binds) {
            dir.mkdirs()
            args += listOf("-b", "${dir.path}:$at")
        }
        args += listOf(
            "-w", "/workspace",
            "/usr/bin/env", "-i",
            "HOME=/root", "USER=root", "LANG=C.UTF-8", "TERM=dumb", "SHELL=/bin/bash", "TMPDIR=/tmp",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        )
        for ((k, v) in env) args += "$k=$v"
        args += listOf("/bin/bash", "-lc", command)
        return ProcessBuilder(args).redirectErrorStream(mergeErrors).apply {
            environment().apply {
                clear()
                put("PROOT_LOADER", "$libDir/libproot-loader.so")
                put("PROOT_TMP_DIR", tmp.path)
                put("PATH", "/system/bin")
            }
        }.start()
    }

    /** A command's exit code and its output (the last [maxChars] of it), killed after [timeoutSec]. */
    fun run(workspace: File, command: String, timeoutSec: Long = 120, maxChars: Int = 16_000): Pair<Int, String> {
        val p = start(workspace, command)
        val out = StringBuilder()
        val reader = Thread {
            runCatching {
                p.inputStream.bufferedReader().use { r ->
                    val buf = CharArray(8192)
                    while (true) {
                        val n = r.read(buf)
                        if (n < 0) break
                        synchronized(out) {
                            out.append(buf, 0, n)
                            if (out.length > maxChars * 2) out.delete(0, out.length - maxChars)
                        }
                    }
                }
            }
        }.apply { start() }
        val finished = p.waitFor(timeoutSec, TimeUnit.SECONDS)
        if (!finished) p.destroyForcibly()
        reader.join(2000)
        val text = synchronized(out) { if (out.length > maxChars) "…" + out.substring(out.length - maxChars) else out.toString() }
        return (if (finished) p.exitValue() else -1) to text
    }

    /** A tar stream's entries (ustar, GNU long names, pax paths). */
    private class Tar(private val input: InputStream) {
        class Entry(val name: String, val type: Char, val mode: Int, val size: Long, val link: String)

        private val header = ByteArray(512)
        private var left = 0L
        private var pad = 0

        fun next(): Entry? {
            skipRest()
            var longName: String? = null
            var longLink: String? = null
            while (true) {
                if (!readFully(header)) return null
                if (header.all { it == 0.toByte() }) return null
                val type = header[156].toInt().toChar()
                val size = number(124, 12)
                val name = longName ?: run {
                    val n = string(0, 100)
                    val prefix = if (string(257, 6).startsWith("ustar")) string(345, 155) else ""
                    if (prefix.isNotEmpty()) "$prefix/$n" else n
                }
                val link = longLink ?: string(157, 100)
                when (type) {
                    'L', 'K', 'x', 'g' -> {
                        val data = ByteArray(size.toInt()).also { readFully(it) }
                        skip(pad(size).toLong())
                        when (type) {
                            'L' -> longName = String(data, Charsets.UTF_8).trimEnd('\u0000')
                            'K' -> longLink = String(data, Charsets.UTF_8).trimEnd('\u0000')
                            'x' -> pax(data).let { m ->
                                m["path"]?.let { longName = it }
                                m["linkpath"]?.let { longLink = it }
                            }
                        }
                        continue
                    }
                }
                if (type == '0' || type == '\u0000' || type == '7') {
                    left = size
                    pad = pad(size)
                } else if (size > 0) {
                    // another entry with data (unusual): not kept
                    skip(size + pad(size))
                }
                return Entry(name, type, number(100, 8).toInt(), size, link)
            }
        }

        fun copyData(out: OutputStream) {
            val buf = ByteArray(1 shl 16)
            while (left > 0) {
                val n = input.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
                if (n < 0) error("the archive ends inside a file")
                out.write(buf, 0, n)
                left -= n
            }
        }

        private fun skipRest() {
            if (left > 0) skip(left)
            left = 0
            if (pad > 0) skip(pad.toLong())
            pad = 0
        }

        private fun pad(size: Long) = ((512 - size % 512) % 512).toInt()

        private fun pax(data: ByteArray): Map<String, String> {
            val m = HashMap<String, String>()
            var at = 0
            while (at < data.size) {
                val sp = (at until data.size).firstOrNull { data[it] == ' '.code.toByte() } ?: break
                val len = String(data, at, sp - at).toIntOrNull() ?: break
                val rec = String(data, sp + 1, len - (sp - at) - 2, Charsets.UTF_8)
                val eq = rec.indexOf('=')
                if (eq > 0) m[rec.substring(0, eq)] = rec.substring(eq + 1)
                at += len
            }
            return m
        }

        private fun string(off: Int, len: Int): String {
            var end = off
            while (end < off + len && header[end] != 0.toByte()) end++
            return String(header, off, end - off, Charsets.UTF_8)
        }

        private fun number(off: Int, len: Int): Long {
            if (header[off].toInt() and 0x80 != 0) {
                var v = 0L
                for (i in off + 1 until off + len) v = (v shl 8) or (header[i].toLong() and 0xff)
                return v
            }
            return string(off, len).trim().ifEmpty { "0" }.toLong(8)
        }

        private fun readFully(b: ByteArray): Boolean {
            var at = 0
            while (at < b.size) {
                val n = input.read(b, at, b.size - at)
                if (n < 0) return false
                at += n
            }
            return true
        }

        private fun skip(n: Long) {
            var left = n
            val buf = ByteArray(8192)
            while (left > 0) {
                val k = input.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
                if (k < 0) return
                left -= k
            }
        }
    }

    companion object {
        private const val MARKER = ".geode-rootfs"
    }
}
