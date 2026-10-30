package xyz.tifu.geode.models

import android.content.Context
import android.os.StatFs
import org.json.JSONObject
import java.io.File

/** What the app needs from a model's geode.json (the engine reads the rest). */
data class PackManifest(
    val name: String,
    val description: String,
    val arch: String,
    val files: List<PackFile>,
    /** The pack has a vision encoder (geode.json's "vision"; its vision.bin may still be missing). */
    val vision: Boolean = false,
) {
    val bytes: Long get() = files.sumOf { it.size }
    /** The engine runs this architecture. */
    val runnable: Boolean get() = arch in ARCHS

    companion object {
        const val FILE = "geode.json"
        /** The engine's architectures: the 35B-A3B MoE and the dense Qwen3.5 (the 2B light model). */
        val ARCHS = setOf("qwen35moe", "qwen35")

        /** Null when the text is not a geode-pack v1 this app can run. */
        fun parse(text: String): PackManifest? = runCatching {
            val o = JSONObject(text)
            if (o.optString("format") != "geode-pack" || o.optInt("version") != 1) return null
            val arr = o.getJSONArray("files")
            PackManifest(
                name = o.optString("name"),
                description = o.optString("description"),
                arch = o.optString("arch"),
                files = (0 until arr.length()).map { i ->
                    val f = arr.getJSONObject(i)
                    PackFile(f.getString("name"), f.getLong("size"), f.optString("sha256"))
                },
                vision = o.has("vision"),
            )
        }.getOrNull()
    }
}

data class PackFile(val name: String, val size: Long, val sha256: String)

data class InstalledModel(val id: String, val dir: File, val manifest: PackManifest) {
    val title: String get() = manifest.name.ifEmpty { id }
    /** The model reads images: its pack has the vision encoder, installed. */
    val vision: Boolean get() = manifest.vision && File(dir, "vision.bin").isFile

    /**
     * What a saved conversation state (StateCache) is of: the pack's folder and its weights - every file's sha256 from
     * geode.json - so a pack replaced in the same folder (new weights, same path) starts its states afresh instead of
     * going on from ones its old weights computed.
     */
    val stateId: String by lazy {
        val d = java.security.MessageDigest.getInstance("SHA-256")
        for (f in manifest.files.sortedBy { it.name }) d.update("${f.name}:${f.sha256}\n".toByteArray())
        dir.path + "#" + d.digest().take(8).joinToString("") { "%02x".format(it) }
    }
}

/** A download or import that stopped before the end; its files stay so it can resume. */
data class PartialModel(val id: String, val dir: File, val source: TransferSource?, val bytes: Long)

/**
 * Models live in the app's own storage, files/models/<id>/: the engine reads experts with O_DIRECT, which needs a
 * real file on /data (shared storage goes through FUSE). A transfer writes into models/.partial-<id>/ and renames
 * it when every file is complete and checked, so a model directory is either whole or absent.
 */
class ModelStore(context: Context) {
    val root = File(context.filesDir, "models").apply { mkdirs() }

    fun dir(id: String) = File(root, id)
    fun partialDir(id: String) = File(root, PARTIAL + id)

    fun installed(): List<InstalledModel> =
        (root.listFiles() ?: emptyArray())
            .filter { it.isDirectory && !it.name.startsWith(".") }
            .mapNotNull { d -> manifest(d)?.let { InstalledModel(d.name, d, it) } }
            .sortedBy { it.title.lowercase() }

    fun partials(): List<PartialModel> =
        (root.listFiles() ?: emptyArray())
            .filter { it.isDirectory && it.name.startsWith(PARTIAL) }
            .map { d ->
                PartialModel(d.name.removePrefix(PARTIAL), d, TransferSource.read(File(d, SOURCE)), sizeOf(d))
            }

    fun find(id: String): InstalledModel? = dir(id).takeIf { it.isDirectory }?.let { d ->
        manifest(d)?.let { InstalledModel(id, d, it) }
    }

    fun delete(id: String) {
        dir(id).deleteRecursively()
    }

    fun deletePartial(id: String) {
        partialDir(id).deleteRecursively()
    }

    fun freeBytes(): Long = StatFs(root.path).availableBytes

    companion object {
        const val PARTIAL = ".partial-"
        /** Where a partial transfer remembers what it was, to resume it. */
        const val SOURCE = ".source.json"

        fun manifest(dir: File): PackManifest? {
            val f = File(dir, PackManifest.FILE)
            return if (f.isFile) PackManifest.parse(f.readText()) else null
        }

        fun sizeOf(f: File): Long = if (f.isDirectory) (f.listFiles() ?: emptyArray()).sumOf { sizeOf(it) } else f.length()

        /** owner/name + a pack inside it -> a directory name: "owner--name" or "owner--name--q3". */
        fun idFor(repo: String, packPath: String): String =
            (repo.replace('/', '-') + if (packPath.isEmpty()) "" else "--" + packPath.replace('/', '-'))
                .replace(Regex("[^A-Za-z0-9._-]"), "_")
                .trimStart('.')
                .take(120)
    }
}
