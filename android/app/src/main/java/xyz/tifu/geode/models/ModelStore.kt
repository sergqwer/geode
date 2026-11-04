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
    /** The pack has an audio encoder (geode.json's "audio": Gemma 4's; its audio.bin may still be missing). */
    val audio: Boolean = false,
    /** The longest context the model was trained for (geode.json's config.max_position_embeddings); 0 = not said. */
    val maxContext: Int = 0,
    /** The pack drafts tokens for speculative decoding: Qwen's MTP layer (geode.json's "mtp") or Gemma 4's drafter
     *  (its "drafter", drafter.bin). */
    val draft: Boolean = false,
) {
    val bytes: Long get() = files.sumOf { it.size }
    /** The engine runs this architecture. */
    val runnable: Boolean get() = arch in ARCHS
    /** Gemma 4 (its chat template, image sizes, video and audio), not Qwen. */
    val gemma: Boolean get() = arch == GEMMA4

    companion object {
        const val FILE = "geode.json"
        const val GEMMA4 = "gemma4"
        /** The engine's architectures: the 35B-A3B MoE, the dense Qwen3.5 (the 2B light model) and Gemma 4 (E2B). */
        val ARCHS = setOf("qwen35moe", "qwen35", GEMMA4)

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
                    PackFile(f.getString("name"), f.getLong("size"), f.optString("sha256"), f.optBoolean("optional"))
                },
                vision = o.has("vision"),
                audio = o.has("audio"),
                maxContext = o.optJSONObject("config")?.optInt("max_position_embeddings") ?: 0,
                draft = o.has("mtp") || o.has("drafter") ||
                    (o.optJSONObject("config")?.optInt("mtp_num_hidden_layers") ?: 0) > 0,
            )
        }.getOrNull()
    }
}

/** A pack's file; [optional]: the model runs without it (an encoder, the drafter), only that part of it missing. */
data class PackFile(val name: String, val size: Long, val sha256: String, val optional: Boolean = false)

/** What the user turned off of a model's inputs and drafting (AppSettings.featuresOff, InstalledModel.off). */
const val OFF_IMAGES = 1
const val OFF_VIDEO = 2
const val OFF_AUDIO = 4
const val OFF_DRAFT = 8

/**
 * An installed model. [off]: what the user turned off in Model settings (OFF_* bits; the settings' featuresOff,
 * applied where the selected model is looked up) - [vision], [audio] and [video] are what it takes as they left it,
 * [hasVision], [hasAudio], [hasVideo] and [hasDraft] what its pack can.
 */
data class InstalledModel(val id: String, val dir: File, val manifest: PackManifest, val off: Int = 0) {
    val title: String get() = manifest.name.ifEmpty { id }
    /** Its pack has the vision encoder, installed. */
    val hasVision: Boolean get() = manifest.vision && File(dir, "vision.bin").isFile
    /** Its pack has the audio encoder, installed. */
    val hasAudio: Boolean get() = manifest.audio && File(dir, "audio.bin").isFile
    /** It can watch video: Gemma 4's vision encoder reads the frames (and the audio encoder the sound). */
    val hasVideo: Boolean get() = manifest.gemma && hasVision
    /** It drafts tokens ahead (speculative decoding): Qwen's MTP layer, Gemma 4's drafter. */
    val hasDraft: Boolean get() = manifest.draft && (!manifest.gemma || File(dir, "drafter.bin").isFile)
    /** The model reads images: its pack can and the user left them on. */
    val vision: Boolean get() = hasVision && off and OFF_IMAGES == 0
    /** The model hears audio: its pack can and the user left it on. */
    val audio: Boolean get() = hasAudio && off and OFF_AUDIO == 0
    /** The model watches video: its pack can and the user left it on. */
    val video: Boolean get() = hasVideo && off and OFF_VIDEO == 0
    /** Speculative decoding: the pack drafts and the user left it on. */
    val draft: Boolean get() = hasDraft && off and OFF_DRAFT == 0

    /**
     * What the engine leaves unread (GeodeNative.OFF_* for geode_config): the drafting when off, an encoder when no
     * input left on needs it (a video needs the vision encoder for its frames, the audio one for its sound).
     */
    val engineOff: Int get() =
        (if (hasDraft && !draft) xyz.tifu.geode.engine.GeodeNative.OFF_DRAFT else 0) or
            (if (hasVision && !vision && !video) xyz.tifu.geode.engine.GeodeNative.OFF_VISION else 0) or
            (if (hasAudio && !audio && !video) xyz.tifu.geode.engine.GeodeNative.OFF_AUDIO else 0)

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
