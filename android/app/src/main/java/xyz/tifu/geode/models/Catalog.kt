package xyz.tifu.geode.models

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/** A pack of Geode's own repository with its geode.json read: what the user sees before downloading it. */
data class CatalogPack(val pack: HfPack, val manifest: PackManifest)

/**
 * Geode's own packs on Hugging Face, offered on the Models screen with one tap: every folder of the repository that
 * holds a geode.json, its name and description read from that file - a pack added there shows up without an update.
 */
object Catalog {
    /**
     * Geode's repositories, each with its packs: the big model first, then the light ones for weaker phones (Gemma 4
     * E4B and E2B, the ones that see video and hear sound, last).
     */
    val REPOS = listOf(
        "Maximilian228/Geode-Qwen3.6-35B-A3B",
        "Maximilian228/Geode-Qwen3.5-2B",
        "Maximilian228/Geode-Gemma-4-E4B",
        "Maximilian228/Geode-Gemma-4-E2B",
    )
    private const val REVISION = "main"

    /**
     * Every repository's packs; one that cannot be read (not there yet, the network) is left out quietly while
     * another gives packs - an error only when none does.
     */
    fun load(token: String): List<CatalogPack> {
        val errors = mutableListOf<Throwable>()
        val packs = REPOS.flatMap { repo ->
            runCatching { packs(repo, token) }.getOrElse { errors += it; emptyList() }
        }
        if (packs.isEmpty() && errors.isNotEmpty()) throw errors.first()
        return packs
    }

    private fun packs(repo: String, token: String): List<CatalogPack> = HfApi.findPacks(repo, REVISION, token).mapNotNull { p ->
        val path = if (p.path.isEmpty()) PackManifest.FILE else "${p.path}/${PackManifest.FILE}"
        PackManifest.parse(HfApi.text(repo, REVISION, path, token))?.let { CatalogPack(p, it) }
    }

    /** Free space a pack needs: its files and a margin (the engine's files beside them, the system's room). */
    fun spaceNeeded(p: CatalogPack): Long = p.pack.bytes + (1L shl 30)

    /**
     * A light model: a dense one (no experts to read from flash), small enough for any phone's memory - it runs on
     * the processor (the NPU reads its prompts where there is one) and needs neither the GPU path nor 8 GB of RAM.
     */
    fun light(p: CatalogPack): Boolean =
        p.manifest.arch == "qwen35" || p.manifest.arch == "gemma4" || p.pack.bytes < (4L shl 30)

    /** A light model too large for a 6 GB phone (Gemma 4 E4B: 2.6 GB of it in memory): it wants 8 GB. */
    fun wantsEightGb(p: CatalogPack): Boolean = light(p) && p.pack.bytes >= (4L shl 30)
}

/** What the phone brings to a model: its memory and whether the engine's GPU path can run on its chip. */
object Device {
    // Qualcomm chips with an Adreno 7xx or newer and its int8 dot products (the engine's gpu8 path): Snapdragon
    // 8 Gen 1 and later, 7+ Gen 2, 7 Gen 3, 7+ Gen 3
    private val GPU_SOCS = setOf("SM8450", "SM8475", "SM8550", "SM8635", "SM8650", "SM8735", "SM8750", "SM8850",
        "SM7475", "SM7550", "SM7675")

    fun ramBytes(context: Context): Long {
        val mi = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(mi)
        return mi.totalMem
    }

    /** False on Android 11, where the chip's name is not given to apps: the engine finds out when it opens a model. */
    fun gpuPath(): Boolean = Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER.equals("QTI", ignoreCase = true) &&
        Build.SOC_MODEL.uppercase() in GPU_SOCS
}
