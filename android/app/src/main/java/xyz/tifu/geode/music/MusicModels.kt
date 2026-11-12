package xyz.tifu.geode.music

import android.content.Context
import android.os.Build
import java.io.File
import xyz.tifu.geode.models.CatalogPack
import xyz.tifu.geode.models.HfApi
import xyz.tifu.geode.models.InstalledModel
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.models.PackManifest

/**
 * The song model (YuE2), kept apart from the chat's models: files/music/models/<id>/, a geode-pack of arch "yue2"
 * (tools/yue2_phone_pack.py). Its packs on Hugging Face: one for every phone (the CPU's weights) and one with the AR's
 * fp16 layers as well, for a phone whose NPU has HMX (the prompt of the acoustic stage on the NPU: 2 s instead of a
 * minute).
 */
object MusicModels {
    const val ARCH = "yue2"
    const val REPO = "Maximilian228/Geode-YuE2"
    private const val REVISION = "main"
    /** The AR's fp16 layers, for the NPU (yue2::NarDsp::has_ar_f16). */
    const val AR_F16 = "ar_f16.bin"

    fun store(context: Context) = ModelStore(File(context.filesDir, "music/models"))

    /** The Q4 weights the engine made on the phone (yue2_q4.hpp: their own manifest beside geode.json) removed from
     * every installed model - the faster composing turned off gives its 1.2 GB back. A pack that has them from the PC
     * (in its geode.json) keeps them. */
    fun dropArQ4(context: Context) {
        for (m in store(context).installed()) {
            val man = File(m.dir, "geode.ar_q4.json")
            if (!man.isFile) continue
            man.delete()   // (first: no manifest ever names a file that is gone)
            File(m.dir, "ar_q4.bin").delete()
        }
    }

    /** The same for the sound's 8-bit weights (yue2_q4.hpp make_nar_q8: the NPU's nar_q8.bin, the processor's
     *  nar_c8.bin, 1.5 GB each). */
    fun dropNarQ8(context: Context) {
        for (m in store(context).installed())
            for ((man, bin) in listOf("geode.nar_q8.json" to "nar_q8.bin", "geode.nar_c8.json" to "nar_c8.bin")) {
                val f = File(m.dir, man)
                if (!f.isFile) continue
                f.delete()   // (first: no manifest ever names a file that is gone)
                File(m.dir, bin).delete()
            }
    }

    fun installed(context: Context): List<InstalledModel> =
        store(context).installed().filter { it.manifest.arch == ARCH }

    /** The model the songs are made with: the one chosen in the settings, else the first installed; null: none. */
    fun selected(context: Context, prefs: MusicPrefs): InstalledModel? {
        val all = installed(context)
        return all.firstOrNull { it.id == prefs.modelId } ?: all.firstOrNull()
    }

    /** The packs of [REPO] (every folder with a geode.json of this arch). */
    fun catalog(token: String): List<CatalogPack> = HfApi.findPacks(REPO, REVISION, token).mapNotNull { p ->
        val path = if (p.path.isEmpty()) PackManifest.FILE else "${p.path}/${PackManifest.FILE}"
        PackManifest.parse(HfApi.text(REPO, REVISION, path, token))?.takeIf { it.arch == ARCH }?.let { CatalogPack(p, it) }
    }

    /** The pack carries the NPU's fp16 AR layers. */
    fun forNpu(p: CatalogPack): Boolean = p.pack.files.any { it.path.substringAfterLast('/') == AR_F16 }

    // Snapdragons whose Hexagon has HMX (the matrix unit the acoustic stage runs on): 8 Gen 2 and newer, 8s Gen 3,
    // 7+ Gen 3 - elsewhere the CPU computes it and the NPU pack's extra layers would only take room
    private val HMX_SOCS = setOf("SM8550", "SM8635", "SM8650", "SM8735", "SM8750", "SM8850", "SM7675")

    /** The NPU pack is this phone's (false on Android 11, where the chip's name is not given to apps). */
    fun phoneHasHmx(): Boolean = Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER.equals("QTI", ignoreCase = true) &&
        Build.SOC_MODEL.uppercase() in HMX_SOCS

    /** The pack to offer first on this phone. */
    fun recommended(packs: List<CatalogPack>): CatalogPack? =
        packs.firstOrNull { forNpu(it) == phoneHasHmx() } ?: packs.firstOrNull()
}
