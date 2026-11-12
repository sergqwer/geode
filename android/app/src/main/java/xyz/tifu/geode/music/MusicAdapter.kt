package xyz.tifu.geode.music

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import xyz.tifu.geode.models.HfApi
import xyz.tifu.geode.models.fetchVerified
import xyz.tifu.geode.models.hashInto
import xyz.tifu.geode.models.toHex
import java.io.File
import java.security.MessageDigest

/**
 * The instrumental adapter: Mothersuperior's YuE2 instrumental cot-full LoRA (rank 64 on the AR's attention and MLP
 * products, derived from m-a-p/YuE2-3B, CC BY-NC 4.0 - huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras).
 * The model writes instrumental music with a section plan: the score first (mode full), the lyrics only bare tags.
 * Downloaded on demand (139 MB, bf16); the engine merges it into the song model's Q8 AR weights the first time a song
 * uses it and keeps that merged copy (about 1.9 GB) in [mergedDir] - the song model's own files are never changed.
 */
object MusicAdapter {
    private const val TAG = "GeodeMusic"
    const val REPO = "Mothersuperior/YuE2-instrumental-cot-full-loras"
    const val REVISION = "947f2f4b28978b2b6c3e316e6a87925c76bf3c4b"
    const val FILE = "ar_lora_inst_v3abc.bf16.safetensors"
    const val SIZE = 139_502_088L
    const val SHA256 = "e408fd3148b75b1165f7ddbf63db575d83bb6402a0b5f876fcb767dbcb2c5414"
    const val LINK = "https://huggingface.co/$REPO"
    /** About what the merged weights take (the song model's Q8 AR with the adapter in). */
    const val MERGED_BYTES = 1_900_000_000L
    /** The structure tags it was trained on: bare, lowercase, one a line. */
    val TAGS = listOf("intro", "verse", "pre-chorus", "chorus", "bridge", "outro")
    /** Its lyrics when no structure is given: the model chooses the sections and the length. */
    const val BARE = "[instrumental]"
    private const val OK = "ok"

    private fun dir(context: Context) = File(context.filesDir, "music/adapters/$SHA256")
    fun file(context: Context) = File(dir(context), FILE)
    fun mergedDir(context: Context) = File(context.filesDir, "music/merged")

    fun installed(context: Context): Boolean = file(context).length() == SIZE && File(dir(context), OK).isFile

    /** The engine's "lora" for a song made with the adapter (yue2_lora.hpp): the file, its scale, where the merged
     *  weights are kept. */
    fun lora(context: Context): JSONObject = JSONObject().put("path", file(context).path).put("scale", 1.0)
        .put("cache", mergedDir(context).path)

    /** [lyrics] as the adapter takes them: blank -> [BARE]. */
    fun lyricsOf(lyrics: String): String = lyrics.trim().ifEmpty { BARE }

    data class Download(val running: Boolean = false, val done: Long = 0, val error: String? = null)
    private val _download = MutableStateFlow(Download())
    val download: StateFlow<Download> = _download
    private var job: Job? = null
    /** The download's own: it goes on when the settings screen is left (cancel() stops it). */
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    fun start(context: Context) {
        if (job?.isActive == true) return
        val app = context.applicationContext
        _download.value = Download(running = true)
        job = scope.launch(Dispatchers.IO) {
            try {
                val d = dir(app).apply { mkdirs() }
                val target = file(app)
                if (!(target.length() == SIZE && sha256(target) == SHA256)) {
                    val part = File(d, "$FILE.part")
                    var have = 0L
                    var shown = 0L
                    HfApi.fetchVerified(HfApi.resolveUrl(REPO, REVISION, FILE), "", part, SIZE, SHA256,
                        badHash = { "$FILE: the file differs from the one expected" }, progress = { n ->
                            have += n
                            if (have - shown >= (1 shl 19) || have < shown) {
                                shown = have
                                _download.value = _download.value.copy(done = have)
                            }
                        })
                    target.delete()
                    if (!part.renameTo(target)) throw java.io.IOException("$FILE: could not be put in place")
                }
                File(d, OK).writeText(REVISION)
                _download.value = Download()
            } catch (e: CancellationException) {
                _download.value = Download()
                throw e
            } catch (e: Exception) {
                Log.w(TAG, "adapter download", e)
                _download.value = Download(error = e.message ?: e.javaClass.simpleName)
            }
        }
    }

    private fun sha256(f: File): String =
        MessageDigest.getInstance("SHA-256").also { d -> f.inputStream().use { hashInto(d, it, f.length()) {} } }.digest().toHex()

    fun cancel() {
        job?.cancel()
    }

    /** The adapter gone, and the weights merged with it. */
    fun delete(context: Context) {
        cancel()
        dir(context).deleteRecursively()
        clearMerged(context)
    }

    /** What the merged weights take now (bytes). */
    fun mergedBytes(context: Context): Long = mergedDir(context).walkBottomUp().filter { it.isFile }.sumOf { it.length() }

    /** The merged weights gone (made again the next time a song uses the adapter). */
    fun clearMerged(context: Context) {
        mergedDir(context).deleteRecursively()
    }

    /** Whether [job] (a song's) was made with the adapter. */
    fun usedBy(job: JSONObject): Boolean = job.optJSONObject("lora") != null
}
