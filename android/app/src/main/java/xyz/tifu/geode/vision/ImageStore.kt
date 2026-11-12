package xyz.tifu.geode.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import xyz.tifu.geode.media.MediaFolder
import xyz.tifu.geode.util.writeAtomic
import java.io.File
import xyz.tifu.geode.tokenizer.TurnImage
import java.util.UUID
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The chats' images. An attached image is kept once, as the model is to see it: its EXIF orientation applied, a
 * transparent background made white, at most [STORE_SIDE] pixels a side, JPEG (files/chats/img/<id>.jpg). The engine
 * gets it at the size the model reads it at ([fit]: Qwen's smart_resize; [fitGemma]: Gemma 4's) as raw RGB in the
 * cache (cache/vision/), written by the UI process and read by the :engine one.
 */
class ImageStore(context: Context) : MediaFolder(File(context.filesDir, "chats/img"), "jpg") {
    private val resolver = context.contentResolver
    private val cache = File(context.cacheDir, "vision").apply { mkdirs() }

    /** A picked image (a content Uri) into the store; its name. */
    fun import(uri: Uri): String {
        val name = UUID.randomUUID().toString() + ".jpg"
        store(ImageDecoder.createSource(resolver, uri), name)
        return name
    }

    /**
     * An image a local server's request sent (PNG, JPEG, WebP... bytes) into the store, named by its content: a client
     * sends a conversation's images again with every request, and the same image keeps its file (and its features,
     * encoded once). Its name. The server's images older than a day go at its start ([dropServerCopies]).
     */
    fun importBytes(bytes: ByteArray): String =
        serverCopy(contentName(bytes, "jpg")) { store(ImageDecoder.createSource(java.nio.ByteBuffer.wrap(bytes)), it) }

    /** An image the web chat sent for a chat's question: kept with the chat (as a picked one is), its name. */
    fun importChatBytes(bytes: ByteArray): String {
        val name = UUID.randomUUID().toString() + ".jpg"
        store(ImageDecoder.createSource(java.nio.ByteBuffer.wrap(bytes)), name)
        return name
    }


    /**
     * The local server's copies of these chat images - the agent sends a chat image's bytes, which the server stores
     * under their content's name - gone with their chat. A forked chat's copy has the same bytes: its next request
     * stores it again.
     */
    fun deleteServerCopies(names: Collection<String>) {
        for (n in names) runCatching { file(contentName(file(n).readBytes(), "jpg")).delete() }
    }

    private fun store(source: ImageDecoder.Source, name: String) {
        val bmp = ImageDecoder.decodeBitmap(source) { d, info, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val w = info.size.width
            val h = info.size.height
            val s = STORE_SIDE.toDouble() / max(w, h)
            if (s < 1.0) d.setTargetSize(max(1, (w * s).roundToInt()), max(1, (h * s).roundToInt()))
        }
        val out = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.WHITE)
            drawBitmap(bmp, 0f, 0f, null)
        }
        writeAtomic(file(name), sync = false) { out.compress(Bitmap.CompressFormat.JPEG, 92, it) }
    }

    /** The stored image's width and height (its pixels not decoded); null when it is gone. */
    fun size(name: String): Pair<Int, Int>? {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file(name).path, o)
        return if (o.outWidth > 0 && o.outHeight > 0) o.outWidth to o.outHeight else null
    }

    /** A small copy for the chat, its longer side about [px]. */
    fun thumbnail(name: String, px: Int): Bitmap? = runCatching {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(file(name))) { d, info, _ ->
            val s = px.toDouble() / max(info.size.width, info.size.height)
            if (s < 1.0) d.setTargetSize(max(1, (info.size.width * s).roundToInt()), max(1, (info.size.height * s).roundToInt()))
        }
    }.getOrNull()

    /**
     * A prompt's images and a video's frames, each at its size (a [fit]), as raw RGB for the engine (made once, kept in
     * the cache): their files, in order. The cache is trimmed once after them.
     */
    fun prepare(images: List<TurnImage>): List<File> = images.map { rgb(it.name, it.width, it.height) }.also { trim() }

    /** The image at [w] x [h] as raw RGB: the cache's file, else made now. */
    private fun rgb(name: String, w: Int, h: Int): File {
        val f = File(cache, "${name.substringBeforeLast('.')}_${w}x$h.rgb")
        if (f.length() == w.toLong() * h * 3) return f.also { it.setLastModified(System.currentTimeMillis()) }
        val bmp = ImageDecoder.decodeBitmap(ImageDecoder.createSource(file(name))) { d, _, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            d.setTargetSize(w, h)
        }
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        val rgb = ByteArray(w * h * 3)
        for (i in px.indices) {
            val c = px[i]
            rgb[3 * i] = (c shr 16).toByte()
            rgb[3 * i + 1] = (c shr 8).toByte()
            rgb[3 * i + 2] = c.toByte()
        }
        writeAtomic(f, sync = false) { it.write(rgb) }
        return f
    }

    private fun trim() {
        // the cache keeps the newest (and the engine's feature files beside them: <name>_<w>x<h>.feat) within a
        // budget of bytes - a video's 32 frames as well as a few large photos - never the last minute's, which a
        // prompt being prepared may still need
        val recent = System.currentTimeMillis() - 60_000L
        var kept = 0L
        (cache.listFiles() ?: emptyArray()).filter { it.name.endsWith(".rgb") || it.name.endsWith(".feat") }
            .sortedByDescending { it.lastModified() }
            .forEach { f ->
                kept += f.length()
                if (kept > CACHE_BYTES && f.lastModified() < recent) f.delete()
            }
    }

    companion object {
        /** The longest side an attached image is kept at. */
        const val STORE_SIDE = 2048
        /** The engine's inputs and features kept in the cache, at most (bytes). */
        private const val CACHE_BYTES = 128L shl 20
        /** A model token covers 2 x 2 patches of 16 pixels (Qwen3.6's vision encoder). */
        const val FACTOR = 32
        const val MIN_PIXELS = 65536L

        /**
         * Qwen's smart_resize: each side a multiple of [FACTOR] near the image's, the pixels at least [MIN_PIXELS] and
         * at most [maxPixels] (the aspect kept as near as that allows). Width, height.
         */
        fun fit(w: Int, h: Int, maxPixels: Long): Pair<Int, Int> {
            val f = FACTOR.toDouble()
            // Python's round(): half to even, as Math.rint
            var hb = max(f, Math.rint(h / f) * f)
            var wb = max(f, Math.rint(w / f) * f)
            if (hb * wb > maxPixels) {
                val beta = sqrt(h.toDouble() * w / maxPixels)
                hb = max(f, floor(h / beta / f) * f)
                wb = max(f, floor(w / beta / f) * f)
            } else if (hb * wb < MIN_PIXELS) {
                val beta = sqrt(MIN_PIXELS / (h.toDouble() * w))
                hb = ceil(h * beta / f) * f
                wb = ceil(w * beta / f) * f
            }
            return wb.toInt() to hb.toInt()
        }

        /** The model tokens of an image read at w x h (a [fit]). */
        fun tokens(w: Int, h: Int): Int = (w / FACTOR) * (h / FACTOR)

        /** The image detail setting's pixel limits: fast, detailed, most. */
        val DETAIL_PIXELS = longArrayOf(512L * 512, 768L * 768, 1024L * 1024)

        /** Gemma 4's vision encoder: 16-pixel patches pooled 3 x 3, one soft token each 48 x 48 pixels. */
        const val GEMMA_PATCH = 16
        const val GEMMA_POOL = 3
        const val GEMMA_FACTOR = GEMMA_PATCH * GEMMA_POOL

        /** The image detail setting's soft token budgets for Gemma 4 (its own budgets; 280 its default). */
        val GEMMA_DETAIL_TOKENS = intArrayOf(140, 280, 560)

        /** A video frame's budget (Gemma4Processor's for video). */
        const val GEMMA_FRAME_TOKENS = 70

        /**
         * Gemma 4's resize (transformers' get_aspect_ratio_preserving_size, as the engine's Vision::fit): the largest
         * size of at most [budget] soft tokens with both sides multiples of [GEMMA_FACTOR], the aspect kept - a small
         * image is made larger. Width, height.
         */
        fun fitGemma(w0: Int, h0: Int, budget: Int): Pair<Int, Int> {
            val f = GEMMA_FACTOR
            val w = max(w0, 1)
            val h = max(h0, 1)
            val factor = sqrt(budget.toDouble() * f * f / (h.toDouble() * w))
            var th = floor(factor * h / f).toInt() * f
            var tw = floor(factor * w / f).toInt() * f
            val longest = budget * f
            if (th == 0 && tw == 0) {
                th = f
                tw = f
            } else if (th == 0) {
                th = f
                tw = minOf(floor(w.toDouble() / h).toInt() * f, longest)
            } else if (tw == 0) {
                tw = f
                th = minOf(floor(h.toDouble() / w).toInt() * f, longest)
            }
            return tw to th
        }

        /** Gemma 4's soft tokens of an image read at w x h (a [fitGemma]). */
        fun gemmaTokens(w: Int, h: Int): Int = (w / GEMMA_FACTOR) * (h / GEMMA_FACTOR)
    }
}
