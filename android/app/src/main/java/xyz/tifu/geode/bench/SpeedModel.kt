package xyz.tifu.geode.bench

import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.pow

/** What a model reads per token, from its geode.json. */
data class ModelShape(
    val name: String,
    val layers: Int,
    val experts: Int,
    val topK: Int,
    val hidden: Int,
    val moeFf: Int,
    /** Quantized weights read for every token from dense.bin (everything but the routed experts), by format. */
    val denseWeights: Map<String, Long>,
    val denseBytesPerToken: Long,
    /** dense.bin as it sits in memory. */
    val denseFileBytes: Long,
    /** BF16 / F32 weights read per token (the routers, the gates). */
    val otherWeights: Long,
    val expertFormat: String,
    val expertBlockBytes: Long,
    val attnLayers: Int,
    /** The KV cache a token: int8 K and V with a float scale a KV head (the engine's default), MTP layer included. */
    val kvBytesPerToken: Long,
    val stateBytes: Long,
    /** Tokens the KV cache grows by at a time (Model::kv_chunk_: 4096 with the Adreno 7xx+ path's tile8 packs). */
    val kvChunk: Int = ContextPlan.KV_CHUNK,
    /** The model's longest context (max_position_embeddings). */
    val maxPositions: Int = 262144,
) {
    val weightsPerExpert: Long get() = 3L * hidden * moeFf
    val expertWeightsPerToken: Long get() = layers.toLong() * topK * weightsPerExpert
    val expertBytesPerToken: Long get() = layers.toLong() * topK * expertBlockBytes

    companion object {
        private val CODE_BYTES = mapOf("q2" to 4, "q3" to 6, "nvfp4" to 8, "q8" to 16)

        fun bitsPerWeight(format: String): Double = ((CODE_BYTES[format] ?: 16) + 1) * 8.0 / 16.0

        fun read(dir: File): ModelShape? =
            runCatching { parse(JSONObject(File(dir, "geode.json").readText()), dir.name) }.getOrNull()

        /** The Q4 pack on Hugging Face (assets/shapes/geode-q4.json): the estimate before a model is downloaded */
        fun bundled(context: android.content.Context): ModelShape? = runCatching {
            parse(JSONObject(context.assets.open("shapes/geode-q4.json").bufferedReader().use { it.readText() }), "")
        }.getOrNull()

        private fun parse(m: JSONObject, fallbackName: String): ModelShape? = runCatching {
            val c = m.getJSONObject("config")
            val hidden = c.getInt("hidden_size")
            val tensors = m.getJSONObject("tensors")
            val dense = HashMap<String, Long>()
            var denseBytes = 0L
            var other = 0L
            for (name in tensors.keys()) {
                val t = tensors.getJSONObject(name)
                if (t.optString("file") != "dense.bin") continue
                val shape = t.getJSONArray("shape")
                var n = 1L
                for (i in 0 until shape.length()) n *= shape.getLong(i)
                val type = t.getString("type")
                val code = CODE_BYTES[type]
                if (code != null) {
                    val rows = shape.getLong(0)
                    val cols = n / rows
                    dense[type] = (dense[type] ?: 0L) + n
                    denseBytes += rows * (cols / 16) * (code + 1)
                } else if (type == "bf16" && shape.length() == 2) {
                    other += n
                }
            }
            val layers = m.getJSONObject("experts").getJSONArray("layers")
            val l0 = layers.getJSONObject(0)
            val format = l0.optString("format", "q2")
            var stride = l0.optLong("stride", 0)
            if (stride == 0L) {
                val offs = l0.getJSONArray("offsets")
                stride = if (offs.length() > 1) offs.getLong(1) - offs.getLong(0) else 0
            }
            val types = c.getJSONArray("layer_types")
            var attn = 0
            for (i in 0 until types.length()) if (types.getString(i) == "full_attention") attn++
            val gdn = types.length() - attn
            val kd = c.getInt("linear_num_key_heads") * c.getInt("linear_key_head_dim")
            val vd = c.getInt("linear_num_value_heads") * c.getInt("linear_value_head_dim")
            val conv = 2 * kd + vd
            val state = gdn.toLong() * (conv.toLong() * (c.getInt("linear_conv_kernel_dim") - 1) +
                c.getLong("linear_num_value_heads") * c.getInt("linear_key_head_dim") * c.getInt("linear_value_head_dim")) * 4
            val files = m.getJSONArray("files")
            var denseFile = 0L
            for (i in 0 until files.length()) {
                val f = files.getJSONObject(i)
                if (f.getString("name") == "dense.bin") denseFile = f.getLong("size")
            }
            ModelShape(
                name = m.optString("name", fallbackName),
                layers = c.getInt("num_hidden_layers"),
                experts = c.getInt("num_experts"),
                topK = c.getInt("num_experts_per_tok"),
                hidden = hidden,
                moeFf = c.getInt("moe_intermediate_size"),
                denseWeights = dense,
                denseBytesPerToken = denseBytes,
                denseFileBytes = denseFile,
                otherWeights = other,
                expertFormat = format,
                expertBlockBytes = stride,
                attnLayers = attn,
                // K and V in int8, a float scale for each KV head's row (Model::kv_scale); the MTP layer's KV too
                kvBytesPerToken = (attn + if (tensors.has("mtp.fc.weight")) 1 else 0).toLong() * 2 *
                    c.getInt("num_key_value_heads") * (c.getInt("head_dim") + 4L),
                stateBytes = state,
                kvChunk = if (m.optJSONObject("formats")?.optJSONObject("q4")?.optString("layout") == "tile8") 4096
                    else ContextPlan.KV_CHUNK,
                maxPositions = c.optInt("max_position_embeddings", 262144),
            )
        }.getOrNull()
    }
}

/** The measured device, as the estimate needs it (from the stored benchmark results). */
data class DeviceSpeed(
    val bigCores: Int,
    /** Gweights/s of the engine's kernel per format: one token on all big cores, four tokens per big core. */
    val decodeGw: Map<String, Double>,
    val prefillGwPerCore: Map<String, Double>,
    val ramGbs: Double,
    val flashMbs: Double,
    val fp32Gflops: Double,
) {
    companion object {
        fun from(results: JSONObject): DeviceSpeed? {
            val cpu = results.optJSONObject("cpu") ?: return null
            val mem = results.optJSONObject("memory") ?: return null
            val flash = results.optJSONObject("flash") ?: return null
            val decode = HashMap<String, Double>()
            val prefill = HashMap<String, Double>()
            val ks = cpu.optJSONArray("kernels") ?: return null
            for (i in 0 until ks.length()) {
                val k = ks.getJSONObject(i)
                decode[k.getString("format")] = k.getDouble("big_cores_gw")
                prefill[k.getString("format")] = k.getDouble("four_tokens_one_core_gw")
            }
            var ram = 0.0
            val cs = mem.optJSONArray("configs")
            for (i in 0 until (cs?.length() ?: 0)) {
                val cfg = cs!!.getJSONObject(i)
                if (cfg.getString("name") == "big_cores" || ram == 0.0) ram = max(ram, cfg.getDouble("read_gbs"))
            }
            // flash: the expert block's best rate (several reads in flight, as the engine reads), else 1 MiB's
            var best = 0.0
            var bestExpert = 0.0
            val rs = flash.optJSONArray("random")
            for (i in 0 until (rs?.length() ?: 0)) {
                val r = rs!!.getJSONObject(i)
                if (r.getDouble("block") >= 1 shl 19) best = max(best, r.getDouble("mbs"))
                if (r.optBoolean("expert_block")) bestExpert = max(bestExpert, r.getDouble("mbs"))
            }
            var fp32 = 0.0
            val ps = cpu.optJSONArray("peaks")
            if (ps != null && ps.length() > 0) fp32 = ps.getJSONObject(0).optDouble("fp32_gflops", 0.0)
            return DeviceSpeed(
                bigCores = cpu.optInt("big_cores", 1),
                decodeGw = decode,
                prefillGwPerCore = prefill,
                ramGbs = ram,
                flashMbs = if (bestExpert > 0) bestExpert else best,
                fp32Gflops = fp32,
            )
        }
    }
}

/**
 * How the engine lays a memory budget out (model.cpp, Model::open): the dense weights, the recurrent state, one chunk
 * of context and the expert cache. Each further KV_CHUNK tokens of context take their memory from the cache, which
 * gives back its least used experts; a new chat returns them.
 */
data class ContextPlan(
    /** Tokens a chunk of context, and its bytes. */
    val chunkTokens: Int,
    val chunkBytes: Long,
    val slotBytes: Long,
    /** Expert-cache slots in a new chat, and how many each further chunk of context takes. */
    val startSlots: Long,
    val slotsPerChunk: Long,
    /** The longest context the budget allows (the engine stops there), at most the one asked for. */
    val maxContext: Int,
) {
    /** The expert cache's slots once the context holds this many tokens. */
    fun slotsAt(tokens: Int): Long {
        val chunks = (max(tokens, 1) + chunkTokens - 1) / chunkTokens
        return (startSlots - (chunks - 1L) * slotsPerChunk).coerceAtLeast(0)
    }

    companion object {
        /** Model::kKvChunk (packs the Adreno 7xx+ path reads: ModelShape.kvChunk). */
        const val KV_CHUNK = 1024
    }
}

/** Where a token's time goes, and the speeds that follow, for one model on one device. */
data class Estimate(
    val cacheExperts: Long,
    val cacheShare: Double,
    val hitRate: Double,
    val hitMeasured: Boolean,
    val computeMs: Double,
    val ramMs: Double,
    val flashMs: Double,
    /** Every resource busy at once (reads fully hidden behind arithmetic, or the other way round). */
    val decodeOverlapped: Double,
    /** Reads not hidden at all. */
    val decodeSerial: Double,
    val prefillComputeMs: Double,
    val prefillFlashMs: Double,
    val prefill: Double,
    val denseMbPerToken: Double,
    val expertMbPerToken: Double,
    val flashMbPerToken: Double,
)

object SpeedModel {
    /** Prompt batches the engine reads at once (Model::kMaxBatch). */
    private const val BATCH = 64

    /**
     * Expert-cache hit rate against the share of all experts the cache holds: the router-trace simulator's EMA
     * policy (PLAN.md, P1) and the phone's own runs (26% held → 82-84%).
     */
    private val HIT_CURVE = listOf(0.0 to 0.0, 0.054 to 0.48, 0.108 to 0.64, 0.16 to 0.73, 0.217 to 0.80,
        0.26 to 0.83, 0.5 to 0.93, 1.0 to 1.0)

    fun hitFor(share: Double): Double {
        val s = share.coerceIn(0.0, 1.0)
        for (i in 1 until HIT_CURVE.size) {
            val (x1, y1) = HIT_CURVE[i]
            val (x0, y0) = HIT_CURVE[i - 1]
            if (s <= x1) return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
        }
        return 1.0
    }

    /** The least budget the engine opens the model with: the dense weights, the state, the scratch, one chunk of
     *  context, and the expert slots one layer and the next one's guesses pin (as [plan] lays them out). */
    fun minBudget(shape: ModelShape): Long {
        val slot = max(1L, (shape.expertBlockBytes + 4095) / 4096 * 4096)
        return shape.denseFileBytes + shape.stateBytes + (32L shl 20) + shape.kvBytesPerToken * shape.kvChunk +
            (2L * shape.topK + 2) * slot
    }

    fun plan(shape: ModelShape, budgetBytes: Long, context: Int): ContextPlan {
        val chunk = shape.kvBytesPerToken * shape.kvChunk
        val slot = max(1L, (shape.expertBlockBytes + 4095) / 4096 * 4096)
        // + batch scratch and read buffers, + the first chunk of the context
        val fixed = shape.denseFileBytes + shape.stateBytes + (32L shl 20) + chunk
        val slots = ((budgetBytes - fixed) / slot).coerceAtLeast(0)
        val perChunk = (chunk + slot - 1) / slot
        // the cache keeps what one layer and the next one's guesses pin
        val chunks = 1 + (slots - (2L * shape.topK + 2)).coerceAtLeast(0) / max(1L, perChunk)
        val longest = minOf(context.toLong(), shape.maxPositions.toLong(), chunks * shape.kvChunk)
        return ContextPlan(shape.kvChunk, chunk, slot, slots, perChunk, longest.toInt())
    }

    /** The speed in a new chat (the context's first chunk; a long one takes cache, see ContextPlan). */
    fun estimate(shape: ModelShape, dev: DeviceSpeed, budgetBytes: Long, measuredHit: Double): Estimate {
        val slots = plan(shape, budgetBytes, shape.kvChunk).startSlots
        val all = shape.layers.toLong() * shape.experts
        val share = slots.toDouble() / all
        val hit = if (measuredHit > 0) measuredHit else hitFor(share)

        fun gw(map: Map<String, Double>, f: String) = (map[f] ?: map["q8"] ?: 1.0).coerceAtLeast(0.01)
        val otherSec = if (dev.fp32Gflops > 0) shape.otherWeights * 2.0 / (dev.fp32Gflops * 0.25 * 1e9) else 0.0

        // generating: every weight once per token, on all big cores
        var compute = otherSec
        for ((f, w) in shape.denseWeights) compute += w / (gw(dev.decodeGw, f) * 1e9)
        compute += shape.expertWeightsPerToken / (gw(dev.decodeGw, shape.expertFormat) * 1e9)
        val ram = (shape.denseBytesPerToken + hit * shape.expertBytesPerToken) / (dev.ramGbs.coerceAtLeast(0.1) * 1e9)
        val flashBytes = (1 - hit) * shape.expertBytesPerToken
        val flash = flashBytes / (dev.flashMbs.coerceAtLeast(1.0) * 1e6)
        val busy = max(compute, ram)

        // a prompt: four tokens per kernel pass, every big core; each batch reads the experts it touches once
        var pc = otherSec
        for ((f, w) in shape.denseWeights) pc += w / (gw(dev.prefillGwPerCore, f) * dev.bigCores * 1e9)
        pc += shape.expertWeightsPerToken / (gw(dev.prefillGwPerCore, shape.expertFormat) * dev.bigCores * 1e9)
        val unique = shape.experts * (1 - (1 - shape.topK.toDouble() / shape.experts).pow(BATCH))
        val pFlash = shape.layers * unique * (1 - hit) * shape.expertBlockBytes / BATCH / (dev.flashMbs.coerceAtLeast(1.0) * 1e6)

        return Estimate(
            cacheExperts = slots,
            cacheShare = share,
            hitRate = hit,
            hitMeasured = measuredHit > 0,
            computeMs = compute * 1e3,
            ramMs = ram * 1e3,
            flashMs = flash * 1e3,
            decodeOverlapped = 1.0 / max(busy, flash),
            decodeSerial = 1.0 / (busy + flash),
            prefillComputeMs = pc * 1e3,
            prefillFlashMs = pFlash * 1e3,
            prefill = 1.0 / (pc + pFlash),
            denseMbPerToken = shape.denseBytesPerToken / 1e6,
            expertMbPerToken = shape.expertBytesPerToken / 1e6,
            flashMbPerToken = flashBytes / 1e6,
        )
    }

    /** The same model with every matrix in another format: the speed side of the quality trade. */
    fun withFormat(shape: ModelShape, format: String): ModelShape {
        val bits = ModelShape.bitsPerWeight(format)
        val denseW = shape.denseWeights.values.sum()
        val block = ((shape.weightsPerExpert * bits / 8 + 4095) / 4096).toLong() * 4096
        return shape.copy(
            denseWeights = mapOf(format to denseW),
            denseBytesPerToken = (denseW * bits / 8).toLong(),
            denseFileBytes = shape.denseFileBytes - shape.denseBytesPerToken + (denseW * bits / 8).toLong(),
            expertFormat = format,
            expertBlockBytes = block,
        )
    }
}
