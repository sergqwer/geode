package xyz.tifu.geode.tokenizer

import xyz.tifu.geode.media.AudioStore
import xyz.tifu.geode.vision.ImageStore

/** A medium of a message as the engine takes it, in the prompt's order: an image or a video's frame, or a clip. */
sealed interface TurnMedia {
    val name: String
}

/**
 * An image of a message at the size the model reads it (ImageStore.fit, fitGemma); [name] is ImageStore's. [frame]:
 * a video's frame (geode_add_video_frame) at [second] of it.
 */
data class TurnImage(override val name: String, val width: Int, val height: Int, val frame: Boolean = false,
                     val second: Int = 0) : TurnMedia {
    /** Qwen's image tokens (Gemma's: ImageStore.gemmaTokens). */
    val tokens: Int get() = ImageStore.tokens(width, height)
}

/** A sound clip of [samples] at 16 kHz mono; [name] is AudioStore's. */
data class TurnAudio(override val name: String, val samples: Int) : TurnMedia {
    val tokens: Int get() = AudioStore.tokens(samples.toLong())
}

/**
 * Token ids, and the media their pads stand for, in their order; [reasoning]: the prompt leaves the model inside its
 * reasoning (it ends by opening it).
 */
class Prompt(val ids: IntArray, val media: List<TurnMedia> = emptyList(), val reasoning: Boolean = false) {
    val size: Int get() = ids.size
    /** The images (and frames) among the media. */
    val images: List<TurnImage> get() = media.filterIsInstance<TurnImage>()
}
