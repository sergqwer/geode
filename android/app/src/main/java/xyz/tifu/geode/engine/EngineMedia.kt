package xyz.tifu.geode.engine

/**
 * A prompt's medium as the :engine process reads it, in the prompt's order: an image or a video's frame (raw RGB,
 * [w] x [h]: ImageStore.prepare), or a sound clip (16 kHz mono float32: AudioStore; [w] its samples).
 */
data class EngineMedia(val kind: Int, val path: String, val w: Int, val h: Int = 0) {
    companion object {
        const val IMAGE = 0
        const val FRAME = 1
        const val AUDIO = 2
    }
}
