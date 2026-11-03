package xyz.tifu.geode.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Gemma 4's image resize against transformers' get_aspect_ratio_preserving_size (its values, computed there). */
class ImageFitTest {
    // w, h, budget -> fitted w, h (transformers 5.x, patch 16, pooling 3, max_patches = budget x 9)
    private val cases = """
        4032,3024,70,432,336 4032,3024,140,624,480 4032,3024,280,912,672 4032,3024,560,1296,960 3024,4032,70,336,432
        3024,4032,140,480,624 3024,4032,280,672,912 3024,4032,560,960,1296 1920,1080,70,528,288 1920,1080,140,720,384
        1920,1080,280,1056,576 1920,1080,560,1488,816 500,500,70,384,384 500,500,140,528,528 500,500,280,768,768
        500,500,560,1104,1104 5000,100,70,2832,48 5000,100,140,3984,48 5000,100,280,5664,96 5000,100,560,8016,144
        100,5000,70,48,2832 100,5000,140,48,3984 100,5000,280,96,5664 100,5000,560,144,8016 48,48,70,384,384
        48,48,140,528,528 48,48,280,768,768 48,48,560,1104,1104 2048,1536,70,432,336 2048,1536,140,624,480
        2048,1536,280,912,672 2048,1536,560,1296,960 777,333,70,576,240 777,333,140,864,336 777,333,280,1200,480
        777,333,560,1728,720 30,30,70,384,384 30,30,140,528,528 30,30,280,768,768 30,30,560,1104,1104
    """.trim().split(Regex("\\s+")).map { c -> c.split(",").map { it.toInt() } }

    @Test
    fun gemmaFitsLikeTransformers() {
        for ((w, h, budget, fw, fh) in cases) {
            assertEquals("$w x $h at $budget", fw to fh, ImageStore.fitGemma(w, h, budget))
            assertTrue(ImageStore.gemmaTokens(fw, fh) <= budget)
        }
    }
}
