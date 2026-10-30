package xyz.tifu.geode.tokenizer

import xyz.tifu.geode.vision.ImageStore

/** An image of a message at the size the model reads it (ImageStore.fit); [name] is ImageStore's. */
data class TurnImage(val name: String, val width: Int, val height: Int) {
    val tokens: Int get() = ImageStore.tokens(width, height)
}

/** One message of a conversation; [content] is the answer without its reasoning; a question's images first. */
data class ChatTurn(val role: String, val content: String, val images: List<TurnImage> = emptyList())

/** Token ids, and the images their image pads stand for, in their order. */
class Prompt(val ids: IntArray, val images: List<TurnImage> = emptyList()) {
    val size: Int get() = ids.size
}

/**
 * Qwen3.6's chat template, token by token (the same ids as tokenizer.apply_chat_template; TokenizerTest checks it):
 *
 *     <|im_start|>system\n{system}<|im_end|>\n
 *     <|im_start|>user\n{images}{text}<|im_end|>\n
 *     <|im_start|>assistant\n{answer}<|im_end|>\n
 *     <|im_start|>assistant\n<think>\n                       (reasoning on)
 *     <|im_start|>assistant\n<think>\n\n</think>\n\n         (reasoning off)
 *
 * an image being <|vision_start|>, one <|image_pad|> for each of its tokens (the engine puts its features there),
 * <|vision_end|>. The engine's conversation only grows (geode.h), so a new question is sent as [nextTurn]: the end of
 * the answer the engine already holds, then the question. After a reset, [conversation] renders it all again, the
 * way the template does (earlier answers without their reasoning).
 */
class ChatFormat(private val tok: Tokenizer) {
    private val imStart = tok.special("<|im_start|>")
    private val imEnd = tok.special("<|im_end|>")
    private val visionStart = runCatching { tok.special("<|vision_start|>") }.getOrNull()
    private val visionEnd = runCatching { tok.special("<|vision_end|>") }.getOrNull()
    private val imagePad = runCatching { tok.special("<|image_pad|>") }.getOrNull()
    val think = tok.special("<think>")
    val thinkEnd = tok.special("</think>")
    val stopTokens = intArrayOf(imEnd, tok.special("<|endoftext|>"))

    /** The whole conversation, ending with the assistant's generation prompt. */
    fun conversation(turns: List<ChatTurn>, thinking: Boolean): Prompt {
        val out = ArrayList<Int>()
        val images = ArrayList<TurnImage>()
        for (t in turns) message(out, images, t)
        generationPrompt(out, thinking)
        return Prompt(out.toIntArray(), images)
    }

    /** After an answer: closes it and asks the next question. */
    fun nextTurn(user: ChatTurn, thinking: Boolean): Prompt {
        val out = ArrayList<Int>()
        val images = ArrayList<TurnImage>()
        out += imEnd
        out += tok.encode("\n").toList()
        message(out, images, user)
        generationPrompt(out, thinking)
        return Prompt(out.toIntArray(), images)
    }

    /**
     * The reasoning budget is spent: Qwen's way to end it (the model card's sentence, then </think>), after which the
     * model writes its answer.
     */
    fun closeThinking(): IntArray {
        val out = ArrayList<Int>()
        out += tok.encode("\n\nConsidering the limited time by the user, I have to give the solution based on the " +
            "thinking directly now.\n").toList()
        out += thinkEnd
        out += tok.encode("\n\n").toList()
        return out.toIntArray()
    }

    private fun message(out: MutableList<Int>, images: MutableList<TurnImage>, t: ChatTurn) {
        out += imStart
        val vs = visionStart
        val ve = visionEnd
        val pad = imagePad
        if (t.images.isEmpty() || vs == null || ve == null || pad == null) {
            out += tok.encode("${t.role}\n${t.content}").toList()
        } else {
            // the special tokens split the text, as Hugging Face's tokenizer splits it at them
            out += tok.encode("${t.role}\n").toList()
            for (im in t.images) {
                out += vs
                repeat(im.tokens) { out += pad }
                out += ve
                images += im
            }
            out += tok.encode(t.content).toList()
        }
        out += imEnd
        out += tok.encode("\n").toList()
    }

    private fun generationPrompt(out: MutableList<Int>, thinking: Boolean) {
        out += imStart
        out += tok.encode("assistant\n").toList()
        out += think
        if (thinking) {
            out += tok.encode("\n").toList()
        } else {
            out += tok.encode("\n\n").toList()
            out += thinkEnd
            out += tok.encode("\n\n").toList()
        }
    }
}
