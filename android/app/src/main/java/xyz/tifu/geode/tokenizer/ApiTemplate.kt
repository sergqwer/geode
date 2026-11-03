package xyz.tifu.geode.tokenizer

/**
 * A model family's whole chat template for the local server's API, and how its answers are read: [ChatTemplate]
 * (Qwen3.6's) or [GemmaTemplate] (Gemma 4's), by the pack's tokenizer ([of]).
 */
interface ApiTemplate {
    /**
     * The conversation from the start: the system turn (tools in it), the messages, the generation prompt - or with
     * [prefill] the last message (the assistant's) left open, the model going on from it.
     */
    fun render(messages: List<TemplateMessage>, tools: List<Any?>, thinking: Boolean, prefill: Boolean = false): Prompt

    /**
     * What continues a conversation the engine holds - [messages]' first [held], the last of them the assistant's
     * answer as the engine generated it, its closing token not read (the model stopped at it): that closing, the
     * messages after it, the generation prompt.
     */
    fun continuation(messages: List<TemplateMessage>, held: Int, thinking: Boolean): Prompt

    /** The tokens an answer ends at. */
    val stopTokens: IntArray

    /** The reasoning's opening and closing tokens. */
    val thinkStart: Int
    val thinkEnd: Int

    /**
     * The model opens its reasoning itself ([thinkStart], then a header line: Gemma's "<|channel>thought\n"); else the
     * generation prompt opens it ([Prompt.reasoning]) and a [thinkStart] the model writes means nothing.
     */
    val modelOpensReasoning: Boolean

    /** The token a tool call begins with. */
    val callStart: Int

    /** The reasoning budget is spent: the tokens that end the reasoning, after which the model writes its answer. */
    fun closeThinking(): IntArray

    /**
     * The calls in the text the model wrote from its first [callStart] on: their names and arguments (typed as
     * [tools]' schemas say, where the format leaves it open).
     */
    fun parseCalls(text: String, tools: List<Map<String, Any?>>): List<Pair<String, Map<String, Any?>>>

    /** The tokens of [p]'s system turn (its closing and newline included), when it starts with one; 0: none. */
    fun systemTurn(p: Prompt): Int

    companion object {
        /** The template the tokenizer's model takes: Gemma 4's has <|turn>, Qwen's <|im_start|>. */
        fun of(tok: Tokenizer): ApiTemplate = if (tok.hasSpecial("<|turn>")) GemmaTemplate(tok) else ChatTemplate(tok)
    }
}
