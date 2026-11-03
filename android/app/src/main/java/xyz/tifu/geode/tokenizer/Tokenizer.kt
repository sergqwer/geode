package xyz.tifu.geode.tokenizer

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * A pack's tokenizer.bin (tools/tokbin.py documents the format), of one of two kinds:
 * - Qwen's byte-level BPE: NFC, the Unicode-class pre-tokenizer regex, then BPE merges on the UTF-8 bytes of each
 *   piece;
 * - SentencePiece-style (Gemma 4): spaces as "\u2581", no pre-split, BPE over the text's characters (those without a
 *   token of their own as their UTF-8 bytes' <0xXX> tokens), merged lowest rank first with a heap (the whole text is
 *   one piece).
 * Special tokens are never parsed out of text by [encode]; the chat template inserts them by id ([ChatFormat]).
 */
class Tokenizer private constructor(
    private val vocabBytes: ByteArray,
    private val vocabOffsets: IntArray,
    private val byteToId: IntArray,
    private val merges: MergeTable,
    private val specialIds: Map<String, Int>,
    private val specialText: Map<Int, ByteArray>,
    private val pattern: Pattern?,
    private val nfc: Boolean,
    // SentencePiece-style: the code points with a token of their own (sorted) and those tokens
    private val charPoints: IntArray?,
    private val charIds: IntArray?,
) {
    /** Whether this is the SentencePiece-style kind (Gemma's). */
    val sentencePiece: Boolean get() = charPoints != null

    val vocabSize: Int
        get() = maxOf(vocabOffsets.size - 1, (specialText.keys.maxOrNull() ?: -1) + 1)

    fun special(text: String): Int = specialIds[text] ?: error("the tokenizer has no $text")

    /** Whether the tokenizer has this special token. */
    fun hasSpecial(text: String): Boolean = text in specialIds

    /** Text -> token ids (the way Hugging Face's tokenizer.encode(text, add_special_tokens=False) does it). */
    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        if (charPoints != null) return encodeSentencePiece(text)
        val s = if (nfc) Normalizer.normalize(text, Normalizer.Form.NFC) else text
        val out = IntList()
        val m = pattern!!.matcher(s)
        var last = 0
        while (m.find()) {
            if (m.start() > last) bpe(s.substring(last, m.start()), out)
            if (m.end() > m.start()) bpe(s.substring(m.start(), m.end()), out)
            last = m.end()
        }
        if (last < s.length) bpe(s.substring(last), out)
        return out.toArray()
    }

    /** The special tokens' texts, longest first (a text is split at them in [encodeWithSpecials]). */
    private val specialsLongestFirst: List<Pair<String, Int>> by lazy {
        specialIds.entries.map { it.key to it.value }.sortedByDescending { it.first.length }
    }

    /**
     * Text -> token ids with the special tokens' texts in it as those tokens, the way Hugging Face's tokenizer splits a
     * rendered chat template at its added tokens (the template's own "<tool_call>" in its instructions, say) - but for
     * [except]: an image pad without its image would leave the engine a hole.
     */
    fun encodeWithSpecials(text: String, except: Set<Int> = emptySet()): IntArray {
        val out = IntList()
        var last = 0
        var i = text.indexOf('<')
        while (i >= 0) {
            val hit = specialsLongestFirst.firstOrNull { (s, id) -> id !in except && text.startsWith(s, i) }
            if (hit == null) {
                i = text.indexOf('<', i + 1)
                continue
            }
            for (id in encode(text.substring(last, i))) out.add(id)
            out.add(hit.second)
            i += hit.first.length
            last = i
            i = text.indexOf('<', i)
        }
        for (id in encode(text.substring(last))) out.add(id)
        return out.toArray()
    }

    /** The raw bytes of a token (a special token's text in UTF-8). */
    fun bytes(id: Int): ByteArray {
        if (id in 0 until vocabOffsets.size - 1)
            return vocabBytes.copyOfRange(vocabOffsets[id], vocabOffsets[id + 1])
        return specialText[id] ?: ByteArray(0)
    }

    fun decode(ids: IntArray): String {
        val b = ByteArrayOutputStream()
        for (id in ids) b.write(bytes(id))
        return b.toString(Charsets.UTF_8.name())
    }

    private fun bpe(piece: String, out: IntList) {
        val utf8 = piece.toByteArray(Charsets.UTF_8)
        val ids = IntArray(utf8.size) { byteToId[utf8[it].toInt() and 0xff] }
        var n = ids.size
        // merge the adjacent pair of the lowest rank until none is left; pieces are short (one word, one run of
        // spaces), so the simple quadratic scan is fine
        while (n > 1) {
            var bestRank = Int.MAX_VALUE
            var best = -1
            for (i in 0 until n - 1) {
                val r = merges.rank(ids[i], ids[i + 1])
                if (r < bestRank) {
                    bestRank = r
                    best = i
                }
            }
            if (best < 0) break
            ids[best] = merges.merged(ids[best], ids[best + 1])
            System.arraycopy(ids, best + 2, ids, best + 1, n - best - 2)
            n--
        }
        for (i in 0 until n) out.add(ids[i])
    }

    /**
     * The SentencePiece-style encode: the symbols (a character's token, or its bytes' fallback tokens), then the
     * adjacent pair of the lowest rank merged until none is left - the leftmost of equal ranks first, as Hugging
     * Face's BPE does. A heap of (rank, position) over a linked list: a text of any length in n log n.
     */
    private fun encodeSentencePiece(text: String): IntArray {
        val points = charPoints!!
        val ids = charIds!!
        val sym = IntList()
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            val c = if (cp == ' '.code) 0x2581 else cp
            val at = points.binarySearch(c)
            if (at >= 0) sym.add(ids[at])
            else for (b in String(Character.toChars(c)).toByteArray(Charsets.UTF_8)) sym.add(byteToId[b.toInt() and 0xff])
            i += Character.charCount(cp)
        }
        val n = sym.size
        val s = sym.toArray()
        if (n < 2) return s
        val next = IntArray(n) { it + 1 }
        val prev = IntArray(n) { it - 1 }
        val alive = BooleanArray(n) { true }
        val heap = LongHeap(n)
        fun push(at: Int) {
            val j = next[at]
            if (j >= n) return
            val r = merges.rank(s[at], s[j])
            if (r != Int.MAX_VALUE) heap.push((r.toLong() shl 32) or at.toLong())
        }
        for (k in 0 until n - 1) push(k)
        while (heap.size > 0) {
            val e = heap.pop()
            val r = (e ushr 32).toInt()
            val at = (e and 0xffffffffL).toInt()
            val j = next[at]
            // stale: a side was merged away or changed since (a pair's rank is unique: the same rank, the same pair)
            if (!alive[at] || j >= n || merges.rank(s[at], s[j]) != r) continue
            s[at] = merges.merged(s[at], s[j])
            alive[j] = false
            next[at] = next[j]
            if (next[j] < n) prev[next[j]] = at
            if (prev[at] >= 0) push(prev[at])
            push(at)
        }
        val out = IntList()
        var k = 0
        while (k < n) {
            out.add(s[k])
            k = next[k]
        }
        return out.toArray()
    }

    /** A min-heap of longs (rank << 32 | position). */
    private class LongHeap(capacity: Int) {
        private var a = LongArray(maxOf(16, capacity))
        var size = 0
            private set

        fun push(v: Long) {
            if (size == a.size) a = a.copyOf(size * 2)
            var i = size++
            a[i] = v
            while (i > 0) {
                val p = (i - 1) / 2
                if (a[p] <= a[i]) break
                val t = a[p]; a[p] = a[i]; a[i] = t
                i = p
            }
        }

        fun pop(): Long {
            val top = a[0]
            a[0] = a[--size]
            var i = 0
            while (true) {
                val l = 2 * i + 1
                if (l >= size) break
                val c = if (l + 1 < size && a[l + 1] < a[l]) l + 1 else l
                if (a[i] <= a[c]) break
                val t = a[c]; a[c] = a[i]; a[i] = t
                i = c
            }
            return top
        }
    }

    /** Turns a stream of tokens into text, holding back the bytes of a character that is not complete yet. */
    inner class StreamDecoder {
        private val pending = ByteArrayOutputStream()

        fun push(id: Int): String {
            pending.write(bytes(id))
            val b = pending.toByteArray()
            val cut = completePrefix(b)
            pending.reset()
            pending.write(b, cut, b.size - cut)
            return String(b, 0, cut, Charsets.UTF_8)
        }

        fun flush(): String {
            val s = pending.toString(Charsets.UTF_8.name())
            pending.reset()
            return s
        }

        /** The length of the longest prefix that does not end inside a UTF-8 sequence. */
        private fun completePrefix(b: ByteArray): Int {
            var i = b.size - 1
            var back = 0
            while (i >= 0 && back < 4) {
                val c = b[i].toInt() and 0xff
                if (c and 0xC0 != 0x80) {
                    val need = when {
                        c < 0x80 -> 1
                        c shr 5 == 0x6 -> 2
                        c shr 4 == 0xE -> 3
                        c shr 3 == 0x1E -> 4
                        else -> 1 // not a lead byte: let the decoder replace it
                    }
                    return if (b.size - i >= need) b.size else i
                }
                i--
                back++
            }
            return b.size
        }
    }

    /** Pair (left, right) -> rank and merged id, open addressing over primitive arrays (~8 MB for Qwen's 248k). */
    private class MergeTable(capacity: Int) {
        private val mask = Integer.highestOneBit(maxOf(4, capacity * 2 - 1)) * 2 - 1
        private val keys = LongArray(mask + 1) { EMPTY }
        private val ranks = IntArray(mask + 1)
        private val mergedIds = IntArray(mask + 1)

        fun put(left: Int, right: Int, rank: Int, merged: Int) {
            val k = key(left, right)
            var i = slot(k)
            while (keys[i] != EMPTY && keys[i] != k) i = (i + 1) and mask
            if (keys[i] == k) return // a duplicate merge: the first (lower rank) wins, as in Hugging Face's BPE
            keys[i] = k
            ranks[i] = rank
            mergedIds[i] = merged
        }

        fun rank(left: Int, right: Int): Int {
            val i = find(left, right)
            return if (i < 0) Int.MAX_VALUE else ranks[i]
        }

        fun merged(left: Int, right: Int): Int = mergedIds[find(left, right)]

        private fun find(left: Int, right: Int): Int {
            val k = key(left, right)
            var i = slot(k)
            while (true) {
                val v = keys[i]
                if (v == k) return i
                if (v == EMPTY) return -1
                i = (i + 1) and mask
            }
        }

        private fun key(left: Int, right: Int) = (left.toLong() shl 32) or (right.toLong() and 0xffffffffL)

        private fun slot(k: Long): Int {
            var h = k * -0x61c8864680b583ebL // Fibonacci hashing
            h = h xor (h ushr 29)
            return h.toInt() and mask
        }

        companion object {
            const val EMPTY = -1L
        }
    }

    private class IntList {
        private var a = IntArray(64)
        private var n = 0
        val size: Int get() = n
        fun add(v: Int) {
            if (n == a.size) a = a.copyOf(n * 2)
            a[n++] = v
        }

        fun toArray(): IntArray = a.copyOf(n)
    }

    companion object {
        const val FILE = "tokenizer.bin"

        fun load(file: File): Tokenizer = load(file.readBytes())

        fun load(data: ByteArray): Tokenizer {
            val b = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { b.get(it) }
            require(String(magic, Charsets.US_ASCII) == "GTOK") { "not a tokenizer.bin" }
            val version = b.int
            require(version == 1 || version == 2) { "tokenizer.bin version $version is not supported" }
            val flags = b.int
            val sp = flags and 2 != 0
            require(!sp || version == 2) { "tokenizer.bin: a SentencePiece-style table needs version 2" }

            val nVocab = b.int
            val offsets = IntArray(nVocab + 1)
            val start = b.position()
            // first pass: the offsets; the bytes are copied in one go after
            var total = 0
            for (i in 0 until nVocab) {
                val len = b.short.toInt() and 0xffff
                offsets[i] = total
                total += len
                b.position(b.position() + len)
            }
            offsets[nVocab] = total
            val vocab = ByteArray(total)
            b.position(start)
            var at = 0
            for (i in 0 until nVocab) {
                val len = b.short.toInt() and 0xffff
                b.get(vocab, at, len)
                at += len
            }

            // (a SentencePiece-style table names its byte-fallback tokens at its end: read below)
            val byteToId = IntArray(256) { -1 }
            if (!sp) {
                for (i in 0 until nVocab)
                    if (offsets[i + 1] - offsets[i] == 1) byteToId[vocab[offsets[i]].toInt() and 0xff] = i
                require(byteToId.all { it >= 0 }) { "tokenizer.bin: a byte has no token" }
            }

            val nMerges = b.int
            val merges = MergeTable(nMerges)
            for (r in 0 until nMerges) {
                val left = b.int
                val right = b.int
                val merged = b.int
                merges.put(left, right, r, merged)
            }

            val nSpecial = b.int
            val ids = HashMap<String, Int>(nSpecial * 2)
            val texts = HashMap<Int, ByteArray>(nSpecial * 2)
            for (i in 0 until nSpecial) {
                val id = b.int
                val len = b.short.toInt() and 0xffff
                val s = ByteArray(len).also { b.get(it) }
                ids[String(s, Charsets.UTF_8)] = id
                texts[id] = s
            }

            val plen = b.int
            val p = ByteArray(plen).also { b.get(it) }
            if (!sp)
                return Tokenizer(vocab, offsets, byteToId, merges, ids, texts,
                    Pattern.compile(String(p, Charsets.UTF_8)), flags and 1 != 0, null, null)
            val nChars = b.int
            val points = IntArray(nChars)
            val charIds = IntArray(nChars)
            for (i in 0 until nChars) {
                points[i] = b.int
                charIds[i] = b.int
                require(i == 0 || points[i] > points[i - 1]) { "tokenizer.bin: the characters are not sorted" }
            }
            for (i in 0 until 256) byteToId[i] = b.int
            require(byteToId.all { it in 0 until nVocab }) { "tokenizer.bin: a byte has no fallback token" }
            return Tokenizer(vocab, offsets, byteToId, merges, ids, texts, null, false, points, charIds)
        }
    }
}
