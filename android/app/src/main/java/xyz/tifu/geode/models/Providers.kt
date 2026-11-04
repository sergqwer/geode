package xyz.tifu.geode.models

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import xyz.tifu.geode.util.Json
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** How a service is spoken to: OpenAI's chat completions, or Anthropic's messages. */
enum class Protocol { OPENAI, ANTHROPIC }

/**
 * A model of an outside service the user has a key for, answering the chats instead of the phone's own: what the
 * chats send goes to that service (Geode's agent still runs on the phone - its Linux, its tools, the files; only the
 * model's part is the service's). [baseUrl] is the API's root: OpenAI's kind up to its /v1 (or without, where the
 * service takes that), Anthropic's its host.
 */
data class Provider(
    val id: String,
    val name: String,
    val baseUrl: String,
    val protocol: Protocol,
    val model: String,
    /** The model reads images (the user says so: a service's list does not always tell). */
    val vision: Boolean = false,
    /** Its context, in tokens: the agent compacts the conversation before it is full. */
    val context: Int = DEFAULT_CONTEXT,
    /** The longest answer asked for, in tokens (some services refuse more than theirs). */
    val maxTokens: Int = DEFAULT_MAX_TOKENS,
) {
    val title: String get() = if (model.isEmpty()) name else "$name · $model"

    /** The service's host, as the user is told where the chats go. */
    val host: String get() = runCatching { URL(baseUrl).host }.getOrNull()?.takeIf { it.isNotEmpty() } ?: baseUrl

    private fun root() = baseUrl.trim().trimEnd('/')
    private fun anthropicRoot() = root().let { if (it.endsWith("/v1")) it else "$it/v1" }

    fun chatUrl(): String = when (protocol) {
        Protocol.OPENAI -> root() + "/chat/completions"
        Protocol.ANTHROPIC -> anthropicRoot() + "/messages"
    }

    fun modelsUrl(): String = when (protocol) {
        Protocol.OPENAI -> root() + "/models"
        Protocol.ANTHROPIC -> anthropicRoot() + "/models"
    }

    /** Whether this is OpenAI's own API (it takes max_completion_tokens and store; others want max_tokens). */
    val openAiItself: Boolean get() = host.endsWith("openai.com")

    companion object {
        const val DEFAULT_CONTEXT = 128_000
        const val DEFAULT_MAX_TOKENS = 8192
    }
}

/** A service ready to fill in: its name, its API's root, its protocol, a model it has, what that model takes. */
data class ProviderPreset(val name: String, val baseUrl: String, val protocol: Protocol, val model: String,
                          val vision: Boolean, val context: Int)

/**
 * The services the user added (files/providers.json) and their keys: each key encrypted with a key of the phone's
 * Keystore that never leaves it (AES-GCM, files/provider_keys.json) - the app keeps no backups (allowBackup off), and
 * a key is never written to the log. A key that cannot be decrypted (the Keystore's key gone: the app's data moved to
 * another phone) is simply missing: the user types it again.
 */
class Providers(context: Context) {
    private val dir = context.filesDir
    private val file = File(dir, "providers.json")
    private val keysFile = File(dir, "provider_keys.json")

    private val _list = MutableStateFlow(load())
    val list: StateFlow<List<Provider>> = _list.asStateFlow()

    fun find(id: String): Provider? = _list.value.firstOrNull { it.id == id }

    /** [p] added or changed; [key]: its new key (null: the one kept stays). */
    @Synchronized
    fun save(p: Provider, key: String?) {
        if (key != null) {
            val keys = keys().toMutableMap()
            if (key.isBlank()) keys.remove(p.id) else keys[p.id] = encrypt(key.trim())
            writeAtomic(keysFile, Json.write(keys))
        }
        val next = _list.value.filter { it.id != p.id } + p
        store(next.sortedBy { it.name.lowercase() })
    }

    @Synchronized
    fun delete(id: String) {
        val keys = keys().toMutableMap()
        if (keys.remove(id) != null) writeAtomic(keysFile, Json.write(keys))
        store(_list.value.filter { it.id != id })
    }

    fun hasKey(id: String): Boolean = keys().containsKey(id)

    /** [id]'s key in the clear, for a request to its service; null: none, or it cannot be read on this phone. */
    fun key(id: String): String? = keys()[id]?.let { runCatching { decrypt(it) }.getOrNull() }

    /** The models [p]'s service lists ([key]: the one typed, else the one kept); an IOException says why not. */
    suspend fun models(p: Provider, key: String?): List<String> = withContext(Dispatchers.IO) {
        val k = key?.trim()?.takeIf { it.isNotEmpty() } ?: key(p.id) ?: throw IOException("no key")
        val c = URL(p.modelsUrl()).openConnection() as HttpURLConnection
        try {
            c.connectTimeout = 15_000
            c.readTimeout = 30_000
            c.setRequestProperty("Accept", "application/json")
            when (p.protocol) {
                Protocol.OPENAI -> c.setRequestProperty("Authorization", "Bearer $k")
                Protocol.ANTHROPIC -> {
                    c.setRequestProperty("x-api-key", k)
                    c.setRequestProperty("anthropic-version", ANTHROPIC_VERSION)
                }
            }
            val code = c.responseCode
            val text = (if (code in 200..299) c.inputStream else c.errorStream)?.use { it.readBytes().decodeToString() }.orEmpty()
            if (code !in 200..299) throw IOException(errorMessage(code, text))
            @Suppress("UNCHECKED_CAST")
            val data = (Json.parse(text) as? Map<String, Any?>)?.get("data") as? List<Any?> ?: emptyList()
            data.mapNotNull { (it as? Map<*, *>)?.get("id") as? String }.distinct().sorted()
        } finally {
            c.disconnect()
        }
    }

    private fun store(list: List<Provider>) {
        writeAtomic(file, Json.write(list.map { p ->
            linkedMapOf("id" to p.id, "name" to p.name, "url" to p.baseUrl, "protocol" to p.protocol.name,
                "model" to p.model, "vision" to p.vision, "context" to p.context, "maxTokens" to p.maxTokens)
        }))
        _list.value = list
    }

    @Suppress("UNCHECKED_CAST")
    private fun load(): List<Provider> = runCatching {
        (Json.parse(file.readText()) as List<Map<String, Any?>>).mapNotNull { o ->
            Provider(
                id = o["id"] as? String ?: return@mapNotNull null,
                name = o["name"] as? String ?: "",
                baseUrl = o["url"] as? String ?: return@mapNotNull null,
                protocol = runCatching { Protocol.valueOf(o["protocol"] as String) }.getOrDefault(Protocol.OPENAI),
                model = o["model"] as? String ?: "",
                vision = o["vision"] == true,
                context = (o["context"] as? Number)?.toInt() ?: Provider.DEFAULT_CONTEXT,
                maxTokens = (o["maxTokens"] as? Number)?.toInt() ?: Provider.DEFAULT_MAX_TOKENS,
            )
        }
    }.getOrDefault(emptyList())

    @Suppress("UNCHECKED_CAST")
    private fun keys(): Map<String, String> = runCatching {
        (Json.parse(keysFile.readText()) as Map<String, Any?>).mapNotNull { (k, v) -> (v as? String)?.let { k to it } }.toMap()
    }.getOrDefault(emptyMap())

    private fun writeAtomic(f: File, text: String) {
        val tmp = File(f.path + ".tmp")
        FileOutputStream(tmp).use {
            it.write(text.toByteArray())
            it.fd.sync()
        }
        if (!tmp.renameTo(f)) throw IOException("cannot write ${f.name}")
    }

    // ---- the keys' encryption (the Keystore's key, made once)

    private fun secret(): SecretKey {
        val ks = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (ks.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val g = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        g.init(KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build())
        return g.generateKey()
    }

    private fun encrypt(s: String): String {
        val c = Cipher.getInstance(TRANSFORMATION)
        c.init(Cipher.ENCRYPT_MODE, secret())
        val iv = c.iv
        val data = c.doFinal(s.toByteArray())
        return Base64.encodeToString(byteArrayOf(iv.size.toByte()) + iv + data, Base64.NO_WRAP)
    }

    private fun decrypt(s: String): String {
        val b = Base64.decode(s, Base64.NO_WRAP)
        val n = b[0].toInt()
        val c = Cipher.getInstance(TRANSFORMATION)
        c.init(Cipher.DECRYPT_MODE, secret(), GCMParameterSpec(128, b, 1, n))
        return c.doFinal(b, 1 + n, b.size - 1 - n).decodeToString()
    }

    companion object {
        private const val KEYSTORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "geode_provider_keys"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        const val ANTHROPIC_VERSION = "2023-06-01"

        /** A new provider's id. */
        fun newId(): String = "p" + ByteArray(6).also { SecureRandom().nextBytes(it) }.joinToString("") { "%02x".format(it) }

        /** A service's error as the user is told it: its own message where its JSON has one. */
        fun errorMessage(code: Int, body: String): String {
            @Suppress("UNCHECKED_CAST")
            val o = runCatching { Json.parse(body) as? Map<String, Any?> }.getOrNull()
            val e = o?.get("error")
            val msg = when (e) {
                is Map<*, *> -> e["message"] as? String
                is String -> e
                else -> null
            } ?: (o?.get("message") as? String) ?: body.take(300).trim()
            return "HTTP $code" + if (msg.isNotEmpty()) ": $msg" else ""
        }

        val PRESETS = listOf(
            ProviderPreset("DeepSeek", "https://api.deepseek.com", Protocol.OPENAI, "deepseek-chat", false, 128_000),
            ProviderPreset("OpenRouter", "https://openrouter.ai/api/v1", Protocol.OPENAI, "", false, 128_000),
            ProviderPreset("OpenAI", "https://api.openai.com/v1", Protocol.OPENAI, "gpt-4.1-mini", true, 128_000),
            ProviderPreset("Anthropic", "https://api.anthropic.com", Protocol.ANTHROPIC, "claude-sonnet-5-5", true, 200_000),
        )
    }
}
