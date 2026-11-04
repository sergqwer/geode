package xyz.tifu.geode.data

import android.content.Context
import android.hardware.biometrics.BiometricManager
import android.os.SystemClock
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyPermanentlyInvalidatedException
import android.security.keystore.KeyProperties
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

/**
 * The app's lock: a PIN or a password before the chats show (or the phone's fingerprint instead of it, when on),
 * asked when the app starts and when it comes back after [Config.afterMs] in the background.
 *
 * Kept apart from the settings' DataStore: whether the app is locked must be known before its first frame (no glimpse
 * of the chats while the settings load), so its few values are in SharedPreferences, read once, synchronously. The code
 * itself is never stored, only its PBKDF2 hash with a random salt; wrong tries are counted there too, so closing the
 * app does not reset the pause they earn. The fingerprint unlocks through a Keystore key that takes a strong biometric
 * for every use and that a fingerprint added to the phone invalidates: someone who knows the phone's own PIN cannot add
 * a finger of theirs and come in with it.
 *
 * What it guards against is a person holding the phone unlocked. The chats are in the app's private storage, which
 * Android encrypts; they are not encrypted with the code.
 */
class AppLock(context: Context) {
    enum class Kind { NONE, PIN, PASSWORD }

    data class Config(val kind: Kind = Kind.NONE, val fingerprint: Boolean = false, val afterMs: Long = DEFAULT_AFTER_MS) {
        val on: Boolean get() = kind != Kind.NONE
    }

    sealed interface Check {
        data object Ok : Check
        data object Wrong : Check
        /** Too many wrong tries: the next one is taken from [untilMs] on (the wall clock). */
        data class Wait(val untilMs: Long) : Check
    }

    private val app = context.applicationContext
    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val _config = MutableStateFlow(read())
    val config: StateFlow<Config> = _config.asStateFlow()

    private val _locked = MutableStateFlow(_config.value.on)
    /** The chats are behind the lock screen. */
    val locked: StateFlow<Boolean> = _locked.asStateFlow()

    private val _locks = MutableStateFlow(0)
    /** The times the app locked since it started: the lock screen offers the fingerprint once for each. */
    val locks: StateFlow<Int> = _locks.asStateFlow()

    private val _fingerprintLost = MutableStateFlow(false)
    /** The fingerprint was on, but its key is gone (the phone's fingerprints changed): it is off now. */
    val fingerprintLost: StateFlow<Boolean> = _fingerprintLost.asStateFlow()

    /** When the app went to the background (elapsedRealtime: sleep counts); 0 while it is in front. */
    private var awaySince = 0L
    /** When the code or the fingerprint was last right (elapsedRealtime). */
    private var unlockedAt = 0L

    private fun read(): Config {
        val kind = runCatching { Kind.valueOf(prefs.getString(K_KIND, null) ?: "") }.getOrDefault(Kind.NONE)
        return Config(
            kind = if (prefs.getString(K_HASH, null) != null) kind else Kind.NONE,
            fingerprint = prefs.getBoolean(K_FINGER, false),
            afterMs = prefs.getLong(K_AFTER, DEFAULT_AFTER_MS),
        )
    }

    private fun changed() {
        _config.value = read()
    }

    /** The app went to the background (MainActivity.onStop): locked at once when [Config.afterMs] is 0. */
    fun away() {
        awaySince = SystemClock.elapsedRealtime()
        if (config.value.afterMs == 0L) lock()
    }

    /**
     * The app is in front again (MainActivity.onStart): locked when it was away long enough - unless it was unlocked
     * meanwhile (a phone whose fingerprint prompt is an activity of its own stops the app while the finger is taken).
     */
    fun back() {
        val since = awaySince
        awaySince = 0
        if (since > 0 && unlockedAt < since && SystemClock.elapsedRealtime() - since >= config.value.afterMs) lock()
    }

    /** Behind the lock screen now (when there is a lock). */
    fun lock() {
        if (!config.value.on || _locked.value) return
        _locked.value = true
        _locks.value++
    }

    /** The code or the fingerprint was right. */
    fun unlock() {
        unlockedAt = SystemClock.elapsedRealtime()
        _locked.value = false
    }

    /** The lock set, or its code changed, to [code] (the tries counted so far forgotten; the fingerprint kept). */
    suspend fun set(kind: Kind, code: String) = withContext(Dispatchers.Default) {
        require(kind != Kind.NONE)
        val salt = ByteArray(16).also { SecureRandom().nextBytes(it) }
        val hash = hash(code, salt, ROUNDS)
        prefs.edit().putString(K_KIND, kind.name).putString(K_HASH, "$ROUNDS:${b64(salt)}:${b64(hash)}")
            .putInt(K_FAILS, 0).putLong(K_UNTIL, 0).commit()
        changed()
    }

    /** No lock: the code's hash, the fingerprint's key and the tries gone (how long away before it locks stays). */
    fun clear() {
        prefs.edit().remove(K_KIND).remove(K_HASH).remove(K_FINGER).remove(K_FAILS).remove(K_UNTIL).commit()
        runCatching { keyStore().deleteEntry(KEY_ALIAS) }
        _fingerprintLost.value = false
        changed()
        _locked.value = false
    }

    fun setAfter(ms: Long) {
        prefs.edit().putLong(K_AFTER, ms).commit()
        changed()
    }

    /** Until when a try is not taken (0: now); a clock turned back does not make the pause longer than its longest. */
    fun waitUntil(): Long {
        val now = System.currentTimeMillis()
        val until = prefs.getLong(K_UNTIL, 0)
        if (until <= now) return 0
        if (until > now + MAX_WAIT_MS) {
            prefs.edit().putLong(K_UNTIL, now + MAX_WAIT_MS).commit()
            return now + MAX_WAIT_MS
        }
        return until
    }

    /**
     * Whether [code] is the lock's. From the [FREE_TRIES]th wrong one in a row on, each earns a pause, doubling from
     * [FIRST_WAIT_MS] up to [MAX_WAIT_MS].
     */
    suspend fun check(code: String): Check = withContext(Dispatchers.Default) {
        waitUntil().takeIf { it > 0 }?.let { return@withContext Check.Wait(it) }
        val stored = prefs.getString(K_HASH, null) ?: return@withContext Check.Ok
        val ok = runCatching {
            val (rounds, salt, hash) = stored.split(':')
            MessageDigest.isEqual(hash(code, unb64(salt), rounds.toInt()), unb64(hash))
        }.getOrDefault(false)
        if (ok) {
            prefs.edit().putInt(K_FAILS, 0).putLong(K_UNTIL, 0).commit()
            return@withContext Check.Ok
        }
        val fails = prefs.getInt(K_FAILS, 0) + 1
        val now = System.currentTimeMillis()
        val wait = if (fails < FREE_TRIES) 0L
        else minOf(MAX_WAIT_MS, FIRST_WAIT_MS shl minOf(fails - FREE_TRIES, 16))
        prefs.edit().putInt(K_FAILS, fails).putLong(K_UNTIL, if (wait > 0) now + wait else 0).commit()
        if (wait > 0) Check.Wait(now + wait) else Check.Wrong
    }

    /** Whether the phone takes a strong biometric now (BiometricManager's answer: BIOMETRIC_SUCCESS or why not). */
    fun biometric(): Int = runCatching {
        app.getSystemService(BiometricManager::class.java)
            ?.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG)
    }.getOrNull() ?: BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE

    /**
     * The cipher the lock screen's fingerprint prompt unlocks; null when the fingerprint is off, or its key is gone -
     * the phone's fingerprints changed - which turns it off ([fingerprintLost]).
     */
    fun fingerprintCipher(): Cipher? {
        if (!config.value.fingerprint) return null
        return try {
            val key = keyStore().getKey(KEY_ALIAS, null) as? SecretKey ?: throw KeyPermanentlyInvalidatedException()
            Cipher.getInstance(TRANSFORMATION).apply { init(Cipher.ENCRYPT_MODE, key) }
        } catch (e: KeyPermanentlyInvalidatedException) {
            fingerprintOff()
            _fingerprintLost.value = true
            null
        } catch (e: Exception) {
            null
        }
    }

    /** A new key for the fingerprint and its cipher, for the prompt that turns it on (throws when none is enrolled). */
    fun newFingerprintCipher(): Cipher {
        runCatching { keyStore().deleteEntry(KEY_ALIAS) }
        val key = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE).apply {
            init(KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setUserAuthenticationRequired(true)
                // a strong biometric for every use of the key, none of the phone's PIN or pattern
                .setUserAuthenticationParameters(0, KeyProperties.AUTH_BIOMETRIC_STRONG)
                .setInvalidatedByBiometricEnrollment(true)
                .build())
        }.generateKey()
        return Cipher.getInstance(TRANSFORMATION).apply { init(Cipher.ENCRYPT_MODE, key) }
    }

    /** The prompt took the finger with the new key: the fingerprint unlocks from now on. */
    fun fingerprintOn() {
        prefs.edit().putBoolean(K_FINGER, true).commit()
        _fingerprintLost.value = false
        changed()
    }

    fun fingerprintOff() {
        runCatching { keyStore().deleteEntry(KEY_ALIAS) }
        prefs.edit().putBoolean(K_FINGER, false).commit()
        changed()
    }

    private fun keyStore() = KeyStore.getInstance(KEYSTORE).apply { load(null) }

    private fun hash(code: String, salt: ByteArray, rounds: Int): ByteArray {
        val spec = PBEKeySpec(code.toCharArray(), salt, rounds, 256)
        try {
            return SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).encoded
        } finally {
            spec.clearPassword()
        }
    }

    private fun b64(b: ByteArray) = Base64.encodeToString(b, Base64.NO_WRAP)
    private fun unb64(s: String) = Base64.decode(s, Base64.NO_WRAP)

    companion object {
        private const val PREFS = "app_lock"

        /**
         * The lock's file read from now on: SharedPreferences reads it on a thread of its own and hands the same object
         * to the lock later, which then finds it read instead of waiting on the main thread for it.
         */
        fun preload(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        }

        const val DEFAULT_AFTER_MS = 60_000L
        /** The shortest code taken (a PIN's digits, a password's characters), and the longest. */
        const val MIN_CODE = 4
        const val MAX_CODE = 64
        private const val ROUNDS = 120_000
        private const val FREE_TRIES = 5
        private const val FIRST_WAIT_MS = 30_000L
        private const val MAX_WAIT_MS = 30 * 60_000L
        private const val KEYSTORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "geode_app_lock"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val K_KIND = "kind"
        private const val K_HASH = "hash"
        private const val K_FINGER = "fingerprint"
        private const val K_AFTER = "after_ms"
        private const val K_FAILS = "fails"
        private const val K_UNTIL = "wait_until"
    }
}
