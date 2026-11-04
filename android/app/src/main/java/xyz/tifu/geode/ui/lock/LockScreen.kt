package xyz.tifu.geode.ui.lock

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.hardware.biometrics.BiometricManager
import android.hardware.biometrics.BiometricPrompt
import android.os.CancellationSignal
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Fingerprint
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import xyz.tifu.geode.R
import xyz.tifu.geode.data.AppLock
import javax.crypto.Cipher

/** The activity a composition's context is, or wraps. */
internal tailrec fun Context.activity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.activity()
    else -> null
}

/** What is left of a pause until [untilMs], as m:ss. */
internal fun pauseLeft(untilMs: Long): String {
    val s = ((untilMs - System.currentTimeMillis() + 999) / 1000).coerceAtLeast(1)
    return "%d:%02d".format(s / 60, s % 60)
}

/**
 * The system's fingerprint prompt over [cipher] (a key that takes a strong biometric for each use): [onOk] once a
 * finger was taken and the key worked after it, [onFail] when the prompt gave up - with the system's message, or null
 * when it was cancelled. Either is called once.
 */
internal fun askFingerprint(activity: Activity, cipher: Cipher, title: String, negative: String,
                            onOk: () -> Unit, onFail: (String?) -> Unit) {
    var done = false
    fun finish(ok: Boolean, message: String?) {
        if (done) return
        done = true
        if (ok) onOk() else onFail(message)
    }
    // a prompt the system will not show (no permission, the sensor busy) is a failure to report, not a crash
    try {
        prompt(activity, cipher, title, negative, ::finish)
    } catch (e: Exception) {
        finish(false, e.message ?: e.javaClass.simpleName)
    }
}

private fun prompt(activity: Activity, cipher: Cipher, title: String, negative: String,
                   finish: (Boolean, String?) -> Unit) {
    val prompt = BiometricPrompt.Builder(activity)
        .setTitle(title)
        .setAllowedAuthenticators(BiometricManager.Authenticators.BIOMETRIC_STRONG)
        .setConfirmationRequired(false)
        .setNegativeButton(negative, activity.mainExecutor) { _, _ -> finish(false, null) }
        .build()
    prompt.authenticate(BiometricPrompt.CryptoObject(cipher), CancellationSignal(), activity.mainExecutor,
        object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                // the key works only after a real biometric: the callback alone is not taken for one
                val ok = runCatching { result.cryptoObject?.cipher?.doFinal(ByteArray(16)) != null }.getOrDefault(false)
                finish(ok, null)
            }

            override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                finish(false, when (errorCode) {
                    BiometricPrompt.BIOMETRIC_ERROR_CANCELED, BiometricPrompt.BIOMETRIC_ERROR_USER_CANCELED -> null
                    else -> errString.toString()
                })
            }
        })
}

/**
 * In place of the whole app while it is locked: the code, the fingerprint (offered by itself once each time the app
 * locks), and, the code forgotten, the one way back in - every chat deleted ([onForgot]), then the lock.
 */
@Composable
fun LockScreen(lock: AppLock, onForgot: suspend () -> Unit) {
    val cfg by lock.config.collectAsState()
    val locks by lock.locks.collectAsState()
    val lost by lock.fingerprintLost.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val pin = cfg.kind == AppLock.Kind.PIN
    // not saveable: the code is never put in the activity's saved state
    var code by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var forgot by remember { mutableStateOf(false) }
    var waitUntil by remember { mutableLongStateOf(lock.waitUntil()) }
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(waitUntil) {
        while (true) {
            now = System.currentTimeMillis()
            if (now >= waitUntil) break
            delay(250)
        }
    }
    val waiting = waitUntil > now
    val focus = remember { FocusRequester() }
    val wrong = stringResource(R.string.lock_wrong)
    val promptTitle = stringResource(R.string.lock_prompt_title)
    val useCode = stringResource(R.string.lock_prompt_code)

    fun submit() {
        if (busy || waiting || code.isEmpty()) return
        busy = true
        scope.launch {
            when (val r = lock.check(code)) {
                AppLock.Check.Ok -> lock.unlock()
                AppLock.Check.Wrong -> error = wrong
                is AppLock.Check.Wait -> {
                    error = null
                    waitUntil = r.untilMs
                }
            }
            code = ""
            busy = false
            // the keyboard stays up for the next try
            runCatching { focus.requestFocus() }
        }
    }

    fun fingerprint() {
        val activity = context.activity() ?: return
        val cipher = lock.fingerprintCipher() ?: return
        askFingerprint(activity, cipher, promptTitle, useCode, onOk = lock::unlock,
            onFail = { message -> if (message != null) error = message })
    }

    // the fingerprint asked for by itself once each time the app locks, when the screen is in front
    var offered by remember(locks) { mutableStateOf(false) }
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    LaunchedEffect(locks) {
        lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
            if (!offered && lock.config.value.fingerprint) {
                offered = true
                fingerprint()
            }
        }
    }
    // without the fingerprint the keyboard comes up for the code at once, and again when a pause is over
    LaunchedEffect(waiting) { if (!waiting && !cfg.fingerprint) runCatching { focus.requestFocus() } }

    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Box(Modifier.fillMaxSize().systemBarsPadding().imePadding(), contentAlignment = Alignment.Center) {
            Column(
                Modifier.widthIn(max = 420.dp).verticalScroll(rememberScrollState()).padding(horizontal = 32.dp, vertical = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Outlined.Lock, contentDescription = null, Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.primary)
                Text(stringResource(R.string.lock_title), style = MaterialTheme.typography.headlineSmall,
                    modifier = Modifier.padding(bottom = 8.dp))
                OutlinedTextField(
                    value = code,
                    onValueChange = { v ->
                        code = (if (pin) v.filter(Char::isDigit) else v).take(AppLock.MAX_CODE)
                        error = null
                    },
                    label = { Text(stringResource(if (pin) R.string.lock_enter_pin else R.string.lock_enter_password)) },
                    singleLine = true,
                    enabled = !waiting,
                    isError = error != null,
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = if (pin) KeyboardType.NumberPassword else KeyboardType.Password,
                        imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = { submit() }),
                    supportingText = {
                        when {
                            waiting -> Text(stringResource(R.string.lock_wait, pauseLeft(waitUntil)))
                            error != null -> Text(error.orEmpty())
                        }
                    },
                    modifier = Modifier.fillMaxWidth().focusRequester(focus),
                )
                Button(onClick = ::submit, enabled = code.isNotEmpty() && !waiting && !busy,
                    modifier = Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.lock_unlock))
                }
                if (cfg.fingerprint) {
                    OutlinedButton(onClick = ::fingerprint, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                        Icon(Icons.Outlined.Fingerprint, contentDescription = null, Modifier.padding(end = 8.dp))
                        Text(stringResource(R.string.lock_fingerprint))
                    }
                }
                if (lost) {
                    Text(stringResource(R.string.lock_fingerprint_changed), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error)
                }
                TextButton(onClick = { forgot = true }, enabled = !busy) { Text(stringResource(R.string.lock_forgot)) }
            }
        }
    }

    if (forgot) {
        AlertDialog(
            onDismissRequest = { if (!busy) forgot = false },
            title = { Text(stringResource(R.string.lock_forgot_title)) },
            text = {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(stringResource(R.string.lock_forgot_text))
                    if (busy) CircularProgressIndicator(Modifier.padding(top = 16.dp).size(24.dp), strokeWidth = 2.dp)
                }
            },
            confirmButton = {
                TextButton(enabled = !busy, onClick = {
                    busy = true
                    scope.launch {
                        onForgot()
                        busy = false
                        forgot = false
                    }
                }) { Text(stringResource(R.string.chat_delete), color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(enabled = !busy, onClick = { forgot = false }) { Text(stringResource(R.string.chat_cancel)) }
            },
        )
    }
}
