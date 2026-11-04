package xyz.tifu.geode.ui.settings

import android.hardware.biometrics.BiometricManager
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.launch
import xyz.tifu.geode.R
import xyz.tifu.geode.data.AppLock
import xyz.tifu.geode.ui.lock.activity
import xyz.tifu.geode.ui.lock.askFingerprint
import xyz.tifu.geode.ui.lock.pauseLeft

/** What the code dialog is for: a new lock, a new code for it, or the lock off (the last two ask the code first). */
private enum class CodeAsk { SET, CHANGE, OFF }

/** How long the app may be in the background before it locks (ms). */
private val LOCK_AFTER = listOf(0L, 60_000L, 5 * 60_000L, 30 * 60_000L)

/** The app's lock: on and off, its code, the fingerprint, how long in the background before it locks. */
@Composable
internal fun LockSection(lock: AppLock) {
    val cfg by lock.config.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var ask by remember { mutableStateOf<CodeAsk?>(null) }
    var message by remember { mutableStateOf<String?>(null) }
    // asked again each time the screen is in front: a fingerprint may have been added in the phone's settings
    var biometric by remember { mutableIntStateOf(lock.biometric()) }
    LifecycleResumeEffect(Unit) {
        biometric = lock.biometric()
        onPauseOrDispose { }
    }

    SwitchRow(stringResource(R.string.lock_setting), stringResource(R.string.lock_setting_desc), cfg.on,
        onChange = { on -> ask = if (on) CodeAsk.SET else CodeAsk.OFF })
    if (cfg.on) {
        TextButton(onClick = { ask = CodeAsk.CHANGE }) { Text(stringResource(R.string.lock_change)) }
        val canFinger = biometric == BiometricManager.BIOMETRIC_SUCCESS
        val title = stringResource(R.string.lock_fingerprint)
        val cancel = stringResource(R.string.chat_cancel)
        SwitchRow(title,
            stringResource(if (canFinger || cfg.fingerprint) R.string.lock_fingerprint_desc else R.string.lock_fingerprint_none),
            cfg.fingerprint, enabled = canFinger || cfg.fingerprint, onChange = fun(on: Boolean) {
                message = null
                if (!on) {
                    lock.fingerprintOff()
                    return
                }
                // on only once a finger was taken with the new key
                val activity = context.activity() ?: return
                val cipher = runCatching { lock.newFingerprintCipher() }.getOrElse {
                    message = it.message ?: it.javaClass.simpleName
                    return
                }
                askFingerprint(activity, cipher, title, cancel, onOk = lock::fingerprintOn, onFail = { m ->
                    lock.fingerprintOff()
                    message = m
                })
            })
        message?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        }
        Text(stringResource(R.string.lock_after), style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.padding(top = 8.dp))
        Choice(LOCK_AFTER.map { ms ->
            if (ms == 0L) stringResource(R.string.lock_after_now)
            else stringResource(R.string.lock_after_min, (ms / 60_000).toInt())
        }, LOCK_AFTER.indexOf(cfg.afterMs).coerceAtLeast(0)) { lock.setAfter(LOCK_AFTER[it]) }
        TextButton(onClick = lock::lock) { Text(stringResource(R.string.lock_now)) }
    }
    Text(stringResource(R.string.lock_note), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
    ask?.let { CodeDialog(it, lock, cfg) { ask = null } }
}

/** The code asked for: the current one first (to change it or turn the lock off), then a new one twice. */
@Composable
private fun CodeDialog(ask: CodeAsk, lock: AppLock, cfg: AppLock.Config, onDone: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var verified by remember { mutableStateOf(ask == CodeAsk.SET) }
    var kind by remember { mutableStateOf(if (cfg.kind == AppLock.Kind.PASSWORD) AppLock.Kind.PASSWORD else AppLock.Kind.PIN) }
    var current by remember { mutableStateOf("") }
    var first by remember { mutableStateOf("") }
    var second by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }

    fun confirm() {
        if (busy) return
        if (!verified) {
            if (current.isEmpty()) return
            busy = true
            scope.launch {
                when (val r = lock.check(current)) {
                    AppLock.Check.Ok -> if (ask == CodeAsk.OFF) {
                        lock.clear()
                        onDone()
                    } else verified = true
                    AppLock.Check.Wrong -> error = context.getString(R.string.lock_wrong)
                    is AppLock.Check.Wait -> error = context.getString(R.string.lock_wait, pauseLeft(r.untilMs))
                }
                current = ""
                busy = false
            }
            return
        }
        error = when {
            first.length < AppLock.MIN_CODE -> context.getString(
                if (kind == AppLock.Kind.PIN) R.string.lock_short_pin else R.string.lock_short_password)
            first != second -> context.getString(R.string.lock_mismatch)
            else -> null
        }
        if (error != null) return
        busy = true
        scope.launch {
            lock.set(kind, first)
            onDone()
        }
    }

    AlertDialog(
        onDismissRequest = { if (!busy) onDone() },
        title = {
            Text(stringResource(when (ask) {
                CodeAsk.SET -> R.string.lock_setting
                CodeAsk.CHANGE -> R.string.lock_change
                CodeAsk.OFF -> R.string.lock_off
            }))
        },
        text = {
            Column {
                if (!verified) {
                    CodeField(current, cfg.kind,
                        stringResource(if (cfg.kind == AppLock.Kind.PIN) R.string.lock_enter_pin else R.string.lock_enter_password),
                        done = ::confirm) {
                        current = it
                        error = null
                    }
                } else {
                    Choice(listOf(stringResource(R.string.lock_kind_pin), stringResource(R.string.lock_kind_password)),
                        if (kind == AppLock.Kind.PIN) 0 else 1) {
                        kind = if (it == 0) AppLock.Kind.PIN else AppLock.Kind.PASSWORD
                        first = ""
                        second = ""
                        error = null
                    }
                    CodeField(first, kind,
                        stringResource(if (kind == AppLock.Kind.PIN) R.string.lock_kind_pin else R.string.lock_kind_password)) {
                        first = it
                        error = null
                    }
                    CodeField(second, kind, stringResource(R.string.lock_repeat), done = ::confirm) {
                        second = it
                        error = null
                    }
                }
                error?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.padding(top = 4.dp))
                }
            }
        },
        confirmButton = {
            TextButton(onClick = ::confirm, enabled = !busy) {
                Text(stringResource(if (verified) R.string.agent_save else R.string.models_ok))
            }
        },
        dismissButton = {
            TextButton(onClick = onDone, enabled = !busy) { Text(stringResource(R.string.chat_cancel)) }
        },
    )
}

/** A masked field for a PIN (digits only, the number pad) or a password. */
@Composable
private fun CodeField(value: String, kind: AppLock.Kind, label: String, done: (() -> Unit)? = null,
                      onChange: (String) -> Unit) {
    val pin = kind == AppLock.Kind.PIN
    OutlinedTextField(
        value = value,
        onValueChange = { v -> onChange((if (pin) v.filter(Char::isDigit) else v).take(AppLock.MAX_CODE)) },
        label = { Text(label) },
        singleLine = true,
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(
            keyboardType = if (pin) KeyboardType.NumberPassword else KeyboardType.Password,
            imeAction = if (done != null) ImeAction.Done else ImeAction.Next),
        keyboardActions = KeyboardActions(onDone = { done?.invoke() }),
        modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
    )
}
