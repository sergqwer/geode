package xyz.tifu.geode.ui

import android.graphics.drawable.ColorDrawable
import android.os.SystemClock
import android.view.ViewGroup
import android.view.WindowManager
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.isImeVisible
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.data.AppSettings
import xyz.tifu.geode.ui.chat.ChatViewModel
import xyz.tifu.geode.ui.lock.activity

/** The last touch on the app (MainActivity.onUserInteraction, and its coming back to the front). */
object Touches {
    /** SystemClock.elapsedRealtime of the last touch. */
    val last = MutableStateFlow(SystemClock.elapsedRealtime())

    fun touched() {
        last.value = SystemClock.elapsedRealtime()
    }
}

/**
 * The screen kept on as the settings say - while Geode answers (the chat, the agent between its tools, a request to the
 * local server), or all the time it is open - and over it the burn-in guard: after [AppSettings.blackScreenMin] minutes
 * without a touch the screen goes black, since the parts of the picture that never move (the bars, the input) would burn
 * into an OLED screen over hours. It is no lock: the touch that ends it shows the app again. Not while the keyboard is
 * up (typing is not a touch on the app's window).
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ScreenGuard(app: GeodeApp, chat: ChatViewModel) {
    val s by app.settings.flow.collectAsStateWithLifecycle(AppSettings())
    val generating by remember(chat) { chat.state.map { it.generating }.distinctUntilChanged() }
        .collectAsStateWithLifecycle(false)
    val holder by app.gate.holder.collectAsStateWithLifecycle()
    // the agent's shared start read ahead holds the engine too, but nobody waits for it
    val engineBusy = holder != null && !app.gate.backgroundIn
    val keep = when (s.keepScreen) {
        AppSettings.KEEP_ALWAYS -> true
        AppSettings.KEEP_ANSWERING -> generating || engineBusy
        else -> false
    }
    val activity = LocalContext.current.activity()
    DisposableEffect(activity, keep) {
        val w = activity?.window
        if (keep) w?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else w?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose { w?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
    }

    var black by remember { mutableStateOf(false) }
    val last by Touches.last.collectAsStateWithLifecycle()
    val typing = WindowInsets.isImeVisible
    val minutes = s.blackScreenMin
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    LaunchedEffect(keep, minutes, last, typing) {
        if (!keep || minutes <= 0 || typing) return@LaunchedEffect
        // only while the app is in front (it comes back to the front as a touch: Touches.touched in onStart)
        lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            val wait = minutes * 60_000L - (SystemClock.elapsedRealtime() - Touches.last.value)
            if (wait > 0) delay(wait)
            black = true
        }
    }
    if (black) BlackScreen(keep) {
        black = false
        Touches.touched()
    }
}

/**
 * Black over everything, the dialogs included (a window of its own, on top), the system bars hidden and the screen as dim
 * as it goes; the first touch anywhere ([onTouch]) ends it and goes no further. Its window keeps the screen on itself
 * while [keep]: a window it covers whole no longer counts.
 */
@Composable
private fun BlackScreen(keep: Boolean, onTouch: () -> Unit) {
    Dialog(onDismissRequest = onTouch, properties = DialogProperties(usePlatformDefaultWidth = false,
        decorFitsSystemWindows = false, dismissOnClickOutside = false)) {
        val window = (LocalView.current.parent as? DialogWindowProvider)?.window
        DisposableEffect(window, keep) {
            window?.let { w ->
                w.setBackgroundDrawable(ColorDrawable(android.graphics.Color.BLACK))
                w.setDimAmount(0f)
                w.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                if (keep) w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else w.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                w.attributes = w.attributes.apply {
                    // as dim as the phone goes: black lights nothing on an OLED, the dimness covers an LCD's backlight
                    screenBrightness = 0f
                    layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
                }
                WindowCompat.getInsetsController(w, w.decorView).apply {
                    systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
                    hide(WindowInsetsCompat.Type.systemBars())
                }
            }
            onDispose {}
        }
        // what it is, said for a moment (a still text for seconds burns nothing in)
        var hint by remember { mutableStateOf(true) }
        LaunchedEffect(Unit) {
            delay(HINT_MS)
            hint = false
        }
        Box(
            Modifier.fillMaxSize().background(Color.Black).pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false).consume()
                    onTouch()
                }
            },
            contentAlignment = Alignment.Center,
        ) {
            AnimatedVisibility(hint, exit = fadeOut()) {
                Text(stringResource(R.string.screen_black_hint), color = Color(0xFF7A7A7A),
                    style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

private const val HINT_MS = 4000L
