package xyz.tifu.geode.ui.chat

import android.os.SystemClock
import android.view.MotionEvent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.Backspace
import androidx.compose.material.icons.automirrored.outlined.KeyboardReturn
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Fullscreen
import androidx.compose.material.icons.outlined.Keyboard
import androidx.compose.material.icons.outlined.Language
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.activity.compose.BackHandler
import androidx.lifecycle.compose.LifecycleStartEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import xyz.tifu.geode.R
import xyz.tifu.geode.agent.browser.AgentBrowser
import kotlin.math.roundToInt

/** The page's frames, fitted (the browser draws them only while a view like this one is on the screen). */
@Composable
private fun Frame(browser: AgentBrowser, modifier: Modifier) {
    val frame by browser.frame.collectAsStateWithLifecycle()
    LifecycleStartEffect(browser) {
        browser.watch(true)
        onStopOrDispose { browser.watch(false) }
    }
    Box(modifier) {
        frame?.let {
            Image(it.asImageBitmap(), contentDescription = null, contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize())
        }
    }
}

private fun title(ui: AgentBrowser.Ui, fallback: String) = ui.title.ifEmpty { ui.url }.ifEmpty { fallback }

/**
 * The agent's browser over the chat, small: a card the user drags anywhere, a tap on the page opens it on the whole
 * screen. Fills the chat's box (only the card takes touches).
 */
@Composable
fun BrowserMini(browser: AgentBrowser, ui: AgentBrowser.Ui) {
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val px = LocalDensity.current
        val aspect = if (ui.width > 0 && ui.height > 0) ui.width.toFloat() / ui.height else 0.45f
        val w = (maxWidth * 0.34f).coerceIn(110.dp, 200.dp)
        val head = 28.dp
        val h = (w / aspect + head).coerceAtMost(maxHeight * 0.85f)
        val margin = with(px) { 8.dp.toPx() }
        val maxX = (constraints.maxWidth - with(px) { w.toPx() } - margin).coerceAtLeast(0f)
        val maxY = (constraints.maxHeight - with(px) { h.toPx() } - margin).coerceAtLeast(0f)
        // where the user left it (top right at first), kept on the screen when the chat's box shrinks (the keyboard)
        var x by rememberSaveable { mutableFloatStateOf(-1f) }
        var y by rememberSaveable { mutableFloatStateOf(-1f) }
        val cx = if (x < 0) maxX else x.coerceIn(margin.coerceAtMost(maxX), maxX)
        val cy = if (y < 0) margin.coerceAtMost(maxY) else y.coerceIn(margin.coerceAtMost(maxY), maxY)
        Surface(
            shape = RoundedCornerShape(12.dp),
            shadowElevation = 8.dp,
            tonalElevation = 2.dp,
            modifier = Modifier.offset { IntOffset(cx.roundToInt(), cy.roundToInt()) }.size(w, h)
                .pointerInput(maxX, maxY, margin) {
                    detectDragGestures { change, drag ->
                        change.consume()
                        // from where it is now (the state's own values: this block outlives the composition it began in)
                        val fromX = if (x < 0) maxX else x.coerceIn(0f, maxX)
                        val fromY = if (y < 0) margin.coerceAtMost(maxY) else y.coerceIn(0f, maxY)
                        x = (fromX + drag.x).coerceIn(0f, maxX)
                        y = (fromY + drag.y).coerceIn(0f, maxY)
                    }
                },
        ) {
            Column {
                Row(Modifier.fillMaxWidth().height(head).padding(start = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    if (ui.loading) CircularProgressIndicator(Modifier.size(12.dp), strokeWidth = 1.5.dp)
                    else Icon(Icons.Outlined.Language, null, Modifier.size(14.dp), tint = MaterialTheme.colorScheme.primary)
                    Text(title(ui, stringResource(R.string.browser_name)), style = MaterialTheme.typography.labelSmall,
                        maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f).padding(start = 4.dp))
                    IconButton(onClick = { browser.show(AgentBrowser.View.FULL) }, Modifier.size(28.dp)) {
                        Icon(Icons.Outlined.Fullscreen, stringResource(R.string.browser_full), Modifier.size(16.dp))
                    }
                    IconButton(onClick = { browser.show(AgentBrowser.View.HIDDEN) }, Modifier.size(28.dp)) {
                        Icon(Icons.Outlined.Close, stringResource(R.string.browser_hide), Modifier.size(16.dp))
                    }
                }
                Frame(browser, Modifier.fillMaxSize().background(Color.White)
                    .clickable { browser.show(AgentBrowser.View.FULL) })
            }
        }
    }
}

/**
 * The agent's browser on the whole screen: the user's touches go to the page (its own display's pixels), their text
 * through a field of the app's (the phone's keyboard cannot reach a page on another display), back returns to the chat.
 * A layer over the chat in the activity's own window (a Dialog's window was sized for the whole display but placed
 * below the status bar: its bottom cut off).
 */
@Composable
fun BrowserFull(browser: AgentBrowser, ui: AgentBrowser.Ui) {
    BackHandler { browser.show(AgentBrowser.View.MINI) }
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
        var typing by rememberSaveable { mutableStateOf(false) }
        Column(Modifier.fillMaxSize().systemBarsPadding().imePadding()) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { browser.show(AgentBrowser.View.MINI) }) {
                    Icon(Icons.AutoMirrored.Outlined.ArrowBack, stringResource(R.string.browser_to_chat))
                }
                Column(Modifier.weight(1f)) {
                    Text(title(ui, stringResource(R.string.browser_name)), style = MaterialTheme.typography.titleSmall,
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
                    if (ui.url.isNotEmpty() && ui.title.isNotEmpty()) {
                        Text(ui.url, style = MaterialTheme.typography.bodySmall, maxLines = 1,
                            overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                IconButton(onClick = browser::reload) {
                    Icon(Icons.Outlined.Refresh, stringResource(R.string.browser_reload))
                }
                IconButton(onClick = { typing = !typing }) {
                    Icon(Icons.Outlined.Keyboard, stringResource(R.string.browser_keyboard),
                        tint = if (typing) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
                }
                IconButton(onClick = { browser.show(AgentBrowser.View.HIDDEN) }) {
                    Icon(Icons.Outlined.Close, stringResource(R.string.browser_hide))
                }
            }
            if (ui.loading) LinearProgressIndicator(Modifier.fillMaxWidth())
            else Box(Modifier.height(4.dp))
            Page(browser, ui, Modifier.weight(1f).fillMaxWidth().background(Color.Black))
            if (typing) Composer(browser)
        }
    }
}

/** The page, its frame fitted, and the user's finger on it as the page's own touches. */
@Composable
private fun Page(browser: AgentBrowser, ui: AgentBrowser.Ui, modifier: Modifier) {
    BoxWithConstraints(modifier) {
        val bw = constraints.maxWidth.toFloat()
        val bh = constraints.maxHeight.toFloat()
        val dw = ui.width.coerceAtLeast(1).toFloat()
        val dh = ui.height.coerceAtLeast(1).toFloat()
        // the fitted frame's scale and place (ContentScale.Fit's)
        val s = minOf(bw / dw, bh / dh)
        val ox = (bw - dw * s) / 2
        val oy = (bh - dh * s) / 2
        Frame(browser, Modifier.fillMaxSize().pointerInput(s, ox, oy) {
            fun send(action: Int, p: Offset, down: Long) = browser.touch(action,
                ((p.x - ox) / s).coerceIn(0f, dw - 1), ((p.y - oy) / s).coerceIn(0f, dh - 1), down)
            awaitEachGesture {
                val first = awaitFirstDown()
                val down = SystemClock.uptimeMillis()
                var last = first.position
                send(MotionEvent.ACTION_DOWN, last, down)
                first.consume()
                var ended = false
                try {
                    while (true) {
                        val ev = awaitPointerEvent()
                        val c = ev.changes.firstOrNull { it.id == first.id } ?: break
                        last = c.position
                        if (!c.pressed) {
                            send(MotionEvent.ACTION_UP, last, down)
                            ended = true
                            c.consume()
                            break
                        }
                        if (c.positionChanged()) send(MotionEvent.ACTION_MOVE, last, down)
                        c.consume()
                    }
                } finally {
                    if (!ended) send(MotionEvent.ACTION_CANCEL, last, down)
                }
            }
        })
    }
}

/** The user's text for the page's focused field, sent whole; Backspace and Enter as the page's keys. */
@Composable
private fun Composer(browser: AgentBrowser) {
    var text by rememberSaveable { mutableStateOf("") }
    fun send() {
        if (text.isNotEmpty()) browser.type(text = text)
        text = ""
    }
    Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(text, { text = it }, Modifier.weight(1f), singleLine = true,
            placeholder = { Text(stringResource(R.string.browser_type_hint)) },
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { send() }))
        IconButton(onClick = ::send) {
            Icon(Icons.AutoMirrored.Filled.Send, stringResource(R.string.browser_send))
        }
        IconButton(onClick = { browser.type(key = "Backspace") }) {
            Icon(Icons.AutoMirrored.Outlined.Backspace, stringResource(R.string.browser_backspace))
        }
        IconButton(onClick = { browser.type(key = "Enter") }) {
            Icon(Icons.AutoMirrored.Outlined.KeyboardReturn, stringResource(R.string.browser_enter))
        }
    }
}
