package xyz.tifu.geode.ui.chat

import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import xyz.tifu.geode.R
import java.io.File

/** How a file of the agent's is shown here. */
enum class FileKind { IMAGE, PAGE, MARKDOWN, TEXT, OTHER }

fun fileKind(name: String): FileKind = when (name.substringAfterLast('.', "").lowercase()) {
    "png", "jpg", "jpeg", "gif", "webp", "bmp" -> FileKind.IMAGE
    "html", "htm", "svg" -> FileKind.PAGE
    "md", "markdown" -> FileKind.MARKDOWN
    "txt", "log", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml", "toml", "ini", "py", "js", "mjs", "ts", "kt",
    "java", "c", "h", "cpp", "rs", "go", "sh", "bash", "sql", "css", "rb", "php", "swift", "r", "tex" -> FileKind.TEXT
    else -> FileKind.OTHER
}

/**
 * A file the agent made, full screen: an image (pinch to zoom), a page (HTML or SVG, its scripts run, the rest of the
 * app's files out of its reach), Markdown rendered, text and code as they are; any other kind goes to an app that
 * opens it.
 */
@Composable
fun AgentFileView(file: File, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val kind = fileKind(file.name)
    LaunchedEffect(file) {
        if (kind == FileKind.OTHER) {
            openWith(context, file)
            onDismiss()
        }
    }
    if (kind == FileKind.OTHER) return
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(Modifier.fillMaxSize()) {
                Row(Modifier.fillMaxWidth().statusBarsPadding().padding(start = 16.dp, end = 4.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Text(file.name, style = MaterialTheme.typography.titleMedium, maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    IconButton(onClick = { share(context, file) }) {
                        Icon(Icons.Outlined.Share, stringResource(R.string.agent_files_share))
                    }
                    IconButton(onClick = onDismiss) { Icon(Icons.Outlined.Close, stringResource(R.string.agent_files_close)) }
                }
                Box(Modifier.weight(1f).fillMaxWidth()) {
                    when (kind) {
                        FileKind.IMAGE -> ImageView(file)
                        FileKind.PAGE -> PageView(file)
                        else -> TextView(file, markdown = kind == FileKind.MARKDOWN)
                    }
                }
            }
        }
    }
}

@Composable
private fun ImageView(file: File) {
    val bitmap by produceBitmap(file)
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    val state = rememberTransformableState { zoom, pan, _ ->
        scale = (scale * zoom).coerceIn(1f, 8f)
        offset = if (scale == 1f) Offset.Zero else offset + pan
    }
    val b = bitmap
    if (b == null) {
        CircularProgressIndicator(Modifier.padding(32.dp))
        return
    }
    Image(b.asImageBitmap(), null, contentScale = ContentScale.Fit,
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.surfaceContainerLowest).transformable(state)
            .graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y))
}

/** The image at about the screen's size (a camera's 50 MP picture is not decoded whole). */
@Composable
private fun produceBitmap(file: File) = androidx.compose.runtime.produceState<android.graphics.Bitmap?>(null, file) {
    value = withContext(Dispatchers.IO) {
        runCatching {
            val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(file.path, o)
            var sample = 1
            while (maxOf(o.outWidth, o.outHeight) / sample > 3000) sample *= 2
            BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
        }.getOrNull()
    }
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
private fun PageView(file: File) {
    AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
        WebView(ctx).apply {
            // the view's own size, not its page's: a WebView whose height wraps its content (AndroidView's default)
            // gives the page a viewport of no height, 100vh = 0
            layoutParams = android.view.ViewGroup.LayoutParams(android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT)
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            // its own folder's file:// URLs load (images, scripts beside it); not other origins' data, not the app's
            settings.allowFileAccess = true
            @Suppress("DEPRECATION")
            settings.allowFileAccessFromFileURLs = false
            @Suppress("DEPRECATION")
            settings.allowUniversalAccessFromFileURLs = false
            settings.allowContentAccess = false
            settings.builtInZoomControls = true
            settings.displayZoomControls = false
            // a page without a viewport tag laid out at the phone's width (the agent writes for this screen), not as
            // a desktop page shrunk; zoom stays
            settings.useWideViewPort = false
            webViewClient = WebViewClient()
            // loaded once it has its size: a page laid out in a view of no height takes 100vh for nothing
            addOnLayoutChangeListener(object : android.view.View.OnLayoutChangeListener {
                override fun onLayoutChange(v: android.view.View, l: Int, t: Int, r: Int, b: Int,
                                            ol: Int, ot: Int, or: Int, ob: Int) {
                    if (b - t <= 0 || r - l <= 0) return
                    removeOnLayoutChangeListener(this)
                    loadUrl("file://" + file.absolutePath)
                }
            })
        }
    }, onRelease = { it.destroy() })
}

@Composable
private fun TextView(file: File, markdown: Boolean) {
    var text by remember(file) { mutableStateOf<String?>(null) }
    LaunchedEffect(file) {
        text = withContext(Dispatchers.IO) {
            runCatching {
                val bytes = file.inputStream().use { s -> ByteArray(minOf(file.length(), MAX_TEXT.toLong()).toInt()).also { s.read(it) } }
                String(bytes, Charsets.UTF_8) + if (file.length() > MAX_TEXT) "\n…" else ""
            }.getOrDefault("")
        }
    }
    val t = text ?: return CircularProgressIndicator(Modifier.padding(32.dp))
    if (markdown) {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
            SelectionContainer { MarkdownText(t) }
        }
    } else {
        SelectionContainer {
            Text(t, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                    .horizontalScroll(rememberScrollState()).padding(12.dp))
        }
    }
}

private const val MAX_TEXT = 1 shl 20
