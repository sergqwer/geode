package xyz.tifu.geode.ui.chat

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.CreateNewFolder
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Folder
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.R
import xyz.tifu.geode.util.formatBytes
import java.io.File
import java.text.DateFormat
import java.util.Date

/** A file of the agent's folder: its path in the folder, its size and time. */
private class Entry(val file: File, val path: String, val size: Long, val modified: Long)

/**
 * The chat's folder (the agent's /workspace) as the user sees it: what the agent made, a text file opened right here,
 * any file handed to another app (share), files from the phone put in for the agent.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AgentFiles(dir: File, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var version by remember { mutableIntStateOf(0) }
    var entries by remember { mutableStateOf<List<Entry>>(emptyList()) }
    var open by remember { mutableStateOf<Entry?>(null) }
    LaunchedEffect(version) { entries = withContext(Dispatchers.IO) { list(dir) } }
    // a whole folder of the phone's in, or the agent's whole folder out to one of the phone's
    val addFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree ->
        if (tree != null) scope.launch {
            withContext(Dispatchers.IO) { runCatching { copyTreeIn(context, tree, dir) } }
            version++
        }
    }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree ->
        if (tree != null) scope.launch {
            val n = withContext(Dispatchers.IO) { runCatching { copyTreeOut(context, dir, tree) }.getOrDefault(-1) }
            android.widget.Toast.makeText(context,
                if (n >= 0) context.getString(R.string.agent_files_exported, n) else context.getString(R.string.agent_files_export_failed),
                android.widget.Toast.LENGTH_SHORT).show()
        }
    }
    val add = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        scope.launch {
            withContext(Dispatchers.IO) { for (u in uris) runCatching { copyIn(context, u, dir) } }
            version++
        }
    }

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.padding(start = 20.dp, end = 12.dp, bottom = 16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.agent_files), style = MaterialTheme.typography.titleLarge)
                    Text(stringResource(R.string.agent_files_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Row(Modifier.padding(top = 8.dp), horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { add.launch(arrayOf("*/*")) }) {
                    Icon(Icons.Outlined.Add, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.agent_files_add), Modifier.padding(start = 6.dp))
                }
                OutlinedButton(onClick = { addFolder.launch(null) }) {
                    Icon(Icons.Outlined.CreateNewFolder, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.agent_files_add_folder), Modifier.padding(start = 6.dp))
                }
                OutlinedButton(onClick = { export.launch(null) }, enabled = entries.isNotEmpty()) {
                    Icon(Icons.Outlined.Download, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.agent_files_export), Modifier.padding(start = 6.dp))
                }
            }
            Spacer(Modifier.size(8.dp))
            if (entries.isEmpty()) {
                Text(stringResource(R.string.agent_files_empty), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(vertical = 24.dp))
            }
            LazyColumn(Modifier.heightIn(max = 480.dp)) {
                items(entries, key = { it.path }) { e ->
                    Row(Modifier.fillMaxWidth().clickable { open = e }.padding(vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Icon(if (e.path.contains('/')) Icons.Outlined.Folder else Icons.Outlined.Description, null,
                            Modifier.size(20.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                        Column(Modifier.weight(1f).padding(start = 12.dp)) {
                            Text(e.path, maxLines = 1, overflow = TextOverflow.Ellipsis,
                                style = MaterialTheme.typography.bodyMedium)
                            Text(formatBytes(e.size) + " · " + DateFormat.getDateTimeInstance(DateFormat.SHORT,
                                DateFormat.SHORT).format(Date(e.modified)), style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        IconButton(onClick = { share(context, e.file) }) {
                            Icon(Icons.Outlined.Share, stringResource(R.string.agent_files_share), Modifier.size(20.dp))
                        }
                    }
                }
            }
        }
    }
    open?.let { e -> FileView(e, onShare = { share(context, e.file) }, onDismiss = { open = null }) }
}

/** A text file's first part right here; anything else goes to an app that opens it. */
@Composable
private fun FileView(e: Entry, onShare: () -> Unit, onDismiss: () -> Unit) {
    val context = LocalContext.current
    var text by remember(e.path) { mutableStateOf<String?>(null) }
    LaunchedEffect(e.path) {
        text = withContext(Dispatchers.IO) { readText(e.file) }
        if (text == null) {
            openWith(context, e.file)
            onDismiss()
        }
    }
    val t = text ?: return
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(e.path, maxLines = 2, overflow = TextOverflow.Ellipsis) },
        text = {
            SelectionContainer {
                Text(t, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.heightIn(max = 460.dp).verticalScroll(rememberScrollState())
                        .horizontalScroll(rememberScrollState()).width(900.dp))
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.agent_files_close)) } },
        dismissButton = { TextButton(onClick = onShare) { Text(stringResource(R.string.agent_files_share)) } },
    )
}

private fun list(dir: File): List<Entry> {
    if (!dir.isDirectory) return emptyList()
    val base = dir.canonicalFile
    return base.walkTopDown()
        .onEnter { it == base || !it.name.startsWith(".") }   // the agent's dot-folders stay out of the way
        .filter { it.isFile && !it.name.startsWith(".") }
        .take(MAX_FILES)
        .map { Entry(it, it.relativeTo(base).path, it.length(), it.lastModified()) }
        .sortedByDescending { it.modified }
        .toList()
}

/** A file's text when it is text (UTF-8, no NULs), its first [MAX_TEXT] bytes; else null. */
private fun readText(f: File): String? {
    val bytes = f.inputStream().use { s -> ByteArray(minOf(f.length(), MAX_TEXT.toLong()).toInt()).also { s.read(it) } }
    if (bytes.any { it == 0.toByte() }) return null
    val s = runCatching { Charsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(bytes)).toString() }.getOrNull()
        ?: String(bytes, Charsets.UTF_8)
    return if (f.length() > MAX_TEXT) "$s\n…" else s
}

private fun uri(context: Context, f: File): Uri =
    FileProvider.getUriForFile(context, context.packageName + ".files", f)

private fun mime(f: File) = MimeTypeMap.getSingleton().getMimeTypeFromExtension(f.extension.lowercase()) ?: "*/*"

internal fun share(context: Context, f: File) {
    val i = Intent(Intent.ACTION_SEND).setType(mime(f)).putExtra(Intent.EXTRA_STREAM, uri(context, f))
        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    runCatching { context.startActivity(Intent.createChooser(i, f.name)) }
}

internal fun openWith(context: Context, f: File) {
    val i = Intent(Intent.ACTION_VIEW).setDataAndType(uri(context, f), mime(f))
        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    runCatching { context.startActivity(Intent.createChooser(i, f.name)) }
}

/** A file the user picked, copied into the folder under its own name (a number added when the name is taken). */
private fun copyIn(context: Context, u: Uri, dir: File) {
    val name = context.contentResolver.query(u, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) c.getString(0) else null
    }?.replace('/', '_')?.takeIf { it.isNotBlank() && !it.startsWith(".") } ?: "file"
    dir.mkdirs()
    var target = File(dir, name)
    var n = 1
    while (target.exists()) target = File(dir, name.substringBeforeLast('.') + " (${n++})" +
        (if ('.' in name) "." + name.substringAfterLast('.') else ""))
    context.contentResolver.openInputStream(u)?.use { input -> target.outputStream().use { input.copyTo(it) } }
}

/** A folder the user picked on the phone (a document tree), copied into the agent's folder under its own name. */
private fun copyTreeIn(context: Context, tree: Uri, dir: File) {
    val root = androidx.documentfile.provider.DocumentFile.fromTreeUri(context, tree) ?: return
    fun walk(d: androidx.documentfile.provider.DocumentFile, to: File) {
        to.mkdirs()
        for (c in d.listFiles()) {
            val name = c.name?.replace('/', '_')?.takeIf { it.isNotBlank() } ?: continue
            if (c.isDirectory) walk(c, File(to, name))
            else context.contentResolver.openInputStream(c.uri)?.use { i -> File(to, name).outputStream().use { i.copyTo(it) } }
        }
    }
    walk(root, File(dir, root.name?.replace('/', '_') ?: "folder"))
}

/** The agent's whole folder copied into a folder the user picked on the phone; the files copied. */
private fun copyTreeOut(context: Context, dir: File, tree: Uri): Int {
    val root = androidx.documentfile.provider.DocumentFile.fromTreeUri(context, tree) ?: return -1
    var n = 0
    fun walk(from: File, to: androidx.documentfile.provider.DocumentFile) {
        for (f in from.listFiles() ?: emptyArray()) {
            if (f.name.startsWith(".")) continue
            if (f.isDirectory) {
                val sub = to.findFile(f.name)?.takeIf { it.isDirectory } ?: to.createDirectory(f.name) ?: continue
                walk(f, sub)
            } else {
                to.findFile(f.name)?.delete()
                val doc = to.createFile(mime(f).takeIf { it != "*/*" } ?: "application/octet-stream", f.name) ?: continue
                context.contentResolver.openOutputStream(doc.uri)?.use { o -> f.inputStream().use { it.copyTo(o) } }
                n++
            }
        }
    }
    walk(dir, root)
    return n
}

private const val MAX_FILES = 500
private const val MAX_TEXT = 256 * 1024
