package xyz.tifu.geode.ui.agent

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.CreateNewFolder
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Folder
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import xyz.tifu.geode.R
import xyz.tifu.geode.agent.AgentConfig
import xyz.tifu.geode.agent.FolderState
import xyz.tifu.geode.agent.PhoneFolders
import xyz.tifu.geode.agent.McpServer
import xyz.tifu.geode.agent.Skill
import xyz.tifu.geode.util.formatBytes

/**
 * The agent: its Linux (there, up to date, or unpacked again), the user's own skills (every chat's agent can use
 * them) and the MCP servers every agent session opens with, and the commands a chat with the agent takes.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AgentScreen(vm: AgentViewModel, openDrawer: () -> Unit) {
    val linux by vm.linux.collectAsStateWithLifecycle()
    val servers by vm.servers.collectAsStateWithLifecycle()
    val skills by vm.skills.collectAsStateWithLifecycle()
    val folders by vm.folders.collectAsStateWithLifecycle()
    val folderStates by vm.folderStates.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var editing by remember { mutableStateOf<McpServer?>(null) }
    var adding by remember { mutableStateOf(false) }
    var creatingSkill by remember { mutableStateOf(false) }
    var viewing by remember { mutableStateOf<Skill?>(null) }
    LaunchedEffect(Unit) { vm.refresh() }

    val imported = stringResource(R.string.agent_skill_imported)
    val notSkill = stringResource(R.string.agent_skill_not_found)
    val onImported: (String?) -> Unit = { name ->
        Toast.makeText(context, if (name != null) imported.format(name) else notSkill, Toast.LENGTH_LONG).show()
    }
    val pickFile = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { vm.importSkill(it, tree = false, onImported) }
    }
    val pickFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        uri?.let { vm.importSkill(it, tree = true, onImported) }
    }
    val notGiven = stringResource(R.string.agent_folder_add_failed)
    val pickShared = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        uri?.let { vm.addFolder(it) { ok -> if (!ok) Toast.makeText(context, notGiven, Toast.LENGTH_LONG).show() } }
    }

    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.nav_agent)) }, navigationIcon = {
            IconButton(onClick = openDrawer) { Icon(Icons.Outlined.Menu, stringResource(R.string.nav_menu)) }
        })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Text(stringResource(R.string.agent_screen_desc), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))

            // ---- its Linux
            Section(stringResource(R.string.agent_linux))
            Text(stringResource(when {
                !linux.installed -> R.string.agent_linux_none
                linux.outdated -> R.string.agent_linux_outdated
                else -> R.string.agent_linux_ready
            }), style = MaterialTheme.typography.bodyMedium)
            linux.installing?.let { p ->
                LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                Text(stringResource(R.string.agent_installing, (p * 100).toInt()), style = MaterialTheme.typography.bodySmall)
            }
            linux.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            if (linux.installing == null && (!linux.installed || linux.outdated)) {
                OutlinedButton(onClick = vm::installLinux, modifier = Modifier.padding(top = 8.dp)) {
                    Text(stringResource(if (linux.installed) R.string.agent_linux_update else R.string.agent_install))
                }
            }

            // ---- skills
            Section(stringResource(R.string.agent_skills))
            Text(stringResource(R.string.agent_skills_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (skills.isEmpty()) {
                Text(stringResource(R.string.agent_skills_none), style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp))
            }
            for (s in skills) {
                Row(Modifier.fillMaxWidth().clickable { viewing = s }.padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(s.name, style = MaterialTheme.typography.titleSmall)
                        if (s.description.isNotEmpty()) {
                            Text(s.description, style = MaterialTheme.typography.bodySmall, maxLines = 2,
                                overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                    IconButton(onClick = { vm.deleteSkill(s) }) {
                        Icon(Icons.Outlined.Delete, stringResource(R.string.agent_delete))
                    }
                }
            }
            Row(Modifier.padding(top = 8.dp).horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { creatingSkill = true }) {
                    Icon(Icons.Outlined.Add, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.agent_skill_new), Modifier.padding(start = 6.dp))
                }
                OutlinedButton(onClick = { pickFile.launch(arrayOf("text/*", "application/octet-stream")) }) {
                    Icon(Icons.Outlined.Description, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.agent_skill_file), Modifier.padding(start = 6.dp))
                }
                OutlinedButton(onClick = { pickFolder.launch(null) }) {
                    Icon(Icons.Outlined.Folder, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.agent_skill_folder), Modifier.padding(start = 6.dp))
                }
            }

            // ---- the phone's folders
            Section(stringResource(R.string.agent_folders))
            Text(stringResource(R.string.agent_folders_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (folders.isEmpty()) {
                Text(stringResource(R.string.agent_folders_none), style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp))
            }
            for (f in folders) {
                val st = folderStates[f.name] ?: FolderState()
                Column(Modifier.fillMaxWidth().padding(top = 8.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Outlined.Folder, null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                        Column(Modifier.weight(1f).padding(start = 12.dp)) {
                            Text("${PhoneFolders.GUEST}/${f.name}", style = MaterialTheme.typography.titleSmall,
                                fontFamily = FontFamily.Monospace, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(f.label, style = MaterialTheme.typography.bodySmall, maxLines = 1,
                                overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(when {
                                st.syncing -> stringResource(R.string.agent_folder_copying)
                                st.error != null -> stringResource(R.string.agent_folder_error, st.error)
                                else -> stringResource(R.string.agent_folder_files, st.files, formatBytes(st.bytes)) +
                                    if (st.skipped > 0) "\n" + stringResource(R.string.agent_folder_skipped, st.skipped,
                                        formatBytes(PhoneFolders.MAX_BYTES)) else ""
                            }, style = MaterialTheme.typography.bodySmall,
                                color = if (st.error != null && !st.syncing) MaterialTheme.colorScheme.error
                                else MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        IconButton(onClick = { vm.removeFolder(f.name) }) {
                            Icon(Icons.Outlined.Delete, stringResource(R.string.agent_folder_remove))
                        }
                    }
                    Row(Modifier.padding(start = 32.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilterChip(selected = !f.write, onClick = { vm.setFolderWrite(f.name, false) },
                            label = { Text(stringResource(R.string.agent_folder_read)) })
                        FilterChip(selected = f.write, enabled = f.canWrite, onClick = { vm.setFolderWrite(f.name, true) },
                            label = { Text(stringResource(R.string.agent_folder_write)) })
                    }
                }
            }
            OutlinedButton(onClick = { pickShared.launch(null) }, modifier = Modifier.padding(top = 8.dp)) {
                Icon(Icons.Outlined.CreateNewFolder, null, Modifier.size(18.dp))
                Text(stringResource(R.string.agent_folders_add), Modifier.padding(start = 6.dp))
            }

            // ---- MCP servers
            Section(stringResource(R.string.agent_mcp))
            Text(stringResource(R.string.agent_mcp_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (servers.isEmpty()) {
                Text(stringResource(R.string.agent_mcp_none), style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp))
            }
            for (s in servers) {
                Row(Modifier.fillMaxWidth().clickable { editing = s }.padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(s.name, style = MaterialTheme.typography.titleSmall)
                        Text(if (s.http) s.url else (listOf(s.command) + s.args).joinToString(" "),
                            style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, maxLines = 1,
                            overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Switch(checked = s.enabled, onCheckedChange = { vm.setEnabled(s.name, it) })
                }
            }
            OutlinedButton(onClick = { adding = true }, modifier = Modifier.padding(top = 8.dp)) {
                Icon(Icons.Outlined.Add, null, Modifier.size(18.dp))
                Text(stringResource(R.string.agent_mcp_add), Modifier.padding(start = 6.dp))
            }

            // ---- commands
            Section(stringResource(R.string.agent_commands))
            Text(stringResource(R.string.agent_commands_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(bottom = 24.dp))
        }
    }

    if (adding || editing != null) {
        ServerDialog(editing, onSave = { s ->
            vm.saveServer(s, editing?.name)
            adding = false
            editing = null
        }, onDelete = editing?.let { e -> {
            vm.deleteServer(e.name)
            editing = null
        } }, onDismiss = {
            adding = false
            editing = null
        })
    }
    if (creatingSkill) {
        SkillDialog(onSave = { n, d, b ->
            vm.createSkill(n, d, b)
            creatingSkill = false
        }, onDismiss = { creatingSkill = false })
    }
    viewing?.let { s ->
        var text by remember(s) { mutableStateOf("") }
        LaunchedEffect(s) { text = vm.skillText(s) }
        AlertDialog(onDismissRequest = { viewing = null }, title = { Text(s.name) },
            text = {
                Text(text, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()))
            },
            confirmButton = { TextButton(onClick = { viewing = null }) { Text(stringResource(R.string.agent_ok)) } })
    }
}

/** An MCP server, new or [old]: a command run in the agent's Linux, or an address reached over HTTP. */
@Composable
private fun ServerDialog(old: McpServer?, onSave: (McpServer) -> Unit, onDelete: (() -> Unit)?, onDismiss: () -> Unit) {
    var name by remember { mutableStateOf(old?.name.orEmpty()) }
    var http by remember { mutableStateOf(old?.http == true) }
    var command by remember { mutableStateOf(old?.command.orEmpty()) }
    var args by remember { mutableStateOf(old?.args?.joinToString(" ") { quote(it) }.orEmpty()) }
    var env by remember { mutableStateOf(old?.env?.joinToString("\n") { (k, v) -> "$k=$v" }.orEmpty()) }
    var url by remember { mutableStateOf(old?.url.orEmpty()) }
    var headers by remember { mutableStateOf(old?.headers?.joinToString("\n") { (k, v) -> "$k: $v" }.orEmpty()) }
    val valid = name.isNotBlank() && if (http) url.startsWith("http://") || url.startsWith("https://") else command.isNotBlank()
    AlertDialog(onDismissRequest = onDismiss,
        title = { Text(stringResource(if (old == null) R.string.agent_mcp_add else R.string.agent_mcp_edit)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                OutlinedTextField(name, { name = it.replace(Regex("[^A-Za-z0-9_-]"), "") }, singleLine = true,
                    label = { Text(stringResource(R.string.agent_mcp_name)) }, modifier = Modifier.fillMaxWidth())
                Row(Modifier.padding(vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = !http, onClick = { http = false }, label = { Text(stringResource(R.string.agent_mcp_stdio)) })
                    FilterChip(selected = http, onClick = { http = true }, label = { Text(stringResource(R.string.agent_mcp_http)) })
                }
                if (http) {
                    OutlinedTextField(url, { url = it.trim() }, singleLine = true, label = { Text("URL") },
                        modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(headers, { headers = it }, label = { Text(stringResource(R.string.agent_mcp_headers)) },
                        minLines = 2, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                } else {
                    OutlinedTextField(command, { command = it.trim() }, singleLine = true,
                        label = { Text(stringResource(R.string.agent_mcp_command)) },
                        placeholder = { Text("npx") }, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(args, { args = it }, singleLine = true,
                        label = { Text(stringResource(R.string.agent_mcp_args)) },
                        placeholder = { Text("-y @modelcontextprotocol/server-memory") },
                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                    OutlinedTextField(env, { env = it }, label = { Text(stringResource(R.string.agent_mcp_env)) },
                        minLines = 2, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                    Text(stringResource(R.string.agent_mcp_stdio_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
                }
                onDelete?.let {
                    TextButton(onClick = it, modifier = Modifier.padding(top = 8.dp)) {
                        Text(stringResource(R.string.agent_delete), color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = valid, onClick = {
                onSave(McpServer(name.trim(), http, command, split(args), lines(env, '='), url, lines(headers, ':'),
                    old?.enabled ?: true))
            }) { Text(stringResource(R.string.agent_save)) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.chat_cancel)) } })
}

/** A skill written here: its name, when to use it, what to do. */
@Composable
private fun SkillDialog(onSave: (String, String, String) -> Unit, onDismiss: () -> Unit) {
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var body by remember { mutableStateOf("") }
    AlertDialog(onDismissRequest = onDismiss, title = { Text(stringResource(R.string.agent_skill_new)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                OutlinedTextField(name, { name = it }, singleLine = true, label = { Text(stringResource(R.string.agent_mcp_name)) },
                    supportingText = { if (name.isNotBlank()) Text(AgentConfig.folderName(name)) },
                    modifier = Modifier.fillMaxWidth())
                OutlinedTextField(description, { description = it }, label = { Text(stringResource(R.string.agent_skill_when)) },
                    minLines = 2, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                OutlinedTextField(body, { body = it }, label = { Text(stringResource(R.string.agent_skill_body)) },
                    minLines = 5, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
            }
        },
        confirmButton = {
            TextButton(enabled = name.isNotBlank() && description.isNotBlank() && body.isNotBlank(),
                onClick = { onSave(name, description, body) }) { Text(stringResource(R.string.agent_save)) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.chat_cancel)) } })
}

@Composable
private fun Section(title: String) {
    HorizontalDivider(Modifier.padding(top = 16.dp))
    Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 12.dp, bottom = 6.dp))
}

/** A command line's arguments: split at spaces, "quoted" or 'quoted' ones kept whole. */
private fun split(s: String): List<String> {
    val out = ArrayList<String>()
    val cur = StringBuilder()
    var quote: Char? = null
    var any = false
    for (c in s) {
        when {
            quote != null && c == quote -> quote = null
            quote != null -> cur.append(c)
            c == '"' || c == '\'' -> { quote = c; any = true }
            c.isWhitespace() -> if (cur.isNotEmpty() || any) { out += cur.toString(); cur.clear(); any = false }
            else -> cur.append(c)
        }
    }
    if (cur.isNotEmpty() || any) out += cur.toString()
    return out
}

private fun quote(a: String) = if (a.isEmpty() || a.any { it.isWhitespace() || it == '"' }) "'$a'" else a

/** `KEY=VALUE` (or `Name: Value`) lines as pairs; others left out. */
private fun lines(s: String, sep: Char) = s.lines().mapNotNull { l ->
    val i = l.indexOf(sep)
    if (i <= 0) null else l.substring(0, i).trim() to l.substring(i + 1).trim()
}
