package xyz.tifu.geode.ui.chat

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Article
import androidx.compose.material.icons.outlined.Build
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.CheckBox
import androidx.compose.material.icons.outlined.CheckBoxOutlineBlank
import androidx.compose.material.icons.outlined.Checklist
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Code
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.EditNote
import androidx.compose.material.icons.outlined.Folder
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.IndeterminateCheckBox
import androidx.compose.material.icons.outlined.InsertDriveFile
import androidx.compose.material.icons.outlined.Language
import androidx.compose.material.icons.outlined.Public
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Terminal
import androidx.compose.material.icons.outlined.Web
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import xyz.tifu.geode.R
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.str

/** The agent's way to its answer: its words between the tools, and each tool's call as a card of its own kind. */
@Composable
fun AgentSteps(steps: List<AgentStep>) {
    Column(Modifier.fillMaxWidth().padding(bottom = 6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        for ((i, s) in steps.withIndex()) {
            if (s.tool.isEmpty()) {
                Text(s.title.trim(), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(horizontal = 4.dp))
            } else {
                StepCard(s, key = "$i:${s.id}")
            }
        }
    }
}

/** The files the agent made for an answer: a tap shows one here. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AgentFileChips(files: List<String>, onOpen: (String) -> Unit) {
    FlowRow(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (f in files) {
            AssistChip(onClick = { onOpen(f) },
                label = { Text(f.substringAfterLast('/'), maxLines = 1, overflow = TextOverflow.Ellipsis) },
                leadingIcon = { Icon(kindIcon(f), null, Modifier.size(18.dp)) })
        }
    }
}

private fun kindIcon(name: String): ImageVector = when (fileKind(name)) {
    FileKind.IMAGE -> Icons.Outlined.Image
    FileKind.PAGE -> Icons.Outlined.Web
    FileKind.MARKDOWN -> Icons.Outlined.Article
    FileKind.TEXT -> Icons.Outlined.Code
    FileKind.OTHER -> Icons.Outlined.InsertDriveFile
}

@Suppress("UNCHECKED_CAST")
private fun args(s: AgentStep): Map<String, Any?> =
    runCatching { Json.parse(s.input) as? Map<String, Any?> }.getOrNull() ?: emptyMap()

/** A path in /workspace as the user knows it: from the chat's folder. */
private fun short(path: String) = path.removePrefix("/workspace/").ifEmpty { path }

/** A tool's call: what it does, in a shape of its own (a terminal, a diff, a checklist), and how it went. */
@Composable
private fun StepCard(s: AgentStep, key: String) {
    var open by rememberSaveable(key) { mutableStateOf(false) }
    val a = remember(s.input) { args(s) }
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, shape = RoundedCornerShape(12.dp),
        modifier = Modifier.fillMaxWidth().clickable(enabled = s.output.isNotEmpty()) { open = !open }) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(icon(s.tool), null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
                Text(label(s.tool), style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(start = 8.dp))
                Text(subject(s, a), style = MaterialTheme.typography.labelMedium, maxLines = 1,
                    overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(start = 8.dp).weight(1f))
                when (s.ok) {
                    null -> Dots(dot = 3.5.dp)
                    true -> Icon(Icons.Outlined.Check, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
                    false -> Icon(Icons.Outlined.Close, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.error)
                }
            }
            Body(s, a, open)
        }
    }
}

/** What the card shows below its head: the command and its output, the new file's start, the edit, the plan. */
@Composable
private fun Body(s: AgentStep, a: Map<String, Any?>, open: Boolean) {
    when (s.tool) {
        "bash", "pwsh", "shell" -> {
            Terminal("$ " + (a.str("command") ?: s.title) + if (open && s.output.isNotEmpty()) "\n" + s.output else "",
                lines = if (open) 400 else 4)
        }
        "write" -> a.str("content")?.let { Code(it, lines = if (open) 400 else 6) }
        "edit" -> {
            val old = a.str("old_string").orEmpty().lines()
            val new = a.str("new_string").orEmpty().lines()
            val n = if (open) 200 else 4
            Column(Modifier.padding(top = 6.dp)) {
                for (l in old.take(n)) DiffLine("- $l", Color(0xFFE5534B))
                for (l in new.take(n)) DiffLine("+ $l", Color(0xFF57AB5A))
            }
        }
        "todo_write" -> Column(Modifier.padding(top = 4.dp)) {
            for (t in a.list("todos").orEmpty()) {
                @Suppress("UNCHECKED_CAST")
                val m = t as? Map<String, Any?> ?: continue
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 1.dp)) {
                    Icon(when (m.str("status")) {
                        "completed" -> Icons.Outlined.CheckBox
                        "in_progress" -> Icons.Outlined.IndeterminateCheckBox
                        else -> Icons.Outlined.CheckBoxOutlineBlank
                    }, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(m.str("content").orEmpty(), style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(start = 6.dp))
                }
            }
        }
        else -> AnimatedVisibility(open && s.output.isNotEmpty()) { Code(s.output, lines = 400) }
    }
    if (s.ok == false && !open && s.output.isNotEmpty() && s.tool !in listOf("bash", "pwsh", "shell")) {
        Text(s.output.lineSequence().firstOrNull().orEmpty(), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error, maxLines = 2, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(top = 4.dp))
    }
}

/** A command and its output as a terminal shows them. */
@Composable
private fun Terminal(text: String, lines: Int) {
    Surface(color = Color(0xFF15141B), shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth().padding(top = 6.dp)) {
        SelectionContainer {
            Text(text, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall, color = Color(0xFFE6E1F2),
                maxLines = lines, overflow = TextOverflow.Ellipsis,
                modifier = Modifier.heightIn(max = 360.dp).verticalScroll(rememberScrollState())
                    .horizontalScroll(rememberScrollState()).padding(horizontal = 10.dp, vertical = 8.dp))
        }
    }
}

@Composable
private fun Code(text: String, lines: Int) {
    SelectionContainer {
        Text(text, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = lines, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(top = 6.dp).heightIn(max = 360.dp).verticalScroll(rememberScrollState())
                .horizontalScroll(rememberScrollState()))
    }
}

@Composable
private fun DiffLine(text: String, color: Color) {
    Text(text, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall, color = color, maxLines = 1,
        overflow = TextOverflow.Ellipsis)
}

/** The browser's tool a call is (open, eval...: BrowserMcp's, under the name the harness gave its server), or null. */
private val BROWSER_TOOL = Regex("^mcp__(?:browser|geode_browser|geode-browser-app)__([a-z]+)$")

private fun browserTool(tool: String): String? = BROWSER_TOOL.find(tool)?.groupValues?.get(1)

/** What the call is about, in the card's head: a file, a pattern, an address, the command's purpose. */
private fun subject(s: AgentStep, a: Map<String, Any?>): String = browserTool(s.tool)?.let { t ->
    when (t) {
        "open" -> a.str("url")
        "eval" -> a.str("expression")?.lineSequence()?.firstOrNull()
        "text" -> a.str("selector")
        "act" -> listOfNotNull(a.str("action"), a.str("selector") ?: a.str("key") ?: a.str("text")).joinToString(" ")
        "cdp" -> a.str("method")
        else -> null
    }.orEmpty()
} ?: when (s.tool) {
    "bash", "pwsh", "shell" -> a.str("description") ?: ""
    "read", "write", "edit", "read_image" -> short(a.str("file_path") ?: s.title)
    "glob", "grep" -> a.str("pattern") ?: s.title
    "web_fetch" -> a.str("url") ?: s.title
    "web_search" -> a.list("queries")?.joinToString(", ") ?: s.title
    "todo_write" -> ""
    else -> s.title
}

@Composable
private fun label(tool: String): String = stringResource(browserTool(tool)?.let {
    when (it) {
        "open" -> R.string.agent_tool_browser_open
        "screenshot" -> R.string.agent_tool_browser_shot
        "eval" -> R.string.agent_tool_browser_eval
        "console" -> R.string.agent_tool_browser_console
        "text" -> R.string.agent_tool_browser_text
        "act" -> R.string.agent_tool_browser_act
        else -> R.string.agent_tool_browser_cdp
    }
} ?: when (tool) {
    "bash", "pwsh", "shell" -> R.string.agent_tool_command
    "read" -> R.string.agent_tool_read
    "write" -> R.string.agent_tool_write
    "edit" -> R.string.agent_tool_edit
    "glob" -> R.string.agent_tool_glob
    "grep" -> R.string.agent_tool_grep
    "web_fetch" -> R.string.agent_tool_fetch
    "web_search" -> R.string.agent_tool_search
    "read_image" -> R.string.agent_tool_image
    "todo_write" -> R.string.agent_tool_plan
    "working_directory" -> R.string.agent_tool_folder
    else -> R.string.agent_tool_other
})

private fun icon(tool: String): ImageVector = if (browserTool(tool) != null) Icons.Outlined.Language else when (tool) {
    "bash", "pwsh", "shell" -> Icons.Outlined.Terminal
    "read" -> Icons.Outlined.Description
    "write", "edit" -> Icons.Outlined.EditNote
    "glob", "grep" -> Icons.Outlined.Search
    "web_fetch", "web_search" -> Icons.Outlined.Public
    "read_image" -> Icons.Outlined.Image
    "todo_write" -> Icons.Outlined.Checklist
    "working_directory" -> Icons.Outlined.Folder
    else -> Icons.Outlined.Build
}
