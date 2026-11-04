package xyz.tifu.geode.agent

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import xyz.tifu.geode.util.Json
import xyz.tifu.geode.util.bool
import xyz.tifu.geode.util.list
import xyz.tifu.geode.util.str
import java.io.File
import java.io.FileOutputStream

/**
 * An MCP server the agent's sessions open with (ACP's mcpServers): one run in the agent's Linux and spoken to over its
 * stdin and stdout ([command], [args], [env]), or one reached over HTTP ([url], [headers]).
 */
data class McpServer(
    val name: String,
    val http: Boolean = false,
    val command: String = "",
    val args: List<String> = emptyList(),
    val env: List<Pair<String, String>> = emptyList(),
    val url: String = "",
    val headers: List<Pair<String, String>> = emptyList(),
    val enabled: Boolean = true,
) {
    /**
     * As ACP's session/new takes it. The harness wants a command's absolute path: a bare one (python3, npx, one pip
     * installed) is found on the chat's PATH by env.
     */
    fun acp(): Map<String, Any?> = if (http) linkedMapOf(
        "type" to "http", "name" to name, "url" to url,
        "headers" to headers.map { (k, v) -> linkedMapOf("name" to k, "value" to v) },
    ) else linkedMapOf(
        "name" to name,
        "command" to if (command.startsWith("/")) command else "/usr/bin/env",
        "args" to if (command.startsWith("/")) args else listOf(command) + args,
        "env" to env.map { (k, v) -> linkedMapOf("name" to k, "value" to v) },
    )
}

/** A skill the agent can use (a SKILL.md: its name and description from the frontmatter, its folder). */
data class Skill(val name: String, val description: String, val dir: File)

/**
 * What the user gives every chat's agent: MCP servers (files/agent/mcp.json) and skills of their own (files/skills,
 * one folder each with its SKILL.md - the harness reads them as the user's skills, beside a chat's project ones).
 */
class AgentConfig(context: Context) {
    private val file = File(context.filesDir, "agent/mcp.json")
    val skillsDir = File(context.filesDir, "skills")
    private val _servers = MutableStateFlow(load())
    val servers: StateFlow<List<McpServer>> = _servers.asStateFlow()
    private val _skills = MutableStateFlow(scanSkills())
    val skills: StateFlow<List<Skill>> = _skills.asStateFlow()

    /** Counts the changes: a harness started before the last one starts again (its servers and skills are read at start). */
    @Volatile
    var version = 0
        private set

    /** The servers a session opens with. */
    fun enabledServers(): List<McpServer> = _servers.value.filter { it.enabled }

    /** The agent has the phone's browser (agent/browser.BrowserMcp); on unless the user turned it off. */
    private val browserOff = File(context.filesDir, "agent/browser-off")
    private val _browser = MutableStateFlow(!browserOff.exists())
    val browser: StateFlow<Boolean> = _browser.asStateFlow()

    fun setBrowser(on: Boolean) {
        if (on == _browser.value) return
        if (on) browserOff.delete() else {
            browserOff.parentFile?.mkdirs()
            browserOff.createNewFile()
        }
        _browser.value = on
        version++
    }

    fun saveServer(server: McpServer, replacing: String? = null) {
        val list = _servers.value.filter { it.name != (replacing ?: server.name) && it.name != server.name } + server
        store(list.sortedBy { it.name.lowercase() })
    }

    fun deleteServer(name: String) = store(_servers.value.filter { it.name != name })

    fun setEnabled(name: String, on: Boolean) = store(_servers.value.map { if (it.name == name) it.copy(enabled = on) else it })

    private fun store(list: List<McpServer>) {
        file.parentFile?.mkdirs()
        val tmp = File(file.path + ".tmp")
        FileOutputStream(tmp).use {
            it.write(Json.write(list.map { s ->
                linkedMapOf("name" to s.name, "http" to s.http, "command" to s.command, "args" to s.args,
                    "env" to s.env.map { (k, v) -> listOf(k, v) }, "url" to s.url,
                    "headers" to s.headers.map { (k, v) -> listOf(k, v) }, "enabled" to s.enabled)
            }).toByteArray())
            it.fd.sync()
        }
        tmp.renameTo(file)
        _servers.value = list
        version++
    }

    @Suppress("UNCHECKED_CAST")
    private fun load(): List<McpServer> = runCatching {
        (Json.parse(file.readText()) as List<Map<String, Any?>>).map { o ->
            fun pairs(k: String) = o.list(k).orEmpty().mapNotNull { p ->
                (p as? List<*>)?.takeIf { it.size == 2 }?.let { it[0].toString() to it[1].toString() }
            }
            McpServer(o.str("name").orEmpty(), o.bool("http") == true, o.str("command").orEmpty(),
                o.list("args").orEmpty().map { it.toString() }, pairs("env"), o.str("url").orEmpty(), pairs("headers"),
                o.bool("enabled") != false)
        }.filter { it.name.isNotBlank() }
    }.getOrDefault(emptyList())

    // ---- skills

    fun refreshSkills() {
        _skills.value = scanSkills()
        version++
    }

    private fun scanSkills(): List<Skill> = (skillsDir.listFiles() ?: emptyArray())
        .filter { it.isDirectory }
        .mapNotNull { d ->
            val md = File(d, "SKILL.md").takeIf { it.isFile } ?: return@mapNotNull null
            val front = frontmatter(md.readText())
            Skill(front["name"] ?: d.name, front["description"].orEmpty(), d)
        }
        .sortedBy { it.name.lowercase() }

    /** A new skill written here: [name] (its folder), [description] (when the agent should use it), [body]. */
    fun createSkill(name: String, description: String, body: String) {
        val id = folderName(name)
        val d = File(skillsDir, id).apply { mkdirs() }
        File(d, "SKILL.md").writeText("---\nname: $id\ndescription: ${description.replace('\n', ' ').trim()}\n---\n" +
            body.trim() + "\n")
        refreshSkills()
    }

    /**
     * A skill from the phone: a SKILL.md (or any .md with the frontmatter) becomes a folder of its own; a folder is
     * copied whole (its scripts and references beside its SKILL.md). The skill's name, or null: none found.
     */
    fun importSkill(context: Context, uri: Uri, tree: Boolean): String? {
        if (tree) {
            val root = DocumentFile.fromTreeUri(context, uri) ?: return null
            val md = root.findFile("SKILL.md") ?: return null
            val text = context.contentResolver.openInputStream(md.uri)?.use { it.readBytes().decodeToString() } ?: return null
            val id = folderName(frontmatter(text)["name"] ?: root.name ?: return null)
            val to = File(skillsDir, id).apply { deleteRecursively(); mkdirs() }
            copyTree(context, root, to)
            refreshSkills()
            return id
        }
        val text = context.contentResolver.openInputStream(uri)?.use { it.readBytes().decodeToString() } ?: return null
        val name = frontmatter(text)["name"] ?: return null
        val to = File(skillsDir, folderName(name)).apply { mkdirs() }
        File(to, "SKILL.md").writeText(text)
        refreshSkills()
        return name
    }

    fun deleteSkill(skill: Skill) {
        if (skill.dir.parentFile == skillsDir) skill.dir.deleteRecursively()
        refreshSkills()
    }

    private fun copyTree(context: Context, from: DocumentFile, to: File) {
        to.mkdirs()
        for (c in from.listFiles()) {
            val n = c.name?.replace('/', '_')?.takeIf { it.isNotBlank() && it != "." && it != ".." } ?: continue
            if (c.isDirectory) copyTree(context, c, File(to, n))
            else context.contentResolver.openInputStream(c.uri)?.use { i -> File(to, n).outputStream().use { i.copyTo(it) } }
        }
    }

    companion object {
        /** A skill's folder name: lower case letters, digits and dashes (the harness's skill names). */
        fun folderName(name: String): String =
            name.trim().lowercase().replace(Regex("[^a-z0-9._-]+"), "-").trim('-', '.').ifEmpty { "skill" }.take(64)

        /** The YAML frontmatter's plain `key: value` lines (what a skill's catalog entry needs). */
        fun frontmatter(text: String): Map<String, String> {
            val lines = text.lines()
            if (lines.firstOrNull()?.trim() != "---") return emptyMap()
            val end = lines.drop(1).indexOfFirst { it.trim() == "---" }.takeIf { it >= 0 } ?: return emptyMap()
            return lines.subList(1, end + 1).mapNotNull { l ->
                val i = l.indexOf(':')
                if (i <= 0 || l.startsWith(" ")) null
                else l.substring(0, i).trim() to l.substring(i + 1).trim().removeSurrounding("\"").removeSurrounding("'")
            }.toMap()
        }
    }
}
