package xyz.tifu.geode.ui.agent

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.agent.McpServer
import xyz.tifu.geode.agent.Skill

/** The agent's Linux as the screen shows it: there or not, the image this build ships or an older one, unpacking. */
data class LinuxState(val installed: Boolean = false, val outdated: Boolean = false, val installing: Float? = null,
                      val error: String? = null)

/** The agent screen: its Linux, the user's skills and MCP servers (every chat's agent's). */
class AgentViewModel(private val app: GeodeApp) : ViewModel() {
    private val config = app.agentConfig
    val servers: StateFlow<List<McpServer>> = config.servers
    val skills: StateFlow<List<Skill>> = config.skills
    private val _linux = MutableStateFlow(LinuxState())
    val linux: StateFlow<LinuxState> = _linux.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val (ready, old) = withContext(Dispatchers.IO) { app.agent.ready to app.agent.outdated }
            _linux.update { it.copy(installed = ready, outdated = old) }
            withContext(Dispatchers.IO) { config.refreshSkills() }
        }
    }

    /** The Linux unpacked (again) from the image this build ships; the chats' folders stay. */
    fun installLinux() {
        if (_linux.value.installing != null || app.agent.busy) return
        _linux.update { it.copy(installing = 0f, error = null) }
        viewModelScope.launch {
            val error = runCatching { app.agent.install { p -> _linux.update { it.copy(installing = p) } } }
                .exceptionOrNull()?.message
            _linux.update { it.copy(installing = null, error = error) }
            refresh()
        }
    }

    fun saveServer(server: McpServer, replacing: String?) = io { config.saveServer(server, replacing) }
    fun deleteServer(name: String) = io { config.deleteServer(name) }
    fun setEnabled(name: String, on: Boolean) = io { config.setEnabled(name, on) }

    fun createSkill(name: String, description: String, body: String) = io { config.createSkill(name, description, body) }
    fun deleteSkill(skill: Skill) = io { config.deleteSkill(skill) }

    /** A SKILL.md, or a folder holding one, from the phone. */
    fun importSkill(uri: Uri, tree: Boolean, done: (String?) -> Unit) {
        viewModelScope.launch {
            val name = withContext(Dispatchers.IO) { runCatching { config.importSkill(app, uri, tree) }.getOrNull() }
            done(name)
        }
    }

    /** A skill's instructions, to read. */
    suspend fun skillText(skill: Skill): String =
        withContext(Dispatchers.IO) { runCatching { java.io.File(skill.dir, "SKILL.md").readText() }.getOrDefault("") }

    private fun io(block: () -> Unit) {
        viewModelScope.launch(Dispatchers.IO) { block() }
    }
}
