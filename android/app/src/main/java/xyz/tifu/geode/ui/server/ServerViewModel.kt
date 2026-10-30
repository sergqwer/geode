package xyz.tifu.geode.ui.server

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.data.AppSettings
import xyz.tifu.geode.models.ModelStore
import xyz.tifu.geode.server.LocalNetwork
import xyz.tifu.geode.server.LocalServer
import xyz.tifu.geode.server.ServerService
import xyz.tifu.geode.server.ServerStatus
import xyz.tifu.geode.server.WebChat
import java.security.SecureRandom

class ServerViewModel(private val app: GeodeApp) : ViewModel() {
    private val store = ModelStore(app)

    val settings: StateFlow<AppSettings> = app.settings.flow.stateIn(viewModelScope, SharingStarted.Eagerly, AppSettings())
    val status: StateFlow<ServerStatus> = LocalServer.status

    /** The phone's addresses on its local networks, looked up again every few seconds while the screen is shown. */
    val addresses: StateFlow<List<String>> = flow {
        while (true) {
            emit(LocalNetwork.addresses())
            delay(3000)
        }
    }.flowOn(Dispatchers.IO).stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    /** The id the server gives the selected model (its directory's name); null when none is selected. */
    val modelId: StateFlow<String?> = app.settings.flow
        .map { s -> s.selectedModel.takeIf { it.isNotEmpty() }?.let(store::find)?.id }
        .flowOn(Dispatchers.IO)
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    fun setOn(on: Boolean) = viewModelScope.launch {
        if (on) {
            if (app.settings.flow.first().serverKey.isEmpty()) app.settings.setServerKey(newKey())
            LocalServer.status.update { it.copy(portBusy = false) }
        }
        app.settings.setServerOn(on)
        apply()
    }

    fun setWebOn(on: Boolean) = viewModelScope.launch {
        if (on) {
            if (app.settings.flow.first().webPassword.isEmpty()) app.settings.setWebPassword(WebChat.newPassword())
            LocalServer.status.update { it.copy(portBusy = false) }
        }
        app.settings.setWebOn(on)
        apply()
    }

    /** A new web chat password: every browser signs in again. */
    fun renewPassword() = viewModelScope.launch { app.settings.setWebPassword(WebChat.newPassword()) }

    /** The server runs while the API or the web chat is on. */
    private suspend fun apply() {
        if (app.settings.flow.first().serverRuns) ServerService.start(app) else ServerService.stop(app)
    }

    fun setKeyOn(on: Boolean) = viewModelScope.launch {
        if (on && app.settings.flow.first().serverKey.isEmpty()) app.settings.setServerKey(newKey())
        app.settings.setServerKeyOn(on)
    }

    fun renewKey() = viewModelScope.launch { app.settings.setServerKey(newKey()) }

    fun setPort(port: Int) = viewModelScope.launch {
        app.settings.setServerPort(port)
        LocalServer.status.update { it.copy(portBusy = false) }
        // the running server moves to the new port
        if (app.settings.flow.first().serverRuns) ServerService.start(app)
    }

    companion object {
        /** "geode-" and 32 letters and digits (~190 bits). */
        fun newKey(): String {
            val abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            val r = SecureRandom()
            return "geode-" + (1..32).map { abc[r.nextInt(abc.length)] }.joinToString("")
        }
    }
}
