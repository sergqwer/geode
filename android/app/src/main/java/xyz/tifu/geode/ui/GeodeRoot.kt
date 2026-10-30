package xyz.tifu.geode.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.outlined.Inventory2
import androidx.compose.material.icons.outlined.Lan
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.Speed
import androidx.compose.material.icons.outlined.Terminal
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.ui.agent.AgentScreen
import xyz.tifu.geode.ui.agent.AgentViewModel
import xyz.tifu.geode.ui.bench.BenchScreen
import xyz.tifu.geode.ui.bench.BenchViewModel
import xyz.tifu.geode.ui.chat.ChatScreen
import xyz.tifu.geode.ui.chat.ChatViewModel
import xyz.tifu.geode.ui.models.ModelsScreen
import xyz.tifu.geode.ui.models.ModelsViewModel
import xyz.tifu.geode.server.QrLogins
import xyz.tifu.geode.ui.server.QrApproval
import xyz.tifu.geode.ui.server.ServerScreen
import xyz.tifu.geode.ui.server.ServerViewModel
import xyz.tifu.geode.ui.settings.SettingsScreen
import xyz.tifu.geode.ui.settings.SettingsViewModel

private enum class Screen(val label: Int, val icon: ImageVector) {
    CHAT(R.string.nav_chat, Icons.AutoMirrored.Outlined.Chat),
    MODELS(R.string.nav_models, Icons.Outlined.Inventory2),
    BENCH(R.string.nav_bench, Icons.Outlined.Speed),
    SERVER(R.string.nav_server, Icons.Outlined.Lan),
    AGENT(R.string.nav_agent, Icons.Outlined.Terminal),
    SETTINGS(R.string.nav_settings, Icons.Outlined.Settings),
}

/**
 * The chat fills the screen; everything else is behind the drawer (☰): no bar at the bottom taking the room the
 * conversation needs. No navigation library: the screens keep their state in their view models, Back from any of
 * them returns to the chat.
 */
@Composable
fun GeodeRoot(app: GeodeApp) {
    var screen by rememberSaveable { mutableIntStateOf(Screen.CHAT.ordinal) }
    val chat: ChatViewModel = viewModel { ChatViewModel(app) }
    val models: ModelsViewModel = viewModel { ModelsViewModel(app) }
    val settings: SettingsViewModel = viewModel { SettingsViewModel(app) }
    val bench: BenchViewModel = viewModel { BenchViewModel(app) }
    // the device check adb started (MainActivity, DeviceReport.ACTION_CHECK): the Benchmark, every test, a file only
    val checkAsked by xyz.tifu.geode.bench.DeviceReport.requested.collectAsStateWithLifecycle()
    LaunchedEffect(checkAsked) {
        if (checkAsked) {
            xyz.tifu.geode.bench.DeviceReport.requested.value = false
            screen = Screen.BENCH.ordinal
            bench.checkAndSend(send = false)
        }
    }
    val server: ServerViewModel = viewModel { ServerViewModel(app) }
    val agent: AgentViewModel = viewModel { AgentViewModel(app) }
    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val openDrawer: () -> Unit = { scope.launch { drawer.open() } }
    fun go(s: Screen) {
        screen = s.ordinal
        scope.launch { drawer.close() }
    }

    BackHandler(enabled = drawer.isClosed && screen != Screen.CHAT.ordinal) { screen = Screen.CHAT.ordinal }

    // a web chat sign-in code (scanned, or a geode:// link the camera app opened): asked about over any screen
    val qr by QrLogins.incoming.collectAsStateWithLifecycle()
    qr?.let { text -> QrApproval(app, text) { QrLogins.incoming.value = null } }

    ModalNavigationDrawer(
        drawerState = drawer,
        drawerContent = {
            ModalDrawerSheet {
                DrawerHeader()
                HorizontalDivider(Modifier.padding(horizontal = 16.dp, vertical = 8.dp))
                Screen.entries.forEach { s ->
                    NavigationDrawerItem(
                        label = { Text(stringResource(s.label)) },
                        icon = { Icon(s.icon, contentDescription = null) },
                        selected = screen == s.ordinal,
                        onClick = { go(s) },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding),
                    )
                }
            }
        },
    ) {
        // the theme's background and its content colour (the Scaffold that held the bottom bar gave them): text and
        // icons without a colour of their own were black on the dark theme's background without it
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            // the system's navigation bar below every screen (the chat's input adds the keyboard on top of it)
            Box(Modifier.fillMaxSize().navigationBarsPadding()) {
                when (Screen.entries[screen]) {
                    Screen.CHAT -> ChatScreen(chat, openModels = { screen = Screen.MODELS.ordinal }, openDrawer = openDrawer)
                    Screen.MODELS -> ModelsScreen(models, openDrawer)
                    Screen.BENCH -> BenchScreen(bench, openDrawer)
                    Screen.SERVER -> ServerScreen(server, openDrawer)
                    Screen.AGENT -> AgentScreen(agent, openDrawer)
                    Screen.SETTINGS -> SettingsScreen(settings, openDrawer)
                }
            }
        }
    }
}

/** The app's icon and name at the top of the drawer */
@Composable
private fun DrawerHeader() {
    Row(Modifier.padding(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        // the adaptive icon's layers, its 72-unit safe zone filling the box (as a launcher shows it)
        Box(Modifier.size(44.dp).clip(RoundedCornerShape(12.dp)), contentAlignment = Alignment.Center) {
            Image(painterResource(R.drawable.ic_launcher_background), null, Modifier.requiredSize(66.dp))
            Image(painterResource(R.drawable.ic_launcher_foreground), null, Modifier.requiredSize(66.dp))
        }
        Column(Modifier.padding(start = 14.dp)) {
            Text(stringResource(R.string.app_name), style = MaterialTheme.typography.titleLarge)
            Text(stringResource(R.string.nav_tagline), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
