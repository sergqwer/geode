package xyz.tifu.geode

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.getValue
import androidx.core.view.WindowCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import xyz.tifu.geode.data.AppSettings
import xyz.tifu.geode.server.QrLogins
import xyz.tifu.geode.server.ServerService
import xyz.tifu.geode.ui.GeodeRoot
import xyz.tifu.geode.ui.theme.GeodeTheme

// AppCompatActivity only for the per-app language picker (AppCompatDelegate.setApplicationLocales);
// Compose Material 3 paints the whole UI.
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Edge to edge with current APIs only: the content under the system bars (Compose pads it by their insets),
        // the bars transparent through the theme on Android 11-14 (Android 15+ draws them so itself), their icons in
        // the app's theme (GeodeTheme). androidx's enableEdgeToEdge() sets the bars' colours and the cutout mode
        // through window calls Android 15 deprecates, which Play flags.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val app = application as GeodeApp
        // the local server comes back with the app while it is on (a foreground service starts only from here)
        lifecycleScope.launch { if (app.settings.flow.first().serverRuns) ServerService.start(app) }
        signIn(intent)
        deviceCheck(intent)
        setContent {
            val settings by app.settings.flow.collectAsStateWithLifecycle(AppSettings())
            GeodeTheme(themeMode = settings.theme, dynamicColor = settings.dynamicColor) {
                GeodeRoot(app)
            }
        }
    }

    /**
     * In the background an idle agent's harness goes (Android would freeze it with its RAM held), and the conversation
     * the engine holds is saved to the storage (the engine's process may be let go).
     */
    override fun onStop() {
        super.onStop()
        if (isChangingConfigurations) return
        val app = application as GeodeApp
        app.agent.stopIfIdle()
        app.appScope.launch { app.gate.flush() }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        signIn(intent)
        deviceCheck(intent)
    }

    /** The device check adb asked for (DeviceReport.ACTION_CHECK): GeodeRoot opens the Benchmark and runs it */
    private fun deviceCheck(intent: Intent?) {
        if (intent?.action == xyz.tifu.geode.bench.DeviceReport.ACTION_CHECK) xyz.tifu.geode.bench.DeviceReport.requested.value = true
    }

    /** A web chat sign-in link (geode://login?c=...) the camera app opened: the app asks about it. */
    private fun signIn(intent: Intent?) {
        val d = intent?.data ?: return
        if (d.scheme == "geode" && d.host == "login") QrLogins.incoming.value = d.toString()
    }
}
