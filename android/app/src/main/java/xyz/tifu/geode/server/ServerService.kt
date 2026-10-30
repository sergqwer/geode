package xyz.tifu.geode.server

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.MainActivity
import xyz.tifu.geode.R
import xyz.tifu.geode.data.AppSettings
import java.io.IOException

/** The local server as its screen shows it. */
data class ServerStatus(
    val running: Boolean = false,
    val port: Int = 0,
    /** The port could not be had (another app has it). */
    val portBusy: Boolean = false,
    /** Requests answered since the server started, and generations running now. */
    val requests: Int = 0,
    val busy: Int = 0,
)

object LocalServer {
    val status = MutableStateFlow(ServerStatus())
}

/**
 * The local server while the user has the API or the web chat on (one HttpServer, one port: /v1 and the API's paths
 * to Api, the page and /web to WebChat, each only while it is on): a special-use foreground service - the phone serves
 * the on-device model to the local network - with a notification that shows the address and turns it off. It holds a
 * partial wake lock and a Wi-Fi lock: a sleeping phone answers nothing. Not sticky: killed, it comes back when the app
 * opens (MainActivity starts it while a setting is on).
 */
class ServerService : Service() {
    private val app get() = application as GeodeApp
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var http: HttpServer? = null
    private var port = 0
    private lateinit var engine: ApiEngine
    private lateinit var api: Api
    private lateinit var web: WebChat
    @Volatile
    private var key: String? = null
    /** The settings as they are now (which of the two is on). */
    @Volatile
    private var current = AppSettings()
    private var wake: PowerManager.WakeLock? = null
    private var wifi: WifiManager.WifiLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        engine = app.apiEngine
        api = Api(app, engine) { key }
        web = WebChat(app, engine) { current.webPassword.takeIf { it.isNotEmpty() } }
        // the key and the password as the screen changes them: the next request needs the new one (a new password
        // signs every browser out); both off, the server goes
        scope.launch {
            var password: String? = null
            app.settings.flow.collect { s ->
                current = s
                key = s.serverKey.takeIf { s.serverKeyOn && it.isNotEmpty() }
                if (password != null && password != s.webPassword) web.forget()
                password = s.webPassword
                if (!s.serverRuns && http != null) stopSelf()
            }
        }
        scope.launch { engine.tidy() }
        wake = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "geode:server")
            .apply { setReferenceCounted(false) }
        @Suppress("DEPRECATION")
        val mode = if (Build.VERSION.SDK_INT >= 29) WifiManager.WIFI_MODE_FULL_LOW_LATENCY else WifiManager.WIFI_MODE_FULL_HIGH_PERF
        wifi = runCatching { getSystemService(WifiManager::class.java).createWifiLock(mode, "geode:server") }.getOrNull()
            ?.apply { setReferenceCounted(false) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // the notification's Turn off (a plain startService: no foreground needed for it)
        if (intent?.action == ACTION_STOP) {
            scope.launch {
                app.settings.setServerOn(false)
                app.settings.setWebOn(false)
                stopSelf()
            }
            return START_NOT_STICKY
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        scope.launch {
            val s = app.settings.flow.first()
            key = s.serverKey.takeIf { s.serverKeyOn && it.isNotEmpty() }
            if (http == null || port != s.serverPort) listen(s.serverPort)
        }
        return START_NOT_STICKY
    }

    private fun listen(p: Int) {
        http?.stop()
        http = null
        val h = HttpServer(p, ::route)
        try {
            h.start()
        } catch (e: IOException) {
            LocalServer.status.update { it.copy(running = false, port = p, portBusy = true) }
            scope.launch {
                app.settings.setServerOn(false)
                app.settings.setWebOn(false)
                stopSelf()
            }
            return
        }
        http = h
        port = p
        LocalServer.status.update { ServerStatus(running = true, port = p) }
        // a day at most each time the server starts (the screen's switch or the app's start renew it)
        wake?.acquire(24 * 3600_000L)
        runCatching { wifi?.acquire() }
        getSystemService(android.app.NotificationManager::class.java).notify(NOTIFICATION_ID, notification())
    }

    /** The page and /web to the web chat, the rest to the API; each only while it is on. */
    private suspend fun route(req: HttpRequest, r: HttpResponder) {
        val s = current
        val webPath = req.path == "/" || req.path == "/index.html" || req.path.startsWith("/web/")
        when {
            webPath && s.webOn -> web.handle(req, r)
            !webPath && s.serverOn -> api.handle(req, r)
            req.path == "/" -> r.text(200, "Geode local server: the web chat is off (Geode's Local server screen).\n")
            else -> r.json(404, mapOf("error" to mapOf("message" to
                (if (webPath) "The web chat is off" else "The API is off") + " (Geode's Local server screen).",
                "type" to "not_found_error")))
        }
    }

    override fun onDestroy() {
        http?.stop()
        http = null
        LocalServer.status.update { it.copy(running = false, busy = 0) }
        runCatching { if (wake?.isHeld == true) wake?.release() }
        runCatching { if (wifi?.isHeld == true) wifi?.release() }
        scope.cancel()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private fun notification(): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 3, Intent(this, ServerService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        val address = LocalNetwork.addresses().firstOrNull()?.let { "http://$it:$port" }
        return NotificationCompat.Builder(this, GeodeApp.CHANNEL_SERVER)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(getString(R.string.server_notif_title))
            .setContentText(if (address != null && port > 0) getString(R.string.server_notif_text, address)
                else getString(R.string.server_no_network))
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(0, getString(R.string.server_notif_stop), stop)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    companion object {
        private const val NOTIFICATION_ID = 3
        private const val ACTION_STOP = "xyz.tifu.geode.server.STOP"

        /** Starts the server, or applies a new port (call it while the app is in front). */
        fun start(context: Context) {
            runCatching { ContextCompat.startForegroundService(context, Intent(context, ServerService::class.java)) }
        }

        fun stop(context: Context) {
            runCatching { context.stopService(Intent(context, ServerService::class.java)) }
        }
    }
}
