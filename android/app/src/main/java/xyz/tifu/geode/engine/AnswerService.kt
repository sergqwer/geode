package xyz.tifu.geode.engine

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.MainActivity
import xyz.tifu.geode.R

/**
 * Keeps Geode alive while the model answers with the app in the background: a special-use foreground service
 * (on-device AI inference the user started). The chat starts it when a generation begins - the user has just
 * sent a message, the app is in front - and stops it a moment after the generation ends. The engine's process is
 * bound from this one with BIND_IMPORTANT, so it rises with it; the answer lands in the chat and is saved as if
 * the app had stayed open. The notification says so and has Stop.
 */
class AnswerService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            (application as GeodeApp).engine.stop()   // the generation ends; the chat then stops this service
            return START_NOT_STICKY
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        return START_NOT_STICKY
    }

    /** Swiped away from the recent apps: nobody is left to read the answer, so the generation stops too. */
    override fun onTaskRemoved(rootIntent: Intent?) {
        (application as GeodeApp).engine.stop()
        stopSelf()
    }

    override fun onDestroy() {
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private fun notification(): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1, Intent(this, AnswerService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, GeodeApp.CHANNEL_ANSWERS)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(getString(R.string.answer_notif_title))
            .setContentText(getString(R.string.answer_notif_text))
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(0, 0, true)
            .addAction(0, getString(R.string.answer_notif_stop), stop)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    companion object {
        private const val NOTIFICATION_ID = 2
        private const val ACTION_STOP = "xyz.tifu.geode.answer.STOP"

        /** A generation began (call it while the app is in front: Android starts no foreground service from behind). */
        fun start(context: Context) {
            runCatching { ContextCompat.startForegroundService(context, Intent(context, AnswerService::class.java)) }
        }

        fun stop(context: Context) {
            runCatching { context.stopService(Intent(context, AnswerService::class.java)) }
        }
    }
}
