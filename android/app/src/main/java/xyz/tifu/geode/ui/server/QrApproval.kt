package xyz.tifu.geode.ui.server

import android.widget.Toast
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import kotlinx.coroutines.flow.first
import xyz.tifu.geode.GeodeApp
import xyz.tifu.geode.R
import xyz.tifu.geode.server.QrLogins
import xyz.tifu.geode.server.QrPending

/**
 * A web chat sign-in code the app was given (scanned here, or opened as a geode:// link from the camera app): the
 * browser it signs in - its kind and address - and the user's yes. A code from anywhere else could sign a stranger's
 * browser in, hence the question.
 */
@Composable
fun QrApproval(app: GeodeApp, text: String, onDone: () -> Unit) {
    val context = LocalContext.current
    // the sign-in it names, and whether the web chat is on (the settings read, not their first default)
    val check by produceState<Pair<QrPending?, Boolean>?>(null, text) {
        value = QrLogins.byScan(text) to app.settings.flow.first().webOn
    }
    val c = check ?: return
    val p = c.first
    LaunchedEffect(c) {
        if (p == null || !c.second) {
            Toast.makeText(context, if (!c.second) R.string.qr_web_off else R.string.qr_invalid, Toast.LENGTH_LONG).show()
            onDone()
        }
    }
    if (p == null || !c.second) return
    val who = listOf(QrLogins.describe(p.agent), p.address).filter { it.isNotEmpty() }.joinToString(" · ")
    AlertDialog(
        onDismissRequest = onDone,
        title = { Text(stringResource(R.string.qr_confirm_title)) },
        text = { Text(stringResource(R.string.qr_confirm_text, who)) },
        confirmButton = {
            TextButton(onClick = {
                QrLogins.approve(p)
                Toast.makeText(context, R.string.qr_done, Toast.LENGTH_SHORT).show()
                onDone()
            }) { Text(stringResource(R.string.qr_confirm_yes)) }
        },
        dismissButton = { TextButton(onClick = onDone) { Text(stringResource(R.string.chat_cancel)) } },
    )
}
