package xyz.tifu.geode.ui.server

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.compose.material.icons.outlined.QrCodeScanner
import xyz.tifu.geode.server.QrLogins
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import xyz.tifu.geode.R

/**
 * The local server: its port and what runs on it (the API on the local network, OpenAI's and Anthropic's at once, and
 * the web chat) first, then their addresses, the key, and a request to try it with.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ServerScreen(vm: ServerViewModel, openDrawer: () -> Unit) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val status by vm.status.collectAsStateWithLifecycle()
    val addresses by vm.addresses.collectAsStateWithLifecycle()
    val model by vm.modelId.collectAsStateWithLifecycle()
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    val copied = stringResource(R.string.server_copied)
    val copy: (String) -> Unit = { text ->
        clipboard.setText(AnnotatedString(text))
        // Android 13+ confirms a copy itself
        if (Build.VERSION.SDK_INT < 33) Toast.makeText(context, copied, Toast.LENGTH_SHORT).show()
    }
    val port = if (status.running) status.port else s.serverPort
    var scanning by remember { mutableStateOf(false) }
    val cameraRefused = stringResource(R.string.qr_camera)
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) scanning = true else Toast.makeText(context, cameraRefused, Toast.LENGTH_LONG).show()
    }
    if (scanning) {
        QrScanner(onCode = { text ->
            scanning = false
            QrLogins.incoming.value = text
        }, onClose = { scanning = false })
    }

    Column(Modifier.fillMaxSize()) {
        TopAppBar(title = { Text(stringResource(R.string.nav_server)) }, navigationIcon = {
            IconButton(onClick = openDrawer) {
                Icon(Icons.Outlined.Menu, contentDescription = stringResource(R.string.nav_menu))
            }
        })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            // ---- the port, then what runs on it: typed before either is turned on ----
            if (status.portBusy && !s.serverRuns) {
                Text(stringResource(R.string.server_port_busy, status.port), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp))
            }
            var portText by remember(s.serverPort) { mutableStateOf(s.serverPort.toString()) }
            // what an app may open without root
            val newPort = portText.toIntOrNull()?.takeIf { it in 1024..65535 }
            Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.Top) {
                OutlinedTextField(
                    value = portText,
                    onValueChange = { v -> portText = v.filter { it.isDigit() }.take(5) },
                    label = { Text(stringResource(R.string.server_port)) },
                    supportingText = { Text(stringResource(R.string.server_port_range)) },
                    singleLine = true,
                    isError = newPort == null,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                )
                OutlinedButton(onClick = { newPort?.let { vm.setPort(it) } },
                    enabled = newPort != null && newPort != s.serverPort,
                    modifier = Modifier.padding(start = 12.dp, top = 8.dp)) {
                    Text(stringResource(R.string.server_port_apply))
                }
            }

            Row(Modifier.padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.server_api), style = MaterialTheme.typography.titleMedium)
                    val state = when {
                        !s.serverOn -> stringResource(R.string.server_state_off)
                        !status.running -> stringResource(R.string.server_state_starting)
                        status.busy > 0 -> stringResource(R.string.server_state_busy, status.requests)
                        status.running -> stringResource(R.string.server_state_on, status.requests)
                        else -> stringResource(R.string.server_state_starting)
                    }
                    Text(state, style = MaterialTheme.typography.bodySmall,
                        color = if (s.serverOn && status.running) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.serverOn, onCheckedChange = { vm.setOn(it) })
            }
            Text(stringResource(R.string.server_api_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
            Row(Modifier.padding(top = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.server_web), style = MaterialTheme.typography.titleMedium)
                    Text(stringResource(R.string.server_web_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.webOn, onCheckedChange = { vm.setWebOn(it) })
            }

            // ---- the API ----
            if (s.serverOn) {
                Section(stringResource(R.string.server_api))
                if (model == null) {
                    Text(stringResource(R.string.server_no_model), style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error)
                } else {
                    Text(stringResource(R.string.server_model, model!!), style = MaterialTheme.typography.bodyMedium)
                }
                if (addresses.isEmpty()) {
                    Text(stringResource(R.string.server_no_network), style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp))
                }
                for (a in addresses) {
                    Address(stringResource(R.string.server_openai_url), "http://$a:$port/v1", copy)
                    Address(stringResource(R.string.server_anthropic_url), "http://$a:$port", copy)
                }
            }

            // ---- the web chat ----
            if (s.webOn) {
                Section(stringResource(R.string.server_section_web))
                for (a in addresses) Address(stringResource(R.string.server_web_url), "http://$a:$port/", copy)
                if (s.webPassword.isNotEmpty()) {
                    Address(stringResource(R.string.server_web_password), s.webPassword, copy)
                    TextButton(onClick = { vm.renewPassword() }) { Text(stringResource(R.string.server_web_password_new)) }
                }
                // or no password at all: the browser's sign-in page shows a QR code, the phone scans it
                OutlinedButton(onClick = {
                    if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                        PackageManager.PERMISSION_GRANTED) scanning = true
                    else camera.launch(Manifest.permission.CAMERA)
                }, modifier = Modifier.padding(top = 4.dp)) {
                    Icon(Icons.Outlined.QrCodeScanner, null, Modifier.size(18.dp))
                    Text(stringResource(R.string.server_web_scan), Modifier.padding(start = 8.dp))
                }
            }

            // ---- access ----
            Section(stringResource(R.string.server_section_access))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.server_key_on), style = MaterialTheme.typography.bodyLarge)
                    Text(stringResource(R.string.server_key_on_desc), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.serverKeyOn, onCheckedChange = { vm.setKeyOn(it) })
            }
            if (s.serverKeyOn && s.serverKey.isNotEmpty()) {
                Address(stringResource(R.string.server_key), s.serverKey, copy)
                TextButton(onClick = { vm.renewKey() }) { Text(stringResource(R.string.server_key_new)) }
            }

            // ---- try it ----
            Section(stringResource(R.string.server_section_try))
            val host = addresses.firstOrNull() ?: "PHONE_IP"
            val auth = if (s.serverKeyOn && s.serverKey.isNotEmpty()) "  -H \"Authorization: Bearer ${s.serverKey}\" \\\n" else ""
            val curl = "curl http://$host:$port/v1/chat/completions \\\n$auth" +
                "  -H \"Content-Type: application/json\" \\\n" +
                "  -d '{\"model\":\"${model ?: "geode"}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
            Surface(color = MaterialTheme.colorScheme.surfaceContainerHigh, shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()) {
                Row(verticalAlignment = Alignment.Top) {
                    SelectionContainer(Modifier.weight(1f)) {
                        Text(curl, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(12.dp))
                    }
                    IconButton(onClick = { copy(curl) }) {
                        Icon(Icons.Outlined.ContentCopy, contentDescription = stringResource(R.string.server_copy),
                            modifier = Modifier.size(20.dp))
                    }
                }
            }
            Text(stringResource(R.string.server_lan_only), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 12.dp))
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** A labelled address (or key) with its copy button. */
@Composable
private fun Address(label: String, value: String, copy: (String) -> Unit) {
    Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            SelectionContainer {
                Text(value, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodyMedium)
            }
        }
        IconButton(onClick = { copy(value) }) {
            Icon(Icons.Outlined.ContentCopy, contentDescription = stringResource(R.string.server_copy),
                modifier = Modifier.size(20.dp))
        }
    }
}

@Composable
private fun Section(title: String) {
    HorizontalDivider(Modifier.padding(top = 16.dp))
    Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 12.dp, bottom = 6.dp))
}
