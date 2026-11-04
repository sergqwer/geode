package xyz.tifu.geode.ui.models

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import xyz.tifu.geode.R
import xyz.tifu.geode.models.Protocol
import xyz.tifu.geode.models.Provider
import xyz.tifu.geode.models.Providers

/**
 * The Models screen's services: the models of outside services the user has keys for, each used for the chats
 * instead of the phone's own at a tap, and the way to add one.
 */
fun LazyListScope.services(providers: List<Provider>, inUse: String, onUse: (Provider) -> Unit,
                           onEdit: (Provider) -> Unit, onAdd: () -> Unit) {
    item(key = "services-title") {
        Column {
            Text(stringResource(R.string.models_services), style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 4.dp))
            Text(stringResource(R.string.models_services_desc), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 2.dp))
        }
    }
    items(providers, key = { "s-" + it.id }) { p -> ServiceCard(p, p.id == inUse, { onUse(p) }, { onEdit(p) }) }
    item(key = "services-add") {
        OutlinedButton(onClick = onAdd) { Text(stringResource(R.string.models_service_add)) }
    }
}

@Composable
private fun ServiceCard(p: Provider, inUse: Boolean, onUse: () -> Unit, onEdit: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text(p.title, style = MaterialTheme.typography.titleMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(p.host + " · " + if (p.protocol == Protocol.ANTHROPIC) "Anthropic" else "OpenAI",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically) {
                if (inUse) Text(stringResource(R.string.models_in_use), color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.labelLarge)
                else Button(onClick = onUse) { Text(stringResource(R.string.models_use)) }
                TextButton(onClick = onEdit) { Text(stringResource(R.string.models_service_edit)) }
            }
        }
    }
}

/**
 * A service added ([initial] null) or changed: its name, the API's address and protocol (a ready start for the usual
 * ones), its key (kept encrypted; left empty when editing: the kept one stays), its model (the service's list fetched
 * to pick from), what the model takes. [onSave] gets the service, the key typed (null: unchanged) and whether to use it
 * now.
 */
@Composable
fun ServiceDialog(initial: Provider?, hasKey: Boolean, listModels: suspend (Provider, String?) -> Result<List<String>>,
                  onSave: (Provider, String?, Boolean) -> Unit, onDelete: (Provider) -> Unit, onDismiss: () -> Unit) {
    var name by remember { mutableStateOf(initial?.name ?: "") }
    var url by remember { mutableStateOf(initial?.baseUrl ?: "") }
    var protocol by remember { mutableStateOf(initial?.protocol ?: Protocol.OPENAI) }
    var key by remember { mutableStateOf("") }
    var model by remember { mutableStateOf(initial?.model ?: "") }
    var vision by remember { mutableStateOf(initial?.vision ?: false) }
    var context by remember { mutableStateOf((initial?.context ?: Provider.DEFAULT_CONTEXT).toString()) }
    var max by remember { mutableStateOf((initial?.maxTokens ?: Provider.DEFAULT_MAX_TOKENS).toString()) }
    var listed by remember { mutableStateOf<List<String>?>(null) }
    var listing by remember { mutableStateOf(false) }
    var listError by remember { mutableStateOf<String?>(null) }
    var menu by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun current() = Provider(
        id = initial?.id ?: "", name = name.trim(), baseUrl = url.trim(), protocol = protocol, model = model.trim(),
        vision = vision, context = context.toIntOrNull()?.coerceIn(1024, 10_000_000) ?: Provider.DEFAULT_CONTEXT,
        maxTokens = max.toIntOrNull()?.coerceIn(256, 1_000_000) ?: Provider.DEFAULT_MAX_TOKENS,
    )
    val keyOk = key.isNotBlank() || hasKey
    val ok = name.isNotBlank() && (url.startsWith("https://") || url.startsWith("http://")) && model.isNotBlank() && keyOk

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(if (initial == null) R.string.models_service_add else R.string.models_service_edit)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                // the usual services filled in at a tap
                if (initial == null) Row(horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.horizontalScroll(rememberScrollState())) {
                    for (pr in Providers.PRESETS) FilterChip(selected = url == pr.baseUrl, onClick = {
                        name = pr.name
                        url = pr.baseUrl
                        protocol = pr.protocol
                        model = pr.model
                        vision = pr.vision
                        context = pr.context.toString()
                        listed = null
                    }, label = { Text(pr.name) })
                }
                OutlinedTextField(name, { name = it }, label = { Text(stringResource(R.string.service_name)) },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(url, { url = it; listed = null }, label = { Text(stringResource(R.string.service_url)) },
                    singleLine = true, modifier = Modifier.fillMaxWidth(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri))
                Text(stringResource(R.string.service_protocol), style = MaterialTheme.typography.labelLarge)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(protocol == Protocol.OPENAI, { protocol = Protocol.OPENAI; listed = null },
                        label = { Text("OpenAI") })
                    FilterChip(protocol == Protocol.ANTHROPIC, { protocol = Protocol.ANTHROPIC; listed = null },
                        label = { Text("Anthropic") })
                }
                OutlinedTextField(key, { key = it }, label = { Text(stringResource(R.string.service_key)) },
                    singleLine = true, modifier = Modifier.fillMaxWidth(), visualTransformation = PasswordVisualTransformation(),
                    supportingText = if (hasKey) ({ Text(stringResource(R.string.service_key_kept)) }) else null,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
                Box {
                    OutlinedTextField(model, { model = it }, label = { Text(stringResource(R.string.service_model)) },
                        singleLine = true, modifier = Modifier.fillMaxWidth())
                    // the service's models, those with what is typed in their name first
                    // (a model already picked from the list shows them all: the user is choosing again)
                    val shown = listed?.let { all ->
                        val q = model.trim().lowercase()
                        val narrow = q.isNotEmpty() && all.none { it.lowercase() == q }
                        (if (narrow) all.filter { q in it.lowercase() }.ifEmpty { all } else all).take(MAX_LISTED)
                    }.orEmpty()
                    DropdownMenu(expanded = menu && shown.isNotEmpty(), onDismissRequest = { menu = false }) {
                        for (m in shown) DropdownMenuItem(text = { Text(m) }, onClick = {
                            model = m
                            menu = false
                        })
                    }
                }
                TextButton(enabled = !listing && url.isNotBlank() && keyOk, onClick = {
                    listing = true
                    listError = null
                    scope.launch {
                        listModels(current(), key.takeIf { it.isNotBlank() }).onSuccess {
                            listed = it
                            menu = true
                        }.onFailure { listError = it.message ?: it.javaClass.simpleName }
                        listing = false
                    }
                }) { Text(stringResource(R.string.service_list)) }
                listError?.let {
                    Text(stringResource(R.string.service_list_failed, it), color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.service_vision), Modifier.weight(1f))
                    Switch(vision, { vision = it })
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(context, { context = it.filter(Char::isDigit) }, singleLine = true,
                        label = { Text(stringResource(R.string.service_context)) }, modifier = Modifier.weight(1f),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number))
                    OutlinedTextField(max, { max = it.filter(Char::isDigit) }, singleLine = true,
                        label = { Text(stringResource(R.string.service_max)) }, modifier = Modifier.weight(1f),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number))
                }
                // where the chats go, said before it is saved
                Text(stringResource(R.string.service_leaves, current().host.ifEmpty { "…" }),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (initial != null) {
                    Spacer(Modifier.height(4.dp))
                    TextButton(onClick = { onDelete(initial) }) {
                        Text(stringResource(R.string.models_delete), color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = ok, onClick = {
                val p = current().let { if (it.id.isEmpty()) it.copy(id = Providers.newId()) else it }
                onSave(p, key.takeIf { it.isNotBlank() }, initial == null)
            }) { Text(stringResource(R.string.agent_save)) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.models_cancel)) } },
    )
}

private const val MAX_LISTED = 60
