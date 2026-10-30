//! MCP servers a chat connects (ACP session/new's mcpServers): stdio servers run in the chat's Linux (through proot,
//! only while the session uses them), HTTP servers reached over streamable HTTP. Ported from @deepseek-ai/dsh-mcp-client
//! and dsh-mcp-resources: a server's tools appear as mcp__<server>__<tool>, its resources through three tools, its
//! instructions in the system prompt, results worded as the harness words them.

use crate::linux::{Cancel, Shell};
use crate::msg::{Block, ToolSpec};
use crate::tools::Output;
use crate::util::{hex, js_stringify};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

const PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_PUBLIC_NAME_LENGTH: usize = 64;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// mcp__<server>__<tool>, made a valid tool name (a hash keeps a long or rewritten one unique).
pub fn public_tool_name(server: &str, raw: &str) -> String {
    let joined = format!("mcp__{server}__{raw}");
    let normalized: String =
        joined.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    if normalized == joined && normalized.len() <= MAX_PUBLIC_NAME_LENGTH {
        return normalized;
    }
    let hash = &hex(&Sha256::digest(format!("{server}\0{raw}").as_bytes()))[..12];
    let keep: String = normalized.chars().take(MAX_PUBLIC_NAME_LENGTH - 12 - 1).collect();
    format!("{keep}_{hash}")
}

enum Conn {
    Stdio {
        child: Child,
        stdin: ChildStdin,
        lines: Receiver<String>,
        next: u64,
    },
    #[cfg(feature = "mcp-http")]
    Http {
        url: String,
        headers: Vec<(String, String)>,
        session: Option<String>,
        next: u64,
    },
}

impl Drop for Conn {
    fn drop(&mut self) {
        if let Conn::Stdio { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct Tool {
    public: String,
    raw: String,
    spec: ToolSpec,
}

struct Server {
    name: String,
    conn: Mutex<Conn>,
    tools: Vec<Tool>,
    instructions: String,
}

#[derive(Default)]
pub struct McpHub {
    servers: Vec<Server>,
}

fn rpc_error(v: &Value) -> String {
    v.get("message").and_then(Value::as_str).unwrap_or("MCP request failed").to_string()
}

impl Conn {
    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        match self {
            Conn::Stdio { stdin, .. } => {
                writeln!(stdin, "{msg}").and_then(|_| stdin.flush()).map_err(|e| format!("the MCP server is gone: {e}"))
            }
            #[cfg(feature = "mcp-http")]
            Conn::Http { .. } => self.http(&msg).map(|_| ()),
        }
    }

    fn request(&mut self, method: &str, params: Value, cancel: &Cancel) -> Result<Value, String> {
        match self {
            Conn::Stdio { stdin, lines, next, .. } => {
                *next += 1;
                let id = *next;
                let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
                writeln!(stdin, "{msg}")
                    .and_then(|_| stdin.flush())
                    .map_err(|e| format!("the MCP server is gone: {e}"))?;
                let started = Instant::now();
                loop {
                    if cancel.is_cancelled() {
                        let _ = writeln!(
                            stdin,
                            "{}",
                            json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": id}})
                        );
                        return Err("MCP request cancelled".into());
                    }
                    if started.elapsed() > REQUEST_TIMEOUT {
                        return Err(format!("MCP request {method} timed out"));
                    }
                    let line = match lines.recv_timeout(Duration::from_millis(50)) {
                        Ok(l) => l,
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => {
                            return Err("the MCP server closed its output".into());
                        }
                    };
                    let Ok(v) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    if v.get("id").and_then(Value::as_u64) == Some(id) && v.get("method").is_none() {
                        if let Some(e) = v.get("error") {
                            return Err(rpc_error(e));
                        }
                        return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                    }
                    // a request from the server: ping answered, anything else refused
                    if let (Some(rid), Some(m)) = (v.get("id"), v.get("method").and_then(Value::as_str)) {
                        let reply = if m == "ping" {
                            json!({"jsonrpc": "2.0", "id": rid, "result": {}})
                        } else {
                            json!({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": format!("no method {m}")}})
                        };
                        let _ = writeln!(stdin, "{reply}").and_then(|_| stdin.flush());
                    }
                }
            }
            #[cfg(feature = "mcp-http")]
            Conn::Http { next, .. } => {
                *next += 1;
                let id = *next;
                let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
                let _ = cancel;
                let replies = self.http(&msg)?;
                for v in replies {
                    if v.get("id").and_then(Value::as_u64) == Some(id) {
                        if let Some(e) = v.get("error") {
                            return Err(rpc_error(e));
                        }
                        return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
                Err(format!("the MCP server sent no answer to {method}"))
            }
        }
    }

    #[cfg(feature = "mcp-http")]
    fn http(&mut self, msg: &Value) -> Result<Vec<Value>, String> {
        let Conn::Http { url, headers, session, .. } = self else { unreachable!() };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .http_status_as_error(false)
            .build()
            .into();
        let mut req = agent
            .post(url.as_str())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", PROTOCOL_VERSION);
        for (k, v) in headers.iter() {
            req = req.header(k.as_str(), v.as_str());
        }
        if let Some(s) = session.as_ref() {
            req = req.header("Mcp-Session-Id", s.as_str());
        }
        let mut resp =
            req.send(js_stringify(msg).as_bytes()).map_err(|e| format!("cannot reach the MCP server: {e}"))?;
        if let Some(s) = resp.headers().get("mcp-session-id").and_then(|h| h.to_str().ok()) {
            *session = Some(s.to_string());
        }
        let status = resp.status().as_u16();
        let ctype = resp.headers().get("content-type").and_then(|h| h.to_str().ok()).unwrap_or("").to_ascii_lowercase();
        let body = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
        if status == 202 || body.trim().is_empty() {
            return Ok(Vec::new());
        }
        if !(200..300).contains(&status) {
            return Err(format!("the MCP server answered HTTP {status}: {}", body.trim()));
        }
        let mut out = Vec::new();
        if ctype.starts_with("text/event-stream") {
            for line in body.lines() {
                if let Some(d) = line.strip_prefix("data:")
                    && let Ok(v) = serde_json::from_str::<Value>(d.trim())
                {
                    out.push(v);
                }
            }
        } else if let Ok(v) = serde_json::from_str::<Value>(&body) {
            match v {
                Value::Array(a) => out.extend(a),
                v => out.push(v),
            }
        }
        Ok(out)
    }
}

impl McpHub {
    /// The servers of an ACP mcpServers list, connected (one that fails is left out, its error returned).
    pub fn connect(configs: &Value, shell: &dyn Shell, cwd: &str) -> (McpHub, Vec<String>) {
        let mut hub = McpHub::default();
        let mut errors = Vec::new();
        for c in configs.as_array().cloned().unwrap_or_default() {
            let name = c.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            match connect_one(&c, shell, cwd) {
                Ok(s) => hub.servers.push(s),
                Err(e) => errors.push(format!("MCP server {name}: {e}")),
            }
        }
        (hub, errors)
    }

    pub fn has_servers(&self) -> bool {
        !self.servers.is_empty()
    }

    pub fn server_names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.servers.iter().map(|s| s.name.clone()).collect();
        n.sort();
        n
    }

    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.servers.iter().flat_map(|s| s.tools.iter().map(|t| t.spec.clone())).collect()
    }

    /// The servers' tools by server (for /mcp).
    pub fn summary(&self) -> Vec<(String, Vec<String>)> {
        self.servers.iter().map(|s| (s.name.clone(), s.tools.iter().map(|t| t.raw.clone()).collect())).collect()
    }

    /// Servers' own instructions for the system prompt.
    pub fn instructions(&self) -> Vec<String> {
        self.servers
            .iter()
            .filter(|s| !s.instructions.is_empty())
            .map(|s| format!("### MCP server: {}\n\n{}", s.name, s.instructions))
            .collect()
    }

    pub fn owns(&self, public: &str) -> bool {
        self.servers.iter().any(|s| s.tools.iter().any(|t| t.public == public))
    }

    pub fn call_tool(&self, public: &str, args: &Value, cancel: &Cancel) -> Result<Output, String> {
        let (server, tool) = self
            .servers
            .iter()
            .find_map(|s| s.tools.iter().find(|t| t.public == public).map(|t| (s, t)))
            .ok_or_else(|| format!("unknown tool \"{public}\""))?;
        let arguments = if args.is_object() { args.clone() } else { json!({}) };
        let result = server.conn.lock().unwrap().request(
            "tools/call",
            json!({"name": tool.raw, "arguments": arguments}),
            cancel,
        )?;
        let content = result.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
        let (text, images) = project(&content, &tool.raw);
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            return Err(text);
        }
        let mut blocks = vec![Block::text(text)];
        blocks.extend(images);
        Ok(Output { blocks, is_error: false, notices: Vec::new() })
    }

    pub fn resource_tool(&self, tool: &str, args: &Value, cancel: &Cancel) -> Result<Output, String> {
        let name = args.get("server").and_then(Value::as_str).ok_or("server must be a string")?;
        let server =
            self.servers.iter().find(|s| s.name == name).ok_or_else(|| format!("unknown MCP server \"{name}\""))?;
        let (method, params) = match tool {
            "list_mcp_resources" | "list_mcp_resource_templates" => {
                let method = if tool == "list_mcp_resources" { "resources/list" } else { "resources/templates/list" };
                let params = match args.get("cursor").and_then(Value::as_str) {
                    Some(c) => json!({"cursor": c}),
                    None => json!({}),
                };
                (method, params)
            }
            _ => {
                let uri = args.get("uri").and_then(Value::as_str).ok_or("uri must be a string")?;
                ("resources/read", json!({"uri": uri}))
            }
        };
        let mut result = server.conn.lock().unwrap().request(method, params, cancel)?;
        hide_blobs(&mut result);
        Ok(Output::text(format!("MCP server: {name}\n{}", js_stringify(&result))))
    }
}

/// A resource's binary data stays out of the model's context.
fn hide_blobs(v: &mut Value) {
    match v {
        Value::Object(m) => {
            for (k, item) in m.iter_mut() {
                if k == "blob" {
                    if let Value::String(s) = item {
                        *item = json!(format!(
                            "[binary resource: {} base64 characters; available to programmatic callers]",
                            s.chars().count()
                        ));
                    }
                } else {
                    hide_blobs(item);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(hide_blobs),
        _ => {}
    }
}

/// An MCP result's content as the model reads it: its text blocks (one line each), images as images.
fn project(content: &[Value], raw: &str) -> (String, Vec<Block>) {
    let mut text: Vec<String> = Vec::new();
    let mut images = Vec::new();
    for b in content {
        let Some(o) = b.as_object() else {
            text.push("[unsupported MCP content block: expected an object]".into());
            continue;
        };
        match o.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                if let Some(t) = o.get("text").and_then(Value::as_str) {
                    text.push(t.to_string());
                }
            }
            "image" => {
                let mime = o.get("mimeType").and_then(Value::as_str).unwrap_or("");
                let data = o.get("data").and_then(Value::as_str).unwrap_or("");
                let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data).ok();
                match decoded.as_deref().and_then(|d| imagesize::blob_size(d).ok()) {
                    Some(size) if ["image/png", "image/jpeg", "image/webp", "image/gif"].contains(&mime) => images
                        .push(Block::Image {
                            mime: mime.into(),
                            data: data.into(),
                            width: size.width as u32,
                            height: size.height as u32,
                            name: String::new(),
                            path: String::new(),
                            id: format!("sha256:{}", hex(&Sha256::digest(decoded.unwrap_or_default()))),
                        }),
                    _ => text.push(format!("[image unavailable: {mime} image could not be read]")),
                }
            }
            "resource_link" => match (o.get("name").and_then(Value::as_str), o.get("uri").and_then(Value::as_str)) {
                (Some(n), Some(u)) => text.push(format!("Resource link: {n} ({u})")),
                _ => text.push("[resource link unavailable: the MCP block is missing its name or URI]".into()),
            },
            "audio" => text.push(format!(
                "[audio result unsupported: {}; raw audio data remains available to programmatic callers]",
                o.get("mimeType").and_then(Value::as_str).unwrap_or("unknown media type")
            )),
            "resource" => text.push(
                "[embedded resource unsupported; raw resource data remains available to programmatic callers]".into(),
            ),
            other => text.push(format!("[unsupported MCP content type: {other}]")),
        }
    }
    if text.is_empty() && images.is_empty() {
        return (format!("({raw} returned no model-visible content)"), images);
    }
    (text.join("\n"), images)
}

fn pairs(v: Option<&Value>, key_field: &str) -> Vec<(String, String)> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|e| Some((e.get(key_field)?.as_str()?.to_string(), e.get("value")?.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

fn connect_one(c: &Value, shell: &dyn Shell, cwd: &str) -> Result<Server, String> {
    let name = c
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .ok_or("an MCP server needs a name")?
        .to_string();
    let kind = c.get("type").and_then(Value::as_str).unwrap_or(if c.get("url").is_some() { "http" } else { "stdio" });
    let mut conn = match kind {
        "http" | "sse" => {
            #[cfg(feature = "mcp-http")]
            {
                let url = c.get("url").and_then(Value::as_str).ok_or("an HTTP MCP server needs a url")?.to_string();
                Conn::Http { url, headers: pairs(c.get("headers"), "name"), session: None, next: 0 }
            }
            #[cfg(not(feature = "mcp-http"))]
            return Err("HTTP MCP servers are not supported in this build".into());
        }
        _ => {
            let command = c
                .get("command")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or("a stdio MCP server needs a command")?;
            let mut argv = vec![command.to_string()];
            argv.extend(
                c.get("args")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect::<Vec<_>>())
                    .unwrap_or_default(),
            );
            let env = pairs(c.get("env"), "name");
            // the Linux counted in for as long as the server runs: its output ending (it exited, crashed or was
            // stopped) counts it out
            let hold = shell.hold();
            let mut child = shell.spawn(&argv, &env, cwd)?;
            let stdin = child.stdin.take().ok_or("no stdin")?;
            let stdout = child.stdout.take().ok_or("no stdout")?;
            let (tx, rx) = channel();
            std::thread::spawn(move || {
                let _hold = hold;
                for line in BufReader::new(stdout).lines() {
                    let Ok(l) = line else { break };
                    if tx.send(l).is_err() {
                        break;
                    }
                }
            });
            Conn::Stdio { child, stdin, lines: rx, next: 0 }
        }
    };
    let cancel = Cancel::new();
    let init = conn.request(
        "initialize",
        json!({"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "geode-agent", "version": env!("CARGO_PKG_VERSION")}}),
        &cancel,
    )?;
    conn.notify("notifications/initialized", json!({}))?;
    let instructions = init.get("instructions").and_then(Value::as_str).unwrap_or("").trim_end().to_string();
    let mut tools = Vec::new();
    if init.pointer("/capabilities/tools").is_some() {
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let r = conn.request("tools/list", params, &cancel)?;
            for t in r.get("tools").and_then(Value::as_array).cloned().unwrap_or_default() {
                let Some(raw) = t.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let public = public_tool_name(&name, raw);
                let spec = ToolSpec {
                    name: public.clone(),
                    description: t.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
                    parameters: t
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                };
                tools.push(Tool { public, raw: raw.to_string(), spec });
            }
            cursor = r.get("nextCursor").and_then(Value::as_str).map(String::from);
            if cursor.is_none() {
                break;
            }
        }
    }
    Ok(Server { name, conn: Mutex::new(conn), tools, instructions })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(public_tool_name("demo", "echo"), "mcp__demo__echo");
        assert_eq!(
            public_tool_name("my server", "do.it"),
            format!("mcp__my_server__do_it_{}", &hex(&Sha256::digest("my server\0do.it".as_bytes()))[..12])
        );
        assert!(public_tool_name("s", &"x".repeat(80)).len() <= 64);
    }

    #[test]
    fn results() {
        let (t, _) = project(&[json!({"type": "text", "text": "HI FROM MCP"})], "echo");
        assert_eq!(t, "HI FROM MCP");
        let (t, _) = project(&[], "echo");
        assert_eq!(t, "(echo returned no model-visible content)");
        let mut v = json!({"contents": [{"uri": "demo://note", "mimeType": "text/plain", "text": "A note from the demo server."}]});
        hide_blobs(&mut v);
        assert_eq!(
            format!("MCP server: demo\n{}", js_stringify(&v)),
            "MCP server: demo\n{\"contents\":[{\"uri\":\"demo://note\",\"mimeType\":\"text/plain\",\"text\":\"A note from the demo server.\"}]}"
        );
    }
}
