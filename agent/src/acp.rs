//! The Agent Client Protocol over stdio, as Geode's Agent.kt speaks it with DeepSeek Harness (dsh-acp): one JSON-RPC
//! message a line. initialize, session/new, session/resume, session/list, session/close, session/prompt (a turn,
//! run on its own thread so session/cancel reaches it), session/cancel; session/update notifications out.
//!
//! Geode's own extension: `geode/linux`. Before the first Linux (proot) process starts the agent asks the app
//! (a request, {"active": true}: the app answers once the model gave back memory for the Linux); after the last one
//! ends it tells it (a notification, {"active": false}). The model never sees any of it.

use crate::agent::{self, Runtime, Sink};
use crate::commands;
use crate::config::Config;
use crate::linux::{Cancel, LinuxClient, Shell};
use crate::mcp::McpHub;
use crate::model::Model;
use crate::paths::Namespace;
use crate::session::Session;
use crate::skills::SkillSet;
use crate::tools::web::Web;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Where messages go out (stdout): whole lines, one writer at a time.
pub struct Out {
    w: Mutex<Box<dyn Write + Send>>,
}

impl Out {
    pub fn new(w: Box<dyn Write + Send>) -> Out {
        Out { w: Mutex::new(w) }
    }

    pub fn send(&self, msg: &Value) {
        let line = format!("{msg}\n");
        let mut w = self.w.lock().unwrap();
        let _ = w.write_all(line.as_bytes());
        let _ = w.flush();
    }
}

impl Sink for Out {
    fn update(&self, session_id: &str, update: Value) {
        self.send(&json!({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": update}}));
    }
}

/// The client as the agent's own peer: requests the agent sends it (their answers routed back by id), notifications.
pub struct Peer {
    out: Arc<Out>,
    pending: Mutex<HashMap<u64, Sender<Value>>>,
    next: AtomicU64,
}

/// The agent's request ids: apart from the client's (a client keeping one table of ids sees no clash).
const PEER_IDS: u64 = 1_000_000;

impl Peer {
    pub fn new(out: Arc<Out>) -> Peer {
        Peer { out, pending: Mutex::new(HashMap::new()), next: AtomicU64::new(PEER_IDS) }
    }

    /// A request and its answer (its result, or its error's message), or None after [timeout].
    pub fn request(&self, method: &str, params: Value, timeout: Duration) -> Option<Result<Value, String>> {
        let id = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = channel();
        self.pending.lock().unwrap().insert(id, tx);
        self.out.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let answer = rx.recv_timeout(timeout).ok();
        self.pending.lock().unwrap().remove(&id);
        answer.map(|m| match m.get("error") {
            Some(e) => Err(e.get("message").and_then(Value::as_str).unwrap_or("error").to_string()),
            None => Ok(m.get("result").cloned().unwrap_or(Value::Null)),
        })
    }

    pub fn notify(&self, method: &str, params: Value) {
        self.out.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    /// A response from the client: to the request waiting for it (false: none was).
    pub fn deliver(&self, msg: &Value) -> bool {
        let Some(id) = msg.get("id").and_then(Value::as_u64) else { return false };
        match self.pending.lock().unwrap().remove(&id) {
            Some(tx) => tx.send(msg.clone()).is_ok(),
            None => false,
        }
    }
}

impl LinuxClient for Peer {
    fn starting(&self, timeout: Duration) {
        // an answer (a result, or an error: a client without the extension) or the timeout: the Linux starts
        let _ = self.request("geode/linux", json!({"active": true}), timeout);
    }

    fn stopped(&self) {
        self.notify("geode/linux", json!({"active": false}));
    }
}

struct Live {
    sess: Mutex<Session>,
    mcp: McpHub,
    cancel: Mutex<Cancel>,
}

pub struct Server {
    pub cfg: Config,
    pub ns: Namespace,
    pub shell: Box<dyn Shell>,
    pub web: Box<dyn Web>,
    pub model: Box<dyn Model>,
    pub out: Arc<Out>,
    pub tool_runner: Option<Box<dyn agent::ToolRunner>>,
    peer: Arc<Peer>,
    sessions: Mutex<HashMap<String, Arc<Live>>>,
}

fn error(id: &Value, code: i64, message: impl Into<String>) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message.into()}})
}

fn result(id: &Value, r: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": r})
}

impl Server {
    pub fn new(
        cfg: Config,
        ns: Namespace,
        shell: Box<dyn Shell>,
        web: Box<dyn Web>,
        model: Box<dyn Model>,
        out: Arc<Out>,
    ) -> Arc<Server> {
        let peer = Arc::new(Peer::new(out.clone()));
        Arc::new(Server {
            cfg,
            ns,
            shell,
            web,
            model,
            out,
            tool_runner: None,
            peer,
            sessions: Mutex::new(HashMap::new()),
        })
    }

    /// The client as a peer (the Linux gate tells it when the Linux runs).
    pub fn peer(&self) -> Arc<Peer> {
        self.peer.clone()
    }

    /// The same server with recorded tool results (parity tests).
    pub fn with_tool_runner(
        cfg: Config,
        ns: Namespace,
        shell: Box<dyn Shell>,
        web: Box<dyn Web>,
        model: Box<dyn Model>,
        out: Arc<Out>,
        runner: Box<dyn agent::ToolRunner>,
    ) -> Arc<Server> {
        let peer = Arc::new(Peer::new(out.clone()));
        Arc::new(Server {
            cfg,
            ns,
            shell,
            web,
            model,
            out,
            tool_runner: Some(runner),
            peer,
            sessions: Mutex::new(HashMap::new()),
        })
    }

    fn runtime(&self) -> Runtime<'_> {
        Runtime {
            cfg: &self.cfg,
            ns: &self.ns,
            shell: self.shell.as_ref(),
            web: self.web.as_ref(),
            model: self.model.as_ref(),
            sink: self.out.as_ref(),
            tool_runner: self.tool_runner.as_deref(),
        }
    }

    fn config_options(&self) -> Value {
        let current = json!(["geode", self.cfg.model]).to_string();
        json!([{"id": "model", "name": "Model", "category": "model", "type": "select", "currentValue": current,
            "options": [{"group": "geode", "name": "geode", "options": [{"value": current, "name": self.cfg.model}]}]}])
    }

    fn advertise(&self, sess: &Session) {
        let skills = SkillSet::discover(&self.ns, &sess.cwd);
        self.out.update(
            &sess.id,
            json!({"sessionUpdate": "available_commands_update", "availableCommands": commands::available(&skills)}),
        );
    }

    /// Reads requests until stdin closes.
    pub fn serve(self: &Arc<Self>, input: impl BufRead) {
        for line in input.lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                self.out.send(&error(&Value::Null, -32700, "Parse error"));
                continue;
            };
            self.dispatch(msg);
        }
        // the app is gone: every session's MCP servers stop with their hubs
        self.sessions.lock().unwrap().clear();
    }

    pub fn dispatch(self: &Arc<Self>, msg: Value) {
        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            // a response: to a request of the agent's own (geode/linux)
            self.peer.deliver(&msg);
            return;
        };
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(json!({}));
        let reply = |v: Value| self.out.send(&v);
        match (method, id) {
            ("session/cancel", _) => {
                if let Some(sid) = params.get("sessionId").and_then(Value::as_str)
                    && let Some(l) = self.sessions.lock().unwrap().get(sid)
                {
                    l.cancel.lock().unwrap().cancel();
                }
            }
            (_, None) => {}
            ("initialize", Some(id)) => reply(result(
                &id,
                json!({"protocolVersion": 1, "agentInfo": {"name": "geode-agent", "version": env!("CARGO_PKG_VERSION")},
                    "agentCapabilities": {"mcpCapabilities": {"http": cfg!(feature = "mcp-http")},
                        "promptCapabilities": {"image": self.cfg.image_input, "audio": false, "embeddedContext": false},
                        "sessionCapabilities": {"close": {}, "list": {}, "resume": {}}},
                    "authMethods": []}),
            )),
            // a session opens on its own thread: its MCP servers may start the Linux, which waits for the client's
            // answer - read by this thread
            ("session/new", Some(id)) => {
                let me = self.clone();
                std::thread::spawn(move || {
                    let cwd = params.get("cwd").and_then(Value::as_str).unwrap_or("/workspace").to_string();
                    match Session::create(&me.cfg.sessions_dir(), &cwd) {
                        Err(e) => me.out.send(&error(&id, -32603, e)),
                        Ok(sess) => {
                            let (mcp, _errors) = McpHub::connect(
                                params.get("mcpServers").unwrap_or(&json!([])),
                                me.shell.as_ref(),
                                &cwd,
                            );
                            let sid = sess.id.clone();
                            let live =
                                Arc::new(Live { sess: Mutex::new(sess), mcp, cancel: Mutex::new(Cancel::new()) });
                            me.sessions.lock().unwrap().insert(sid.clone(), live.clone());
                            me.out.send(&result(&id, json!({"sessionId": sid, "configOptions": me.config_options()})));
                            me.advertise(&live.sess.lock().unwrap());
                        }
                    }
                });
            }
            ("session/resume", Some(id)) => {
                let me = self.clone();
                std::thread::spawn(move || {
                    let sid = params.get("sessionId").and_then(Value::as_str).unwrap_or("").to_string();
                    match Session::load(&me.cfg.sessions_dir(), &sid) {
                        Err(e) => me.out.send(&error(&id, -32002, e)),
                        Ok(sess) => {
                            let cwd = sess.cwd.clone();
                            let (mcp, _errors) = McpHub::connect(
                                params.get("mcpServers").unwrap_or(&json!([])),
                                me.shell.as_ref(),
                                &cwd,
                            );
                            let live =
                                Arc::new(Live { sess: Mutex::new(sess), mcp, cancel: Mutex::new(Cancel::new()) });
                            me.sessions.lock().unwrap().insert(sid, live.clone());
                            me.out.send(&result(&id, json!({"configOptions": me.config_options()})));
                            me.advertise(&live.sess.lock().unwrap());
                        }
                    }
                });
            }
            ("session/list", Some(id)) => {
                reply(result(&id, json!({"sessions": Session::list(&self.cfg.sessions_dir())})))
            }
            ("session/close", Some(id)) => {
                if let Some(sid) = params.get("sessionId").and_then(Value::as_str)
                    && let Some(l) = self.sessions.lock().unwrap().remove(sid)
                {
                    l.cancel.lock().unwrap().cancel();
                }
                reply(result(&id, json!({})));
            }
            ("session/prompt", Some(id)) => {
                let sid = params.get("sessionId").and_then(Value::as_str).unwrap_or("").to_string();
                let Some(live) = self.sessions.lock().unwrap().get(&sid).cloned() else {
                    return reply(error(&id, -32002, format!("Resource not found: session {sid}")));
                };
                let me = self.clone();
                std::thread::spawn(move || {
                    let cancel = Cancel::new();
                    *live.cancel.lock().unwrap() = cancel.clone();
                    let mut sess = live.sess.lock().unwrap();
                    let blocks = agent::prompt_blocks(params.get("prompt").unwrap_or(&json!([])));
                    let rt = me.runtime();
                    let only_text = blocks.iter().all(|b| matches!(b, crate::msg::Block::Text { .. }));
                    let text = agent::prompt_text(&blocks);
                    if only_text && let Some(answer) = commands::handle(&rt, &mut sess, &live.mcp, &text, &cancel) {
                        me.out.update(&sid, json!({"sessionUpdate": "agent_message_chunk", "messageId": crate::util::uuid(), "content": {"type": "text", "text": answer}}));
                        me.advertise(&sess);
                        return me.out.send(&result(&id, json!({"stopReason": "end_turn"})));
                    }
                    let r = agent::run_turn(&rt, &mut sess, &live.mcp, blocks, &cancel);
                    me.out.send(&match r {
                        Ok(stop) => result(&id, json!({"stopReason": stop})),
                        Err(e) => error(&id, -32603, e),
                    });
                });
            }
            (m, Some(id)) => reply(error(&id, -32601, format!("Method not found: {m}"))),
        }
    }
}
