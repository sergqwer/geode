//! geode/linux through the ACP server, against a fake client: the agent asks before the first Linux process (an MCP
//! server's, a command's) and waits for the answer (or its timeout), tells the client after the last one ends; the
//! model never hears of it. Tool calls run one after another in a turn, so a command adds to an MCP server's count
//! and never overlaps another command.

use geode_agent::acp::{Out, Peer, Server};
use geode_agent::config::Config;
use geode_agent::linux::{Cancel, Captured, LinuxGate, LinuxHold, Outcome, Shell};
use geode_agent::model::{Accumulator, Answer, Delta, Model};
use geode_agent::paths::Namespace;
use geode_agent::tools::web::{FetchResult, SearchResult, Web};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A Linux that counts itself in as proot does: a command holds the gate while it runs, an MCP server while it lives
/// (the demo server, run by the PC's Python).
struct GatedShell {
    gate: Arc<LinuxGate>,
    ran: Mutex<Vec<usize>>,
}

impl Shell for GatedShell {
    fn run(&self, command: &str, _cwd: &str, _t: u64, _c: &Cancel) -> Result<Outcome, String> {
        let _hold = self.gate.hold();
        self.ran.lock().unwrap().push(self.gate.running());
        std::thread::sleep(Duration::from_millis(30));
        Ok(Outcome {
            stdout: Captured { text: format!("ran {command}\n"), ..Default::default() },
            exit_code: Some(0),
            ..Default::default()
        })
    }

    fn spawn(&self, _argv: &[String], _env: &[(String, String)], _cwd: &str) -> Result<std::process::Child, String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let py = root.join(".venv/Scripts/python.exe");
        let py = if py.exists() { py } else { PathBuf::from("python3") };
        std::process::Command::new(py)
            .arg("-I")
            .arg(root.join("tools/linux/trace/mcp_demo.py"))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
    }

    fn hold(&self) -> Option<LinuxHold> {
        Some(self.gate.hold())
    }
}

/// One bash call, then "Done.".
struct OneCommand {
    seen: Arc<Mutex<Vec<Value>>>,
}

impl Model for OneCommand {
    fn stream(&self, body: &Value, _: &Cancel, d: &mut dyn FnMut(Delta)) -> Result<Answer, String> {
        self.seen.lock().unwrap().push(body.clone());
        let called = body["messages"].as_array().unwrap().iter().any(|m| m.get("tool_calls").is_some());
        let mut acc = Accumulator::default();
        let delta = if called {
            json!({"content": "Done."})
        } else {
            json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"description\":\"d\",\"command\":\"true\"}"}}]})
        };
        acc.push(&json!({"choices": [{"index": 0, "delta": delta, "finish_reason": if called { "stop" } else { "tool_calls" }}]}), d);
        Ok(acc.finish())
    }
}

struct NoWeb;
impl Web for NoWeb {
    fn search(&self, _: &str, _: usize, _: &Cancel) -> Result<SearchResult, String> {
        Err("no".into())
    }
    fn fetch(&self, _: &str, _: &Cancel) -> Result<FetchResult, String> {
        Err("no".into())
    }
}

#[derive(Clone)]
struct Capture(Arc<Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Capture {
    fn lines(&self) -> Vec<Value> {
        let t = String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned();
        let whole = &t[..t.rfind('\n').map(|i| i + 1).unwrap_or(0)];
        whole.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }
}

/// The client's side: answers geode/linux requests after [delay] (None: never), records what it saw.
fn client(
    server: Arc<Server>,
    out: Capture,
    delay: Option<Duration>,
    stop: Arc<Mutex<bool>>,
) -> std::thread::JoinHandle<Vec<String>> {
    std::thread::spawn(move || {
        let mut seen = Vec::new();
        let mut done = 0;
        while !*stop.lock().unwrap() {
            let lines = out.lines();
            for l in &lines[done..] {
                if l["method"] == "geode/linux" {
                    let active = l["params"]["active"].as_bool().unwrap();
                    match l.get("id") {
                        Some(id) => {
                            seen.push(format!("request active={active}"));
                            if let Some(d) = delay {
                                std::thread::sleep(d);
                                server.dispatch(json!({"jsonrpc": "2.0", "id": id, "result": {}}));
                            }
                        }
                        None => seen.push(format!("notification active={active}")),
                    }
                }
            }
            done = lines.len();
            std::thread::sleep(Duration::from_millis(5));
        }
        seen
    })
}

fn call(s: &Arc<Server>, out: &Capture, id: u64, method: &str, params: Value) -> Value {
    s.dispatch(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
    let t = Instant::now();
    loop {
        if let Some(r) = out.lines().into_iter().find(|l| l["id"] == id && l.get("method").is_none()) {
            return r;
        }
        assert!(t.elapsed() < Duration::from_secs(15), "no response to {method}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The server, what it writes, its Linux, the requests its model got.
type Setup = (Arc<Server>, Capture, Arc<GatedShell>, Arc<Mutex<Vec<Value>>>);

fn setup(timeout: Duration) -> Setup {
    let chat = std::env::temp_dir().join(format!("geode-agent-gate-{}", geode_agent::util::uuid()));
    std::fs::create_dir_all(chat.join("workspace")).unwrap();
    let cfg = Config::for_chat(chat.clone());
    let ns = Namespace::new(&chat, None, None);
    let gate = LinuxGate::new(timeout);
    let shell = Arc::new(GatedShell { gate: gate.clone(), ran: Mutex::new(Vec::new()) });
    struct Shared(Arc<GatedShell>);
    impl Shell for Shared {
        fn run(&self, c: &str, w: &str, t: u64, x: &Cancel) -> Result<Outcome, String> {
            self.0.run(c, w, t, x)
        }
        fn spawn(&self, a: &[String], e: &[(String, String)], w: &str) -> Result<std::process::Child, String> {
            self.0.spawn(a, e, w)
        }
        fn hold(&self) -> Option<LinuxHold> {
            self.0.hold()
        }
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let out = Capture(Arc::new(Mutex::new(Vec::new())));
    let s = Server::new(
        cfg,
        ns,
        Box::new(Shared(shell.clone())),
        Box::new(NoWeb),
        Box::new(OneCommand { seen: seen.clone() }),
        Arc::new(Out::new(Box::new(out.clone()))),
    );
    gate.attach(s.peer());
    (s, out, shell, seen)
}

#[test]
fn reserve_follows_the_linux() {
    let (s, out, shell, seen) = setup(Duration::from_secs(5));
    let stop = Arc::new(Mutex::new(false));
    let c = client(s.clone(), out.clone(), Some(Duration::from_millis(200)), stop.clone());
    // the MCP server starts the Linux: the session is ready only after the client made room
    let t = Instant::now();
    let r = call(
        &s,
        &out,
        1,
        "session/new",
        json!({"cwd": "/workspace", "mcpServers": [{"name": "demo", "command": "python3", "args": ["mcp_demo.py"], "env": []}]}),
    );
    assert!(t.elapsed() >= Duration::from_millis(200), "waited for the answer");
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    // a command while the server lives: counted in with it, no second request
    let r = call(&s, &out, 2, "session/prompt", json!({"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]}));
    assert_eq!(r["result"]["stopReason"], "end_turn");
    assert_eq!(*shell.ran.lock().unwrap(), [2]);
    // the server stops with its session: the last Linux process ended
    call(&s, &out, 3, "session/close", json!({"sessionId": sid}));
    let t = Instant::now();
    while shell.gate.running() > 0 {
        assert!(t.elapsed() < Duration::from_secs(5), "the MCP server's exit counted out");
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(50));
    *stop.lock().unwrap() = true;
    assert_eq!(c.join().unwrap(), ["request active=true", "notification active=false"]);
    // and none of it reached the model
    for body in seen.lock().unwrap().iter() {
        assert!(!body.to_string().contains("geode/linux"));
    }
}

#[test]
fn a_command_alone_and_a_silent_client() {
    // a client that never answers: the Linux starts after the timeout anyway
    let (s, out, shell, _) = setup(Duration::from_millis(300));
    let stop = Arc::new(Mutex::new(false));
    let c = client(s.clone(), out.clone(), None, stop.clone());
    let sid = call(&s, &out, 1, "session/new", json!({"cwd": "/workspace", "mcpServers": []}))["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let t = Instant::now();
    let r = call(&s, &out, 2, "session/prompt", json!({"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]}));
    assert_eq!(r["result"]["stopReason"], "end_turn");
    assert!(t.elapsed() >= Duration::from_millis(300));
    assert_eq!(*shell.ran.lock().unwrap(), [1]);
    std::thread::sleep(Duration::from_millis(50));
    *stop.lock().unwrap() = true;
    assert_eq!(c.join().unwrap(), ["request active=true", "notification active=false"]);
}

#[test]
fn peer_answers_errors_and_timeouts() {
    let out = Capture(Arc::new(Mutex::new(Vec::new())));
    let peer = Arc::new(Peer::new(Arc::new(Out::new(Box::new(out.clone())))));
    let answer = |reply: Value| {
        let (p, o) = (peer.clone(), out.clone());
        std::thread::spawn(move || {
            let t = Instant::now();
            loop {
                if let Some(req) = o.lines().into_iter().rev().find(|l| l["method"] == "x/y") {
                    let mut r = reply.clone();
                    r["id"] = req["id"].clone();
                    assert!(p.deliver(&r));
                    return;
                }
                assert!(t.elapsed() < Duration::from_secs(5));
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };
    let h = answer(json!({"jsonrpc": "2.0", "result": {"ok": true}}));
    assert_eq!(peer.request("x/y", json!({}), Duration::from_secs(5)), Some(Ok(json!({"ok": true}))));
    h.join().unwrap();
    out.0.lock().unwrap().clear();
    let h = answer(json!({"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found: x/y"}}));
    assert_eq!(peer.request("x/y", json!({}), Duration::from_secs(5)), Some(Err("Method not found: x/y".into())));
    h.join().unwrap();
    let t = Instant::now();
    assert_eq!(peer.request("x/y", json!({}), Duration::from_millis(120)), None);
    assert!(t.elapsed() >= Duration::from_millis(120));
    // a late answer finds nobody waiting
    assert!(!peer.deliver(&json!({"jsonrpc": "2.0", "id": 1_000_003, "result": {}})));
    // the agent's ids stay apart from the client's
    let ids: Vec<u64> = out.lines().iter().filter_map(|l| l["id"].as_u64()).collect();
    assert!(ids.iter().all(|&i| i > 1_000_000));
}
