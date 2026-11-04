//! Parity with DeepSeek Harness: the runs recorded on the phone (tools/linux/trace/capture*.sh: dsh as Geode ran it,
//! driven by script_model.py) replayed against this agent. The same scripted model answers, the tools give the
//! results dsh's tools gave; every request the agent sends the model must be the one dsh sent, byte for byte, and
//! the ACP stream must tell the client the same things.

use geode_agent::acp::{Out, Server};
use geode_agent::agent::ToolRunner;
use geode_agent::config::Config;
use geode_agent::linux::{Cancel, Outcome, Shell};
use geode_agent::model::{Accumulator, Answer, Delta, Model};
use geode_agent::msg::{Block, ToolCall};
use geode_agent::paths::Namespace;
use geode_agent::tools::Output;
use geode_agent::tools::web::{FetchResult, SearchResult, Web};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn ref_dir(run: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("ref").join(run)
}

/// The reference's request bodies, rebuilt (each line keeps a prefix of the previous messages, adds the rest).
fn reference_requests(run: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(ref_dir(run).join("requests.jsonl")).unwrap();
    let tools: Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir(run).join("tools.json")).unwrap()).unwrap();
    let mut prev: Vec<Value> = Vec::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let l: Value = serde_json::from_str(line).unwrap();
        let keep = l["keep"].as_u64().unwrap() as usize;
        let mut msgs: Vec<Value> = prev[..keep].to_vec();
        msgs.extend(l["new"].as_array().unwrap().iter().cloned());
        let mut body = serde_json::Map::new();
        body.insert("model".into(), l["params"]["model"].clone());
        body.insert("messages".into(), Value::Array(msgs.clone()));
        for (k, v) in l["params"].as_object().unwrap() {
            if k != "model" {
                body.insert(k.clone(), v.clone());
            }
        }
        let names: Vec<&str> = l["tools"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
        let all: Vec<Value> = tools
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| names.contains(&t["function"]["name"].as_str().unwrap()))
            .cloned()
            .collect();
        body.insert("tools".into(), Value::Array(all));
        out.push(Value::Object(body));
        prev = msgs;
    }
    out
}

fn reference_acp(run: &str) -> Vec<Value> {
    std::fs::read_to_string(ref_dir(run).join("acp.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// script_model.py's steps (--full, and --features as its tools resolved in the recorded run).
fn steps(run: &str) -> Vec<Vec<(&'static str, Value)>> {
    if run == "features" {
        return vec![
            vec![("skill", json!({"name": "hello"}))],
            vec![("mcp__demo__echo", json!({"text": "hi from mcp"}))],
            vec![("list_mcp_resource_templates", json!({"server": "demo"}))],
            vec![("read_mcp_resource", json!({"server": "demo", "uri": "demo://note"}))],
        ];
    }
    vec![
        vec![(
            "bash",
            json!({"description": "write a file", "command": "echo hi > a.txt && ls -la && python3 -c 'print(6*7)'"}),
        )],
        vec![("read", json!({"file_path": "/workspace/a.txt"}))],
        vec![("write", json!({"file_path": "/workspace/b.py", "content": "print('hi')\n"}))],
        vec![("edit", json!({"file_path": "/workspace/b.py", "old_string": "hi", "new_string": "hello"}))],
        vec![("glob", json!({"pattern": "*.py"}))],
        vec![("grep", json!({"pattern": "hello"}))],
        vec![("todo_write", json!({"todos": [{"content": "check", "status": "completed"}]}))],
        vec![("working_directory", json!({}))],
        vec![("web_fetch", json!({"url": "https://example.com/"}))],
        vec![(
            "bash",
            json!({"description": "an image", "command": "echo iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg== | base64 -d > img.png"}),
        )],
        vec![("read_image", json!({"file_path": "/workspace/img.png"}))],
        vec![("web_search", json!({"queries": ["geode phone llm"]}))],
        vec![("read", json!({"file_path": "/workspace/missing.txt"}))],
        vec![("edit", json!({"file_path": "/workspace/b.py", "old_string": "zzz", "new_string": "y"}))],
        vec![("write", json!({"file_path": "/workspace/new/dir/c.txt", "content": "line 1\nline 2\n"}))],
        vec![("bash", json!({"description": "fails", "command": "echo out; echo err >&2; exit 3"}))],
        vec![("bash", json!({"description": "long output", "command": "seq 1 40000"}))],
        vec![("read", json!({"file_path": "/workspace/new/dir/c.txt"})), ("glob", json!({"pattern": "**/*.txt"}))],
        vec![("write", json!({"file_path": "/workspace/a.txt", "content": "overwritten\n"}))],
        vec![("edit", json!({"file_path": "/workspace/missing.py", "old_string": "a", "new_string": "b"}))],
    ]
}

/// Python's json.dumps (", " and ": " between items): the arguments as the scripted model streamed them.
fn py_dumps(v: &Value) -> String {
    match v {
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, v)| format!("{}: {}", json!(k), py_dumps(v))).collect::<Vec<_>>().join(", ")
        ),
        Value::Array(a) => format!("[{}]", a.iter().map(py_dumps).collect::<Vec<_>>().join(", ")),
        Value::String(s) => {
            // json.dumps escapes non-ASCII; serde writes it as is: these steps are ASCII
            serde_json::to_string(s).unwrap()
        }
        other => other.to_string(),
    }
}

struct ScriptModel {
    run: &'static str,
    seen: Mutex<Vec<Value>>,
}

impl Model for ScriptModel {
    fn stream(&self, body: &Value, _cancel: &Cancel, on_delta: &mut dyn FnMut(Delta)) -> Result<Answer, String> {
        self.seen.lock().unwrap().push(body.clone());
        let msgs = body["messages"].as_array().unwrap();
        let step = msgs.iter().filter(|m| m["role"] == "assistant" && m.get("tool_calls").is_some()).count();
        let steps = steps(self.run);
        let mut chunks = vec![json!({"role": "assistant", "content": ""})];
        let mut finish = "stop";
        if body.get("tools").is_some() && step < steps.len() {
            let calls: Vec<Value> = steps[step]
                .iter()
                .enumerate()
                .map(|(i, (name, args))| json!({"index": i, "id": format!("call_{step}_{i}"), "type": "function", "function": {"name": name, "arguments": py_dumps(args)}}))
                .collect();
            chunks.push(json!({"reasoning_content": format!("Thinking about step {}.", step + 1)}));
            chunks.push(json!({"content": format!("Step {}.", step + 1)}));
            chunks.push(json!({"tool_calls": calls}));
            finish = "tool_calls";
        } else {
            chunks.push(json!({"content": "Done."}));
        }
        let mut acc = Accumulator::default();
        let n = chunks.len();
        for (i, d) in chunks.into_iter().enumerate() {
            let fin = if i == n - 1 { json!(finish) } else { Value::Null };
            acc.push(&json!({"choices": [{"index": 0, "delta": d, "finish_reason": fin}]}), on_delta);
        }
        acc.push(
            &json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
            on_delta,
        );
        Ok(acc.finish())
    }
}

/// The tools' results as dsh's tools gave them (the reference requests' tool messages; failed as ACP reported).
struct Replay {
    results: HashMap<String, (String, bool)>,
}

impl Replay {
    fn new(run: &str) -> Replay {
        let mut failed = HashMap::new();
        for r in reference_acp(run) {
            let u = &r["msg"]["params"]["update"];
            if u["sessionUpdate"] == "tool_call_update" {
                failed.insert(u["toolCallId"].as_str().unwrap().to_string(), u["status"] == "failed");
            }
        }
        let mut results = HashMap::new();
        for body in reference_requests(run) {
            for m in body["messages"].as_array().unwrap() {
                if m["role"] == "tool" {
                    let id = m["tool_call_id"].as_str().unwrap().to_string();
                    let f = failed.get(&id).copied().unwrap_or(false);
                    results.insert(id, (m["content"].as_str().unwrap().to_string(), f));
                }
            }
        }
        Replay { results }
    }
}

impl ToolRunner for Replay {
    fn run(&self, call: &ToolCall) -> Option<Output> {
        let (text, failed) = self.results.get(&call.id)?.clone();
        Some(Output { blocks: vec![Block::text(text)], is_error: failed, notices: Vec::new() })
    }
}

struct NoShell;
impl Shell for NoShell {
    fn run(&self, _: &str, _: &str, _: u64, _: &Cancel) -> Result<Outcome, String> {
        Err("no Linux in tests".into())
    }
    fn spawn(&self, argv: &[String], _env: &[(String, String)], _cwd: &str) -> Result<std::process::Child, String> {
        // the demo MCP server, run by the PC's Python
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/linux/trace/mcp_demo.py");
        if argv.iter().any(|a| a.ends_with("mcp_demo.py")) {
            let py = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/Scripts/python.exe");
            let py = if py.exists() { py } else { PathBuf::from("python3") };
            return std::process::Command::new(py)
                .arg("-I")
                .arg(script)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| e.to_string());
        }
        Err("no Linux in tests".into())
    }
}

struct NoWeb;
impl Web for NoWeb {
    fn search(&self, _: &str, _: usize, _: &Cancel) -> Result<SearchResult, String> {
        Err("no network".into())
    }
    fn fetch(&self, _: &str, _: &Cancel) -> Result<FetchResult, String> {
        Err("no network".into())
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
        let bytes = self.0.lock().unwrap().clone();
        let text = String::from_utf8_lossy(&bytes);
        // whole lines only: one may be in the middle of being written
        let whole = &text[..text.rfind('\n').map(|i| i + 1).unwrap_or(0)];
        whole.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }
}

struct Process {
    server: Arc<Server>,
    out: Capture,
}

/// The persona the run was recorded with (its system prompt's first section): the one thing the reference may say
/// otherwise than the agent does today, everything after it compared byte for byte.
fn reference_persona(run: &str) -> String {
    let first = reference_requests(run).remove(0);
    let system = first["messages"][0]["content"].as_str().unwrap().to_string();
    system.split("\n\n").next().unwrap().to_string()
}

fn start(chat: &Path, run: &'static str, context: usize, model: Arc<ScriptModel>) -> Process {
    let mut cfg = Config::for_chat(chat.to_path_buf());
    cfg.persona_prefix = reference_persona(run);
    cfg.context_window = context;
    cfg.skills_tool = run == "features";
    cfg.auto_compact = false;
    // the recordings' model (Qwen) reads images, hears nothing, watches no video
    cfg.audio_input = false;
    cfg.video_input = false;
    let ns = Namespace::new(&cfg.chat, None, None);
    let out = Capture(Arc::new(Mutex::new(Vec::new())));
    struct Shared(Arc<ScriptModel>);
    impl Model for Shared {
        fn stream(&self, body: &Value, c: &Cancel, d: &mut dyn FnMut(Delta)) -> Result<Answer, String> {
            self.0.stream(body, c, d)
        }
    }
    let server = Server::with_tool_runner(
        cfg,
        ns,
        Box::new(NoShell),
        Box::new(NoWeb),
        Box::new(Shared(model)),
        Arc::new(Out::new(Box::new(out.clone()))),
        Box::new(Replay::new(run)),
    );
    Process { server, out }
}

impl Process {
    /// A request, then its response (a prompt runs on its own thread).
    fn call(&self, msg: Value) -> Value {
        let id = msg["id"].clone();
        self.server.dispatch(msg);
        let t = Instant::now();
        loop {
            if let Some(r) =
                self.out.lines().into_iter().find(|l| l.get("id") == Some(&id) && l.get("method").is_none())
            {
                return r;
            }
            assert!(t.elapsed() < Duration::from_secs(20), "no response to {id}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// The client's side of a recorded run: its requests, with the session id this agent gave.
fn drive(run: &'static str, context: usize, chat: &Path, model: &Arc<ScriptModel>) -> Vec<Value> {
    let reference = reference_acp(run);
    let outs: Vec<Value> = reference.iter().filter(|r| r["dir"] == "out").map(|r| r["msg"].clone()).collect();
    let ref_sid = outs.iter().find_map(|m| m["params"]["sessionId"].as_str()).unwrap_or("").to_string();
    let mut transcript = Vec::new();
    let mut p = start(chat, run, context, model.clone());
    let mut sid = String::new();
    let mut seen_lines = 0;
    for m in outs {
        if m["method"] == "initialize" && !sid.is_empty() {
            transcript.extend(p.out.lines().into_iter().skip(seen_lines));
            p = start(chat, run, context, model.clone()); // the app restarted
            seen_lines = 0;
        }
        let mut m = m;
        if m["params"].get("sessionId").is_some() {
            m["params"]["sessionId"] = json!(sid);
        }
        if m["method"] == "session/new" && run == "features" {
            m["params"]["mcpServers"] = json!([{"name": "demo", "command": "/usr/bin/python3", "args": ["/workspace/.ref-tools/mcp_demo.py"], "env": []}]);
        }
        let r = p.call(m);
        if let Some(s) = r["result"]["sessionId"].as_str() {
            sid = s.to_string();
        }
        let _ = &ref_sid;
        // a prompt's updates come before its response: wait for the response, then take what came
        transcript.extend(p.out.lines().into_iter().skip(seen_lines));
        seen_lines = p.out.lines().len();
    }
    transcript
}

/// The stream as the client reads it: chunks of one message joined, ids and identities taken out, our own command
/// advertisement left aside.
fn normalize(msgs: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for mut m in msgs {
        if let Some(u) = m.pointer_mut("/params/update") {
            let kind = u["sessionUpdate"].as_str().unwrap_or("").to_string();
            if kind == "available_commands_update" {
                continue;
            }
            if let Some(o) = u.as_object_mut() {
                o.remove("messageId");
            }
            if (kind == "agent_message_chunk" || kind == "agent_thought_chunk")
                && let Some(prev) = out.last_mut().and_then(|p| p.pointer_mut("/params/update"))
                && prev["sessionUpdate"] == kind.as_str()
            {
                let joined = format!(
                    "{}{}",
                    prev["content"]["text"].as_str().unwrap(),
                    m["params"]["update"]["content"]["text"].as_str().unwrap()
                );
                prev["content"]["text"] = json!(joined);
                continue;
            }
        }
        if let Some(p) = m.get_mut("params").and_then(Value::as_object_mut)
            && p.contains_key("sessionId")
        {
            p.insert("sessionId".into(), json!("S"));
        }
        if let Some(r) = m.get_mut("result").and_then(Value::as_object_mut) {
            if r.contains_key("sessionId") {
                r.insert("sessionId".into(), json!("S"));
            }
            r.remove("agentInfo");
        }
        if let Some(o) = m.as_object_mut() {
            o.remove("id");
        }
        out.push(m);
    }
    out
}

fn parity(run: &'static str, context: usize, compare_requests: usize) {
    let chat = std::env::temp_dir().join(format!("geode-agent-parity-{run}-{}", geode_agent::util::uuid()));
    for d in ["workspace", "home", "local", "tmp"] {
        std::fs::create_dir_all(chat.join(d)).unwrap();
    }
    if run == "features" {
        std::fs::create_dir_all(chat.join("home/.dsh/skills/hello")).unwrap();
        std::fs::create_dir_all(chat.join("workspace/.agents/skills/proj")).unwrap();
        std::fs::write(chat.join("home/.dsh/skills/hello/SKILL.md"), "---\nname: hello\ndescription: Greets the user in their language. Use it when the user asks for a greeting.\n---\nGreet the user warmly, in one short sentence, in the language they wrote in.\n").unwrap();
        std::fs::write(chat.join("workspace/.agents/skills/proj/SKILL.md"), "---\nname: proj\ndescription: How this project's files are laid out.\n---\nScripts go in /workspace/scripts, results in /workspace/out.\n").unwrap();
    }
    let model = Arc::new(ScriptModel { run, seen: Mutex::new(Vec::new()) });
    let transcript = drive(run, context, &chat, &model);
    // every request, byte for byte
    let ours = model.seen.lock().unwrap().clone();
    let theirs = reference_requests(run);
    let n = compare_requests.min(theirs.len());
    assert!(ours.len() >= n, "{run}: {} requests, the reference has {}", ours.len(), theirs.len());
    for i in 0..n {
        let (a, b) = (serde_json::to_string(&ours[i]).unwrap(), serde_json::to_string(&theirs[i]).unwrap());
        if a != b {
            let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            let from = at.saturating_sub(300);
            panic!(
                "{run}: request {} differs at byte {at}:\n ours: …{}\n dsh:  …{}",
                i + 1,
                &a[from..(at + 300).min(a.len())],
                &b[from..(at + 300).min(b.len())]
            );
        }
    }
    if compare_requests >= theirs.len() {
        assert_eq!(ours.len(), theirs.len(), "{run}: request count");
        // the client's view, message by message
        let ours = normalize(transcript);
        let theirs =
            normalize(reference_acp(run).into_iter().filter(|r| r["dir"] == "in").map(|r| r["msg"].clone()).collect());
        assert_eq!(ours.len(), theirs.len(), "{run}: ACP message count");
        for (i, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
            assert_eq!(a, b, "{run}: ACP message {i}");
        }
    }
}

#[test]
fn phone_run() {
    parity("phone", 32768, usize::MAX);
}

#[test]
fn small_context_run() {
    parity("compact", 8192, usize::MAX);
}

#[test]
fn skills_and_mcp_run() {
    // up to the "/compact" prompt: dsh sent it to the model as text, this agent runs the command
    parity("features", 32768, 5);
}
