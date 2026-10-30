//! Chat commands and user-invoked skills, through the ACP server: what the user types, what the agent answers,
//! what the model is sent.

use geode_agent::acp::{Out, Server};
use geode_agent::config::Config;
use geode_agent::linux::{Cancel, Outcome, Shell};
use geode_agent::model::{Answer, Delta, Model};
use geode_agent::paths::Namespace;
use geode_agent::tools::web::{FetchResult, SearchResult, Web};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Echo {
    seen: Arc<Mutex<Vec<Value>>>,
}
impl Model for Echo {
    fn stream(&self, body: &Value, _: &Cancel, d: &mut dyn FnMut(Delta)) -> Result<Answer, String> {
        self.seen.lock().unwrap().push(body.clone());
        let last = body["messages"].as_array().unwrap().last().unwrap();
        let text =
            if last["content"].as_str().is_some_and(|c| c.starts_with("You are now acting as a compaction engine")) {
                "## Primary Request and Intent\n- talk".to_string()
            } else {
                "Fine.".to_string()
            };
        d(Delta::Text(text.clone()));
        Ok(Answer {
            text,
            finish_reason: "stop".into(),
            usage: Some(geode_agent::msg::Usage { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 }),
            ..Default::default()
        })
    }
}

struct NoShell;
impl Shell for NoShell {
    fn run(&self, _: &str, _: &str, _: u64, _: &Cancel) -> Result<Outcome, String> {
        Err("no Linux".into())
    }
    fn spawn(&self, _: &[String], _: &[(String, String)], _: &str) -> Result<std::process::Child, String> {
        Err("no Linux".into())
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

fn call(s: &Arc<Server>, out: &Capture, id: u64, method: &str, params: Value) -> Value {
    s.dispatch(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
    let t = Instant::now();
    loop {
        if let Some(r) = out.lines().into_iter().find(|l| l["id"] == id && l.get("method").is_none()) {
            return r;
        }
        assert!(t.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The text the agent sent for a prompt (its message chunks after the [from]th line).
fn said(out: &Capture, from: usize) -> String {
    out.lines()[from..]
        .iter()
        .filter(|l| l.pointer("/params/update/sessionUpdate") == Some(&json!("agent_message_chunk")))
        .map(|l| l.pointer("/params/update/content/text").unwrap().as_str().unwrap().to_string())
        .collect()
}

#[test]
fn commands_and_skills() {
    let chat = std::env::temp_dir().join(format!("geode-agent-cmd-{}", geode_agent::util::uuid()));
    std::fs::create_dir_all(chat.join("workspace/.agents/skills/hello")).unwrap();
    std::fs::write(
        chat.join("workspace/.agents/skills/hello/SKILL.md"),
        "---\nname: hello\ndescription: Says hello.\n---\nSay hello.\n",
    )
    .unwrap();
    let mut cfg = Config::for_chat(chat.clone());
    // small: the history below is worth compacting (the newest 16% of the context is kept)
    cfg.context_window = 8192;
    // the manual command alone (on its own the conversation would compact itself at 80%)
    cfg.auto_compact = false;
    let ns = Namespace::new(&chat, None, None);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let out = Capture(Arc::new(Mutex::new(Vec::new())));
    let s = Server::new(
        cfg,
        ns,
        Box::new(NoShell),
        Box::new(NoWeb),
        Box::new(Echo { seen: seen.clone() }),
        Arc::new(Out::new(Box::new(out.clone()))),
    );
    call(&s, &out, 1, "initialize", json!({"protocolVersion": 1}));
    let sid = call(&s, &out, 2, "session/new", json!({"cwd": "/workspace", "mcpServers": []}))["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    // the commands are advertised after session/new answers (from the session's own thread): waited for
    let started = std::time::Instant::now();
    let adv = loop {
        let found = out
            .lines()
            .into_iter()
            .find(|l| l.pointer("/params/update/sessionUpdate") == Some(&json!("available_commands_update")));
        if let Some(l) = found {
            break l;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(10), "no available_commands_update");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    let names: Vec<&str> = adv
        .pointer("/params/update/availableCommands")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["compact", "clear", "skills", "mcp", "help", "hello"]);

    let prompt = |id: u64, text: &str| {
        let from = out.lines().len();
        let r =
            call(&s, &out, id, "session/prompt", json!({"sessionId": sid, "prompt": [{"type": "text", "text": text}]}));
        assert_eq!(r["result"]["stopReason"], "end_turn");
        said(&out, from)
    };
    assert_eq!(prompt(3, "/compact"), "No compactable history yet.");
    assert!(seen.lock().unwrap().is_empty(), "a command reaches no model");
    // a user's "/hello": the skill's instructions follow the message, the catalog before it
    assert_eq!(prompt(4, "please /hello"), "Fine.");
    let first = seen.lock().unwrap()[0].clone();
    let msgs = first["messages"].as_array().unwrap();
    assert_eq!(msgs[1]["content"], "please /hello");
    assert!(msgs[2]["content"].as_str().unwrap().starts_with("Current runtime context."));
    assert!(msgs[3]["content"].as_str().unwrap().contains("- `hello`: Says hello."));
    assert!(msgs[4]["content"].as_str().unwrap().starts_with("<skill_content name=\"hello\">"));
    for i in 0..6 {
        prompt(10 + i, &format!("message {i} {}", "words ".repeat(400)));
    }
    let r = prompt(20, "/compact");
    assert!(r.starts_with("Compacted ") && r.ends_with(" tokens)."), "{r}");
    let after = prompt(21, "go on");
    assert_eq!(after, "Fine.");
    let last = seen.lock().unwrap().last().unwrap().clone();
    let has_checkpoint = last["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["content"].as_str().is_some_and(|c| c.contains("<compacted-summary>## Primary Request")));
    assert!(has_checkpoint);
    assert!(prompt(22, "/skills").contains("/hello — Says hello."));
    assert!(prompt(23, "/help").contains("/compact — Compact older conversation history"));
    assert_eq!(prompt(24, "/mcp"), "No MCP servers are connected to this chat.");
    assert!(prompt(25, "/clear").starts_with("Cleared."));
    let n = seen.lock().unwrap().len();
    prompt(26, "fresh");
    let fresh = seen.lock().unwrap()[n].clone();
    assert_eq!(fresh["messages"].as_array().unwrap()[1]["content"], "fresh");
    // a "/word" that is no command goes to the model
    assert_eq!(prompt(27, "/unknown thing"), "Fine.");
    // session/list knows it
    let l = call(&s, &out, 30, "session/list", json!({}));
    assert_eq!(l["result"]["sessions"][0]["sessionId"], sid.as_str());
}

#[test]
fn compacts_itself_near_the_context() {
    let chat = std::env::temp_dir().join(format!("geode-agent-auto-{}", geode_agent::util::uuid()));
    let mut cfg = Config::for_chat(chat.clone());
    // the least an agent works in: its prompt and tools are ~3K tokens, the request reserve 4K
    cfg.context_window = 16384;
    let ns = Namespace::new(&chat, None, None);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let out = Capture(Arc::new(Mutex::new(Vec::new())));
    let s = Server::new(
        cfg,
        ns,
        Box::new(NoShell),
        Box::new(NoWeb),
        Box::new(Echo { seen: seen.clone() }),
        Arc::new(Out::new(Box::new(out.clone()))),
    );
    let sid = call(&s, &out, 1, "session/new", json!({"cwd": "/workspace", "mcpServers": []}))["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    for i in 0..16 {
        let r = call(
            &s,
            &out,
            10 + i,
            "session/prompt",
            json!({"sessionId": sid, "prompt": [{"type": "text", "text": format!("message {i} {}", "words ".repeat(400))}]}),
        );
        assert_eq!(r["result"]["stopReason"], "end_turn");
    }
    let seen = seen.lock().unwrap();
    let summarized = seen.iter().any(|b| {
        b["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .is_some_and(|c| c.starts_with("You are now acting as a compaction engine"))
    });
    assert!(summarized, "a summarization request");
    let last = seen.last().unwrap()["messages"].as_array().unwrap().clone();
    assert!(last.iter().any(|m| { m["content"].as_str().is_some_and(|c| c.contains("<compacted-summary>")) }));
    // and every request stayed within the context
    for (i, b) in seen.iter().enumerate() {
        let msgs = b["messages"].as_array().unwrap();
        let last = msgs.last().unwrap()["content"].as_str().unwrap_or("").chars().take(30).collect::<String>();
        assert!(
            b["max_completion_tokens"].as_u64().unwrap() > 1,
            "request {i}: {} messages, last {last:?}, budget {}",
            msgs.len(),
            b["max_completion_tokens"]
        );
    }
}
