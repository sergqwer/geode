//! The agent's tools: their schemas (the harness's own, word for word), their dispatch, and the inline-size policy
//! every result passes (dsh-spill-policy: a result over 12 500 tokens keeps its head and tail, the whole of it saved
//! to a file the model can read back).

pub mod bash;
pub mod fs;
pub mod misc;
pub mod search;
pub mod web;

use crate::config::Config;
use crate::linux::{Cancel, Shell};
use crate::mcp::McpHub;
use crate::msg::{Block, Message, ToolCall, ToolSpec, price_block};
use crate::paths::Namespace;
use crate::session::Session;
use crate::skills::SkillSet;
use crate::util::{hex, js_len, js_slice_head, js_slice_tail, random_bytes};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::OnceLock;

const SPECS: &str = include_str!("specs.json");
pub const MAX_INLINE_TOKENS: usize = 12_500;

/// Everything a tool may use.
pub struct Env<'a> {
    pub cfg: &'a Config,
    pub ns: &'a Namespace,
    pub shell: &'a dyn Shell,
    pub web: &'a dyn web::Web,
    pub skills: &'a SkillSet,
    pub mcp: &'a McpHub,
    pub cancel: &'a Cancel,
}

/// A tool's result: what the model reads, whether it failed, and context the harness adds after it.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub blocks: Vec<Block>,
    pub is_error: bool,
    pub notices: Vec<Message>,
}

impl Output {
    pub fn text(s: impl Into<String>) -> Output {
        Output { blocks: vec![Block::text(s)], is_error: false, notices: Vec::new() }
    }
    pub fn error(s: impl AsRef<str>) -> Output {
        Output { blocks: vec![Block::text(format!("Error: {}", s.as_ref()))], is_error: true, notices: Vec::new() }
    }
}

fn base_specs() -> &'static [ToolSpec] {
    static S: OnceLock<Vec<ToolSpec>> = OnceLock::new();
    S.get_or_init(|| {
        let v: Vec<Value> = serde_json::from_str(SPECS).expect("embedded tool schemas");
        v.into_iter()
            .map(|t| ToolSpec {
                name: t["name"].as_str().unwrap_or_default().to_string(),
                description: t["description"].as_str().unwrap_or_default().to_string(),
                parameters: t["parameters"].clone(),
            })
            .collect()
    })
}

const MCP_RESOURCE_TOOLS: [&str; 3] = ["list_mcp_resource_templates", "list_mcp_resources", "read_mcp_resource"];

/// The tools the model is offered, by name (the harness's order): the phone's eleven, the skill tool when there are
/// skills to load, an MCP server's tools and its resource tools when one is connected.
pub fn specs(env: &Env) -> Vec<ToolSpec> {
    let mut out: Vec<ToolSpec> = base_specs()
        .iter()
        .filter(|s| {
            if MCP_RESOURCE_TOOLS.contains(&s.name.as_str()) {
                return env.mcp.has_servers();
            }
            if s.name == "skill" {
                return env.cfg.skills_tool;
            }
            true
        })
        .cloned()
        .collect();
    out.extend(env.mcp.tool_specs());
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// One call run: its result, bounded.
pub fn execute(env: &Env, sess: &mut Session, call: &ToolCall) -> Output {
    let args: Value = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);
    let out = run(env, sess, &call.name, &args);

    match call.name.as_str() {
        "read" => out,
        _ => bound(env, sess, &call.name, out),
    }
}

fn run(env: &Env, sess: &mut Session, name: &str, args: &Value) -> Output {
    if !args.is_object() {
        return Output::error(format!("{name} arguments must be a JSON object"));
    }
    let r = match name {
        "bash" => bash::bash(env, sess, args),
        "read" => fs::read(env, sess, args),
        "write" => fs::write(env, sess, args),
        "edit" => fs::edit(env, sess, args),
        "read_image" => fs::read_image(env, sess, args),
        "glob" => search::glob(env, sess, args),
        "grep" => search::grep(env, sess, args),
        "todo_write" => misc::todo_write(env, sess, args),
        "working_directory" => misc::working_directory(env, sess, args),
        "web_search" => web::web_search(env, args),
        "web_fetch" => web::web_fetch(env, args),
        "skill" => crate::skills::skill_tool(env, sess, args),
        "list_mcp_resources" | "list_mcp_resource_templates" | "read_mcp_resource" => {
            env.mcp.resource_tool(name, args, env.cancel)
        }
        _ if env.mcp.owns(name) => env.mcp.call_tool(name, args, env.cancel),
        _ => Err(format!("unknown tool \"{name}\"")),
    };
    r.unwrap_or_else(Output::error)
}

/// A string argument.
pub fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

pub fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    arg_str(args, key).ok_or_else(|| format!("{key} must be a string"))
}

/// A positive whole number argument (JSON numbers, as the model sends them).
pub fn arg_positive(args: &Value, key: &str) -> Result<Option<usize>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(f) if f.fract() == 0.0 && f >= 1.0 => Ok(Some(f as usize)),
            _ => Err(format!("{key} must be a positive integer")),
        },
    }
}

// ---- the spill store (dsh-spill-local) ----

pub struct SpillRef {
    pub locator: String,
}

pub const RETRIEVAL_HINT: &str = "Use read with offset/limit, or grep this path to search within it.";

/// A text saved for the model to read back: the chat's /tmp/geode-spill-*/session-<sha256(id)[:12]>/<id>-<name>.
pub fn save_spill(env: &Env, session_id: &str, name: &str, content: &str) -> Option<SpillRef> {
    static ROOT: OnceLock<String> = OnceLock::new();
    let root = ROOT.get_or_init(|| format!("geode-spill-{}", hex(&random_bytes::<3>())));
    let session_dir = format!("session-{}", &hex(&Sha256::digest(session_id.as_bytes()))[..12]);
    let guest = format!("/tmp/{root}/{session_dir}/{}-{name}", hex(&random_bytes::<6>()));
    let host: PathBuf = env.ns.host_of(&guest)?;
    std::fs::create_dir_all(host.parent()?).ok()?;
    std::fs::write(&host, content).ok()?;
    Some(SpillRef { locator: guest })
}

const GAP: &str = "\n\n[...]\n\n";

fn price_text(t: &str) -> usize {
    price_block(&Block::text(t))
}

/// The longest head (or tail) of [text] whose price fits [budget].
fn fit_text(text: &str, budget: usize, tail: bool) -> String {
    let (mut low, mut high) = (0usize, js_len(text));
    let slice = |n: usize| {
        if tail { js_slice_tail(text, n) } else { js_slice_head(text, n) }
    };
    while low < high {
        let middle = (low + high).div_ceil(2);
        let candidate = slice(middle);
        if candidate.is_empty() || price_text(candidate) <= budget {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    slice(low).to_string()
}

/// dsh-spill-policy: a result over [MAX_INLINE_TOKENS] keeps a head and a tail around a gap, the whole saved.
fn bound(env: &Env, sess: &Session, name: &str, out: Output) -> Output {
    if !out.blocks.iter().all(|b| matches!(b, Block::Text { .. })) {
        return out;
    }
    let price: usize = out.blocks.iter().map(price_block).sum();
    if price <= MAX_INLINE_TOKENS {
        return out;
    }
    let full: String = out
        .blocks
        .iter()
        .map(|b| match b {
            Block::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();
    let Some(r) = save_spill(env, &sess.id, &format!("{name}.txt"), &full) else {
        return out;
    };
    let total_bytes = full.len();
    let notice = |bytes: usize| {
        format!("(Omitted {bytes} bytes. Full formatted result stored at: {}. {RETRIEVAL_HINT})", r.locator)
    };
    let reserved = price_text(GAP) + price_text(&format!("\n\n{}", notice(total_bytes)));
    let budget = MAX_INLINE_TOKENS.saturating_sub(reserved);
    // head then tail, over the blocks (the results here are one text block)
    let texts: Vec<&str> = out
        .blocks
        .iter()
        .map(|b| match b {
            Block::Text { text } => text.as_str(),
            _ => "",
        })
        .collect();
    let mut head: Vec<String> = Vec::new();
    let mut tail: Vec<String> = Vec::new();
    let (mut first, mut last) = (0usize, texts.len() as isize - 1);
    let mut head_chars = 0usize;
    let mut remaining = budget.div_ceil(2);
    while (first as isize) <= last {
        let cost = price_text(texts[first]);
        if cost <= remaining {
            head.push(texts[first].to_string());
            remaining -= cost;
            first += 1;
            continue;
        }
        let t = fit_text(texts[first], remaining, false);
        head_chars = js_len(&t);
        if !t.is_empty() {
            head.push(t);
        }
        break;
    }
    remaining = budget / 2;
    while last >= first as isize {
        let original = texts[last as usize];
        let block =
            if last as usize == first { js_slice_tail(original, js_len(original) - head_chars) } else { original };
        let cost = price_text(block);
        if cost <= remaining {
            tail.push(block.to_string());
            remaining -= cost;
            last -= 1;
            continue;
        }
        let t = fit_text(block, remaining, true);
        if !t.is_empty() {
            tail.push(t);
        }
        break;
    }
    tail.reverse();
    let kept: usize = head.iter().chain(tail.iter()).map(|t| t.len()).sum();
    let footer = notice(total_bytes - kept);
    let text = if head.is_empty() && tail.is_empty() {
        footer
    } else {
        format!("{}{GAP}{}\n\n{footer}", head.concat(), tail.concat())
    };
    Output { blocks: vec![Block::text(text)], is_error: out.is_error, notices: out.notices }
}

#[cfg(test)]
pub mod testenv {
    //! A tool environment over temporary folders: no Linux, no network.
    use super::*;
    use crate::linux::Outcome;
    use std::sync::Mutex;

    pub struct FakeShell {
        pub outcomes: Mutex<Vec<(String, Outcome)>>,
    }

    impl Shell for FakeShell {
        fn run(&self, command: &str, _cwd: &str, _t: u64, _c: &Cancel) -> Result<Outcome, String> {
            let mut o = self.outcomes.lock().unwrap();
            let i = o.iter().position(|(c, _)| c == command).ok_or_else(|| format!("unexpected command {command}"))?;
            Ok(o.remove(i).1)
        }
        fn spawn(&self, _: &[String], _: &[(String, String)], _: &str) -> Result<std::process::Child, String> {
            Err("no Linux in tests".into())
        }
    }

    pub struct NoWeb;
    impl web::Web for NoWeb {
        fn search(&self, _q: &str, _m: usize, _c: &Cancel) -> Result<web::SearchResult, String> {
            Err("no network in tests".into())
        }
        fn fetch(&self, _u: &str, _c: &Cancel) -> Result<web::FetchResult, String> {
            Err("no network in tests".into())
        }
    }

    pub struct World {
        pub dir: PathBuf,
        pub cfg: Config,
        pub ns: Namespace,
        pub shell: FakeShell,
        pub skills: SkillSet,
        pub mcp: McpHub,
        pub cancel: Cancel,
    }

    impl World {
        pub fn new(name: &str) -> World {
            let dir = std::env::temp_dir().join(format!("geode-agent-{name}-{}", hex(&random_bytes::<4>())));
            for d in ["chat/workspace", "chat/home", "chat/local", "chat/tmp", "rootfs/usr/bin"] {
                std::fs::create_dir_all(dir.join(d)).unwrap();
            }
            let mut cfg = Config::for_chat(dir.join("chat"));
            cfg.rootfs = Some(dir.join("rootfs"));
            let ns = Namespace::new(&cfg.chat, cfg.rootfs.as_deref(), None);
            World {
                ns,
                cfg,
                dir,
                shell: FakeShell { outcomes: Mutex::new(Vec::new()) },
                skills: SkillSet::default(),
                mcp: McpHub::default(),
                cancel: Cancel::new(),
            }
        }

        pub fn env<'a>(&'a self, web: &'a dyn web::Web) -> Env<'a> {
            Env {
                cfg: &self.cfg,
                ns: &self.ns,
                shell: &self.shell,
                web,
                skills: &self.skills,
                mcp: &self.mcp,
                cancel: &self.cancel,
            }
        }

        pub fn ws(&self, rel: &str) -> PathBuf {
            self.dir.join("chat/workspace").join(rel)
        }

        pub fn session(&self) -> Session {
            Session::create(&self.cfg.sessions_dir(), "/workspace").unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testenv::*;
    use super::*;

    #[test]
    fn schemas_are_the_harness() {
        let names: Vec<_> = base_specs().iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"bash") && names.contains(&"skill") && names.contains(&"read_mcp_resource"));
    }

    #[test]
    fn spill_keeps_head_and_tail() {
        let w = World::new("spill");
        let s = w.session();
        let env = w.env(&NoWeb);
        let big: String = (1..40000).map(|n| format!("{n}\n")).collect();
        let out = bound(&env, &s, "bash", Output::text(big.clone()));
        let Block::Text { text } = &out.blocks[0] else { panic!() };
        assert!(text.starts_with("1\n2\n"));
        assert!(text.contains("\n\n[...]\n\n"));
        assert!(text.contains("Full formatted result stored at: /tmp/geode-spill-"));
        assert!(price_text(text) <= MAX_INLINE_TOKENS);
    }
}
