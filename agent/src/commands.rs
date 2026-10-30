//! Commands typed in the chat ("/compact", "/clear", "/skills", "/mcp", "/help"), answered by the agent itself
//! without the model, and advertised to the app (ACP available_commands_update) with the skills a user may load as
//! "/name". DeepSeek Harness has a command registry (dsh-commands, dsh-command-compact) its ACP bridge never wired:
//! there "/compact" reached the model as text. Here commands work; anything else starting with "/" still goes to
//! the model (a skill's "/name" with its instructions loaded).

use crate::agent::Runtime;
use crate::compact;
use crate::linux::Cancel;
use crate::mcp::McpHub;
use crate::msg::Message;
use crate::session::Session;
use crate::skills::SkillSet;
use crate::tools;
use serde_json::{Value, json};

pub const COMMANDS: [(&str, &str); 5] = [
    ("compact", "Compact older conversation history"),
    ("clear", "Start the conversation afresh (the chat's files stay)"),
    ("skills", "List the skills this chat can use"),
    ("mcp", "List the connected MCP servers and their tools"),
    ("help", "List the commands"),
];

/// The commands and the skills a user may invoke, as ACP lists them.
pub fn available(skills: &SkillSet) -> Vec<Value> {
    let mut out: Vec<Value> =
        COMMANDS.iter().map(|(n, d)| json!({"name": n, "description": d, "input": null})).collect();
    for s in skills.skills.iter().filter(|s| s.user_invocable) {
        if !COMMANDS.iter().any(|(n, _)| *n == s.name) {
            out.push(
                json!({"name": s.name, "description": s.description, "input": {"hint": "what to do with this skill"}}),
            );
        }
    }
    out
}

/// A command's answer (None: not a command - the text goes to the model).
pub fn handle(rt: &Runtime, sess: &mut Session, mcp: &McpHub, text: &str, cancel: &Cancel) -> Option<String> {
    let t = text.trim();
    let rest = t.strip_prefix('/')?;
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (rest, ""),
    };
    if !COMMANDS.iter().any(|(n, _)| *n == name) {
        return None;
    }
    let skills = SkillSet::discover(rt.ns, &sess.cwd);
    Some(match name {
        "compact" => {
            if !args.is_empty() {
                return Some("Usage: /compact (no arguments)".into());
            }
            crate::agent::ensure_system(rt, sess, mcp);
            let env = rt.env(&skills, mcp, cancel);
            let specs = tools::specs(&env);
            match compact::compact(rt, sess, &specs, cancel, None) {
                Ok(None) => "No compactable history yet.".into(),
                Ok(Some(c)) => format!("Compacted {} history items (~{} tokens).", c.items, c.tokens),
                Err(_) if cancel.is_cancelled() => "Compaction cancelled.".into(),
                Err(e) => format!("Compaction could not produce a useful summary. ({e})"),
            }
        }
        "clear" => {
            let head: Vec<Message> =
                sess.messages.iter().take(1).filter(|m| matches!(m, Message::System { .. })).cloned().collect();
            sess.todos.clear();
            sess.observed.clear();
            match sess.replace_messages(head) {
                Ok(()) => {
                    "Cleared. The next message starts a fresh conversation; the chat's files stay as they are.".into()
                }
                Err(e) => e,
            }
        }
        "skills" => {
            if skills.skills.is_empty() {
                "No skills are installed. Put a skill's folder (with its SKILL.md) in the chat's /workspace/.agents/skills, or add one to Geode's shared skills.".into()
            } else {
                let lines: Vec<String> =
                    skills.skills.iter().map(|s| format!("- /{} — {} ({})", s.name, s.description, s.base)).collect();
                format!("Skills in this chat:\n{}", lines.join("\n"))
            }
        }
        "mcp" => {
            let servers = mcp.summary();
            if servers.is_empty() {
                "No MCP servers are connected to this chat.".into()
            } else {
                let lines: Vec<String> = servers
                    .iter()
                    .map(|(n, t)| {
                        format!("- {n}: {}", if t.is_empty() { "(no tools)".to_string() } else { t.join(", ") })
                    })
                    .collect();
                format!("MCP servers:\n{}", lines.join("\n"))
            }
        }
        _ => {
            let mut lines: Vec<String> = COMMANDS.iter().map(|(n, d)| format!("/{n} — {d}")).collect();
            for s in skills.skills.iter().filter(|s| s.user_invocable) {
                lines.push(format!("/{} — load the skill: {}", s.name, s.description));
            }
            lines.join("\n")
        }
    })
}
