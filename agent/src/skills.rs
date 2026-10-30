//! Skills: SKILL.md bundles (or flat <name>.md files) the agent loads when a task matches one. Ported from
//! @deepseek-ai/dsh-skill, dsh-skill-filesystem and dsh-tool-skill: the same roots and ranks (the project's
//! .dsh/skills and .agents/skills, the harness home's skills, the user's - in Geode the skills shared by every chat,
//! at /root/.agents/skills), the same catalog the model is shown, the same `skill` tool, and "/name" in the user's
//! message loading a skill directly.

use crate::msg::{Message, Source};
use crate::paths::Namespace;
use crate::session::Session;
use crate::tools::{Env, Output, required_str};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub model_invocable: bool,
    pub user_invocable: bool,
    pub content: String,
    /// The skill's folder as the model reaches it.
    pub base: String,
    pub rank: u32,
}

/// The skills found for a working directory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkillSet {
    pub skills: Vec<Skill>,
}

const DESCRIPTION_MAX: usize = 500;

fn skill_name_ok(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 64
        && n.split('-').all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()))
}

/// A frontmatter's flat keys (strings, quoted or not; folded `>`/`|` blocks), and the body after it.
pub fn parse_frontmatter(text: &str) -> Option<(BTreeMap<String, String>, String)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let rest = text.strip_prefix("---")?.trim_start_matches([' ', '\t']);
    let rest = rest.strip_prefix("\r\n").or_else(|| rest.strip_prefix('\n'))?;
    let mut keys = BTreeMap::new();
    let mut lines = rest.split('\n');
    let mut body_start = None;
    let mut consumed = 0usize;
    let mut pending: Option<(String, bool, Vec<String>)> = None;
    let flush = |keys: &mut BTreeMap<String, String>, p: Option<(String, bool, Vec<String>)>| {
        if let Some((k, folded, ls)) = p {
            let v = if folded { ls.join(" ") } else { ls.join("\n") };
            keys.insert(k, v.trim().to_string());
        }
    };
    for line in lines.by_ref() {
        consumed += line.len() + 1;
        let l = line.trim_end_matches('\r');
        if l.trim_end() == "---" {
            flush(&mut keys, pending.take());
            body_start = Some(consumed);
            break;
        }
        if let Some((_, _, ls)) = pending.as_mut() {
            if l.starts_with(' ') || l.starts_with('\t') || l.is_empty() {
                ls.push(l.trim().to_string());
                continue;
            }
            flush(&mut keys, pending.take());
        }
        if l.starts_with(' ') || l.starts_with('#') || l.trim().is_empty() {
            continue;
        }
        let Some((k, v)) = l.split_once(':') else {
            continue;
        };
        let v = v.trim();
        if matches!(v, ">" | ">-" | ">+" | "|" | "|-" | "|+") {
            pending = Some((k.trim().to_string(), v.starts_with('>'), Vec::new()));
            continue;
        }
        let v = if (v.starts_with('"') && v.ends_with('"') && v.len() >= 2)
            || (v.starts_with('\'') && v.ends_with('\'') && v.len() >= 2)
        {
            v[1..v.len() - 1].to_string()
        } else {
            v.to_string()
        };
        keys.insert(k.trim().to_string(), v);
    }
    let body = &rest[body_start?.min(rest.len())..];
    Some((keys, body.trim().to_string()))
}

fn truthy(v: Option<&String>, default: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s == "true" => true,
        Some(s) if s == "false" => false,
        _ => default,
    }
}

fn load_skill(file: &Path, base_guest: String, rank: u32) -> Option<Skill> {
    let text = std::fs::read_to_string(file).ok()?;
    let (keys, body) = parse_frontmatter(&text)?;
    let name = keys.get("name")?.trim().to_string();
    let description = keys.get("description")?.trim().to_string();
    if !skill_name_ok(&name) || description.is_empty() {
        return None;
    }
    Some(Skill {
        name,
        description,
        model_invocable: !truthy(keys.get("disable-model-invocation"), false),
        user_invocable: truthy(keys.get("user-invocable"), true),
        content: body,
        base: base_guest,
        rank,
    })
}

/// The nearest folder holding .git, from [cwd] up; [cwd] itself when there is none.
fn project_root(ns: &Namespace, cwd: &str) -> String {
    let mut p = cwd.trim_end_matches('/').to_string();
    loop {
        if ns.host_of(&format!("{p}/.git")).is_some_and(|h| h.exists()) {
            return if p.is_empty() { "/".into() } else { p };
        }
        match p.rfind('/') {
            Some(0) | None => return cwd.to_string(),
            Some(i) => p.truncate(i),
        }
    }
}

impl SkillSet {
    /// The skills for a working directory, a name's lowest-ranked root winning.
    pub fn discover(ns: &Namespace, cwd: &str) -> SkillSet {
        let project = project_root(ns, cwd);
        let p = project.trim_end_matches('/');
        let roots = [
            (100, format!("{p}/.dsh/skills")),
            (200, format!("{p}/.agents/skills")),
            (400, "/root/.dsh/skills".to_string()),
            (500, "/root/.agents/skills".to_string()),
        ];
        let mut by_name: BTreeMap<String, Skill> = BTreeMap::new();
        for (rank, root) in roots {
            let Some(host) = ns.host_of(&root) else {
                continue;
            };
            let Ok(rd) = std::fs::read_dir(&host) else {
                continue;
            };
            let mut entries: Vec<_> = rd.flatten().collect();
            entries.sort_by_key(|e| e.file_name());
            for e in entries {
                let fname = e.file_name().to_string_lossy().into_owned();
                if rank == 400 && fname == ".system" {
                    continue;
                }
                let path = e.path();
                let skill = if path.is_dir() {
                    let f = path.join("SKILL.md");
                    if !f.is_file() {
                        continue;
                    }
                    load_skill(&f, format!("{root}/{fname}"), rank)
                } else if fname.ends_with(".md") {
                    load_skill(&path, root.clone(), rank)
                } else {
                    None
                };
                if let Some(s) = skill {
                    by_name.entry(s.name.clone()).or_insert(s);
                }
            }
        }
        SkillSet { skills: by_name.into_values().collect() }
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    /// The catalog's entries: the skills the model may load, by name.
    pub fn catalog_entries(&self) -> Vec<(String, String)> {
        self.skills
            .iter()
            .filter(|s| s.model_invocable)
            .map(|s| (s.name.clone(), catalog_description(&s.description)))
            .collect()
    }
}

pub fn escape_text(v: &str) -> String {
    v.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn escape_attr(v: &str) -> String {
    escape_text(v).replace('"', "&quot;")
}

fn catalog_description(v: &str) -> String {
    let normalized = v.split_whitespace().collect::<Vec<_>>().join(" ");
    if crate::util::js_len(&normalized) <= DESCRIPTION_MAX {
        normalized
    } else {
        format!("{}...", crate::util::js_slice_head(&normalized, DESCRIPTION_MAX - 3))
    }
}

/// A loaded skill as the model reads it (the `skill` tool's result, and a user's "/name").
pub fn render_skill(s: &Skill) -> String {
    [
        format!("<skill_content name=\"{}\">", escape_attr(&s.name)),
        "<skill_resources>".into(),
        format!("Base directory for this skill: {}", s.base),
        "Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.".into(),
        "</skill_resources>".into(),
        String::new(),
        "<skill_instructions>".into(),
        s.content.clone(),
        "</skill_instructions>".into(),
        "</skill_content>".into(),
    ]
    .join("\n")
}

fn entries_text(entries: &[(String, String)]) -> Vec<String> {
    entries.iter().map(|(n, d)| format!("- `{n}`: {}", escape_text(d))).collect()
}

/// The catalog the model is shown (first time), or its replacement when the skills changed.
pub fn catalog_message(entries: &[(String, String)], update: bool) -> String {
    let mut lines = vec!["<system-reminder>".to_string()];
    if !update {
        lines.push("A skill is a reusable set of task-specific instructions. The following skills are available in this session:".into());
    } else {
        lines.push("The available skill catalog changed. This complete catalog replaces every earlier available-skills list in this session:".into());
    }
    lines.push(String::new());
    lines.push("<available_skills>".into());
    lines.extend(entries_text(entries));
    lines.push("</available_skills>".into());
    lines.push(String::new());
    if !update {
        lines.push("If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.".into());
        lines.push("A user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.".into());
    } else if entries.is_empty() {
        lines.push(
            "No skills are currently available through the `skill` tool. Do not use names from earlier skill catalogs."
                .into(),
        );
        lines.push("A user may still invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool for it.".into());
    } else {
        lines.push("Use only names in this replacement catalog. If the user names a listed skill, or the task clearly matches its description, call the `skill` tool with the exact name before acting.".into());
        lines.push("A user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.".into());
    }
    lines.push("</system-reminder>".into());
    lines.join("\n")
}

/// The catalog message this step needs (none when nothing changed since the last one shown).
pub fn catalog_for_step(set: &SkillSet, messages: &[Message]) -> Option<Message> {
    let entries = set.catalog_entries();
    let last = messages
        .iter()
        .rev()
        .find(|m| m.source() == Some(Source::SkillCatalog))
        .and_then(|m| m.only_text().map(String::from));
    match last {
        None if entries.is_empty() => None,
        None => Some(Message::user_text(catalog_message(&entries, false), Source::SkillCatalog)),
        Some(text) => {
            // the entries the last catalog listed
            let shown: Vec<String> = text.lines().filter(|l| l.starts_with("- `")).map(String::from).collect();
            if shown == entries_text(&entries) {
                None
            } else {
                Some(Message::user_text(catalog_message(&entries, true), Source::SkillCatalog))
            }
        }
    }
}

/// "/name" in the user's words: the skills it loads, in order (each once).
pub fn invoked_names(text: &str) -> Vec<String> {
    let re = regex::Regex::new(r"(^|\s)/([a-z0-9]+(?:-[a-z0-9]+)*)(\s|$)").unwrap();
    let mut names: Vec<String> = Vec::new();
    let mut at = 0;
    while let Some(c) = re.captures_at(text, at) {
        let m = c.get(2).unwrap();
        if !names.iter().any(|n| n == m.as_str()) {
            names.push(m.as_str().to_string());
        }
        at = m.end();
    }
    names
}

pub fn skill_tool(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let name = required_str(args, "name")?;
    if !skill_name_ok(name) {
        return Err(format!("invalid skill name \"{name}\""));
    }
    let set = SkillSet::discover(env.ns, &sess.cwd);
    let skill = set.get(name).ok_or_else(|| format!("skill \"{name}\" is unknown or no longer available"))?;
    if !skill.model_invocable {
        return Err(format!("skill \"{name}\" is not available for model invocation"));
    }
    Ok(Output::text(render_skill(skill)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter() {
        let (k, body) = parse_frontmatter("---\nname: hello\ndescription: >-\n  Greets the user\n  warmly.\nuser-invocable: false\n---\nGreet them.\n").unwrap();
        assert_eq!(k["name"], "hello");
        assert_eq!(k["description"], "Greets the user warmly.");
        assert_eq!(k["user-invocable"], "false");
        assert_eq!(body, "Greet them.");
        assert!(parse_frontmatter("no frontmatter").is_none());
    }

    #[test]
    fn the_reference_catalog_and_skill() {
        let dir = std::env::temp_dir().join(format!("geode-agent-skills-{}", crate::util::uuid()));
        let chat = dir.join("chat");
        std::fs::create_dir_all(chat.join("home/.dsh/skills/hello")).unwrap();
        std::fs::create_dir_all(chat.join("workspace/.agents/skills/proj")).unwrap();
        std::fs::write(chat.join("home/.dsh/skills/hello/SKILL.md"), "---\nname: hello\ndescription: Greets the user in their language. Use it when the user asks for a greeting.\n---\nGreet the user warmly, in one short sentence, in the language they wrote in.\n").unwrap();
        std::fs::write(chat.join("workspace/.agents/skills/proj/SKILL.md"), "---\nname: proj\ndescription: How this project's files are laid out.\n---\nScripts go in /workspace/scripts, results in /workspace/out.\n").unwrap();
        let ns = Namespace::new(&chat, None, None);
        let set = SkillSet::discover(&ns, "/workspace");
        let reference = "<system-reminder>\nA skill is a reusable set of task-specific instructions. The following skills are available in this session:\n\n<available_skills>\n- `hello`: Greets the user in their language. Use it when the user asks for a greeting.\n- `proj`: How this project's files are laid out.\n</available_skills>\n\nIf the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.\nA user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.\n</system-reminder>";
        assert_eq!(catalog_message(&set.catalog_entries(), false), reference);
        let hello = set.get("hello").unwrap();
        assert_eq!(hello.base, "/root/.dsh/skills/hello");
        assert_eq!(
            render_skill(hello),
            "<skill_content name=\"hello\">\n<skill_resources>\nBase directory for this skill: /root/.dsh/skills/hello\nResolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.\n</skill_resources>\n\n<skill_instructions>\nGreet the user warmly, in one short sentence, in the language they wrote in.\n</skill_instructions>\n</skill_content>"
        );
        let msgs = vec![catalog_for_step(&set, &[]).unwrap()];
        assert!(catalog_for_step(&set, &msgs).is_none());
    }

    #[test]
    fn invocations() {
        assert_eq!(invoked_names("please /hello and /proj, /hello again"), vec!["hello"]);
        assert_eq!(invoked_names("/hello /proj"), vec!["hello", "proj"]);
        assert!(invoked_names("a/b /Up").is_empty());
    }
}
