//! A session: the conversation the model sees, the working directory, the todo list - kept in the chat's own Linux
//! (/root/.geode-agent/sessions/<id>.jsonl) so a new process (the app restarted) goes on with it. Messages are
//! appended a line each; a history that is rewritten (compaction, /clear) is written whole aside and renamed into
//! place. No hard links: the phone does not allow them (dsh's own store needed proot's emulation).

use crate::msg::Message;
use crate::util::{iso_now, uuid};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Todo {
    pub content: String,
    pub status: String,
}

/// What a file tool last saw of a file (dsh-fs-observation-policy: kept for the process, not the session log).
#[derive(Clone, Debug, PartialEq)]
pub enum Observed {
    Absent,
    Present(String),
}

pub struct Session {
    pub id: String,
    pub cwd: String,
    pub title: String,
    pub created: String,
    pub updated: String,
    pub messages: Vec<Message>,
    pub todos: Vec<Todo>,
    pub observed: HashMap<String, Observed>,
    path: PathBuf,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
enum Line {
    Meta { id: String, cwd: String, created: String },
    Msg { m: Message },
    State { cwd: String, title: String, updated: String, todos: Vec<Todo> },
}

impl Session {
    pub fn create(dir: &Path, cwd: &str) -> Result<Session, String> {
        fs::create_dir_all(dir).map_err(|e| format!("cannot keep sessions in {}: {e}", dir.display()))?;
        let id = uuid();
        let now = iso_now();
        let s = Session {
            path: dir.join(format!("{id}.jsonl")),
            id,
            cwd: cwd.to_string(),
            title: String::new(),
            created: now.clone(),
            updated: now,
            messages: Vec::new(),
            todos: Vec::new(),
            observed: HashMap::new(),
        };
        s.rewrite()?;
        Ok(s)
    }

    pub fn load(dir: &Path, id: &str) -> Result<Session, String> {
        if id.is_empty() || id.contains(['/', '\\', '.']) {
            return Err(format!("no session {id}"));
        }
        let path = dir.join(format!("{id}.jsonl"));
        let f = File::open(&path).map_err(|_| format!("Resource not found: session {id}"))?;
        let mut s = Session {
            id: id.to_string(),
            cwd: "/workspace".into(),
            title: String::new(),
            created: String::new(),
            updated: String::new(),
            messages: Vec::new(),
            todos: Vec::new(),
            observed: HashMap::new(),
            path,
        };
        for line in BufReader::new(f).lines() {
            let Ok(line) = line else { break };
            // a line cut short by a crash is the end of what was written
            let Ok(l) = serde_json::from_str::<Line>(&line) else {
                continue;
            };
            match l {
                Line::Meta { cwd, created, .. } => {
                    s.cwd = cwd;
                    s.created = created.clone();
                    s.updated = created;
                }
                Line::Msg { m } => s.messages.push(m),
                Line::State { cwd, title, updated, todos } => {
                    s.cwd = cwd;
                    s.title = title;
                    s.updated = updated;
                    s.todos = todos;
                }
            }
        }
        Ok(s)
    }

    fn append_lines(&self, lines: &[Line]) {
        if let Ok(mut f) = OpenOptions::new().append(true).create(true).open(&self.path) {
            let mut buf = String::new();
            for l in lines {
                if let Ok(s) = serde_json::to_string(l) {
                    buf.push_str(&s);
                    buf.push('\n');
                }
            }
            let _ = f.write_all(buf.as_bytes());
        }
    }

    fn state_line(&self) -> Line {
        Line::State {
            cwd: self.cwd.clone(),
            title: self.title.clone(),
            updated: self.updated.clone(),
            todos: self.todos.clone(),
        }
    }

    /// A message into the conversation (and the log).
    pub fn push(&mut self, m: Message) {
        self.updated = iso_now();
        self.append_lines(&[Line::Msg { m: m.clone() }]);
        self.messages.push(m);
    }

    /// The working directory, the title, the todo list: saved.
    pub fn save_state(&mut self) {
        self.updated = iso_now();
        self.append_lines(&[self.state_line()]);
    }

    /// The whole log written again (a history rewritten), aside and renamed into place.
    pub fn rewrite(&self) -> Result<(), String> {
        let tmp = self.path.with_extension("jsonl.tmp");
        let mut out = String::new();
        let mut put = |l: &Line| {
            out.push_str(&serde_json::to_string(l).unwrap_or_default());
            out.push('\n');
        };
        put(&Line::Meta { id: self.id.clone(), cwd: self.cwd.clone(), created: self.created.clone() });
        for m in &self.messages {
            put(&Line::Msg { m: m.clone() });
        }
        put(&self.state_line());
        let mut f = File::create(&tmp).map_err(|e| format!("cannot save the session: {e}"))?;
        f.write_all(out.as_bytes()).and_then(|_| f.sync_all()).map_err(|e| format!("cannot save the session: {e}"))?;
        drop(f);
        fs::rename(&tmp, &self.path).map_err(|e| format!("cannot save the session: {e}"))
    }

    pub fn replace_messages(&mut self, messages: Vec<Message>) -> Result<(), String> {
        self.messages = messages;
        self.updated = iso_now();
        self.rewrite()
    }

    /// The sessions kept in [dir], newest first (session/list).
    pub fn list(dir: &Path) -> Vec<Value> {
        let mut out: Vec<(String, Value)> = Vec::new();
        let Ok(rd) = fs::read_dir(dir) else {
            return Vec::new();
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(id) = name.strip_suffix(".jsonl") else {
                continue;
            };
            if let Ok(s) = Session::load(dir, id) {
                let mut v = json!({"sessionId": s.id, "cwd": s.cwd});
                if !s.title.is_empty() {
                    v["title"] = json!(s.title);
                }
                v["updatedAt"] = json!(s.updated);
                out.push((s.updated.clone(), v));
            }
        }
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out.into_iter().map(|(_, v)| v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msg::{Message, Source};

    #[test]
    fn persists() {
        let dir = std::env::temp_dir().join(format!("geode-agent-sess-{}", crate::util::uuid()));
        let mut s = Session::create(&dir, "/workspace").unwrap();
        s.push(Message::System { text: "sys".into() });
        s.push(Message::user_text("hi", Source::User));
        s.cwd = "/workspace/sub".into();
        s.title = "hi".into();
        s.save_state();
        let id = s.id.clone();
        let r = Session::load(&dir, &id).unwrap();
        assert_eq!(r.messages.len(), 2);
        assert_eq!(r.cwd, "/workspace/sub");
        let mut r = r;
        r.replace_messages(vec![Message::System { text: "sys".into() }]).unwrap();
        let r2 = Session::load(&dir, &id).unwrap();
        assert_eq!(r2.messages.len(), 1);
        assert_eq!(r2.title, "hi");
        assert_eq!(Session::list(&dir).len(), 1);
    }
}
