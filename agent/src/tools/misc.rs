//! todo_write and working_directory (ported from @deepseek-ai/dsh-tool-todo, dsh-tool-working-directory and
//! dsh-working-directory: a changed directory is announced to the model and the runtime context follows it).

use super::{Env, Output, arg_str};
use crate::msg::{Message, Source};
use crate::session::{Session, Todo};
use serde_json::{Value, json};

const STATUSES: [&str; 3] = ["pending", "in_progress", "completed"];

pub fn todo_write(_env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let raw = args.get("todos").and_then(Value::as_array).ok_or("todos must be an array")?;
    let mut todos: Vec<Todo> = Vec::new();
    for item in raw {
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .ok_or("invalid todo: `content` must be a non-empty string")?
            .trim();
        if content.is_empty() {
            return Err("invalid todo: `content` must be a non-empty string".into());
        }
        if todos.iter().any(|t| t.content == content) {
            return Err(format!("invalid todos: duplicate content {}", json!(content)));
        }
        let status = item
            .get("status")
            .and_then(Value::as_str)
            .filter(|s| STATUSES.contains(s))
            .ok_or("invalid todo: `status` must be pending, in_progress or completed")?;
        todos.push(Todo { content: content.to_string(), status: status.to_string() });
    }
    let count = |s: &str| todos.iter().filter(|t| t.status == s).count();
    let text = format!(
        "Updated todo list: {} pending, {} in progress, {} completed.",
        count("pending"),
        count("in_progress"),
        count("completed")
    );
    sess.todos = todos;
    sess.save_state();
    Ok(Output::text(text))
}

pub fn working_directory(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let Some(cd) = arg_str(args, "cd") else {
        return Ok(Output::text(sess.cwd.clone()));
    };
    if cd.is_empty() {
        return Err("working-directory: cd must not be empty".into());
    }
    let r = env
        .ns
        .resolve(cd, &sess.cwd, true)
        .map_err(|_| format!("working-directory: directory does not exist: {}", json!(cd)))?;
    if !r.host.is_dir() {
        return Err(format!("working-directory: directory does not exist: {}", json!(cd)));
    }
    let previous = sess.cwd.clone();
    let mut out = Output::text(r.guest.clone());
    if previous != r.guest {
        sess.cwd = r.guest.clone();
        sess.save_state();
        out.notices.push(Message::user_text(
            format!("The working directory changed from {} to {}.", json!(previous), json!(r.guest)),
            Source::Notice,
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::testenv::*;
    use super::*;
    use crate::msg::Block;

    #[test]
    fn todos_and_cwd() {
        let w = World::new("misc");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        let o = todo_write(&env, &mut s, &json!({"todos": [{"content": "check", "status": "completed"}]})).unwrap();
        assert_eq!(o.blocks[0], Block::text("Updated todo list: 0 pending, 0 in progress, 1 completed."));
        assert_eq!(working_directory(&env, &mut s, &json!({})).unwrap().blocks[0], Block::text("/workspace"));
        std::fs::create_dir_all(w.ws("sub")).unwrap();
        let o = working_directory(&env, &mut s, &json!({"cd": "sub"})).unwrap();
        assert_eq!(o.blocks[0], Block::text("/workspace/sub"));
        assert_eq!(
            o.notices[0].only_text().unwrap(),
            "The working directory changed from \"/workspace\" to \"/workspace/sub\"."
        );
        assert_eq!(
            working_directory(&env, &mut s, &json!({"cd": "nope"})).unwrap_err(),
            "working-directory: directory does not exist: \"nope\""
        );
    }
}
