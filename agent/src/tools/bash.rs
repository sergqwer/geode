//! bash: a command in the chat's Linux, started for it and gone after it (ported from @deepseek-ai/dsh-tool-bash:
//! its checks, its result - stdout, then [stderr], then the markers the model reads: timed out, killed, exit code).

use super::{Env, Output, arg_str, required_str};
use crate::linux::{Captured, DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS, Outcome};
use crate::paths::join_guest;
use crate::session::Session;
use serde_json::Value;

fn stream_text(c: &Captured) -> String {
    if !c.truncated {
        return c.text.clone();
    }
    format!("{}\n[output truncated; full output: {}]", c.text, c.spill_path.as_deref().unwrap_or("(unavailable)"))
}

/// The result the model reads (renderResult).
pub fn render(r: &Outcome) -> String {
    let out = stream_text(&r.stdout);
    let err = stream_text(&r.stderr);
    let mut body = out;
    if !err.is_empty() {
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(&format!("[stderr]\n{err}"));
    }
    if body.is_empty() {
        body = "(no output)".into();
    }
    let mut markers = Vec::new();
    if r.timed_out {
        markers.push(format!("[timed out after {}ms]", r.timeout_ms));
    }
    if r.aborted {
        markers.push("[stopped: tool call aborted]".to_string());
    }
    if let Some(s) = &r.signal {
        markers.push(format!("[killed by signal: {s}]"));
    } else if let Some(code) = r.exit_code.filter(|&c| c != 0) {
        markers.push(format!("[exit code: {code}]"));
    }
    if markers.is_empty() {
        return body;
    }
    if !body.ends_with('\n') {
        body.push('\n');
    }
    body + &markers.join("\n")
}

pub fn bash(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let command =
        required_str(args, "command").map_err(|_| "invalid command: expected a non-empty string".to_string())?;
    if command.trim().is_empty() {
        return Err("invalid command: expected a non-empty string".into());
    }
    let description = arg_str(args, "description").unwrap_or("");
    if description.trim().is_empty() {
        return Err("invalid description: expected a non-empty string".into());
    }
    let timeout = match args.get("timeoutMs") {
        None | Some(Value::Null) => DEFAULT_TIMEOUT_MS,
        Some(v) => match v.as_f64() {
            Some(t) if t.is_finite() && t > 0.0 => (t as u64).min(MAX_TIMEOUT_MS),
            _ => {
                return Err(format!("invalid timeoutMs: expected a positive number, got {v}"));
            }
        },
    };
    let workdir = match arg_str(args, "workdir") {
        None => sess.cwd.clone(),
        Some(w) => join_guest(&sess.cwd, w),
    };
    let resolved = env.ns.resolve(&workdir, "/", true)?;
    if !resolved.host.is_dir() {
        return Err(format!("working directory does not exist: {}", resolved.guest));
    }
    let outcome = env.shell.run(command, &resolved.guest, timeout, env.cancel)?;
    Ok(Output::text(render(&outcome)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linux::MAX_OUTPUT_BYTES;

    fn ok(stdout: &str, stderr: &str, code: i32) -> Outcome {
        Outcome {
            stdout: Captured { text: stdout.into(), ..Default::default() },
            stderr: Captured { text: stderr.into(), ..Default::default() },
            exit_code: Some(code),
            timeout_ms: 120_000,
            ..Default::default()
        }
    }

    #[test]
    fn the_reference() {
        assert_eq!(render(&ok("out\n", "err\n", 3)), "out\n[stderr]\nerr\n[exit code: 3]");
        assert_eq!(render(&ok("", "", 0)), "(no output)");
        assert_eq!(render(&ok("42\n", "", 0)), "42\n");
        let mut t = ok("partial", "", 0);
        t.timed_out = true;
        t.exit_code = None;
        assert_eq!(render(&t), "partial\n[timed out after 120000ms]");
    }

    #[test]
    fn long_output_keeps_its_tail() {
        // seq 1 40000, as the reference's step 17 ran it: the last 64 000 bytes and the log's path
        let full: String = (1..=40000).map(|n| format!("{n}\n")).collect();
        let tail = &full[full.len() - MAX_OUTPUT_BYTES..];
        let o = Outcome {
            stdout: Captured { text: tail.into(), truncated: true, spill_path: Some("/tmp/x/stdout.log".into()) },
            exit_code: Some(0),
            ..Default::default()
        };
        let r = render(&o);
        assert!(r.starts_with("334\n29335\n"));
        assert!(r.ends_with("40000\n\n[output truncated; full output: /tmp/x/stdout.log]"));
    }
}
