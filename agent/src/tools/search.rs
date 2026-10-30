//! glob and grep. The harness runs ripgrep for them; ripgrep is these crates (ignore, globset, regex), so the same
//! rules hold without a ripgrep binary: glob = `rg --files --glob=P --sort=modified --no-ignore --hidden` minus
//! version-control folders, grep = `rg --json --regexp=P [--glob=I]` (ignore files honoured, hidden files skipped).
//! Ported from @deepseek-ai/dsh-tool-fs-search: its caps, its words, its spill of an over-cap list.

use super::{Env, Output, RETRIEVAL_HINT, arg_str, required_str, save_spill};
use crate::session::Session;
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const GLOB_MAX_RESULTS: usize = 100;
const GREP_MAX_MATCHES: usize = 250;
const GREP_MAX_LINE_BYTES: usize = 2000;
const VCS: [&str; 6] = [".git", ".svn", ".hg", ".bzr", ".jj", ".sl"];

/// Where a search runs: its host folder, how its paths are shown (the path argument as given, or none), the
/// working directory they are made relative to.
struct Root {
    host: PathBuf,
    shown: Option<String>,
}

fn root(env: &Env, sess: &Session, tool: &str, path: Option<&str>) -> Result<Root, String> {
    match path {
        None => {
            let host = env
                .ns
                .host_of(&sess.cwd)
                .ok_or_else(|| format!("{tool} search failed (exit 2): no working directory"))?;
            Ok(Root { host, shown: None })
        }
        Some(p) => {
            let r = env.ns.resolve(p, &sess.cwd, true).map_err(|e| format!("{tool} search failed (exit 2): {e}"))?;
            if !r.host.exists() {
                return Err(format!(
                    "{tool} search failed (exit 2): rg: {p}: IO error for operation on {p}: No such file or directory (os error 2)"
                ));
            }
            Ok(Root { host: r.host, shown: Some(p.to_string()) })
        }
    }
}

/// A found path as ripgrep prints it (under the path argument as given), then made relative to the working
/// directory when it is absolute and inside it.
fn display(env: &Env, sess: &Session, root: &Root, host: &Path) -> String {
    let rel: Vec<String> = host
        .strip_prefix(&root.host)
        .map(|r| r.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    let printed = match &root.shown {
        None => rel.join("/"),
        Some(p) if rel.is_empty() => p.clone(),
        Some(p) => format!("{}/{}", p.trim_end_matches('/'), rel.join("/")),
    };
    if printed.starts_with('/') {
        let cwd = sess.cwd.trim_end_matches('/');
        if printed == cwd {
            return ".".into();
        }
        if let Some(r) = printed.strip_prefix(&format!("{cwd}/")) {
            return r.to_string();
        }
        let _ = env;
    }
    printed
}

pub fn glob(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let pattern = required_str(args, "pattern")?;
    if pattern.trim().is_empty() {
        return Err("pattern must be a non-empty string".into());
    }
    let path = arg_str(args, "path");
    if path.is_some_and(|p| p.trim().is_empty()) {
        return Err("path must be a non-empty string when given".into());
    }
    let root = root(env, sess, "glob", path)?;
    let mut ob = OverrideBuilder::new(&root.host);
    ob.add(pattern).map_err(|e| format!("glob pattern rejected by ripgrep: error parsing glob '{pattern}': {e}"))?;
    for v in VCS {
        let _ = ob.add(&format!("!**/{v}"));
        let _ = ob.add(&format!("!**/{v}/**"));
    }
    let overrides = ob.build().map_err(|e| format!("glob pattern rejected by ripgrep: {e}"))?;
    let walker = WalkBuilder::new(&root.host)
        .hidden(false)
        .ignore(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .overrides(overrides)
        .build();
    let mut found: Vec<(SystemTime, String)> = Vec::new();
    for entry in walker {
        if env.cancel.is_cancelled() {
            return Err("glob was aborted before completion (tool timeout or caller cancellation)".into());
        }
        let Ok(e) = entry else { continue };
        if !e.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let mtime = e.metadata().ok().and_then(|m| m.modified().ok()).unwrap_or(SystemTime::UNIX_EPOCH);
        found.push((mtime, display(env, sess, &root, e.path())));
    }
    found.sort();
    let paths: Vec<String> = found.into_iter().map(|(_, p)| p).collect();
    if paths.is_empty() {
        return Ok(Output::text("No files found"));
    }
    if paths.len() <= GLOB_MAX_RESULTS {
        return Ok(Output::text(paths.join("\n")));
    }
    let spill = save_spill(env, &sess.id, "glob-results.txt", &paths.join("\n"));
    let recovery = match spill {
        Some(r) => format!("Full sorted result stored at: {}. {RETRIEVAL_HINT}", r.locator),
        None => "The complete result could not be saved; narrow pattern or path to see more.".into(),
    };
    Ok(Output::text(format!(
        "{}\n\n(Showing {GLOB_MAX_RESULTS} of {} paths. {recovery})",
        paths[..GLOB_MAX_RESULTS].join("\n"),
        paths.len()
    )))
}

struct Match {
    path: String,
    line_number: usize,
    line: String,
}

/// A line kept to its first [GREP_MAX_LINE_BYTES] bytes.
fn preview(line: &str) -> String {
    if line.len() <= GREP_MAX_LINE_BYTES {
        return line.to_string();
    }
    let mut end = GREP_MAX_LINE_BYTES;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} (line truncated)", &line[..end])
}

fn format_matches(matches: &[Match]) -> String {
    let mut sections: Vec<String> = Vec::new();
    let mut i = 0;
    while i < matches.len() {
        let path = &matches[i].path;
        let mut lines = Vec::new();
        while i < matches.len() && &matches[i].path == path {
            lines.push(format!("Line {}: {}", matches[i].line_number, matches[i].line));
            i += 1;
        }
        sections.push(format!("{path}\n{}", lines.join("\n")));
    }
    sections.join("\n\n")
}

fn noun(n: usize) -> &'static str {
    if n == 1 { "match" } else { "matches" }
}

pub fn grep(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let pattern = required_str(args, "pattern")?;
    if pattern.is_empty() {
        return Err("pattern must be a non-empty string".into());
    }
    let path = arg_str(args, "path");
    if path.is_some_and(|p| p.trim().is_empty()) {
        return Err("path must be a non-empty string when given".into());
    }
    let include = arg_str(args, "include");
    if let Some(i) = include {
        if i.trim().is_empty() {
            return Err("include must be a non-empty glob when given".into());
        }
        if i.starts_with('!') {
            return Err("include must be a positive glob filter; negated patterns (\"!…\") are not supported".into());
        }
        let mut depth = 0i32;
        for c in i.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth = (depth - 1).max(0),
                ',' if depth == 0 => {
                    return Err(
                        "include must be one glob, not a comma-separated list (use {a,b} alternation instead)".into()
                    );
                }
                _ => {}
            }
        }
    }
    let re =
        regex::Regex::new(pattern).map_err(|e| format!("grep pattern rejected by ripgrep: regex parse error:\n{e}"))?;
    let root = root(env, sess, "grep", path)?;
    let mut builder = WalkBuilder::new(&root.host);
    if let Some(i) = include {
        let mut ob = OverrideBuilder::new(&root.host);
        ob.add(i).map_err(|e| format!("grep pattern rejected by ripgrep: error parsing glob '{i}': {e}"))?;
        builder.overrides(ob.build().map_err(|e| e.to_string())?);
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in builder.build() {
        if env.cancel.is_cancelled() {
            return Err("grep was aborted before completion (tool timeout or caller cancellation)".into());
        }
        let Ok(e) = entry else { continue };
        if e.file_type().is_some_and(|t| t.is_file()) {
            files.push(e.into_path());
        }
    }
    files.sort();
    let mut matches: Vec<Match> = Vec::new();
    for f in &files {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        if bytes.contains(&0) {
            continue; // a binary file: ripgrep stops at its first NUL
        }
        let shown = display(env, sess, &root, f);
        let mut lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
        if bytes.ends_with(b"\n") || bytes.is_empty() {
            lines.pop(); // the empty segment after a final newline is not a line
        }
        for (n, raw) in lines.into_iter().enumerate() {
            let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
            match std::str::from_utf8(raw) {
                Ok(s) => {
                    if re.is_match(s) {
                        matches.push(Match { path: shown.clone(), line_number: n + 1, line: preview(s) });
                    }
                }
                Err(_) => {
                    let lossy = String::from_utf8_lossy(raw);
                    if re.is_match(&lossy) {
                        matches.push(Match {
                            path: shown.clone(),
                            line_number: n + 1,
                            line: "(line is not valid UTF-8)".into(),
                        });
                    }
                }
            }
        }
    }
    let seen = matches.len();
    if seen == 0 {
        return Ok(Output::text("No matches found"));
    }
    if seen <= GREP_MAX_MATCHES {
        return Ok(Output::text(format!("Found {seen} {}\n\n{}", noun(seen), format_matches(&matches))));
    }
    let spill = save_spill(
        env,
        &sess.id,
        "grep-results.txt",
        &format!("Found {seen} {}\n\n{}", noun(seen), format_matches(&matches)),
    );
    let recovery = match spill {
        Some(r) => format!("Full grep result stored at: {}. {RETRIEVAL_HINT}", r.locator),
        None => "The complete result could not be saved; narrow pattern, path, or include to see more.".into(),
    };
    Ok(Output::text(format!(
        "Found {GREP_MAX_MATCHES} of {seen} matches\n\n{}\n\n({recovery})",
        format_matches(&matches[..GREP_MAX_MATCHES])
    )))
}

#[cfg(test)]
mod tests {
    use super::super::testenv::*;
    use super::*;
    use crate::msg::Block;
    use serde_json::json;
    use std::time::Duration;

    fn text(o: Result<Output, String>) -> String {
        match o {
            Ok(o) => match &o.blocks[0] {
                Block::Text { text } => text.clone(),
                _ => panic!(),
            },
            Err(e) => format!("Error: {e}"),
        }
    }

    fn touch(p: &Path, secs: u64) {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs)).unwrap();
    }

    #[test]
    fn the_reference() {
        let w = World::new("search");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        std::fs::create_dir_all(w.ws(".ref-tools")).unwrap();
        std::fs::create_dir_all(w.ws("new/dir")).unwrap();
        for (f, c, t) in [
            (".ref-tools/script_model.py", "x", 100),
            (".ref-tools/acp_drive.py", "y", 101),
            ("b.py", "print('hello')\n", 200),
            ("a.txt", "hi\n", 150),
            ("new/dir/c.txt", "line 1\n", 300),
        ] {
            std::fs::write(w.ws(f), c).unwrap();
            touch(&w.ws(f), t);
        }
        assert_eq!(
            text(glob(&env, &mut s, &json!({"pattern": "*.py"}))),
            ".ref-tools/script_model.py\n.ref-tools/acp_drive.py\nb.py"
        );
        assert_eq!(
            text(grep(&env, &mut s, &json!({"pattern": "hello"}))),
            "Found 1 match\n\nb.py\nLine 1: print('hello')"
        );
        assert_eq!(text(glob(&env, &mut s, &json!({"pattern": "**/*.txt"}))), "a.txt\nnew/dir/c.txt");
        assert_eq!(text(glob(&env, &mut s, &json!({"pattern": "*.zzz"}))), "No files found");
        assert_eq!(text(grep(&env, &mut s, &json!({"pattern": "zzz"}))), "No matches found");
        assert_eq!(text(glob(&env, &mut s, &json!({"pattern": "*.txt", "path": "new"}))), "new/dir/c.txt");
        assert_eq!(text(glob(&env, &mut s, &json!({"pattern": "*.txt", "path": "/workspace/new"}))), "new/dir/c.txt");
        assert!(
            text(grep(&env, &mut s, &json!({"pattern": "("})))
                .starts_with("Error: grep pattern rejected by ripgrep: regex parse error")
        );
        assert!(
            text(grep(&env, &mut s, &json!({"pattern": "x", "include": "a,b"}))).contains("not a comma-separated list")
        );
    }

    #[test]
    fn over_cap() {
        let w = World::new("search-cap");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        for i in 0..120 {
            std::fs::write(w.ws(&format!("f{i:03}.md")), "needle\n").unwrap();
        }
        let g = text(glob(&env, &mut s, &json!({"pattern": "*.md"})));
        assert!(g.contains("(Showing 100 of 120 paths. Full sorted result stored at: /tmp/geode-spill-"));
        for i in 0..2 {
            std::fs::write(w.ws(&format!("many{i}.txt")), "needle\n".repeat(140)).unwrap();
        }
        let r = text(grep(&env, &mut s, &json!({"pattern": "needle", "include": "*.txt"})));
        assert!(r.starts_with("Found 250 of 280 matches\n\n"));
        assert!(r.contains("Full grep result stored at: /tmp/geode-spill-"));
    }
}
