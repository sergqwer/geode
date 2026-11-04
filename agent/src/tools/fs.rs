//! read, write, edit, read_image: on the chat's own folders directly (no proot), the paths the model sees being the
//! Linux's. Ported from @deepseek-ai/dsh-tool-fs + dsh-fs-local + dsh-fs-observation-policy: the same windows and
//! caps, the same words, and the same rule - a file is read before it is replaced or edited, and not changed under
//! the agent since.

use super::{Env, Output, arg_positive, required_str};
use crate::msg::Block;
use crate::paths::Resolved;
use crate::session::{Observed, Session};
use crate::util::{js_len, js_slice_head};
use base64::Engine;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;

const READ_LIMIT: usize = 2000;
const READ_MAX_LINE_LENGTH: usize = 2000;
const READ_MAX_BYTES: usize = 50 * 1024;
const BINARY_SAMPLE_BYTES: usize = 8192;
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// A file's version as the observation policy compares it: its size and modification time.
pub fn version_of(path: &Path) -> Option<String> {
    let m = fs::metadata(path).ok()?;
    let mtime = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    Some(format!("{}:{mtime}", m.len()))
}

fn resolve(env: &Env, sess: &Session, path: &str, follow_last: bool) -> Result<Resolved, String> {
    if path.trim().is_empty() {
        return Err("file_path must be a non-empty string".into());
    }
    env.ns.resolve(path, &sess.cwd, follow_last)
}

/// The target of a read: a regular file (a missing one is remembered as absent).
fn regular(sess: &mut Session, r: &Resolved) -> Result<fs::Metadata, String> {
    match fs::metadata(&r.host) {
        Err(_) => {
            sess.observed.insert(r.guest.clone(), Observed::Absent);
            Err(format!("cannot read \"{}\": not found", r.guest))
        }
        Ok(m) if !m.is_file() => Err(format!("cannot read \"{}\": not a regular file", r.guest)),
        Ok(m) => Ok(m),
    }
}

fn decode(bytes: Vec<u8>, verb: &str, display: &str) -> Result<String, String> {
    if bytes[..bytes.len().min(BINARY_SAMPLE_BYTES)].contains(&0) {
        return Err(format!("cannot {verb} \"{display}\": binary file"));
    }
    String::from_utf8(bytes).map_err(|_| format!("cannot {verb} \"{display}\": invalid UTF-8 text"))
}

struct Line {
    number: usize,
    text: String,
}

fn truncate_line(line: &str) -> String {
    if js_len(line) > READ_MAX_LINE_LENGTH {
        format!("{}... (line truncated to {READ_MAX_LINE_LENGTH} chars)", js_slice_head(line, READ_MAX_LINE_LENGTH))
    } else {
        line.to_string()
    }
}

pub fn read(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let file_path = required_str(args, "file_path")?;
    if file_path.trim().is_empty() {
        return Err("file_path must be a non-empty string".into());
    }
    let offset = arg_positive(args, "offset")?.unwrap_or(1);
    let limit = arg_positive(args, "limit")?.unwrap_or(READ_LIMIT);
    if limit > READ_LIMIT {
        return Err(format!("limit must be less than or equal to {READ_LIMIT}"));
    }
    let r = resolve(env, sess, file_path, true)?;
    regular(sess, &r)?;
    let bytes = fs::read(&r.host).map_err(|e| format!("cannot read \"{}\": {e}", r.guest))?;
    let text = decode(bytes, "read", &r.guest)?;
    // the window: lines split on \n, a trailing \r dropped, each capped; the bytes shown capped
    let mut lines: Vec<Line> = Vec::new();
    let mut total = 0usize;
    let mut out_bytes = 0usize;
    let mut by_bytes = false;
    let segments: Vec<&str> = text.split('\n').collect();
    let n = segments.len();
    for (i, raw) in segments.into_iter().enumerate() {
        if i == n - 1 && raw.is_empty() {
            break; // the text after the last newline, when there is none
        }
        total += 1;
        if by_bytes || total < offset || lines.len() >= limit {
            continue;
        }
        let raw = raw.strip_suffix('\r').unwrap_or(raw);
        let t = truncate_line(raw);
        let bytes = t.len() + usize::from(!lines.is_empty());
        if out_bytes + bytes > READ_MAX_BYTES {
            by_bytes = true;
            continue;
        }
        out_bytes += bytes;
        lines.push(Line { number: total, text: t });
    }
    if !by_bytes && offset > total && !(total == 0 && offset == 1) {
        return Err(format!("offset {offset} is out of range for \"{}\" ({total} lines)", r.guest));
    }
    if let Some(v) = super::fs::version_of(&r.host) {
        sess.observed.insert(r.guest.clone(), Observed::Present(v));
    }
    let end = lines.last().map(|l| l.number).unwrap_or(offset.saturating_sub(1));
    let truncated_by_bytes = lines.len() < limit && end < total;
    let footer = if truncated_by_bytes {
        format!("(Output capped. Showing lines {offset}-{end}. Use offset={} to continue.)", end + 1)
    } else if end < total {
        format!("(Showing lines {offset}-{end} of {total}. Use offset={} to continue.)", end + 1)
    } else {
        format!("(End of file - total {total} lines)")
    };
    let body = if lines.is_empty() {
        footer
    } else {
        let numbered: Vec<String> = lines.iter().map(|l| format!("{}: {}", l.number, l.text)).collect();
        format!("{}\n\n{footer}", numbered.join("\n"))
    };
    Ok(Output::text(format!("<path>{}</path>\n<type>file</type>\n<content>\n{body}\n</content>", r.guest)))
}

/// Writes [content] beside [path] and renames it over (the file is whole or as it was; no hard links).
fn write_atomic(path: &Path, content: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", crate::util::hex(&crate::util::random_bytes::<4>())));
    let mode = fs::metadata(path).ok().map(|m| m.permissions());
    let r = (|| {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(content)?;
        f.sync_all()?;
        drop(f);
        if let Some(p) = mode {
            fs::set_permissions(&tmp, p)?;
        }
        fs::rename(&tmp, path)
    })();
    if r.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    r
}

/// A file in a folder of the phone the user gave read-only is not changed.
fn writable(env: &Env, verb: &str, r: &Resolved) -> Result<(), String> {
    match env.ns.read_only(&r.guest) {
        Some(folder) => Err(format!(
            "cannot {verb} \"{}\": {folder} is a folder of the user's phone given to you read-only — write your result in /workspace instead",
            r.guest
        )),
        None => Ok(()),
    }
}

fn not_observed(display: &str) -> String {
    format!("cannot modify \"{display}\": file has not been read — read the file, then retry")
}

pub fn write(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let file_path = required_str(args, "file_path")?;
    let content = required_str(args, "content")?;
    if file_path.trim().is_empty() {
        return Err("file_path must be a non-empty string".into());
    }
    let r = resolve(env, sess, file_path, false)?;
    writable(env, "write", &r)?;
    let existing = fs::symlink_metadata(&r.host).ok();
    if let Some(m) = &existing
        && m.is_dir()
    {
        return Err(format!("cannot write \"{}\": not a regular file", r.guest));
    }
    match sess.observed.get(&r.guest) {
        Some(Observed::Present(v)) => match &existing {
            None => {
                return Err(format!(
                    "cannot write \"{}\": file no longer exists — re-read the file, then retry",
                    r.guest
                ));
            }
            Some(_) if version_of(&r.host).as_ref() != Some(v) => {
                return Err(format!(
                    "cannot write \"{}\": file changed since it was read — re-read the file, then retry",
                    r.guest
                ));
            }
            _ => {}
        },
        _ => {
            if existing.is_some() {
                return Err(not_observed(&r.guest));
            }
        }
    }
    let created = existing.is_none();
    write_atomic(&r.host, content.as_bytes()).map_err(|e| format!("cannot write \"{}\": {e}", r.guest))?;
    if let Some(v) = version_of(&r.host) {
        sess.observed.insert(r.guest.clone(), Observed::Present(v));
    }
    Ok(Output::text(format!(
        "<path>{}</path>\n<type>file</type>\n<content>\n{} file\n</content>",
        r.guest,
        if created { "Created" } else { "Updated" }
    )))
}

pub fn edit(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let file_path = required_str(args, "file_path")?;
    let old = required_str(args, "old_string")?;
    let new = required_str(args, "new_string")?;
    let replace_all = args.get("replace_all").and_then(Value::as_bool).unwrap_or(false);
    if file_path.trim().is_empty() {
        return Err("file_path must be a non-empty string".into());
    }
    if old.is_empty() {
        return Err("old_string must be a non-empty string".into());
    }
    if old == new {
        return Err("old_string and new_string must differ".into());
    }
    let r = resolve(env, sess, file_path, true)?;
    writable(env, "edit", &r)?;
    let expected = match sess.observed.get(&r.guest) {
        None => return Err(not_observed(&r.guest)),
        Some(Observed::Absent) => return Err(format!("cannot edit \"{}\": not found", r.guest)),
        Some(Observed::Present(v)) => v.clone(),
    };
    let stale =
        || format!("cannot edit \"{}\": file changed since it was read — re-read the file, then retry", r.guest);
    let meta = fs::metadata(&r.host).map_err(|_| stale())?;
    if !meta.is_file() {
        return Err(format!("cannot edit \"{}\": not a regular file", r.guest));
    }
    if version_of(&r.host).as_ref() != Some(&expected) {
        return Err(stale());
    }
    let bytes = fs::read(&r.host).map_err(|e| format!("cannot edit \"{}\": {e}", r.guest))?;
    if bytes.contains(&0) {
        return Err(format!("cannot edit \"{}\": binary file", r.guest));
    }
    let raw = String::from_utf8(bytes).map_err(|_| format!("cannot edit \"{}\": invalid UTF-8 text", r.guest))?;
    let crlf = {
        let sample = js_slice_head(&raw, 4096);
        let crlf = sample.matches("\r\n").count();
        crlf > sample.matches('\n').count() - crlf
    };
    let content = raw.replace("\r\n", "\n");
    let old_n = old.replace("\r\n", "\n");
    let new_n = new.replace("\r\n", "\n");
    let count = content.matches(old_n.as_str()).count();
    if count == 0 {
        return Err(format!("old_string was not found in \"{}\"", r.guest));
    }
    if !replace_all && count > 1 {
        return Err(format!(
            "old_string matched {count} times in \"{}\"; provide a more specific old_string or set replace_all to true",
            r.guest
        ));
    }
    let mut edited = content.replace(old_n.as_str(), &new_n);
    if crlf {
        edited = edited.split('\n').collect::<Vec<_>>().join("\r\n");
    }
    write_atomic(&r.host, edited.as_bytes()).map_err(|e| format!("cannot edit \"{}\": {e}", r.guest))?;
    if let Some(v) = version_of(&r.host) {
        sess.observed.insert(r.guest.clone(), Observed::Present(v));
    }
    Ok(Output::text(if replace_all {
        format!("The file {} has been updated. All occurrences were successfully replaced.", r.guest)
    } else {
        format!("The file {} has been updated successfully.", r.guest)
    }))
}

fn sniff(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]) {
        Some("image/png")
    } else if data.starts_with(&[255, 216, 255]) {
        Some("image/jpeg")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

pub fn read_image(env: &Env, sess: &mut Session, args: &Value) -> Result<Output, String> {
    let file_path = required_str(args, "file_path")?;
    if file_path.trim().is_empty() {
        return Err("file_path must be a non-empty string".into());
    }
    let ext = Path::new(file_path)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let declared = match ext.as_str() {
        ".png" => Some("image/png"),
        ".jpg" | ".jpeg" => Some("image/jpeg"),
        ".webp" => Some("image/webp"),
        ".gif" => Some("image/gif"),
        _ => None,
    };
    if declared.is_none() && !ext.is_empty() {
        return Err(format!(
            "cannot read \"{file_path}\": the {ext} extension does not declare a supported image format; read_image accepts PNG/JPEG/WebP/GIF files, including extension-less files in those formats"
        ));
    }
    if !env.cfg.image_input {
        return Err(format!(
            "cannot read \"{file_path}\" as an image: model \"{}\" does not declare image input; switch to an image-capable model to read images",
            env.cfg.model
        ));
    }
    let r = resolve(env, sess, file_path, true)?;
    let meta = regular(sess, &r)?;
    if meta.len() as usize > MAX_IMAGE_BYTES {
        return Err(format!(
            "cannot read \"{}\": the image cannot be stored within the deployment's byte limits; downscale the image and read the smaller copy",
            r.guest
        ));
    }
    let data = fs::read(&r.host).map_err(|e| format!("cannot read \"{}\": {e}", r.guest))?;
    let sniffed = sniff(&data);
    let mime = match (declared, sniffed) {
        (_, None) => {
            return Err(format!(
                "cannot read \"{}\": the file content is not a supported image format; read_image accepts PNG/JPEG/WebP/GIF",
                r.guest
            ));
        }
        (Some(d), Some(s)) if d != s => {
            return Err(format!(
                "cannot read \"{}\": the {ext} extension declares {d}, but the bytes use a different image format; rename the file to match its actual format if it is PNG/JPEG/WebP/GIF, or convert it to one of those formats",
                r.guest
            ));
        }
        (_, Some(s)) => s,
    };
    let size = imagesize::blob_size(&data).map_err(|_| {
        format!("cannot read \"{}\": the bytes do not decode as a supported PNG/JPEG/WebP/GIF image; the file may be truncated or corrupt", r.guest)
    })?;
    if let Some(v) = version_of(&r.host) {
        sess.observed.insert(r.guest.clone(), Observed::Present(v));
    }
    let name = Path::new(&r.guest).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let id = format!("sha256:{}", crate::util::hex(&Sha256::digest(&data)));
    let text = format!(
        "<path>{}</path>\n<type>image</type>\n<content>\n{mime} image, {}x{} px, {} bytes\n</content>",
        r.guest,
        size.width,
        size.height,
        data.len()
    );
    Ok(Output {
        blocks: vec![
            Block::text(text),
            Block::Image {
                mime: mime.into(),
                data: base64::engine::general_purpose::STANDARD.encode(&data),
                width: size.width as u32,
                height: size.height as u32,
                name,
                path: r.guest,
                id,
            },
        ],
        is_error: false,
        notices: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::testenv::*;
    use super::*;
    use serde_json::json;

    fn text(o: &Result<Output, String>) -> String {
        match o {
            Ok(o) => match &o.blocks[0] {
                Block::Text { text } => text.clone(),
                _ => panic!(),
            },
            Err(e) => format!("Error: {e}"),
        }
    }

    #[test]
    fn the_reference_sequence() {
        let w = World::new("fs");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        std::fs::write(w.ws("a.txt"), "hi\n").unwrap();
        assert_eq!(
            text(&read(&env, &mut s, &json!({"file_path": "/workspace/a.txt"}))),
            "<path>/workspace/a.txt</path>\n<type>file</type>\n<content>\n1: hi\n\n(End of file - total 1 lines)\n</content>"
        );
        assert_eq!(
            text(&write(&env, &mut s, &json!({"file_path": "/workspace/b.py", "content": "print('hi')\n"}))),
            "<path>/workspace/b.py</path>\n<type>file</type>\n<content>\nCreated file\n</content>"
        );
        assert_eq!(
            text(&edit(
                &env,
                &mut s,
                &json!({"file_path": "/workspace/b.py", "old_string": "hi", "new_string": "hello"})
            )),
            "The file /workspace/b.py has been updated successfully."
        );
        assert_eq!(std::fs::read_to_string(w.ws("b.py")).unwrap(), "print('hello')\n");
        assert_eq!(
            text(&read(&env, &mut s, &json!({"file_path": "/workspace/missing.txt"}))),
            "Error: cannot read \"/workspace/missing.txt\": not found"
        );
        assert_eq!(
            text(&edit(&env, &mut s, &json!({"file_path": "/workspace/b.py", "old_string": "zzz", "new_string": "y"}))),
            "Error: old_string was not found in \"/workspace/b.py\""
        );
        assert_eq!(
            text(&write(
                &env,
                &mut s,
                &json!({"file_path": "/workspace/new/dir/c.txt", "content": "line 1\nline 2\n"})
            )),
            "<path>/workspace/new/dir/c.txt</path>\n<type>file</type>\n<content>\nCreated file\n</content>"
        );
        assert_eq!(
            text(&read(&env, &mut s, &json!({"file_path": "/workspace/new/dir/c.txt"}))),
            "<path>/workspace/new/dir/c.txt</path>\n<type>file</type>\n<content>\n1: line 1\n2: line 2\n\n(End of file - total 2 lines)\n</content>"
        );
        assert_eq!(
            text(&write(&env, &mut s, &json!({"file_path": "/workspace/a.txt", "content": "overwritten\n"}))),
            "<path>/workspace/a.txt</path>\n<type>file</type>\n<content>\nUpdated file\n</content>"
        );
        assert_eq!(
            text(&edit(
                &env,
                &mut s,
                &json!({"file_path": "/workspace/missing.py", "old_string": "a", "new_string": "b"})
            )),
            "Error: cannot modify \"/workspace/missing.py\": file has not been read — read the file, then retry"
        );
    }

    #[test]
    fn observation_rules() {
        let w = World::new("fs-obs");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        std::fs::write(w.ws("x.txt"), "one\r\ntwo\r\n").unwrap();
        assert!(
            text(&write(&env, &mut s, &json!({"file_path": "x.txt", "content": "z"})))
                .contains("file has not been read")
        );
        read(&env, &mut s, &json!({"file_path": "x.txt"})).unwrap();
        assert!(
            edit(&env, &mut s, &json!({"file_path": "x.txt", "old_string": "one\ntwo", "new_string": "1\n2"})).is_ok()
        );
        assert_eq!(std::fs::read_to_string(w.ws("x.txt")).unwrap(), "1\r\n2\r\n");
        assert!(
            text(&edit(&env, &mut s, &json!({"file_path": "x.txt", "old_string": "1", "new_string": "1"})))
                .contains("must differ")
        );
        std::fs::write(w.ws("y.txt"), "a a").unwrap();
        read(&env, &mut s, &json!({"file_path": "y.txt"})).unwrap();
        assert_eq!(
            text(&edit(&env, &mut s, &json!({"file_path": "y.txt", "old_string": "a", "new_string": "b"}))),
            "Error: old_string matched 2 times in \"/workspace/y.txt\"; provide a more specific old_string or set replace_all to true"
        );
        assert_eq!(
            text(&edit(
                &env,
                &mut s,
                &json!({"file_path": "y.txt", "old_string": "a", "new_string": "b", "replace_all": true})
            )),
            "The file /workspace/y.txt has been updated. All occurrences were successfully replaced."
        );
    }

    #[test]
    fn windows() {
        let w = World::new("fs-win");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        let body: String = (1..=10).map(|n| format!("l{n}\n")).collect();
        std::fs::write(w.ws("t.txt"), body).unwrap();
        assert!(
            text(&read(&env, &mut s, &json!({"file_path": "t.txt", "offset": 3, "limit": 2})))
                .ends_with("3: l3\n4: l4\n\n(Showing lines 3-4 of 10. Use offset=5 to continue.)\n</content>")
        );
        assert_eq!(
            text(&read(&env, &mut s, &json!({"file_path": "t.txt", "offset": 30}))),
            "Error: offset 30 is out of range for \"/workspace/t.txt\" (10 lines)"
        );
        std::fs::write(w.ws("e.txt"), "").unwrap();
        assert!(
            text(&read(&env, &mut s, &json!({"file_path": "e.txt"})))
                .contains("<content>\n(End of file - total 0 lines)\n</content>")
        );
        std::fs::write(w.ws("bin"), b"a\0b").unwrap();
        assert_eq!(
            text(&read(&env, &mut s, &json!({"file_path": "bin"}))),
            "Error: cannot read \"/workspace/bin\": binary file"
        );
    }

    #[test]
    fn images() {
        let w = World::new("fs-img");
        let mut s = w.session();
        let env = w.env(&NoWeb);
        let png = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
            .unwrap();
        std::fs::write(w.ws("img.png"), &png).unwrap();
        let o = read_image(&env, &mut s, &json!({"file_path": "/workspace/img.png"})).unwrap();
        assert_eq!(
            text(&Ok(o.clone())),
            format!(
                "<path>/workspace/img.png</path>\n<type>image</type>\n<content>\nimage/png image, 1x1 px, {} bytes\n</content>",
                png.len()
            )
        );
        assert!(matches!(o.blocks[1], Block::Image { width: 1, height: 1, .. }));
        assert!(
            text(&read_image(&env, &mut s, &json!({"file_path": "a.txt"})))
                .contains("the .txt extension does not declare")
        );
    }
}
