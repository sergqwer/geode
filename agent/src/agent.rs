//! A turn: the user's message in, then steps - context the harness adds, a request, the model's answer streamed out,
//! its tool calls run one by one - until the model answers without calling a tool. Ported from @deepseek-ai/dsh-agent
//! and dsh-agent-loop, with dsh-acp's updates (thought and message chunks, usage, tool call and tool result).

use crate::compact;
use crate::config::Config;
use crate::linux::{Cancel, Shell};
use crate::mcp::McpHub;
use crate::model::{Delta, Model};
use crate::msg::{Block, Message, Source, completion_budget, occupancy, request_body};
use crate::paths::Namespace;
use crate::prompt;
use crate::session::Session;
use crate::skills::{self, SkillSet};
use crate::tools::{self, Env, web::Web};
use crate::util::uuid;
use serde_json::{Value, json};

/// Where a turn's updates go (ACP session/update notifications).
pub trait Sink: Send + Sync {
    fn update(&self, session_id: &str, update: Value);
}

/// A stand-in for the tools (a replay of recorded results: the parity tests).
pub trait ToolRunner: Send + Sync {
    fn run(&self, call: &crate::msg::ToolCall) -> Option<tools::Output>;
}

/// What a turn runs with.
pub struct Runtime<'a> {
    pub cfg: &'a Config,
    pub ns: &'a Namespace,
    pub shell: &'a dyn Shell,
    pub web: &'a dyn Web,
    pub model: &'a dyn Model,
    pub sink: &'a dyn Sink,
    pub tool_runner: Option<&'a dyn ToolRunner>,
}

impl Runtime<'_> {
    pub fn env<'b>(&'b self, skills: &'b SkillSet, mcp: &'b McpHub, cancel: &'b Cancel) -> Env<'b> {
        Env { cfg: self.cfg, ns: self.ns, shell: self.shell, web: self.web, skills, mcp, cancel }
    }
}

/// The system prompt at the head of the conversation, kept current (an MCP server's instructions may change it).
pub fn ensure_system(rt: &Runtime, sess: &mut Session, mcp: &McpHub) {
    let text = prompt::system_prompt(rt.cfg, mcp);
    match sess.messages.first() {
        Some(Message::System { text: t }) if *t == text => {}
        Some(Message::System { .. }) => {
            let mut m = sess.messages.clone();
            m[0] = Message::System { text };
            let _ = sess.replace_messages(m);
        }
        _ => {
            if sess.messages.is_empty() {
                sess.push(Message::System { text });
            } else {
                let mut m = vec![Message::System { text }];
                m.extend(sess.messages.clone());
                let _ = sess.replace_messages(m);
            }
        }
    }
}

/// The context the harness adds before a step, when it changed: the runtime context, the workspace instructions,
/// the skill catalog.
fn pre_step(rt: &Runtime, sess: &mut Session, skills: &SkillSet) {
    if let Some(m) = prompt::runtime_context_for_step(&sess.cwd, &sess.messages) {
        sess.push(m);
    }
    if let Some(m) = prompt::instructions_for_step(rt.ns, &sess.cwd, &sess.messages) {
        sess.push(m);
    }
    if rt.cfg.skills_tool
        && let Some(m) = skills::catalog_for_step(skills, &sess.messages)
    {
        sess.push(m);
    }
}

fn acp_content(b: &Block) -> Value {
    match b {
        Block::Text { text } => json!({"type": "text", "text": text}),
        Block::Image { mime, data, .. } => json!({"type": "image", "data": data, "mimeType": mime}),
        Block::Audio { mime, data, .. } => json!({"type": "audio", "data": data, "mimeType": mime}),
        Block::Video { frames, audio, ms, .. } => {
            let frames: Vec<Value> =
                frames.iter().map(|f| json!({"data": f.data, "mimeType": f.mime, "second": f.second})).collect();
            let mut v = json!({"type": "video", "frames": frames, "durationMs": ms});
            if !audio.is_empty() {
                v["audio"] = json!({"data": audio, "mimeType": "audio/wav"});
            }
            v
        }
    }
}

/// The user's message: its text and images (ACP prompt blocks).
pub fn prompt_blocks(prompt: &Value) -> Vec<Block> {
    let mut out = Vec::new();
    for b in prompt.as_array().cloned().unwrap_or_default() {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => out.push(Block::text(b.get("text").and_then(Value::as_str).unwrap_or(""))),
            Some("image") => {
                let data = b.get("data").and_then(Value::as_str).unwrap_or("").to_string();
                let mime = b.get("mimeType").and_then(Value::as_str).unwrap_or("image/png").to_string();
                let decoded =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &data).unwrap_or_default();
                let (w, h) =
                    imagesize::blob_size(&decoded).map(|s| (s.width as u32, s.height as u32)).unwrap_or((0, 0));
                let id = format!(
                    "sha256:{}",
                    crate::util::hex(&sha2::Digest::finalize(<sha2::Sha256 as sha2::Digest>::new_with_prefix(
                        &decoded
                    )))
                );
                // a copy the client keeps in the guest (ACP's uri, file:///...): the model is told its path, which its
                // tools read (without one it was left to guess a path)
                let path = b.get("uri").and_then(Value::as_str).and_then(|u| u.strip_prefix("file://")).unwrap_or("");
                let name = path.rsplit('/').next().unwrap_or("").to_string();
                out.push(Block::Image { mime, data, width: w, height: h, name, path: path.to_string(), id });
            }
            Some("audio") => {
                let data = b.get("data").and_then(Value::as_str).unwrap_or("").to_string();
                let mime = b.get("mimeType").and_then(Value::as_str).unwrap_or("audio/wav").to_string();
                let decoded =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &data).unwrap_or_default();
                let id = format!(
                    "sha256:{}",
                    crate::util::hex(&sha2::Digest::finalize(<sha2::Sha256 as sha2::Digest>::new_with_prefix(
                        &decoded
                    )))
                );
                let path = b.get("uri").and_then(Value::as_str).and_then(|u| u.strip_prefix("file://")).unwrap_or("");
                let name = path.rsplit('/').next().unwrap_or("").to_string();
                out.push(Block::Audio { mime, data, ms: wav_ms(&decoded), name, path: path.to_string(), id });
            }
            // Geode's own block (no ACP type carries a video): the app's frames of it, its sound, its length
            Some("video") => {
                let mut digest = <sha2::Sha256 as sha2::Digest>::new();
                let mut frames = Vec::new();
                for fr in b.get("frames").and_then(Value::as_array).into_iter().flatten() {
                    let data = fr.get("data").and_then(Value::as_str).unwrap_or("").to_string();
                    if data.is_empty() {
                        continue;
                    }
                    sha2::Digest::update(&mut digest, data.as_bytes());
                    let mime = fr.get("mimeType").and_then(Value::as_str).unwrap_or("image/jpeg").to_string();
                    let second = fr.get("second").and_then(Value::as_u64).unwrap_or(0).min(u32::MAX as u64) as u32;
                    frames.push(crate::msg::VideoFrame { mime, data, second });
                }
                if frames.is_empty() {
                    continue;
                }
                let audio = b.get("audio").and_then(|a| a.get("data")).and_then(Value::as_str).unwrap_or("").to_string();
                sha2::Digest::update(&mut digest, audio.as_bytes());
                let ms = b.get("durationMs").and_then(Value::as_u64).unwrap_or(0).min(u32::MAX as u64) as u32;
                let id = format!("sha256:{}", crate::util::hex(&sha2::Digest::finalize(digest)));
                let path = b.get("uri").and_then(Value::as_str).and_then(|u| u.strip_prefix("file://")).unwrap_or("");
                let name = path.rsplit('/').next().unwrap_or("").to_string();
                out.push(Block::Video { frames, audio, ms, name, path: path.to_string(), id });
            }
            Some("resource_link") => {
                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                let uri = b.get("uri").and_then(Value::as_str).unwrap_or("");
                out.push(Block::text(format!("Resource link: {name} ({uri})")));
            }
            Some("resource") => {
                if let Some(t) = b.pointer("/resource/text").and_then(Value::as_str) {
                    out.push(Block::text(t));
                }
            }
            _ => {}
        }
    }
    out
}

/// A WAV's length in milliseconds from its fmt and data chunks (0: not a WAV, or not known).
fn wav_ms(b: &[u8]) -> u32 {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return 0;
    }
    let (mut at, mut bytes_per_s, mut data) = (12usize, 0u64, 0u64);
    while at + 8 <= b.len() {
        let len = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        let body = at + 8;
        match &b[at..at + 4] {
            b"fmt " if body + 12 <= b.len() => {
                bytes_per_s = u32::from_le_bytes([b[body + 8], b[body + 9], b[body + 10], b[body + 11]]) as u64
            }
            b"data" => {
                // a streamed WAV says 0 (or more than there is): the rest of the file, and no chunk after it
                let rest = b.len().saturating_sub(body);
                let streamed = len == 0 || len > rest;
                data = (if streamed { rest } else { len }) as u64;
                if streamed {
                    break;
                }
            }
            _ => {}
        }
        if data > 0 && bytes_per_s > 0 {
            break;
        }
        at = body.saturating_add(len).saturating_add(len & 1);
    }
    if bytes_per_s == 0 { 0 } else { (data * 1000 / bytes_per_s).min(u32::MAX as u64) as u32 }
}

pub fn prompt_text(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// One turn. Returns how it ended (ACP stopReason: end_turn, max_tokens, cancelled).
pub fn run_turn(
    rt: &Runtime,
    sess: &mut Session,
    mcp: &McpHub,
    blocks: Vec<Block>,
    cancel: &Cancel,
) -> Result<String, String> {
    ensure_system(rt, sess, mcp);
    let text = prompt_text(&blocks);
    if sess.title.is_empty() && !text.trim().is_empty() {
        sess.title = text.trim().lines().next().unwrap_or("").chars().take(80).collect();
        sess.save_state();
    }
    let mut skills = SkillSet::discover(rt.ns, &sess.cwd);
    sess.push(Message::User { blocks, source: Source::User });
    pre_step(rt, sess, &skills);
    // "/name": the user loads a skill directly
    if rt.cfg.skills_tool {
        for name in skills::invoked_names(&text) {
            if let Some(s) = skills.get(&name).filter(|s| s.user_invocable) {
                sess.push(Message::user_text(skills::render_skill(s), Source::SkillInvocation));
            }
        }
    }
    let mut overflow_retried = false;
    loop {
        if cancel.is_cancelled() {
            return Ok("cancelled".into());
        }
        let env = rt.env(&skills, mcp, cancel);
        let specs = tools::specs(&env);
        compact::auto(rt, sess, &specs, cancel);
        let budget = completion_budget(rt.cfg.context_window, rt.cfg.max_tokens, &sess.messages, &specs, rt.cfg.input());
        let body = request_body(&rt.cfg.model, &sess.messages, &specs, budget, rt.cfg.input());
        let message_id = uuid();
        let sid = sess.id.clone();
        let mut on_delta = |d: Delta| {
            let (kind, t) = match d {
                Delta::Text(t) => ("agent_message_chunk", t),
                Delta::Reasoning(t) => ("agent_thought_chunk", t),
            };
            rt.sink.update(
                &sid,
                json!({"sessionUpdate": kind, "messageId": message_id, "content": {"type": "text", "text": t}}),
            );
        };
        let answer = match rt.model.stream(&body, cancel, &mut on_delta) {
            Ok(a) => a,
            Err(_) if cancel.is_cancelled() => return Ok("cancelled".into()),
            Err(e) if !overflow_retried && compact::is_overflow(&e) => {
                // the conversation no longer fits: compacted, then the same step again
                overflow_retried = true;
                if compact::compact(rt, sess, &specs, cancel, None).is_ok() {
                    continue;
                }
                return Err(e);
            }
            Err(e) => return Err(e),
        };
        let calls = answer.tool_calls.clone();
        sess.push(Message::Assistant {
            id: message_id.clone(),
            reasoning: answer.reasoning,
            reasoning_field: answer.reasoning_field,
            text: answer.text,
            tool_calls: calls.clone(),
            usage: answer.usage,
        });
        if answer.usage.is_some() {
            rt.sink.update(&sess.id, json!({"sessionUpdate": "usage_update", "used": occupancy(&sess.messages, &specs), "size": rt.cfg.context_window}));
        }
        if answer.finish_reason == "length" && calls.is_empty() {
            return Ok("max_tokens".into());
        }
        if calls.is_empty() {
            return Ok("end_turn".into());
        }
        let mut notices = Vec::new();
        for (i, call) in calls.iter().enumerate() {
            let raw: Value =
                serde_json::from_str(&call.arguments).unwrap_or_else(|_| Value::String(call.arguments.clone()));
            rt.sink.update(&sess.id, json!({"sessionUpdate": "tool_call", "toolCallId": call.id, "title": call.name, "kind": "other", "status": "in_progress", "rawInput": raw}));
            let out = if cancel.is_cancelled() {
                tools::Output::error("tool call skipped: the turn was cancelled")
            } else {
                rt.tool_runner.and_then(|r| r.run(call)).unwrap_or_else(|| tools::execute(&env, sess, call))
            };
            let content: Vec<Value> =
                out.blocks.iter().map(|b| json!({"type": "content", "content": acp_content(b)})).collect();
            rt.sink.update(&sess.id, json!({"sessionUpdate": "tool_call_update", "toolCallId": call.id, "status": if out.is_error { "failed" } else { "completed" }, "content": content}));
            sess.push(Message::Tool {
                call_id: call.id.clone(),
                name: call.name.clone(),
                blocks: out.blocks,
                is_error: out.is_error,
            });
            notices.extend(out.notices);
            if cancel.is_cancelled() {
                // the calls not run still get results: the conversation keeps every call answered
                for rest in &calls[i + 1..] {
                    sess.push(Message::Tool {
                        call_id: rest.id.clone(),
                        name: rest.name.clone(),
                        blocks: vec![Block::text("Error: tool call skipped: the turn was cancelled")],
                        is_error: true,
                    });
                }
                for n in notices {
                    sess.push(n);
                }
                return Ok("cancelled".into());
            }
        }
        for n in notices {
            sess.push(n);
        }
        if answer.finish_reason == "length" {
            return Ok("max_tokens".into());
        }
        skills = SkillSet::discover(rt.ns, &sess.cwd);
        pre_step(rt, sess, &skills);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_image_takes_its_guest_path() {
        // a 1x1 PNG
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let v = json!([
            {"type": "image", "mimeType": "image/png", "data": png, "uri": "file:///workspace/.geode/images/a.png"},
            {"type": "image", "mimeType": "image/png", "data": png},
            {"type": "text", "text": "hi"}
        ]);
        let b = prompt_blocks(&v);
        match &b[0] {
            Block::Image { path, name, width, .. } => {
                assert_eq!(path, "/workspace/.geode/images/a.png");
                assert_eq!(name, "a.png");
                assert_eq!(*width, 1);
            }
            _ => panic!("not an image"),
        }
        match &b[1] {
            Block::Image { path, name, .. } => assert!(path.is_empty() && name.is_empty()),
            _ => panic!("not an image"),
        }
    }

    #[test]
    fn prompt_audio_knows_its_length() {
        // a WAV of 1.5 s, 16 kHz mono PCM16 (the samples left out: the length is the data chunk's)
        let mut wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        wav.extend(16u32.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend(16000u32.to_le_bytes());
        wav.extend(32000u32.to_le_bytes());
        wav.extend(2u16.to_le_bytes());
        wav.extend(16u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend(48000u32.to_le_bytes());
        wav.extend(vec![0u8; 48000]);
        let data = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &wav);
        let v = json!([
            {"type": "audio", "mimeType": "audio/wav", "data": data, "uri": "file:///workspace/.geode/audio/v.wav"},
            {"type": "text", "text": "hi"}
        ]);
        let b = prompt_blocks(&v);
        match &b[0] {
            Block::Audio { ms, name, path, mime, .. } => {
                assert_eq!(*ms, 1500);
                assert_eq!(name, "v.wav");
                assert_eq!(path, "/workspace/.geode/audio/v.wav");
                assert_eq!(mime, "audio/wav");
            }
            _ => panic!("not a clip"),
        }
        let user = crate::msg::Message::User { blocks: b, source: crate::msg::Source::User };
        let req = crate::msg::openai_messages(&[user], true);
        let parts = req[0]["content"].as_array().unwrap();
        assert!(parts[0]["text"].as_str().unwrap().contains("1.5 s"), "{}", parts[0]);
        assert_eq!(parts[1]["type"], "input_audio");
        assert_eq!(parts[1]["input_audio"]["format"], "wav");
        assert_eq!(parts[1]["input_audio"]["data"].as_str().unwrap(), data);
        assert_eq!(wav_ms(b"not a wav"), 0);
        // a streamed WAV (its data's length 0): the rest of the file
        let mut streamed = wav.clone();
        streamed[40..44].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(wav_ms(&streamed), 1500);
    }

    #[test]
    fn prompt_video_is_its_frames_and_sound() {
        let v = json!([
            {"type": "video", "durationMs": 6000, "frames": [
                {"data": "AAAA", "mimeType": "image/jpeg", "second": 0},
                {"data": "BBBB", "mimeType": "image/jpeg", "second": 1}],
             "audio": {"data": "CCCC", "mimeType": "audio/wav"}},
            {"type": "video", "frames": []},
            {"type": "text", "text": "what happens?"}
        ]);
        let b = prompt_blocks(&v);
        // the empty one left out
        assert_eq!(b.len(), 2);
        match &b[0] {
            Block::Video { frames, audio, ms, .. } => {
                assert_eq!(frames.len(), 2);
                assert_eq!(frames[1].second, 1);
                assert_eq!(audio, "CCCC");
                assert_eq!(*ms, 6000);
            }
            _ => panic!("not a video"),
        }
        let user = crate::msg::Message::User { blocks: b.clone(), source: crate::msg::Source::User };
        let req = crate::msg::openai_messages(&[user], true);
        let parts = req[0]["content"].as_array().unwrap();
        let handle = parts[0]["text"].as_str().unwrap();
        assert!(handle.contains("2 frames over 6.0 s, with its sound"), "{handle}");
        assert_eq!(parts[1]["type"], "input_video");
        assert_eq!(parts[1]["input_video"]["frames"][1]["data"], "BBBB");
        assert_eq!(parts[1]["input_video"]["frames"][1]["second"], 1);
        assert_eq!(parts[1]["input_video"]["audio"]["format"], "wav");
        // a model that does not watch (the user switched to it): the handle alone, said so
        let user = crate::msg::Message::User { blocks: b, source: crate::msg::Source::User };
        let req = crate::msg::openai_messages(&[user], crate::msg::Input { image: true, audio: true, video: false });
        let content = &req[0]["content"];
        let text = content.as_str().map(str::to_string).unwrap_or_else(|| content.to_string());
        assert!(text.contains("cannot take it"), "{text}");
        assert!(!text.contains("input_video"), "{text}");
    }
}
