//! The conversation as the harness keeps it, the two token estimates it makes from it (the occupancy it reports, the
//! completion budget it asks for), and its translation into an OpenAI chat-completions request.
//!
//! Ported from @deepseek-ai/dsh-token-meter (occupancy: 4 characters a token, 4 tokens a block and a message),
//! @earendil-works/pi-ai utils/estimate + api/simple-options (budget: 3.5 characters a token, a 4096-token reserve)
//! and api/openai-completions convertMessages, through dsh-llm-pi-ai's history conversion.

use crate::util::{ceil_div, ceil_div_f, js_len, js_stringify};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
    },
    /// An image the model sees: its bytes (base64), what it is, and where it came from (a guest path, or a name).
    Image {
        mime: String,
        data: String,
        width: u32,
        height: u32,
        #[serde(default)]
        name: String,
        #[serde(default)]
        path: String,
        #[serde(default)]
        id: String,
    },
}

impl Block {
    pub fn text(s: impl Into<String>) -> Block {
        Block::Text { text: s.into() }
    }
}

/// Where a user-role message came from: the person, or the harness's own context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    #[default]
    User,
    RuntimeContext,
    SkillCatalog,
    SkillInvocation,
    AgentInstructions,
    Notice,
    Checkpoint,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// The arguments as the harness keeps them: the model's JSON, re-serialized compact (or its raw text when it
    /// was not JSON).
    pub arguments: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        text: String,
    },
    User {
        blocks: Vec<Block>,
        #[serde(default)]
        source: Source,
    },
    Assistant {
        id: String,
        #[serde(default)]
        reasoning: String,
        /// The stream field the reasoning came in (reasoning_content, reasoning): it goes back under the same name.
        #[serde(default)]
        reasoning_field: String,
        #[serde(default)]
        text: String,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        usage: Option<Usage>,
    },
    Tool {
        call_id: String,
        name: String,
        blocks: Vec<Block>,
        #[serde(default)]
        is_error: bool,
    },
}

impl Message {
    pub fn user_text(text: impl Into<String>, source: Source) -> Message {
        Message::User { blocks: vec![Block::text(text)], source }
    }

    pub fn source(&self) -> Option<Source> {
        match self {
            Message::User { source, .. } => Some(*source),
            _ => None,
        }
    }

    /// The text of a one-text-block message (the harness's context messages are).
    pub fn only_text(&self) -> Option<&str> {
        let blocks = match self {
            Message::User { blocks, .. } | Message::Tool { blocks, .. } => blocks,
            Message::System { text } => return Some(text),
            _ => return None,
        };
        match blocks.as_slice() {
            [Block::Text { text }] => Some(text),
            _ => None,
        }
    }
}

/// The arguments as the harness stores a call's: re-serialized compact when they parse, as given otherwise.
pub fn normalize_arguments(raw: &str) -> String {
    if raw.trim().is_empty() {
        return "{}".into();
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(v) => js_stringify(&v),
        Err(_) => raw.to_string(),
    }
}

fn parse_arguments(raw: &str) -> Value {
    if raw.is_empty() {
        return json!({});
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// A tool as the model is told of it.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolSpec {
    /// pi-ai's and the request header's form: {name, description, parameters}.
    pub fn plain(&self) -> Value {
        json!({"name": self.name, "description": self.description, "parameters": self.parameters})
    }

    pub fn openai(&self) -> Value {
        json!({"type": "function", "function": self.plain()})
    }
}

pub fn tools_json(tools: &[ToolSpec]) -> String {
    js_stringify(&Value::Array(tools.iter().map(ToolSpec::plain).collect()))
}

// ---- occupancy: dsh-token-meter's fixed heuristic ----

const CHARS: usize = 4;
const OVERHEAD: usize = 4;

fn image_reference_json(b: &Block) -> String {
    match b {
        Block::Image { mime, width, height, name, id, data, .. } => {
            let mut att = Map::new();
            att.insert("attachmentId".into(), json!(id));
            att.insert("mediaType".into(), json!(mime));
            att.insert("bytes".into(), json!(base64_len(data)));
            att.insert("width".into(), json!(width));
            att.insert("height".into(), json!(height));
            if !name.is_empty() {
                att.insert("name".into(), json!(name));
            }
            js_stringify(&json!({"type": "image", "attachment": Value::Object(att)}))
        }
        Block::Text { .. } => String::new(),
    }
}

fn base64_len(b64: &str) -> usize {
    let pad = b64.bytes().rev().take_while(|&c| c == b'=').count();
    b64.len() / 4 * 3 - pad.min(2)
}

/// One block's price (text, or an image's structural reference).
pub fn price_block(b: &Block) -> usize {
    match b {
        Block::Text { text } => ceil_div(js_len(text), CHARS) + OVERHEAD,
        Block::Image { .. } => OVERHEAD + ceil_div(js_len(&image_reference_json(b)), CHARS),
    }
}

pub fn price_blocks(blocks: &[Block]) -> usize {
    blocks.iter().map(price_block).sum()
}

/// One message's occupancy price.
pub fn meter_message(m: &Message) -> usize {
    match m {
        Message::System { text } => {
            if text.is_empty() {
                0
            } else {
                ceil_div(js_len(text), CHARS) + 4
            }
        }
        Message::User { blocks, .. } | Message::Tool { blocks, .. } => price_blocks(blocks) + 4,
        Message::Assistant { reasoning, text, tool_calls, .. } => {
            let mut t = 0;
            if !reasoning.is_empty() {
                t += ceil_div(js_len(reasoning), CHARS) + OVERHEAD;
            }
            if !text.is_empty() {
                t += ceil_div(js_len(text), CHARS) + OVERHEAD;
            }
            for c in tool_calls {
                t += ceil_div(js_len(&c.name), CHARS) + ceil_div(js_len(&c.arguments), CHARS) + OVERHEAD;
            }
            t + 4
        }
    }
}

pub fn meter_tools(tools: &[ToolSpec]) -> usize {
    if tools.is_empty() {
        return 0;
    }
    ceil_div(js_len(&tools_json(tools)), CHARS) + OVERHEAD
}

/// The context occupancy the harness reports (usage_update.used). With a provider usage anchor (the last answer's
/// reported total, when it is no lower than the estimate of the same request), the anchor plus what changed since;
/// the estimate otherwise.
pub fn occupancy(messages: &[Message], tools: &[ToolSpec]) -> usize {
    let surface: usize = messages.iter().map(meter_message).sum();
    let tools_tokens = meter_tools(tools);
    // the anchor: the newest answer with usage; its price includes everything up to it
    if let Some(i) = messages.iter().rposition(|m| matches!(m, Message::Assistant { usage: Some(_), .. }))
        && let Message::Assistant { usage: Some(u), .. } = &messages[i]
    {
        let anchor_surface: usize = messages[..=i].iter().map(meter_message).sum();
        let estimated_anchor = tools_tokens + anchor_surface;
        let total = u.total_tokens as usize;
        if total >= estimated_anchor {
            return total + (surface - anchor_surface);
        }
    }
    tools_tokens + surface
}

// ---- completion budget: pi-ai's estimate ----

const BUDGET_CHARS: f64 = 3.5;
const ESTIMATED_IMAGE_CHARS: usize = 4800;
pub const CONTEXT_SAFETY_TOKENS: usize = 4096;

fn budget_content_chars(blocks: &[Block]) -> usize {
    blocks
        .iter()
        .map(|b| match b {
            Block::Text { text } => js_len(text),
            Block::Image { .. } => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

/// pi-ai's estimate of one request's prompt tokens (it never trusts provider usage here: the harness replays
/// answers without it).
pub fn budget_estimate(messages: &[Message], tools: &[ToolSpec]) -> usize {
    let mut tokens = 0;
    for m in messages {
        tokens += match m {
            Message::System { text } => {
                ceil_div_f(js_len(text), BUDGET_CHARS)
                    + if tools.is_empty() { 0 } else { ceil_div_f(js_len(&tools_json(tools)), BUDGET_CHARS) }
            }
            Message::User { blocks, .. } => {
                // user content is flattened to one string when it is all text
                ceil_div_f(budget_content_chars(&user_blocks_for_request(blocks)), BUDGET_CHARS)
            }
            Message::Tool { blocks, .. } => {
                ceil_div_f(budget_content_chars(&tool_blocks_for_request(blocks)), BUDGET_CHARS)
            }
            Message::Assistant { reasoning, text, tool_calls, .. } => {
                let mut chars = js_len(text) + js_len(reasoning);
                for c in tool_calls {
                    chars += js_len(&c.name) + js_len(&js_stringify(&parse_arguments(&c.arguments)));
                }
                ceil_div_f(chars, BUDGET_CHARS)
            }
        };
    }
    tokens
}

/// max_completion_tokens for a request: the model's output cap, clamped to what the context leaves.
pub fn completion_budget(context_window: usize, max_tokens: usize, messages: &[Message], tools: &[ToolSpec]) -> usize {
    if context_window == 0 {
        return max_tokens.max(1);
    }
    let est = budget_estimate(messages, tools) + CONTEXT_SAFETY_TOKENS;
    let available = context_window.saturating_sub(est).max(1);
    max_tokens.min(available)
}

// ---- the request: pi-ai's openai-completions conversion ----

/// What follows the handle of an image the user attached: the model sees it in the message, and read_image on it only
/// cost a step (Geode's; DSH's handle alone left the model to read the file again).
const USER_IMAGE_NOTE: &str = "You can see this image here, in the user's message: answer about it directly, without      read_image. Its file path is only for tools that need the file itself (a script, a conversion).";
const USER_IMAGE_NOTE_NO_FILE: &str =
    "You can see this image here, in the user's message: answer about it directly. It has no file to read.";

/// A user message's content as the request carries it: its text blocks (empty ones dropped) and, before each image,
/// its handle text and that the model sees it.
fn user_blocks_for_request(blocks: &[Block]) -> Vec<Block> {
    blocks_for_request(blocks, true)
}

/// [blocks] as a request carries them, [user]'s (a user message's) or a tool result's.
fn blocks_for_request(blocks: &[Block], user: bool) -> Vec<Block> {
    let mut out = Vec::new();
    for b in blocks {
        match b {
            Block::Text { text } => {
                if !text.is_empty() {
                    out.push(b.clone());
                }
            }
            Block::Image { path, .. } => {
                let handle = image_handle_text(b);
                out.push(Block::text(if !user {
                    handle
                } else if path.is_empty() {
                    format!("{handle} {USER_IMAGE_NOTE_NO_FILE}")
                } else {
                    format!("{handle} {USER_IMAGE_NOTE}")
                }));
                out.push(b.clone());
            }
        }
    }
    out
}

fn tool_blocks_for_request(blocks: &[Block]) -> Vec<Block> {
    if blocks.iter().all(|b| matches!(b, Block::Text { .. })) {
        let joined: String = blocks
            .iter()
            .map(|b| match b {
                Block::Text { text } => text.as_str(),
                _ => "",
            })
            .collect();
        return vec![Block::text(if joined.is_empty() { "(no output)".to_string() } else { joined })];
    }
    blocks_for_request(blocks, false)
}

/// The text the harness puts before an image it sends (dsh-llm requestImageHandleText).
pub fn image_handle_text(b: &Block) -> String {
    if let Block::Image { mime, width, height, name, path, id, .. } = b {
        let identity = if name.is_empty() { id.clone() } else { format!("{} ({id})", js_stringify(&json!(name))) };
        let preview = format!("Image {identity}; request preview {width}x{height}px.");
        if path.is_empty() {
            format!("{preview} It may be resized or re-encoded; source dimensions, format, and byte size may differ.")
        } else {
            let ext = match mime.as_str() {
                "image/png" => ".png",
                "image/jpeg" => ".jpg",
                "image/webp" => ".webp",
                "image/gif" => ".gif",
                _ => "",
            };
            format!(
                "{preview} Normalized copy (read-only; may be resized or re-encoded): {} ({width}x{height}px, {mime}). Source dimensions, format, and byte size may differ. Copy to a writable path ending in {ext} before editing.",
                js_stringify(&json!(path))
            )
        }
    } else {
        String::new()
    }
}

fn image_url(b: &Block) -> Value {
    match b {
        Block::Image { mime, data, .. } => {
            json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{data}")}})
        }
        Block::Text { text } => json!({"type": "text", "text": text}),
    }
}

/// A content as the request carries it: one string when it is all text, parts otherwise.
fn content_value(blocks: &[Block]) -> Value {
    if blocks.iter().all(|b| matches!(b, Block::Text { .. })) {
        let s: String = blocks
            .iter()
            .map(|b| match b {
                Block::Text { text } => text.as_str(),
                _ => "",
            })
            .collect();
        return Value::String(s);
    }
    Value::Array(blocks.iter().map(image_url).collect())
}

/// The OpenAI messages for a history (pi-ai convertMessages, the harness's compat: no developer role, null content
/// on a tool-only answer, reasoning back under its stream field).
pub fn openai_messages(messages: &[Message], image_input: bool) -> Vec<Value> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        match &messages[i] {
            Message::System { text } => {
                if !text.is_empty() {
                    out.push(json!({"role": "system", "content": text}));
                }
            }
            Message::User { blocks, .. } => {
                let parts = user_blocks_for_request(blocks);
                if !parts.is_empty() {
                    out.push(json!({"role": "user", "content": content_value(&parts)}));
                }
            }
            Message::Assistant { reasoning, reasoning_field, text, tool_calls, .. } => {
                let mut m = Map::new();
                m.insert("role".into(), json!("assistant"));
                m.insert("content".into(), Value::Null);
                if !text.trim().is_empty() {
                    m.insert("content".into(), json!(text));
                }
                if !reasoning.trim().is_empty() {
                    let field = if reasoning_field.is_empty() { "reasoning_content" } else { reasoning_field.as_str() };
                    m.insert(field.into(), json!(reasoning));
                }
                if !tool_calls.is_empty() {
                    let calls: Vec<Value> = tool_calls
                        .iter()
                        .map(|c| {
                            json!({"id": c.id, "type": "function", "function": {
                                "name": c.name, "arguments": js_stringify(&parse_arguments(&c.arguments))}})
                        })
                        .collect();
                    m.insert("tool_calls".into(), Value::Array(calls));
                }
                let has_content = !m["content"].is_null();
                if has_content || !tool_calls.is_empty() {
                    out.push(Value::Object(m));
                }
            }
            Message::Tool { .. } => {
                // a run of tool results, then the images among them as one user message
                let mut images = Vec::new();
                while let Some(Message::Tool { call_id, blocks, .. }) = messages.get(i) {
                    let blocks = tool_blocks_for_request(blocks);
                    let text: Vec<&str> = blocks
                        .iter()
                        .filter_map(|b| match b {
                            Block::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    let text = text.join("\n");
                    let has_images = blocks.iter().any(|b| matches!(b, Block::Image { .. }));
                    let shown = if !text.trim().is_empty() {
                        text
                    } else if has_images {
                        "(see attached image)".into()
                    } else {
                        "(no tool output)".into()
                    };
                    out.push(json!({"role": "tool", "content": shown, "tool_call_id": call_id}));
                    if has_images && image_input {
                        images.extend(blocks.iter().filter(|b| matches!(b, Block::Image { .. })).map(image_url));
                    }
                    i += 1;
                }
                if !images.is_empty() {
                    let mut parts = vec![json!({"type": "text", "text": "Attached image(s) from tool result:"})];
                    parts.extend(images);
                    out.push(json!({"role": "user", "content": parts}));
                }
                continue;
            }
        }
        i += 1;
    }
    out
}

/// The whole request body, its fields in the harness's order.
pub fn request_body(
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    max_completion_tokens: usize,
    image_input: bool,
) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(openai_messages(messages, image_input)));
    body.insert("stream".into(), json!(true));
    body.insert("stream_options".into(), json!({"include_usage": true}));
    body.insert("store".into(), json!(false));
    body.insert("max_completion_tokens".into(), json!(max_completion_tokens));
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools.iter().map(ToolSpec::openai).collect()));
    } else if messages.iter().any(|m| matches!(m, Message::Tool { .. })) {
        body.insert("tools".into(), json!([]));
    }
    Value::Object(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_compact() {
        assert_eq!(normalize_arguments("{\"a\": 1, \"b\": \"x\"}"), "{\"a\":1,\"b\":\"x\"}");
        assert_eq!(normalize_arguments(""), "{}");
        assert_eq!(normalize_arguments("not json"), "not json");
    }

    #[test]
    fn assistant_null_content() {
        let m = Message::Assistant {
            id: "a".into(),
            reasoning: String::new(),
            reasoning_field: String::new(),
            text: "  ".into(),
            tool_calls: vec![ToolCall { id: "c".into(), name: "read".into(), arguments: "{}".into() }],
            usage: None,
        };
        let v = openai_messages(&[m], true);
        assert_eq!(
            js_stringify(&v[0]),
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"read","arguments":"{}"}}]}"#
        );
    }

    #[test]
    fn user_images_say_they_are_seen() {
        let img = |path: &str| Block::Image {
            mime: "image/jpeg".into(),
            data: "AAAA".into(),
            width: 1,
            height: 1,
            name: "a.jpg".into(),
            path: path.into(),
            id: "sha256:x".into(),
        };
        let user = Message::User {
            blocks: vec![img("/workspace/.geode/images/a.jpg"), Block::text("what is it?")],
            source: Source::User,
        };
        let v = openai_messages(&[user], true);
        let handle = v[0]["content"][0]["text"].as_str().unwrap();
        assert!(handle.contains("/workspace/.geode/images/a.jpg") && handle.ends_with(USER_IMAGE_NOTE), "{handle}");
        let bare = Message::User { blocks: vec![img(""), Block::text("and this?")], source: Source::User };
        let v = openai_messages(&[bare], true);
        assert!(v[0]["content"][0]["text"].as_str().unwrap().ends_with(USER_IMAGE_NOTE_NO_FILE));
        let tool = Message::Tool {
            call_id: "c".into(),
            name: "read_image".into(),
            blocks: vec![Block::text("t"), img("/w/a.jpg")],
            is_error: false,
        };
        let v = openai_messages(&[tool], true);
        assert!(!v[0]["content"].as_str().unwrap().contains("You can see this image"));
    }

    #[test]
    fn tool_images_follow() {
        let img = Block::Image {
            mime: "image/png".into(),
            data: "AAAA".into(),
            width: 1,
            height: 1,
            name: "a.png".into(),
            path: "/workspace/a.png".into(),
            id: "sha256:x".into(),
        };
        let m = Message::Tool {
            call_id: "c".into(),
            name: "read_image".into(),
            blocks: vec![Block::text("t"), img],
            is_error: false,
        };
        let v = openai_messages(&[m], true);
        assert_eq!(v.len(), 2);
        assert!(v[0]["content"].as_str().unwrap().starts_with(
            "t
Image \"a.png\" (sha256:x); request preview 1x1px."
        ));
        assert_eq!(v[1]["content"][0]["text"], "Attached image(s) from tool result:");
    }
}
