//! Compaction: the older part of the conversation condensed by the model into a checkpoint, the recent part kept.
//! Ported from @deepseek-ai/dsh-compaction-basic (the range it picks, the summarization instruction, the checkpoint's
//! framing, the rule that a checkpoint must be smaller than what it replaces) and dsh-command-compact (/compact).
//!
//! One change for phones: the harness reserves 64K tokens of headroom before it compacts on its own, more than a
//! phone's whole context, so it never did; here the conversation is compacted when it reaches 80% of the context
//! (and when the model says it no longer fits), keeping the newest 16%.

use crate::agent::Runtime;
use crate::linux::Cancel;
use crate::msg::{Block, Message, Source, ToolSpec, completion_budget, meter_message, occupancy, request_body};
use serde_json::json;

const THRESHOLD_RATIO: f64 = 0.8;
const RETAIN_RATIO: f64 = 0.16;
const SUMMARY_MAX_TOKENS: usize = 65536;
/// Room a summary needs; and the room an answer needs, below which the conversation is compacted first.
const MIN_SUMMARY_TOKENS: usize = 512;

fn min_answer_tokens(context: usize) -> usize {
    (context / 8).clamp(512, 4096)
}

pub const SUMMARY_OPEN_TAG: &str = "<compacted-summary>";
pub const SUMMARY_CLOSE_TAG: &str = "</compacted-summary>";
pub const CHECKPOINT_PREAMBLE: &str = "This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.";

pub fn compaction_instruction() -> String {
    [
        "You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.",
        "",
        "Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write \"(none)\" for an empty section — never drop a section.",
        "",
        "## Primary Request and Intent",
        "- [the user's original and evolving goals; quote verbatim where the exact wording matters]",
        "",
        "## Key Technical Concepts",
        "- [technologies, frameworks, patterns, and conventions in play]",
        "",
        "## Files and Code",
        "- [exact path: why it matters, key changes or snippets]",
        "",
        "## Errors and Fixes",
        "- [error: how it was resolved, plus any related user feedback]",
        "",
        "## Pending Jobs",
        "- [explicitly requested work not yet completed]",
        "",
        "## Current Work",
        "- [precisely what was in progress at this checkpoint]",
        "",
        "## Next Step",
        "- [the single next action, directly in line with the most recent request, or \"(none)\"]",
        "",
        "## Critical Context",
        "- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]",
        "",
        "Rules:",
        "- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.",
        "- Capture user feedback and explicit instructions faithfully, especially corrections.",
        "- Do NOT mention this summarization request or that the context was compacted.",
        "- Output only the checkpoint text: do not call any tool or take any other action.",
        "- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.",
    ]
    .join("\n")
}

/// The model said the request no longer fits its context.
pub fn is_overflow(e: &str) -> bool {
    let l = e.to_ascii_lowercase();
    (l.contains("context")
        && (l.contains("exceed") || l.contains("too long") || l.contains("full") || l.contains("length")))
        || l.contains("too many tokens")
}

/// What a compaction did: how many messages it replaced and their estimated tokens.
#[derive(Debug, PartialEq)]
pub struct Compacted {
    pub items: usize,
    pub tokens: usize,
}

/// The range to replace: after the system prompt, up to the newest messages worth [retain] tokens, never between a
/// tool call and its results. None: nothing worth compacting.
pub fn select_range(messages: &[Message], retain: usize) -> Option<(usize, usize)> {
    let first = usize::from(matches!(messages.first(), Some(Message::System { .. })));
    let mut keep_from = messages.len();
    let mut acc = 0;
    for i in (0..messages.len()).rev() {
        acc += meter_message(&messages[i]);
        keep_from = i;
        if acc >= retain {
            break;
        }
    }
    if keep_from <= first {
        return None;
    }
    // a kept tool result keeps the call it answers
    while keep_from > first && matches!(messages[keep_from], Message::Tool { .. }) {
        keep_from -= 1;
    }
    if keep_from <= first {
        return None;
    }
    Some((first, keep_from - 1))
}

/// Compact now (/compact, or a conversation too big). [retain]: tokens of newest history to keep (the policy's
/// share of the context by default).
pub fn compact(
    rt: &Runtime,
    sess: &mut crate::session::Session,
    tools: &[ToolSpec],
    cancel: &Cancel,
    retain: Option<usize>,
) -> Result<Option<Compacted>, String> {
    let retain = retain.unwrap_or((rt.cfg.context_window as f64 * RETAIN_RATIO) as usize);
    let Some((start, mut end)) = select_range(&sess.messages, retain) else {
        return Ok(None);
    };
    // a region too big to summarize in one request: its older part first (the rest at the next compaction)
    loop {
        let mut probe: Vec<Message> = sess.messages[..start].to_vec();
        probe.extend(sess.messages[start..=end].iter().cloned());
        probe.push(Message::user_text(compaction_instruction(), Source::User));
        if completion_budget(rt.cfg.context_window, SUMMARY_MAX_TOKENS.min(rt.cfg.max_tokens), &probe, tools)
            >= MIN_SUMMARY_TOKENS
            || end <= start + 1
        {
            break;
        }
        let mut half = start + (end - start) / 2;
        // never between a call and its results
        while half > start && matches!(sess.messages.get(half + 1), Some(Message::Tool { .. })) {
            half -= 1;
        }
        if half <= start {
            break;
        }
        end = half;
    }
    let region = &sess.messages[start..=end];
    let shadowed_tokens: usize = region.iter().map(meter_message).sum();
    let mut input: Vec<Message> = Vec::new();
    if let Some(sys @ Message::System { .. }) = sess.messages.first() {
        input.push(sys.clone());
    }
    input.extend(region.iter().cloned());
    input.push(Message::user_text(compaction_instruction(), Source::User));
    let budget = completion_budget(rt.cfg.context_window, SUMMARY_MAX_TOKENS.min(rt.cfg.max_tokens), &input, tools);
    let body = request_body(&rt.cfg.model, &input, tools, budget, rt.cfg.image_input);
    let answer = rt.model.stream(&body, cancel, &mut |_| {}).map_err(|e| format!("compaction failed: {e}"))?;
    if answer.finish_reason == "length" {
        return Err("summarization truncated at the token cap (incomplete checkpoint)".into());
    }
    if answer.text.trim().is_empty() {
        return Err("Compaction could not produce a useful summary.".into());
    }
    let checkpoint = Message::User {
        blocks: vec![
            Block::text(format!("{CHECKPOINT_PREAMBLE}\n\n{SUMMARY_OPEN_TAG}")),
            Block::text(answer.text),
            Block::text(SUMMARY_CLOSE_TAG),
        ],
        source: Source::Checkpoint,
    };
    if meter_message(&checkpoint) >= shadowed_tokens {
        return Err(format!(
            "summary is not smaller than the shadowed content ({} estimated framed tokens >= {shadowed_tokens})",
            meter_message(&checkpoint)
        ));
    }
    let mut next: Vec<Message> = sess.messages[..start].to_vec();
    next.push(checkpoint);
    next.extend(sess.messages[end + 1..].iter().cloned());
    let items = end + 1 - start;
    sess.replace_messages(next)?;
    rt.sink.update(&sess.id, json!({"sessionUpdate": "usage_update", "used": occupancy(&sess.messages, tools), "size": rt.cfg.context_window}));
    Ok(Some(Compacted { items, tokens: shadowed_tokens }))
}

/// Compaction when the conversation has grown to the threshold.
pub fn auto(rt: &Runtime, sess: &mut crate::session::Session, tools: &[ToolSpec], cancel: &Cancel) {
    let ctx = rt.cfg.context_window;
    if !rt.cfg.auto_compact || ctx == 0 {
        return;
    }
    let threshold = (ctx as f64 * THRESHOLD_RATIO) as usize;
    let crowded = occupancy(&sess.messages, tools) >= threshold;
    let no_room = completion_budget(ctx, rt.cfg.max_tokens, &sess.messages, tools) < min_answer_tokens(ctx);
    if !crowded && !no_room {
        return;
    }
    let _ = compact(rt, sess, tools, cancel, None);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msg::ToolCall;

    #[test]
    fn range_keeps_tool_pairs() {
        let mut m = vec![Message::System { text: "s".into() }];
        for i in 0..6 {
            m.push(Message::user_text(format!("q{i} {}", "x".repeat(400)), Source::User));
            m.push(Message::Assistant {
                id: "a".into(),
                reasoning: String::new(),
                reasoning_field: String::new(),
                text: String::new(),
                tool_calls: vec![ToolCall { id: format!("c{i}"), name: "read".into(), arguments: "{}".into() }],
                usage: None,
            });
            m.push(Message::Tool {
                call_id: format!("c{i}"),
                name: "read".into(),
                blocks: vec![Block::text("y".repeat(400))],
                is_error: false,
            });
        }
        let (s, e) = select_range(&m, 150).unwrap();
        assert_eq!(s, 1);
        assert!(!matches!(m[e + 1], Message::Tool { .. }));
        assert!(select_range(&m[..2], 10).is_none());
    }

    #[test]
    fn overflow_words() {
        assert!(is_overflow("the model answered HTTP 400: This model's maximum context length is 32768 tokens"));
        assert!(!is_overflow("cannot reach the model"));
    }
}
