//! The model: an OpenAI chat-completions endpoint (Geode's, on the phone's loopback), streamed. Plain HTTP/1.1 over
//! a socket: no TLS, no async runtime. A cancelled turn shuts the socket down.

use crate::linux::Cancel;
use crate::msg::{ToolCall, Usage, normalize_arguments};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// What the model streamed, as it comes.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Text(String),
    Reasoning(String),
}

/// A finished answer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Answer {
    pub text: String,
    pub reasoning: String,
    pub reasoning_field: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

pub trait Model: Send + Sync {
    /// One request ([body]: the whole chat-completions body); [on_delta] gets the words as they come.
    fn stream(&self, body: &Value, cancel: &Cancel, on_delta: &mut dyn FnMut(Delta)) -> Result<Answer, String>;
}

/// Gathers a stream's chunks into an answer (pi-ai's accumulation: content, reasoning under whichever field it
/// came in, tool calls by index, the finish reason, the usage chunk).
#[derive(Default)]
pub struct Accumulator {
    answer: Answer,
    calls: Vec<(String, String, String)>,
}

impl Accumulator {
    pub fn push(&mut self, chunk: &Value, on_delta: &mut dyn FnMut(Delta)) {
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
            let (p, c) = (n("prompt_tokens"), n("completion_tokens"));
            let t = u.get("total_tokens").and_then(Value::as_u64).unwrap_or(p + c);
            self.answer.usage = Some(Usage { prompt_tokens: p, completion_tokens: c, total_tokens: t });
        }
        let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
            return;
        };
        if let Some(f) = choice.get("finish_reason").and_then(Value::as_str) {
            self.answer.finish_reason = f.to_string();
        }
        let Some(delta) = choice.get("delta") else {
            return;
        };
        for field in ["reasoning_content", "reasoning", "reasoning_text"] {
            if let Some(r) = delta.get(field).and_then(Value::as_str).filter(|r| !r.is_empty()) {
                if self.answer.reasoning_field.is_empty() {
                    self.answer.reasoning_field = field.to_string();
                }
                self.answer.reasoning.push_str(r);
                on_delta(Delta::Reasoning(r.to_string()));
                break;
            }
        }
        if let Some(t) = delta.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            self.answer.text.push_str(t);
            on_delta(Delta::Text(t.to_string()));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for (pos, c) in calls.iter().enumerate() {
                let index = c.get("index").and_then(Value::as_u64).map(|i| i as usize).unwrap_or(pos);
                while self.calls.len() <= index {
                    self.calls.push(Default::default());
                }
                let slot = &mut self.calls[index];
                if let Some(id) = c.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    slot.0 = id.to_string();
                }
                if let Some(f) = c.get("function") {
                    if let Some(n) = f.get("name").and_then(Value::as_str) {
                        slot.1.push_str(n);
                    }
                    if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                        slot.2.push_str(a);
                    }
                }
            }
        }
    }

    pub fn finish(mut self) -> Answer {
        self.answer.tool_calls = self
            .calls
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .enumerate()
            .map(|(i, (id, name, args))| ToolCall {
                id: if id.is_empty() { format!("call_{i}") } else { id },
                name,
                arguments: normalize_arguments(&args),
            })
            .collect();
        self.answer
    }
}

/// The real endpoint.
pub struct HttpModel {
    pub base_url: String,
    pub key: String,
}

fn parse_url(base: &str) -> Result<(String, u16, String), String> {
    let rest = base.strip_prefix("http://").ok_or_else(|| format!("the model's URL must be http:// (got {base})"))?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/').to_string()),
        None => (rest, String::new()),
    };
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().map_err(|_| format!("bad port in {base}"))?),
        None => (hostport.to_string(), 80),
    };
    Ok((host, port, path))
}

impl Model for HttpModel {
    fn stream(&self, body: &Value, cancel: &Cancel, on_delta: &mut dyn FnMut(Delta)) -> Result<Answer, String> {
        let (host, port, path) = parse_url(&self.base_url)?;
        let stream = TcpStream::connect((host.as_str(), port)).map_err(|e| format!("cannot reach the model: {e}"))?;
        stream.set_nodelay(true).ok();
        // cancel: a watcher shuts the socket down, the read below ends
        let watch = stream.try_clone().map_err(|e| e.to_string())?;
        let done = Arc::new(Mutex::new(false));
        let d = done.clone();
        let c = cancel.clone();
        let watcher = thread::spawn(move || {
            while !*d.lock().unwrap() {
                if c.is_cancelled() {
                    let _ = watch.shutdown(Shutdown::Both);
                    return;
                }
                thread::sleep(Duration::from_millis(30));
            }
        });
        let result = (|| {
            let payload = serde_json::to_vec(body).map_err(|e| e.to_string())?;
            let mut head = format!(
                "POST {path}/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nUser-Agent: geode-agent/{}\r\nContent-Length: {}\r\nConnection: close\r\n",
                env!("CARGO_PKG_VERSION"),
                payload.len()
            );
            if !self.key.is_empty() {
                head.push_str(&format!("Authorization: Bearer {}\r\n", self.key));
            }
            head.push_str("\r\n");
            let mut w = &stream;
            w.write_all(head.as_bytes())
                .and_then(|_| w.write_all(&payload))
                .map_err(|e| format!("cannot reach the model: {e}"))?;
            let mut r = BufReader::new(&stream);
            let mut status = String::new();
            r.read_line(&mut status).map_err(|e| e.to_string())?;
            let code: u16 = status.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
            let mut chunked = false;
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).map_err(|e| e.to_string())? == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
                let l = line.to_ascii_lowercase();
                if l.starts_with("transfer-encoding:") && l.contains("chunked") {
                    chunked = true;
                }
            }
            let mut body_reader: Box<dyn BufRead> = if chunked {
                Box::new(BufReader::new(Chunked { inner: r, left: 0, done: false }))
            } else {
                Box::new(r)
            };
            if code != 200 {
                let mut text = String::new();
                let _ = body_reader.read_to_string(&mut text);
                let msg = serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(String::from))
                    .unwrap_or_else(|| text.trim().to_string());
                return Err(format!("the model answered HTTP {code}: {msg}"));
            }
            let mut acc = Accumulator::default();
            let mut line = String::new();
            loop {
                line.clear();
                match body_reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => {
                        if cancel.is_cancelled() {
                            break;
                        }
                        return Err(format!("the model's stream broke: {e}"));
                    }
                }
                let l = line.trim_end();
                let Some(data) = l.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data == "[DONE]" {
                    break;
                }
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    if let Some(e) = v.get("error") {
                        return Err(e.get("message").and_then(Value::as_str).unwrap_or("the model failed").to_string());
                    }
                    acc.push(&v, on_delta);
                }
            }
            Ok(acc.finish())
        })();
        *done.lock().unwrap() = true;
        let _ = watcher.join();
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        result
    }
}

/// An HTTP/1.1 chunked body as a plain one.
struct Chunked<R: BufRead> {
    inner: R,
    left: usize,
    done: bool,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done {
            return Ok(0);
        }
        if self.left == 0 {
            let mut size = String::new();
            self.inner.read_line(&mut size)?;
            if size.trim().is_empty() {
                size.clear();
                self.inner.read_line(&mut size)?;
            }
            self.left = usize::from_str_radix(size.trim().split(';').next().unwrap_or("0"), 16).unwrap_or(0);
            if self.left == 0 {
                self.done = true;
                return Ok(0);
            }
        }
        let want = buf.len().min(self.left);
        let n = self.inner.read(&mut buf[..want])?;
        self.left -= n;
        if self.left == 0 {
            let mut crlf = String::new();
            self.inner.read_line(&mut crlf)?;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accumulates() {
        let mut a = Accumulator::default();
        let mut seen = Vec::new();
        let mut d = |x: Delta| seen.push(x);
        a.push(&json!({"choices":[{"delta":{"role":"assistant","content":""}}]}), &mut d);
        a.push(&json!({"choices":[{"delta":{"reasoning_content":"think"}}]}), &mut d);
        a.push(&json!({"choices":[{"delta":{"content":"Step 1."}}]}), &mut d);
        a.push(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"bash","arguments":"{\"command\": "}}]}}]}), &mut d);
        a.push(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]},"finish_reason":"tool_calls"}]}), &mut d);
        a.push(&json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}), &mut d);
        let r = a.finish();
        assert_eq!(r.text, "Step 1.");
        assert_eq!(r.reasoning, "think");
        assert_eq!(r.reasoning_field, "reasoning_content");
        assert_eq!(r.tool_calls[0].arguments, "{\"command\":\"ls\"}");
        assert_eq!(r.finish_reason, "tool_calls");
        assert_eq!(r.usage.unwrap().total_tokens, 7);
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn urls() {
        assert_eq!(parse_url("http://127.0.0.1:18099/v1").unwrap(), ("127.0.0.1".into(), 18099, "/v1".into()));
        assert!(parse_url("https://x").is_err());
    }
}
