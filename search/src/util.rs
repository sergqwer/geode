// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small helpers SearXNG takes from Python's standard library: base64, hex, str.title / str.islower, and the
//! result text rules of searx.result_types._base (whitespace and length).

/// Python's str.strip() / str.split(): Unicode whitespace.
pub fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// _normalize_text_field: runs of space, tab and newline become one space, the ends are stripped, and a text
/// longer than `max` characters is cut at a word with " …" added.
pub fn normalize_text_field(text: &str, max: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = false;
    for c in text.chars() {
        if matches!(c, ' ' | '\t' | '\n') {
            if !run {
                out.push(' ');
            }
            run = true;
        } else {
            out.push(c);
            run = false;
        }
    }
    let text = out.trim();
    if text.chars().count() > max {
        let cut: String = text.chars().take(max).collect();
        let cut = match cut.rfind(' ') {
            Some(i) => &cut[..i],
            None => cut.as_str(),
        };
        return format!("{cut} …");
    }
    text.to_string()
}

/// geode_search.py's `_text`: one line of at most `n` characters ending in "…" when cut.
pub fn clip(s: &str, n: usize) -> String {
    let s = squeeze(s);
    if s.chars().count() <= n {
        return s;
    }
    let cut: String = s.chars().take(n - 1).collect();
    format!("{}…", cut.trim_end())
}

/// A computation on a thread of its own, awaited without blocking the (single-threaded) runtime.
pub fn on_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> impl std::future::Future<Output = T> {
    use std::sync::{Arc, Mutex};
    use std::task::{Poll, Waker};
    let shared: Arc<Mutex<(Option<T>, Option<Waker>)>> = Arc::new(Mutex::new((None, None)));
    let s = shared.clone();
    std::thread::spawn(move || {
        let v = f();
        let mut g = s.lock().unwrap();
        g.0 = Some(v);
        if let Some(w) = g.1.take() {
            w.wake();
        }
    });
    std::future::poll_fn(move |cx| {
        let mut g = shared.lock().unwrap();
        match g.0.take() {
            Some(v) => Poll::Ready(v),
            None => {
                g.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    })
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n =
            (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if i <= chunk.len() { B64[(n >> shift & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// base64.urlsafe_b64decode / b64decode (either alphabet; padding optional)
pub fn b64decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            _ => return None,
        } as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits & 0xff) as u8);
        }
    }
    Some(out)
}

pub fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

fn is_cased(c: char) -> bool {
    c.is_lowercase() || c.is_uppercase()
}

/// str.islower: at least one cased character, and no upper case one
pub fn py_islower(s: &str) -> bool {
    s.chars().any(is_cased) && !s.chars().any(char::is_uppercase)
}

/// str.title: a letter after a cased one in lower case, any other in upper case
pub fn py_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_cased = false;
    for c in s.chars() {
        if prev_cased {
            out.extend(c.to_lowercase());
        } else {
            out.extend(c.to_uppercase());
        }
        prev_cased = is_cased(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_fields() {
        assert_eq!(normalize_text_field("  a \n\t b  ", 200), "a b");
        assert_eq!(normalize_text_field("hello world", 9), "hello …");
        assert_eq!(clip("abcdef", 4), "abc…");
        assert_eq!(clip(" a  b ", 4), "a b");
    }

    #[test]
    fn base64_roundtrip() {
        for s in ["", "f", "fo", "foo", "foob", "hello world"] {
            assert_eq!(b64decode(&b64encode(s.as_bytes())).unwrap(), s.as_bytes());
        }
        assert_eq!(b64encode(b"foob"), "Zm9vYg==");
        assert_eq!(b64decode("aHR0cHM6Ly9leGFtcGxlLmNvbS8_YT0x").unwrap(), b"https://example.com/?a=1");
    }

    #[test]
    fn python_case() {
        assert!(py_islower("kyiv city"));
        assert!(!py_islower("Kyiv"));
        assert!(!py_islower("123"));
        assert_eq!(py_title("kyiv city's"), "Kyiv City'S");
        assert_eq!(py_title("київ"), "Київ");
    }
}
