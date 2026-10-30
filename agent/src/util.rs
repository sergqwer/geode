//! Small helpers: ids, hex, time, the JavaScript string length the harness's estimates count in.

use std::time::{SystemTime, UNIX_EPOCH};

/// Random bytes from the OS (a failure leaves a time-seeded fallback: ids only need to be unique).
pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut buf = [0u8; N];
    if getrandom::fill(&mut buf).is_err() {
        let mut x = now_millis() as u64 ^ 0x9e37_79b9_7f4a_7c15;
        for b in buf.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = x as u8;
        }
    }
    buf
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A random (v4) UUID.
pub fn uuid() -> String {
    let mut b = random_bytes::<16>();
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h = hex(&b);
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

pub fn now_millis() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// UTC time as ISO 8601 (`2026-10-28T17:54:01.123Z`).
pub fn iso_now() -> String {
    iso_millis(now_millis())
}

pub fn iso_millis(ms: u128) -> String {
    let secs = (ms / 1000) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3600, rem % 3600 / 60, rem % 60, ms % 1000)
}

/// A string's length as JavaScript counts it (UTF-16 code units): the unit of every estimate the harness makes.
pub fn js_len(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// `Math.ceil(n / d)` for the harness's fractional densities (3.5 characters a token).
pub fn ceil_div_f(n: usize, d: f64) -> usize {
    (n as f64 / d).ceil() as usize
}

pub fn ceil_div(n: usize, d: usize) -> usize {
    n.div_ceil(d)
}

/// The first `n` UTF-16 units of `s` (a cut inside a surrogate pair keeps the pair whole, rounding down).
pub fn js_slice_head(s: &str, n: usize) -> &str {
    let mut units = 0;
    for (i, c) in s.char_indices() {
        let w = c.len_utf16();
        if units + w > n {
            return &s[..i];
        }
        units += w;
    }
    s
}

/// The last `n` UTF-16 units of `s` (a cut inside a surrogate pair keeps the pair whole, rounding down).
pub fn js_slice_tail(s: &str, n: usize) -> &str {
    let total = js_len(s);
    if n >= total {
        return s;
    }
    let skip = total - n;
    let mut units = 0;
    for (i, c) in s.char_indices() {
        if units >= skip {
            return &s[i..];
        }
        units += c.len_utf16();
    }
    ""
}

/// JSON text as `JSON.stringify` writes it (compact, keys in insertion order, non-ASCII as is).
pub fn js_stringify(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso() {
        assert_eq!(iso_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso_millis(1_793_205_140_123), "2026-10-28T16:32:20.123Z");
    }

    #[test]
    fn utf16() {
        assert_eq!(js_len("a—b"), 3);
        assert_eq!(js_len("😀"), 2);
        assert_eq!(js_slice_head("ab😀c", 3), "ab");
        assert_eq!(js_slice_tail("ab😀c", 2), "c");
        assert_eq!(js_slice_tail("ab😀c", 3), "😀c");
    }

    #[test]
    fn uuids() {
        let u = uuid();
        assert_eq!(u.len(), 36);
        assert_eq!(&u[14..15], "4");
    }
}
