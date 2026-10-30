// SPDX-License-Identifier: AGPL-3.0-or-later
//! The fetch rules that need no network (src/fetch.rs): which destinations are public, which content a page may be.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

fn v4_public(a: Ipv4Addr) -> bool {
    let o = a.octets();
    !(a.is_private()
        || a.is_loopback()
        || a.is_link_local()
        || a.is_unspecified()
        || a.is_broadcast()
        || a.is_multicast()
        || a.is_documentation()
        || o[0] == 0
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // shared address space (CGNAT)
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // IETF protocol assignments
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // benchmarking
        || o[0] >= 240)
}

pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => v4_public(a),
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return v4_public(v4);
            }
            let s = a.segments();
            // NAT64 (64:ff9b::/96) carries an IPv4 address: that one decides
            if s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0, 0, 0, 0] {
                return v4_public(Ipv4Addr::new((s[6] >> 8) as u8, s[6] as u8, (s[7] >> 8) as u8, s[7] as u8));
            }
            !(a.is_loopback()
                || a.is_unspecified()
                || a.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00 // unique local
                || (s[0] & 0xffc0) == 0xfe80 // link local
                || (s[0] == 0x2001 && s[1] == 0x0db8)) // documentation
                && a != Ipv6Addr::LOCALHOST
        }
    }
}

pub fn kind_of(content_type: Option<&str>) -> Option<&'static str> {
    let mime = content_type?.split(';').next()?.trim().to_ascii_lowercase();
    if mime == "text/html" || mime == "application/xhtml+xml" {
        Some("html")
    } else if mime.starts_with("text/") || mime == "application/json" || mime == "application/xml" || mime.ends_with("+json") || mime.ends_with("+xml") {
        Some("text")
    } else {
        None
    }
}

/// No content type, or one that only says "bytes": the body decides (looks_like_text).
pub fn may_be_text(content_type: Option<&str>) -> bool {
    let mime = content_type.and_then(|c| c.split(';').next()).map(|m| m.trim().to_ascii_lowercase()).unwrap_or_default();
    matches!(mime.as_str(), "" | "unknown" | "application/octet-stream" | "binary/octet-stream")
}

/// UTF-8 without NUL bytes in its first 64 KiB (a character cut at that edge is fine).
pub fn looks_like_text(body: &[u8]) -> bool {
    let head = &body[..body.len().min(65536)];
    if head.contains(&0) {
        return false;
    }
    match std::str::from_utf8(head) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none() && head.len() == 65536,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_that_are_text() {
        assert!(may_be_text(Some("application/octet-stream")) && may_be_text(None));
        assert!(!may_be_text(Some("image/png")));
        assert!(looks_like_text(b"{\"version\": \"0.2.7\"}\n"));
        assert!(looks_like_text("Привіт".as_bytes()));
        assert!(!looks_like_text(b"PK\x03\x04\0\0binary"));
        assert!(!looks_like_text(&[0xff, 0xfe, 0x41]));
    }

    #[test]
    fn destinations() {
        for bad in ["127.0.0.1", "10.1.2.3", "192.168.1.113", "169.254.1.1", "100.64.0.1", "::1", "fd00::1", "fe80::1", "::ffff:10.0.0.1", "64:ff9b::a00:1"] {
            assert!(!is_public(bad.parse().unwrap()), "{bad}");
        }
        for ok in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111", "64:ff9b::808:808"] {
            assert!(is_public(ok.parse().unwrap()), "{ok}");
        }
    }

    #[test]
    fn kinds() {
        assert_eq!(kind_of(Some("text/html; charset=utf-8")), Some("html"));
        assert_eq!(kind_of(Some("application/ld+json")), Some("text"));
        assert_eq!(kind_of(Some("image/png")), None);
        assert_eq!(kind_of(None), None);
    }
}
