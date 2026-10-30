// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx/engines/mojeek.py (web search): Mojeek wants the cookie its Altcha captcha gives (a PBKDF2 proof of
//! work, solved here once and kept), its language (lb) and region (arc) in cookies.

use super::{Ctx, Params};
use crate::html::{attr, children, parent, parse, sel, text_all};
use crate::http::{EngineError, HttpRequest, HttpResponse, Method};
use crate::results::{Item, MainResult};
use crate::url::urlencode;
use crate::util::{b64encode, hex, unhex};
use sha2::{Digest, Sha256};

const BASE_URL: &str = "https://www.mojeek.com";

pub async fn search(ctx: &Ctx<'_>, name: &str, query: &str, p: &Params) -> Result<Vec<Item>, EngineError> {
    let token = captcha_token(ctx, name, p).await?;
    let resp = ctx.send(name, request(query, p, &token), true).await?;
    response(&resp)
}

pub fn request(query: &str, p: &Params, token: &str) -> HttpRequest {
    let mut args = vec![("q", query.to_string()), ("safe", p.safesearch.min(1).to_string())];
    if p.pageno > 1 {
        args.push(("s", (10 * (p.pageno - 1)).to_string()));
    }
    let mut req = p.request(format!("{BASE_URL}/search?{}", urlencode(&args)));
    // params["cookies"] = {...}: these and no others
    req.cookies = vec![
        ("lb".into(), p.row.str("mj_lb").into()),
        ("arc".into(), p.row.str("mj_arc").into()),
        ("chllg".into(), token.into()),
    ];
    req
}

/// _captcha_token: the cached cookie, or the challenge solved and verified
async fn captcha_token(ctx: &Ctx<'_>, name: &str, p: &Params) -> Result<String, EngineError> {
    if let Some(t) = ctx.cache.get(name, "chllg") {
        return Ok(t);
    }
    let mut req = p.request(format!("{BASE_URL}/captcha/challenge"));
    req.follow_redirects = true; // searx.network.get
    let challenge = ctx.send(name, req, false).await?.json()?;
    let params = challenge.get("parameters").ok_or_else(|| EngineError::Parse("KeyError: 'parameters'".into()))?;
    // seconds of CPU at worst: on a thread of its own, the other engines' requests go on meanwhile
    let p2 = params.clone();
    let solved = crate::util::on_thread(move || solve_altcha(&p2, 1000)).await;
    let (key, counter) = solved.ok_or_else(|| EngineError::Api("failed to solve CAPTCHA".into()))?;
    let time = 100 + (crate::cache::unix_now() % 101);
    let solution =
        serde_json::json!({"challenge": challenge, "solution": {"counter": counter, "derivedKey": key, "time": time}});
    let encoded = b64encode(solution.to_string().as_bytes());
    let boundary = format!("geode{:016x}", crate::cache::unix_now().wrapping_mul(0x9E3779B97F4A7C15));
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"altcha\"\r\n\r\n{encoded}\r\n--{boundary}--\r\n"
    );
    let mut req = p.request(format!("{BASE_URL}/captcha/verify"));
    req.method = Method::Post;
    req.body = Some((format!("multipart/form-data; boundary={boundary}"), body.into_bytes()));
    let resp = ctx.send(name, req, false).await?;
    let token = resp.cookie("chllg").ok_or_else(|| EngineError::Parse("KeyError: 'chllg'".into()))?.to_string();
    ctx.cache.set(name, "chllg", &token, None);
    Ok(token)
}

/// HMAC-SHA256 with its key's inner and outer blocks hashed once: each round then costs two compressions.
struct Hmac {
    inner: Sha256,
    outer: Sha256,
}

impl Hmac {
    fn new(key: &[u8]) -> Hmac {
        let mut k = [0u8; 64];
        if key.len() > 64 {
            k[..32].copy_from_slice(&Sha256::digest(key));
        } else {
            k[..key.len()].copy_from_slice(key);
        }
        let (mut ipad, mut opad) = ([0x36u8; 64], [0x5cu8; 64]);
        for i in 0..64 {
            ipad[i] ^= k[i];
            opad[i] ^= k[i];
        }
        Hmac { inner: Sha256::new().chain_update(ipad), outer: Sha256::new().chain_update(opad) }
    }

    fn mac(&self, data: &[u8]) -> [u8; 32] {
        let inner = self.inner.clone().chain_update(data).finalize();
        self.outer.clone().chain_update(inner).finalize().into()
    }
}

/// hashlib.pbkdf2_hmac("sha256", ...)
pub fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32, dklen: usize) -> Vec<u8> {
    let h = Hmac::new(password);
    let mut out = Vec::with_capacity(dklen);
    let mut block = 1u32;
    while out.len() < dklen {
        let mut s = salt.to_vec();
        s.extend_from_slice(&block.to_be_bytes());
        let mut u = h.mac(&s);
        let mut t = u;
        for _ in 1..iterations {
            u = h.mac(&u);
            for (a, b) in t.iter_mut().zip(u.iter()) {
                *a ^= b;
            }
        }
        out.extend_from_slice(&t);
        block += 1;
    }
    out.truncate(dklen);
    out
}

/// searx.utils.solve_altcha: the counter whose derived key starts with the challenge's prefix
pub fn solve_altcha(params: &serde_json::Value, max_counter: u32) -> Option<(String, u32)> {
    let nonce = unhex(params.get("nonce")?.as_str()?)?;
    let salt = unhex(params.get("salt")?.as_str()?)?;
    let prefix = unhex(params.get("keyPrefix")?.as_str()?)?;
    let cost = params.get("cost")?.as_u64()? as u32;
    let key_len = params.get("keyLength")?.as_u64()? as usize;
    let algo = params.get("algorithm")?.as_str()?.rsplit('/').next()?.replace('-', "").to_lowercase();
    if algo != "sha256" {
        return None;
    }
    for counter in 0..max_counter {
        let mut secret = nonce.clone();
        secret.extend_from_slice(&counter.to_be_bytes());
        let key = pbkdf2_sha256(&secret, &salt, cost, key_len);
        if key.starts_with(&prefix) {
            return Some((hex(&key), counter));
        }
    }
    None
}

pub fn response(resp: &HttpResponse) -> Result<Vec<Item>, EngineError> {
    let doc = parse(&resp.text);
    let mut items = Vec::new();
    // //ul[@class="results-standard"]/li/a[@class="ob"]: the url its href, the title ../h2/a, the snippet ..//p[@class="s"]
    for a in doc.select(&sel("ul[class=\"results-standard\"] > li > a[class=\"ob\"]")) {
        let url = attr(a, "href").unwrap_or_default();
        let (title, content) = match parent(a) {
            Some(li) => (
                text_all(children(li, "h2").flat_map(|h| children(h, "a"))),
                text_all(li.select(&sel("p[class=\"s\"]"))),
            ),
            None => (String::new(), String::new()),
        };
        items.push(Item::Main(MainResult::new(url, title, content)));
    }
    for s in doc.select(&sel("div[class=\"top-info\"] > p[class=\"top-info spell\"] > em > a")) {
        items.push(Item::Suggestion(crate::html::text(s)));
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbkdf2_vectors() {
        // RFC 7914 §11
        assert_eq!(hex(&pbkdf2_sha256(b"passwd", b"salt", 1, 64))[..32], *"55ac046e56e3089fec1691c22544b605");
        assert_eq!(
            hex(&pbkdf2_sha256(b"password", b"salt", 2, 32)),
            "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43"
        );
    }

    #[test]
    fn altcha() {
        let nonce = "00112233";
        let salt = "a1b2";
        // the counter 3's key, its first byte as the prefix
        let mut secret = unhex(nonce).unwrap();
        secret.extend_from_slice(&3u32.to_be_bytes());
        let key = pbkdf2_sha256(&secret, &unhex(salt).unwrap(), 10, 16);
        let p = serde_json::json!({"nonce": nonce, "salt": salt, "keyPrefix": hex(&key[..2]), "cost": 10, "keyLength": 16, "algorithm": "PBKDF2/SHA-256"});
        let (k, c) = solve_altcha(&p, 1000).unwrap();
        assert!(c <= 3);
        assert!(k.starts_with(&hex(&key[..2])));
    }
}
