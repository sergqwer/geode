// SPDX-License-Identifier: AGPL-3.0-or-later
//! searx.utils' JavaScript helpers, as Brave's engine needs them: a JS object literal made JSON
//! (js_obj_str_to_json_str), bracket matching that skips strings, argument splitting and the free variables of
//! an object literal (to substitute an IIFE's arguments). The regular expressions of the Python code are matched
//! by hand here, with the same leftmost, non-overlapping replacement.

fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// js_obj_str_to_json_str
pub fn js_obj_to_json(js: &str) -> Result<String, String> {
    if js.is_empty() {
        return Err("js_obj_str can't be an empty string".into());
    }
    // _JS_STRING_DELIMITERS.split: text and the quote characters, alternating
    let mut parts: Vec<String> = vec![String::new()];
    for c in js.chars() {
        if matches!(c, '"' | '\'' | '`') {
            parts.push(c.to_string());
            parts.push(String::new());
        } else {
            parts.last_mut().unwrap().push(c);
        }
    }
    let mut in_string: Option<String> = None;
    let mut backslash_before = false;
    for i in 0..parts.len() {
        let mut p = parts[i].clone();
        if in_string.as_deref() == Some(p.as_str()) && !backslash_before {
            in_string = None;
            parts[i] = "\"".into();
        } else if let Some(delim) = in_string.clone() {
            p = p.replace(':', "\u{1}");
            p = string_escapes(&p);
            if delim == "'" {
                p = p.replace('"', "\\\"");
            }
            parts[i] = p.clone();
            if backslash_before && p.starts_with('\'') {
                let prev = &parts[i - 1];
                let n = prev.len() - prev.trim_end_matches('\\').len();
                if n % 2 == 1 {
                    let mut prev = prev.clone();
                    prev.pop();
                    parts[i - 1] = prev;
                }
            }
        } else if matches!(p.as_str(), "\"" | "'" | "`") {
            in_string = Some(p.clone());
            parts[i] = "\"".into();
        } else {
            p = void_to_null(&p);
            p = decimal(&p);
            p = decimal2(&p);
            p = extra_comma(&p);
            parts[i] = p.clone();
        }
        backslash_before = p.ends_with('\\');
    }
    let s = quote_keys(&parts.concat());
    Ok(s.replace('\u{1}', ":").trim().to_string())
}

/// _JS_STRING_ESCAPE_RE.sub(_j2p_process_escape): JS escapes made JSON ones
fn string_escapes(p: &str) -> String {
    let c: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '\\' && i + 1 < c.len() && c[i + 1] != '\n' {
            let e = c[i + 1];
            if "\"\\bfnrtu".contains(e) {
                out.push('\\');
                out.push(e);
            } else if e == 'x' {
                out.push_str("\\u00");
            } else {
                out.push(e);
            }
            i += 2;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

/// void\s+[0-9]+ | void\s*\([0-9]+\) | undefined  →  null
fn void_to_null(p: &str) -> String {
    let c: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len());
    let mut i = 0;
    while i < c.len() {
        if starts(&c, i, "undefined") {
            out.push_str("null");
            i += 9;
            continue;
        }
        if starts(&c, i, "void") {
            let mut j = i + 4;
            let ws = c[j..].iter().take_while(|x| is_ws(**x)).count();
            j += ws;
            let digits = c[j..].iter().take_while(|x| x.is_ascii_digit()).count();
            if ws > 0 && digits > 0 {
                out.push_str("null");
                i = j + digits;
                continue;
            }
            if j < c.len() && c[j] == '(' {
                let d = c[j + 1..].iter().take_while(|x| x.is_ascii_digit()).count();
                if d > 0 && c.get(j + 1 + d) == Some(&')') {
                    out.push_str("null");
                    i = j + d + 2;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

fn starts(c: &[char], i: usize, s: &str) -> bool {
    let n = s.chars().count();
    i + n <= c.len() && c[i..i + n].iter().copied().eq(s.chars())
}

/// At `i` (after one of [ , :): \s*(-?)\s*  — the sign and where the number starts.
fn sign_at(c: &[char], i: usize) -> (bool, usize) {
    let mut j = i;
    while j < c.len() && is_ws(c[j]) {
        j += 1;
    }
    let neg = j < c.len() && c[j] == '-';
    if neg {
        j += 1;
    }
    while j < c.len() && is_ws(c[j]) {
        j += 1;
    }
    (neg, j)
}

/// ([\[\,:])\s*(\-?)\s*([0-9_]*)\.([0-9_]*)  →  \1\2(\3 or 0).(\4 or 0), underscores dropped
fn decimal(p: &str) -> String {
    let c: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len());
    let mut i = 0;
    let digits = |j: usize| c[j..].iter().take_while(|x| x.is_ascii_digit() || **x == '_').count();
    while i < c.len() {
        if matches!(c[i], '[' | ',' | ':') {
            let (neg, j) = sign_at(&c, i + 1);
            let a = digits(j);
            if j + a < c.len() && c[j + a] == '.' {
                let b = digits(j + a + 1);
                let int: String = c[j..j + a].iter().filter(|x| **x != '_').collect();
                let frac: String = c[j + a + 1..j + a + 1 + b].iter().filter(|x| **x != '_').collect();
                out.push(c[i]);
                if neg {
                    out.push('-');
                }
                out.push_str(if int.is_empty() { "0" } else { &int });
                out.push('.');
                out.push_str(if frac.is_empty() { "0" } else { &frac });
                i = j + a + 1 + b;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// ([\[\,:])\s*(\-?)\s*([0-9_]+)  →  \1\2\3, underscores dropped
fn decimal2(p: &str) -> String {
    let c: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len());
    let mut i = 0;
    while i < c.len() {
        if matches!(c[i], '[' | ',' | ':') {
            let (neg, j) = sign_at(&c, i + 1);
            let n = c[j..].iter().take_while(|x| x.is_ascii_digit() || **x == '_').count();
            if n > 0 {
                out.push(c[i]);
                if neg {
                    out.push('-');
                }
                out.extend(c[j..j + n].iter().filter(|x| **x != '_'));
                i = j + n;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// \s*,\s*([\]\}])  →  \1
fn extra_comma(p: &str) -> String {
    let c: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len());
    let mut i = 0;
    while i < c.len() {
        let mut j = i;
        while j < c.len() && is_ws(c[j]) {
            j += 1;
        }
        if j < c.len() && c[j] == ',' {
            let mut k = j + 1;
            while k < c.len() && is_ws(c[k]) {
                k += 1;
            }
            if k < c.len() && matches!(c[k], ']' | '}') {
                out.push(c[k]);
                i = k + 1;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// ([\{\s,])([\$_\w][\$_\w0-9]*)(:)  →  \1"\2"\3
fn quote_keys(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + s.len() / 8);
    let mut i = 0;
    while i < c.len() {
        if c[i] == '{' || c[i] == ',' || is_ws(c[i]) {
            let j = i + 1;
            if j < c.len() && (c[j] == '$' || is_word(c[j])) {
                let n = c[j..].iter().take_while(|x| **x == '$' || is_word(**x)).count();
                if c.get(j + n) == Some(&':') {
                    out.push(c[i]);
                    out.push('"');
                    out.extend(&c[j..j + n]);
                    out.push_str("\":");
                    i = j + n + 1;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// js_find_matching_bracket: the index of the bracket closing the one at `start`, strings skipped.
pub fn find_matching_bracket(c: &[char], start: usize, open: char, close: char) -> Result<usize, String> {
    let (mut depth, mut in_str, mut esc) = (0i32, None::<char>, false);
    for (i, &ch) in c.iter().enumerate().skip(start) {
        if let Some(q) = in_str {
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == q {
                in_str = None;
            }
        } else if matches!(ch, '"' | '\'' | '`') {
            in_str = Some(ch);
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Ok(i);
            }
        }
    }
    Err(format!("unbalanced {open}{close} starting at {start}"))
}

/// js_split_args
pub fn split_args(args: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let (mut last, mut in_str, mut esc, mut depth) = (0usize, None::<char>, false, 0i32);
    let mut with_comma = args.to_vec();
    with_comma.push(',');
    for (idx, &ch) in with_comma.iter().enumerate() {
        if let Some(q) = in_str {
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == q {
                in_str = None;
            }
        } else if matches!(ch, '"' | '\'' | '`') {
            in_str = Some(ch);
        } else if matches!(ch, '(' | '[' | '{') {
            depth += 1;
        } else if matches!(ch, ')' | ']' | '}') {
            depth -= 1;
        } else if ch == ',' && depth == 0 {
            let item: String = args[last.min(args.len())..idx.min(args.len())].iter().collect();
            let item = item.trim();
            if !item.is_empty() {
                out.push(item.to_string());
            }
            last = idx + 1;
        }
    }
    out
}

/// js_find_variable_positions: the variables an object literal's values name, with their positions.
pub fn find_variable_positions(js: &[char]) -> Vec<(String, usize)> {
    let mut matches = Vec::new();
    let (mut in_str, mut esc, mut flat, mut in_value) = (None::<char>, false, false, false);
    let mut cur: Option<(String, usize)> = None;
    let ident_start = |c: char| c.is_ascii_alphabetic() || c == '_' || c == '$';
    for (pos, &c) in js.iter().enumerate() {
        if let Some(q) = in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == q {
                in_str = None;
            }
        } else if matches!(c, '\'' | '"' | '`') {
            in_str = Some(c);
        } else if c == ':' {
            in_value = true;
        } else if matches!(c, ',' | '{' | '}' | '[' | ']') {
            if c == '[' {
                flat = true;
            } else if c != ',' {
                flat = false;
            }
            in_value = flat;
            if let Some(v) = cur.take()
                && !["void0", "true", "false", "null", "undefined"].contains(&v.0.as_str())
            {
                matches.push(v);
            }
        } else if in_value {
            if let Some(v) = cur.as_mut().filter(|_| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
                v.0.push(c);
            } else if ident_start(c) {
                cur = Some((c.to_string(), pos));
            }
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(js: &str) -> serde_json::Value {
        serde_json::from_str(&js_obj_to_json(js).unwrap())
            .unwrap_or_else(|e| panic!("{e}: {}", js_obj_to_json(js).unwrap()))
    }

    #[test]
    fn object_literals() {
        assert_eq!(v("{a:1,b:'x',c:\"y\"}"), serde_json::json!({"a":1,"b":"x","c":"y"}));
        assert_eq!(v("{ a: [1,2,3,], b: .5, c: -1.}"), serde_json::json!({"a":[1,2,3],"b":0.5,"c":-1.0}));
        assert_eq!(v("{a:void 0,b:undefined,c:void(0)}"), serde_json::json!({"a":null,"b":null,"c":null}));
        assert_eq!(
            v("{a:'it\\'s',b:\"q\\\"t\",c:'say \"hi\"'}"),
            serde_json::json!({"a":"it's","b":"q\"t","c":"say \"hi\""})
        );
        assert_eq!(v("{url:'https://x.y/z',t:`a:b`}"), serde_json::json!({"url":"https://x.y/z","t":"a:b"}));
        assert_eq!(v("{s:'\\x41\\n'}"), serde_json::json!({"s":"A\n"}));
        assert_eq!(v("{n:1_000}"), serde_json::json!({"n":1000}));
    }

    #[test]
    fn brackets_and_args() {
        let t: Vec<char> = "{a:\"}\",b:{c:1}} tail".chars().collect();
        assert_eq!(find_matching_bracket(&t, 0, '{', '}').unwrap(), 14);
        let a: Vec<char> = "1, \"a,b\", {x:[1,2]}, f(3,4)".chars().collect();
        assert_eq!(split_args(&a), ["1", "\"a,b\"", "{x:[1,2]}", "f(3,4)"]);
    }

    #[test]
    fn variables() {
        let o: Vec<char> = "{a:b,c:\"d\",e:[f,g],h:{i:true},j:null}".chars().collect();
        let names: Vec<String> = find_variable_positions(&o).into_iter().map(|x| x.0).collect();
        assert_eq!(names, ["b", "f", "g"]);
    }
}
