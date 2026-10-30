// SPDX-License-Identifier: AGPL-3.0-or-later
//! geode-search [--max N] [--lang xx] [--engines a,b] [--categories c] [--cache DIR] [--stats] [--] QUERY
//! geode-search --fetch URL
//!
//! One SearXNG search; its merged and ranked results as one JSON object on stdout. Exit 0 when an engine
//! answered, 1 (and one line on stderr) when none did, 2 for a usage error. --stats: the engines' times, the
//! total and the peak memory on stderr. --dump DIR (debug): every response saved there.
//!
//! --fetch URL: one page for the agent's web_fetch, as JSON {url, statusCode, kind, content, truncated}; exit 1 and
//! the reason on stderr when it cannot be fetched (DeepSeek Harness's fetch rules, src/fetch.rs).

#[cfg(feature = "net")]
fn main() {
    use geode_search::search::{Options, run};
    use std::io::Write;

    let t0 = std::time::Instant::now();
    if std::env::args().nth(1).as_deref() == Some("--fetch") {
        let url = std::env::args().nth(2).unwrap_or_default();
        let rt = tokio::runtime::Builder::new_current_thread().enable_time().enable_io().build().expect("runtime");
        match rt.block_on(geode_search::fetch::fetch(&url)) {
            Ok(json) => {
                let mut o = std::io::stdout().lock();
                let _ = writeln!(o, "{json}");
                let _ = o.flush();
                std::process::exit(0)
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1)
            }
        }
    }
    let mut opts = Options { max: 10, phone_lang: std::env::var("GEODE_SEARCH_LANG").ok(), ..Default::default() };
    let mut stats = false;
    let mut words: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    let usage = |m: &str| -> ! {
        eprintln!(
            "geode-search: {m}\nusage: geode-search [--max N] [--lang xx] [--engines a,b] [--categories c] [--cache DIR] [--stats] [--] QUERY"
        );
        std::process::exit(2)
    };
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().unwrap_or_else(|| usage(&format!("{name} needs a value")));
        match a.as_str() {
            "--" => {
                words.extend(args.by_ref());
                break;
            }
            "--max" => opts.max = value("--max").parse().unwrap_or_else(|_| usage("--max wants a number")),
            "--lang" => opts.lang = Some(value("--lang")),
            "--engines" => opts.engines = Some(value("--engines").split(',').map(String::from).collect()),
            "--categories" => opts.categories = Some(value("--categories").split(',').map(String::from).collect()),
            "--cache" => opts.cache = Some(value("--cache").into()),
            "--dump" => opts.dump = Some(value("--dump").into()),
            "--stats" => stats = true,
            "-h" | "--help" => usage("one web search, as JSON"),
            s if s.starts_with("--") => usage(&format!("unknown option {s}")),
            _ => words.push(a),
        }
    }
    opts.query = words.join(" ");

    let rt = tokio::runtime::Builder::new_current_thread().enable_time().enable_io().build().expect("runtime");
    let out = rt.block_on(async {
        let net = match geode_search::net::Net::new(std::time::Duration::from_secs(10)) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("geode-search: network: {e}");
                std::process::exit(1)
            }
        };
        run(opts, &net).await
    });
    if let Some(json) = &out.json {
        let mut o = std::io::stdout().lock();
        let _ = writeln!(o, "{json}");
        let _ = o.flush();
    }
    if stats {
        for (e, ms, err) in &out.timings {
            eprintln!("{e:>14} {ms:>5} ms {}", err.as_deref().unwrap_or("ok"));
        }
        let hwm = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
        });
        eprintln!(
            "{:>14} {:>5} ms, peak RSS {} kB",
            "total",
            t0.elapsed().as_millis(),
            hwm.unwrap_or_else(|| "?".into())
        );
    }
    if let Some(m) = &out.message {
        eprintln!("{m}");
    }
    // the runtime's connections need no orderly goodbye: the answer is out
    std::process::exit(out.code);
}

#[cfg(not(feature = "net"))]
fn main() {
    eprintln!("geode-search: built without the network (the `net` feature)");
    std::process::exit(2);
}
