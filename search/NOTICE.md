# geode-search — notice

geode-search is a port to Rust of parts of **SearXNG** (https://github.com/searxng/searxng, commit
`c360868e5b7c01e463a362a53518190d5a3b1097`), a free metasearch engine licensed under the GNU Affero General Public
License, version 3 or later. geode-search is licensed the same way: [LICENSE](LICENSE) is the AGPL-3.0 text taken
from that commit. Copyright of the ported code stays with the SearXNG authors; the port is © Geode's author.

## How it relates to Geode

geode-search is a program of its own. The Geode app ships it in its native library folder and runs it as a
separate process (`geode-search [options] QUERY`), reading one JSON object from its standard output — the same way
it would run any command-line tool. Geode's own code is not derived from it and is not covered by this license.
The complete corresponding source of geode-search is this folder of Geode's repository.

## What was ported (SearXNG → this crate)

| SearXNG (Python) | geode-search (Rust) |
|---|---|
| `searx/query.py` (RawTextQuery: `<timeout`, `:lang`, `!engine` / `!category`, `!!`) | `src/query.rs` |
| `searx/search/__init__.py` (Search: engine selection, timeouts, threads) | `src/search.rs` (one async task per engine) |
| `searx/search/processors/{abstract,online}.py` (request parameters, suspension, errors) | `src/engines/mod.rs`, `src/http.rs`, `src/cache.rs` |
| `searx/network/raise_for_httperror.py`, `searx/exceptions.py` | `src/http.rs` |
| `searx/network` (curl_cffi with browser impersonation) | `src/net.rs` (wreq + wreq-util: Chrome 149 on Android) |
| `searx/results.py`, `searx/result_types/_base.py` (merge, score, order, normalisation) | `src/results.rs`, `src/url.rs` |
| `searx/utils.py` (html_to_text, extract_text, JS object → JSON, solve_altcha, extr) | `src/html.rs`, `src/js.rs`, `src/engines/mojeek.rs`, `src/engines/startpage.rs` |
| `searx/enginelib/__init__.py` EngineCache, SuspendedStatus | `src/cache.rs` (a JSON file in `--cache DIR`) |
| `searx/enginelib/traits.py`, `searx/locales.py`, `data/engine_traits.json`, Babel | `src/locales.rs` + `src/data/locales.json` (see below) |
| `searx/engines/{duckduckgo,brave,mojeek,wikipedia,github,stackexchange,arxiv,bing,google,startpage}.py` | `src/engines/*.rs` |
| geode's `engines/pypi_json.py` (from the slim Python tree) | `src/engines/pypi.rs` |
| geode's `settings.yml` | `src/settings.rs` |

`src/data/locales.json` holds SearXNG's own answers, per locale, for what the engines derive from it (the
Accept-Language header, DuckDuckGo's region, Brave's country and UI language, Google's hl/lr/cr, Bing's market,
Startpage's region and language, Mojeek's lb/arc, the Wikipedia host), computed by running SearXNG's code
(`tools/search/gen_rust_data.py` over the slim tree of `tools/search/slim.py`), so the port needs no Babel.

## Differences from upstream

- One browser identity for every request (Chrome 149 on Android, TLS/HTTP2/headers consistent): no Nokia
  User-Agent for Google, no desktop Firefox User-Agent for DuckDuckGo. Bing then serves its mobile markup, whose
  title link wraps the `<h2>`; both markups are read.
- Wikipedia's summary API answers a redirect page with a 302; up to two redirects are followed.
- External bangs (`!!g` …) are left out (they only redirect a browser); DuckDuckGo quotes every `!word`.
- Engine suspensions and engine state persist between runs in the `--cache` folder (SearXNG keeps them in memory
  for its server's lifetime).
- A category bang (`!general`, `!it`) leaves the engines disabled by default out, as SearXNG's preferences do.
- No web server, UI, plugins, answerers, translations or metrics.

## Checks

- `cargo test --no-default-features` (no network, on the PC): unit tests, the engines' parsers on pages saved
  from the phone (`tests/fixtures`, `geode-search --dump DIR`), and `tests/parity.rs`: the merged, scored and
  ordered results equal SearXNG's own for the same pages (`tools/search/parity.py` writes
  `tests/fixtures/parity.json`).
- Build for the phone: `CARGO_TARGET_DIR=D:/gst cargo build --release --target aarch64-linux-android`.
