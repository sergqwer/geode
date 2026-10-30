// SPDX-License-Identifier: AGPL-3.0-or-later
//! geode-search: SearXNG's metasearch (https://github.com/searxng/searxng, commit c360868e5) ported to Rust: the
//! query syntax, the engines Geode keeps, the merging and ranking of their results. It runs as a program of its
//! own beside Geode (see NOTICE.md). Everything but the network builds without the `net` feature, so the parsers
//! and the ranking are tested on the PC host against pages saved from the phone.

pub mod cache;
pub mod date;
pub mod engines;
pub mod fetchrules;
pub mod html;
pub mod http;
pub mod js;
pub mod locales;
pub mod output;
pub mod query;
pub mod results;
pub mod settings;
pub mod url;
pub mod util;

#[cfg(feature = "net")]
pub mod fetch;
#[cfg(feature = "net")]
pub mod net;
#[cfg(feature = "net")]
pub mod search;
