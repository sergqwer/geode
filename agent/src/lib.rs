//! Geode's agent: DeepSeek Harness (MIT, @deepseek-ai/dsh 0.2.1-rc.2) rewritten in Rust. The app runs it from its
//! native library folder and speaks ACP with it over stdio; it reaches the model on the phone's loopback, works on
//! the chat's folders directly, and starts the chat's Linux (proot) only to run a command.

pub mod acp;
pub mod agent;
pub mod commands;
pub mod compact;
pub mod config;
pub mod linux;
pub mod mcp;
pub mod model;
pub mod msg;
pub mod paths;
pub mod prompt;
pub mod session;
pub mod skills;
pub mod tools;
pub mod util;
