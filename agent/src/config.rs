//! What the app tells the agent (environment variables set by Agent.kt), and the harness settings Geode's profile
//! (tools/linux/dsh-geode.yml) gave DeepSeek Harness.

use std::env;
use std::path::PathBuf;

/// Who the agent is and where it runs: the Linux starts bare, the agent installs what a task needs (the one place
/// this is said; the parity tests put the persona their recordings were made with in its stead).
pub const PERSONA_PREFIX: &str = "You are Geode's agent. You run on the user's Android phone, in a small Debian Linux of your own (arm64). It starts bare (bash and the basic tools): install what a task needs with apt, quietly: every line a command prints is read, and reading is slow on a phone (e.g. `apt-get update -qq && apt-get install -y -qq python3 >/dev/null`; nodejs, npm, git, curl the same way), then pip (in a venv, or with --break-system-packages) and npm. What apt installs is shared by all the user's chats; /workspace, /root, /usr/local and /tmp are this chat's own. web_search asks several search engines at once; a bang in the query picks one: !gh GitHub, !so Stack Overflow, !arx arXiv, !pypi, !wp.";
pub const PERSONA_SUFFIX: &str = "Your folder is /workspace: it belongs to this chat and the user sees it. To give the user something to look at (a page, a chart, a picture, a document), save it in /workspace: the files you create appear under your answer and open right in the chat with a tap (HTML and SVG pages, PNG and JPEG images, Markdown, text) - no browser needed. For anything visual write one self-contained HTML file for a phone's screen. The phone's screen is small: keep answers short. Answer in the user's language.";

/// The model's output cap when the route declares none (dsh-llm-pi-ai's default).
pub const DEFAULT_MAX_TOKENS: usize = 32768;

#[derive(Clone, Debug)]
pub struct Config {
    /// The model's OpenAI-compatible API (Geode's, on the phone's loopback) and its key.
    pub base_url: String,
    pub key: String,
    pub model: String,
    pub context_window: usize,
    pub max_tokens: usize,
    pub image_input: bool,
    /// The app's native library folder: libproot.so, libproot-loader.so, libgeodesearch.so.
    pub libdir: Option<PathBuf>,
    /// The Linux image (shared by every chat, read-mostly).
    pub rootfs: Option<PathBuf>,
    /// The chat's own folders: workspace (/workspace), home (/root), local (/usr/local), tmp (/tmp).
    pub chat: PathBuf,
    pub proot_tmp: Option<PathBuf>,
    pub search_cache: Option<PathBuf>,
    /// The user's skills shared by every chat (presented as /root/.agents/skills).
    pub skills: Option<PathBuf>,
    pub persona_prefix: String,
    pub persona_suffix: String,
    pub search_max_results: usize,
    pub search_max_queries: usize,
    pub fetch_max_output_chars: usize,
    pub bash_timeout_ms: u64,
    /// The skill tool and catalog (on: Geode's agent loads skills; off: the profile the phone ran dsh with).
    pub skills_tool: bool,
    /// Compaction when the conversation nears the context (off: as the harness ran on a phone, never).
    pub auto_compact: bool,
}

fn path(name: &str) -> Option<PathBuf> {
    env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

impl Config {
    pub fn from_env() -> Config {
        let chat = path("GEODE_CHAT").unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        Config {
            base_url: env::var("GEODE_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into()),
            key: env::var("GEODE_KEY").unwrap_or_default(),
            model: env::var("GEODE_MODEL").ok().filter(|m| !m.is_empty()).unwrap_or_else(|| "geode".into()),
            context_window: env::var("GEODE_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(32768),
            max_tokens: env::var("GEODE_MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MAX_TOKENS),
            image_input: env::var("GEODE_INPUT").map(|v| v.split(',').any(|s| s.trim() == "image")).unwrap_or(false),
            libdir: path("GEODE_LIBDIR"),
            rootfs: path("GEODE_ROOTFS"),
            proot_tmp: path("GEODE_PROOT_TMP"),
            search_cache: path("GEODE_SEARCH_CACHE"),
            skills: path("GEODE_SKILLS"),
            persona_prefix: env::var("GEODE_PERSONA_PREFIX").unwrap_or_else(|_| PERSONA_PREFIX.into()),
            persona_suffix: env::var("GEODE_PERSONA_SUFFIX").unwrap_or_else(|_| PERSONA_SUFFIX.into()),
            search_max_results: 6,
            search_max_queries: 2,
            fetch_max_output_chars: 12000,
            bash_timeout_ms: 120_000,
            skills_tool: env::var("GEODE_SKILLS_TOOL").map(|v| v != "0").unwrap_or(true),
            auto_compact: env::var("GEODE_AUTO_COMPACT").map(|v| v != "0").unwrap_or(true),
            chat,
        }
    }

    /// A configuration for tests: a chat folder, no Linux, no network.
    pub fn for_chat(chat: PathBuf) -> Config {
        Config {
            base_url: "http://127.0.0.1:1/v1".into(),
            key: "test".into(),
            model: "geode".into(),
            context_window: 32768,
            max_tokens: DEFAULT_MAX_TOKENS,
            image_input: true,
            libdir: None,
            rootfs: None,
            proot_tmp: None,
            search_cache: None,
            skills: None,
            persona_prefix: PERSONA_PREFIX.into(),
            persona_suffix: PERSONA_SUFFIX.into(),
            search_max_results: 6,
            search_max_queries: 2,
            fetch_max_output_chars: 12000,
            bash_timeout_ms: 120_000,
            skills_tool: true,
            auto_compact: true,
            chat,
        }
    }

    pub fn workspace(&self) -> PathBuf {
        self.chat.join("workspace")
    }

    pub fn home(&self) -> PathBuf {
        self.chat.join("home")
    }

    /// Where the agent keeps its sessions (the chat's /root/.geode-agent).
    pub fn sessions_dir(&self) -> PathBuf {
        self.home().join(".geode-agent").join("sessions")
    }

    /// Long outputs saved for the model to read back (the chat's /tmp).
    pub fn tmp(&self) -> PathBuf {
        self.chat.join("tmp")
    }
}
