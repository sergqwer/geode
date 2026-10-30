# geode-agent — notice

geode-agent is DeepSeek Harness rewritten in Rust for Geode: the agent the Geode app runs natively on the phone
(`libgeodeagent.so`), speaking the Agent Client Protocol with the app over stdio. Its behaviour, prompts, tool
schemas, tool result wording and limits are ported from:

- **DeepSeek Harness** `@deepseek-ai/dsh` 0.2.1-rc.2 — Copyright (c) 2026 DeepSeek, MIT License (`LICENSE-MIT`).
- **pi-ai** `@earendil-works/pi-ai` 1.4.0 (the harness's model adapter) — Copyright (c) Mario Zechner, MIT License
  (same terms as `LICENSE-MIT`, with that copyright).

What came from where (dsh package → Rust module):

| DeepSeek Harness / pi-ai | geode-agent |
|---|---|
| dsh-acp (ACP bridge: initialize, session new/resume/list/close/prompt/cancel, updates) | `src/acp.rs` |
| dsh-agent, dsh-agent-loop (turn, steps, tool execution, runtime-context projection) | `src/agent.rs` |
| dsh-system-prompt, the tool packages' guidance sections, dsh-working-directory, dsh-sandbox-policy, dsh-user-approval, dsh-agent-instructions | `src/prompt.rs` |
| dsh-token-meter (occupancy), pi-ai utils/estimate + api/simple-options (completion budget), pi-ai api/openai-completions + dsh-llm-pi-ai (request conversion) | `src/msg.rs` |
| dsh-llm, dsh-llm-pi-ai, pi-ai openai-completions (streamed chat completions) | `src/model.rs` |
| dsh-tool-fs, dsh-fs-local, dsh-fs-observation-policy (read, write, edit, read_image) | `src/tools/fs.rs` |
| dsh-tool-fs-search (glob, grep; ripgrep's rules through the ignore/globset/regex crates) | `src/tools/search.rs` |
| dsh-tool-bash, dsh-bash-local, dsh-subprocess-local (bash, output capture) | `src/tools/bash.rs`, `src/linux.rs` |
| dsh-tool-todo, dsh-tool-working-directory | `src/tools/misc.rs` |
| dsh-tool-web, dsh-web (web_search, web_fetch rendering; turndown's rules through htmd) | `src/tools/web.rs` |
| dsh-spill-policy, dsh-spill-local, dsh-output-retention | `src/tools/mod.rs` |
| dsh-skill, dsh-skill-filesystem, dsh-tool-skill | `src/skills.rs` |
| dsh-mcp-client, dsh-mcp-resources | `src/mcp.rs` |
| dsh-compaction, dsh-compaction-basic, dsh-command-compact, dsh-commands | `src/compact.rs`, `src/commands.rs` |
| dsh-session, dsh-session-persistence-jsonl (sessions kept for resume) | `src/session.rs` |
| the 15 tool schemas the harness sent the model (recorded on a phone) | `src/tools/specs.json` |

Deliberate differences from the harness:

- Commands typed in the chat (`/compact`, `/clear`, `/skills`, `/mcp`, `/help`) are answered by the agent and
  advertised over ACP; dsh's ACP bridge passed them to the model as text.
- The conversation is compacted on its own at 80% of the context (and when too little room is left for an answer,
  or the model reports an overflow); dsh's 64K-token headroom kept that from ever happening in a phone's context.
- Words stream to the client as the model writes them (dsh sent each message's text whole when it was committed).
- read_image sends the image as it is (no re-encoding or downscaling); Geode's own API scales images.
- The file tools work on the chat's folders directly; only bash and stdio MCP servers start the chat's Linux
  (proot), exactly as Geode binds it.
- Geode's ACP extension `geode/linux`: a request ({"active": true}) before the first Linux process starts (the
  app gives the Linux memory back from the model; 5 s at most), a notification ({"active": false}) after the last one
  ends. dsh has no such thing.
- Sessions are kept in the agent's own JSONL (written whole aside and renamed, never hard-linked).
- Web search and page fetching go through geode-search (`geode/search`, AGPL-3.0), which this program runs as a
  separate process and never links.

`tests/parity.rs` replays runs recorded from the real harness on a phone (`tests/ref/`) and requires every request
sent to the model to be byte-identical and the ACP stream to tell the client the same things.
