//! The MCP client against a real server (tools/linux/trace/mcp_demo.py, run by the PC's Python): what the features
//! reference recorded dsh's client returning, word for word.

use geode_agent::linux::{Cancel, Outcome, Shell};
use geode_agent::mcp::McpHub;
use geode_agent::msg::Block;
use serde_json::json;
use std::path::{Path, PathBuf};

struct PcPython;
impl Shell for PcPython {
    fn run(&self, _: &str, _: &str, _: u64, _: &Cancel) -> Result<Outcome, String> {
        Err("no Linux".into())
    }
    fn spawn(&self, argv: &[String], _env: &[(String, String)], _cwd: &str) -> Result<std::process::Child, String> {
        assert_eq!(argv[0], "/usr/bin/python3");
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let py = root.join(".venv/Scripts/python.exe");
        let py = if py.exists() { py } else { PathBuf::from("python3") };
        std::process::Command::new(py)
            .arg("-I")
            .arg(root.join("tools/linux/trace/mcp_demo.py"))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
    }
}

fn text(o: &geode_agent::tools::Output) -> &str {
    match &o.blocks[0] {
        Block::Text { text } => text,
        _ => panic!(),
    }
}

#[test]
fn the_demo_server() {
    let (hub, errors) = McpHub::connect(
        &json!([{"name": "demo", "command": "/usr/bin/python3", "args": ["/workspace/.ref-tools/mcp_demo.py"], "env": []}]),
        &PcPython,
        "/workspace",
    );
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(hub.server_names(), vec!["demo"]);
    let specs = hub.tool_specs();
    assert_eq!(specs[0].name, "mcp__demo__echo");
    assert_eq!(
        serde_json::to_string(&specs[0].parameters).unwrap(),
        r#"{"type":"object","properties":{"text":{"type":"string","description":"What to echo."}},"required":["text"]}"#
    );
    let c = Cancel::new();
    assert_eq!(text(&hub.call_tool("mcp__demo__echo", &json!({"text": "hi from mcp"}), &c).unwrap()), "HI FROM MCP");
    assert_eq!(
        hub.resource_tool("list_mcp_resource_templates", &json!({"server": "demo"}), &c).unwrap_err(),
        "no method resources/templates/list"
    );
    assert_eq!(
        text(&hub.resource_tool("read_mcp_resource", &json!({"server": "demo", "uri": "demo://note"}), &c).unwrap()),
        "MCP server: demo\n{\"contents\":[{\"uri\":\"demo://note\",\"mimeType\":\"text/plain\",\"text\":\"A note from the demo server.\"}]}"
    );
    assert!(
        text(&hub.resource_tool("list_mcp_resources", &json!({"server": "demo"}), &c).unwrap())
            .starts_with("MCP server: demo\n{\"resources\":[{\"uri\":\"demo://note\"")
    );
    assert!(hub.resource_tool("list_mcp_resources", &json!({"server": "nope"}), &c).is_err());
}
