"""Builds geode-agent (geode/agent: DeepSeek Harness rewritten in Rust) for Android arm64 and puts it in the app as an
executable native library.

    python tools/build_agent.py

The app runs it from its native library folder (where Android lets an app exec files: useLegacyPackaging keeps it
there, extracted): android/app/src/main/jniLibs/arm64-v8a/libgeodeagent.so (gitignored). It is pure Rust but for
ring (rustls, for MCP servers over HTTPS), built by the NDK's clang (geode/agent/.cargo/config.toml).
"""
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATE = ROOT / "agent"
TARGET_DIR = pathlib.Path(os.environ.get("GEODE_AGENT_TARGET", r"D:\gat"))
OUT = ROOT / "android" / "app" / "src" / "main" / "jniLibs" / "arm64-v8a" / "libgeodeagent.so"


def main() -> None:
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET_DIR))
    r = subprocess.run(["cargo", "build", "--release", "--target", "aarch64-linux-android"], cwd=CRATE, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)
    built = TARGET_DIR / "aarch64-linux-android" / "release" / "geode-agent"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, OUT)
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
