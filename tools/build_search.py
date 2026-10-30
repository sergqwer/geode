"""Builds geode-search (geode/search, Rust) for Android arm64 and puts it in the app as an executable native library.

    python tools/build_search.py

The app runs it from its native library folder (where Android lets an app exec files: useLegacyPackaging keeps it
there, extracted): android/app/src/main/jniLibs/arm64-v8a/libgeodesearch.so (gitignored). The cross-build itself
(BoringSSL through the NDK, static libc++) is configured in geode/search/.cargo/config.toml; the target folder is
short (D:/gst) because BoringSSL's build trips over Windows' path limit from a deep one.
"""
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATE = ROOT / "search"
TARGET_DIR = pathlib.Path(os.environ.get("GEODE_SEARCH_TARGET", r"D:\gst"))
OUT = ROOT / "android" / "app" / "src" / "main" / "jniLibs" / "arm64-v8a" / "libgeodesearch.so"


def main() -> None:
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET_DIR))
    r = subprocess.run(["cargo", "build", "--release", "--target", "aarch64-linux-android"], cwd=CRATE, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)
    built = TARGET_DIR / "aarch64-linux-android" / "release" / "geode-search"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, OUT)
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
