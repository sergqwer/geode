"""Fetches sherpa-onnx's Android library (the runtime of Geode's own voice, Supertonic 3) into the app.

    python tools/fetch_sherpa.py

sherpa-onnx v2.0.1's release build with ONNX Runtime linked in statically (one library, Apache-2.0 / MIT), arm64 only:
android/app/src/main/jniLibs/arm64-v8a/libsherpa-onnx-jni.so (gitignored). The archive and the library are checked
against the hashes below; its Kotlin side (com.k2fsa.sherpa.onnx, the JNI's own names) is kept in the app's sources.
"""
import hashlib
import io
import pathlib
import sys
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = "2.0.1"
URL = (f"https://github.com/k2-fsa/sherpa-onnx/releases/download/v{VERSION}/"
       f"sherpa-onnx-v{VERSION}-android-static-link-onnxruntime.tar.bz2")
ARCHIVE_SHA256 = "fe1c9ceffd86770218a1d4bba6b410d6f726b2b8dda6a5488499b05250b53aea"
MEMBER = "jniLibs/arm64-v8a/libsherpa-onnx-jni.so"
LIB_SHA256 = "491b1b6e287fdfe5e0e1ffa918f5e5556299cb14bea7b7b8a0c19468f4156193"
OUT = ROOT / "android" / "app" / "src" / "main" / "jniLibs" / "arm64-v8a" / "libsherpa-onnx-jni.so"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    if OUT.is_file() and sha256(OUT.read_bytes()) == LIB_SHA256:
        print(f"{OUT} is there")
        return
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    if sha256(data) != ARCHIVE_SHA256:
        sys.exit("the archive's hash differs from the one pinned here")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as t:
        # (its members may start with "./")
        name = next(n for n in t.getnames() if n.removeprefix("./") == MEMBER)
        lib = t.extractfile(name).read()
    if sha256(lib) != LIB_SHA256:
        sys.exit("the library's hash differs from the one pinned here")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(lib)
    print(f"{OUT} ({len(lib) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
