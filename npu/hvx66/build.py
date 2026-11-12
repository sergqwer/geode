"""npu/hvx66/build.py - builds Geode's cDSP module for Hexagon v66 class DSPs (libgeode_hvx_skel_v66.so: HVX integer
vectors, no HMX, no HVX floats - Snapdragon 7xx / 6xx) and its adb probe (hvx_probe).

The SDK's hexagon-clang 21 no longer targets v66, so the DSP side is built by zig's LLVM (pip package ziglang, in a
venv: GEODE_ZIG_PYTHON names its python.exe) - a position-independent shared library for an unsigned PD, its libc /
QuRT symbols resolved by the DSP's loader. The interface (geode_hvx.idl) goes through the SDK's qaic as npu/build.py's.

    py -3.13 -I npu/hvx66/build.py [--push] [--serial 8d40db98]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SDK = pathlib.Path(os.environ.get("HEXAGON_SDK_ROOT", r"C:\Hexagon_SDK\6.7.0.0"))
TOOLS = SDK / "tools" / "HEXAGON_Tools" / "21.0.04" / "Tools"   # (only its DSP libc headers: types.h for QuRT's posix)
NDK = pathlib.Path(os.environ.get("ANDROID_NDK", r"D:\android-tools\sdk\ndk\29.0.14206865"))
NDK_CC = NDK / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin" / "aarch64-linux-android30-clang.cmd"
ZIG_PY = pathlib.Path(os.environ.get("GEODE_ZIG_PYTHON", sys.executable))
OUT = HERE.parent / "out"


def run(cmd: list) -> None:
    print(" ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode:
        sys.exit(r.returncode)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--push", action="store_true", help="the skel and the probe to /data/local/tmp/npu")
    p.add_argument("--app", action="store_true", help="the skel into the app's jniLibs (the device report opens it)")
    p.add_argument("--serial", default="8d40db98")
    a = p.parse_args(argv)
    gen = OUT / "gen66"
    gen.mkdir(parents=True, exist_ok=True)
    incs = [SDK / "incs", SDK / "incs" / "stddef"]
    run([SDK / "ipc" / "fastrpc" / "qaic" / "WinNT" / "qaic.exe", "-mdll", "-o", gen,
         *[x for i in incs for x in ("-I", i)], HERE / "geode_hvx.idl"])

    # the DSP side: v66 + 128-byte HVX (zig takes them as -mcpu features), no small data (-G0), PIC; QuRT's v68 headers (the API v66 has too)
    qurt = SDK / "rtos" / "qurt" / "computev68" / "include"
    dsp_inc = [gen, HERE, *incs, qurt, qurt / "qurt", qurt / "posix"]
    skel = OUT / "libgeode_hvx_skel_v66.so"
    objs = []
    for src in (HERE / "hvx66.c", gen / "geode_hvx_skel.c"):
        obj = OUT / (src.stem + ".v66.o")
        run([ZIG_PY, "-m", "ziglang", "cc", "--target=hexagon-freestanding-none", "-mcpu=hexagonv66+hvxv66+hvx_length128b",
             "-O2", "-G0", "-fPIC", "-Wall", "-fno-zero-initialized-in-bss", "-fdata-sections",
             "-ffunction-sections", "-fno-stack-protector", *[x for i in dsp_inc for x in ("-I", i)],
             "-idirafter", TOOLS / "target" / "hexagon" / "include",
             "-c", src, "-o", obj])
        objs.append(obj)
    # (zig's lld directly: its cc links no shared library for a freestanding target; undefined symbols - QuRT, libc -
    # left to the DSP's loader)
    # the layout the DSP's loader takes (as hexagon-link's): two segments, RX and RW, 4 KB pages, no RELRO (its
    # loader "cannot protect segment 1" of lld's default four at 64 KB, nor a segment not page-aligned in the file)
    run([ZIG_PY, "-m", "ziglang", "ld.lld", "-shared", "-Bsymbolic", "--build-id", "-soname", skel.name,
         "--no-rosegment", "-z", "norelro", "-z", "nognustack", "-z", "separate-loadable-segments",
         "-z", "max-page-size=4096",
         "-z", "common-page-size=4096", "--hash-style=sysv", "-o", skel, *objs])

    host_inc = [x for i in [gen, *incs] for x in ("-I", i)]
    rpc = f"-L{SDK / 'ipc' / 'fastrpc' / 'remote' / 'ship' / 'android_aarch64'}"
    probe = OUT / "hvx_probe"
    run([NDK_CC, "-O2", "-Wall", *host_inc, HERE / "hvx_probe.c", gen / "geode_hvx_stub.c", rpc, "-lcdsprpc", "-lm", "-o", probe])
    print(", ".join(f"{f} ({f.stat().st_size} B)" for f in (skel, probe)))
    if a.app:
        dst = HERE.parent.parent / "android" / "app" / "src" / "main" / "jniLibs" / "arm64-v8a"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / skel.name).write_bytes(skel.read_bytes())
        print(f"copied into {dst}")
    if a.push:
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        adb = [r"D:\android-tools\sdk\platform-tools\adb.exe", "-s", a.serial]
        subprocess.run(adb + ["shell", "mkdir -p /data/local/tmp/npu"], env=env, check=True)
        for f in (skel, probe):
            subprocess.run(adb + ["push", str(f), f"/data/local/tmp/npu/{f.name}"], env=env, check=True)
        subprocess.run(adb + ["shell", "chmod 755 /data/local/tmp/npu/hvx_probe"], env=env, check=True)


if __name__ == "__main__":
    main()
