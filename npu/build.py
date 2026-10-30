"""npu/build.py - builds Geode's cDSP module (libgeode_npu_skel.so, Hexagon v79+), the adb probe (npu_probe) and
the app's debug hook library (libgeode_npu.so, npu_app.c).

The interface (geode_npu.idl) goes through the SDK's qaic into a stub (the Android side) and a skel (the DSP side);
the DSP side is built by the SDK's hexagon-clang as a position-independent shared library for an unsigned PD; the
probe and the hook library by the NDK's clang, linked against the SDK's libcdsprpc.so (the phone's own is used at run
time). One skel per Hexagon version (libgeode_npu_skel_vNN.so; the host picks the phone's at run time, npu_api.c) and
the newest one also under the plain name the adb probe opens; v73 / v75 lack the IEEE single-precision HVX the decode
products use, so their skels have the prompt's HMX products only. --app copies the skels and libgeode_npu.so into the
app's jniLibs (android/app/src/main/jniLibs: every build ships them, the release included).

    python npu/build.py [--arch all|v73|v75|v79|v81] [--push] [--app]      # --push: to /data/local/tmp/npu
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SDK = pathlib.Path(os.environ.get("HEXAGON_SDK_ROOT", r"C:\Hexagon_SDK\6.7.0.0"))
TOOLS = SDK / "tools" / "HEXAGON_Tools" / "21.0.04" / "Tools"
NDK = pathlib.Path(os.environ.get("ANDROID_NDK", r"D:\android-tools\sdk\ndk\29.0.14206865"))
NDK_CC = NDK / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin" / "aarch64-linux-android30-clang.cmd"
OUT = HERE / "out"
ARCHS = ["v73", "v75", "v79", "v81"]   # Snapdragon 8 Gen 2, 8 Gen 3, 8 Elite, 8 Elite Gen 5


def run(cmd: list) -> None:
    print(" ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode:
        sys.exit(r.returncode)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", default="all", help="all, or one of " + ", ".join(ARCHS))
    p.add_argument("--push", action="store_true")
    p.add_argument("--app", action="store_true", help="the skels and libgeode_npu.so into the app's jniLibs")
    p.add_argument("--serial", default="A22511110088")
    a = p.parse_args(argv)
    gen = OUT / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    incs = [SDK / "incs", SDK / "incs" / "stddef"]
    run([SDK / "ipc" / "fastrpc" / "qaic" / "WinNT" / "qaic.exe", "-mdll", "-o", gen,
         *[x for i in incs for x in ("-I", i)], HERE / "geode_npu.idl"])

    archs = ARCHS if a.arch == "all" else [a.arch]
    skels = []
    for arch in archs:
        qurt = SDK / "rtos" / "qurt" / f"compute{arch}" / "include"
        dsp_inc = [gen, HERE, *incs, qurt, qurt / "qurt", qurt / "posix"]
        skel = OUT / f"libgeode_npu_skel_{arch}.so"
        run([TOOLS / "bin" / "hexagon-clang.exe", f"-m{arch}", "-mhvx", "-mhvx-length=128B", "-mhmx", "-O2", "-G0",
             "-fpic", "-Wall", "-fno-zero-initialized-in-bss", "-fdata-sections",
             *[x for i in dsp_inc for x in ("-I", i)],
             "-shared", "-Wl,-Bsymbolic", "-Wl,--no-threads", "-Wl,--build-id",
             f"-Wl,-L{TOOLS / 'target' / 'hexagon' / 'lib' / arch / 'G0' / 'pic'}",
             "-o", skel, f"-Wl,-soname,{skel.name}",
             HERE / "dsp" / "npu_imp.c", HERE / "dsp" / "dec.c", gen / "geode_npu_skel.c", "-lc"])
        skels.append(skel)
    skel = OUT / "libgeode_npu_skel.so"
    skel.write_bytes((OUT / "libgeode_npu_skel_v79.so" if "v79" in archs else skels[-1]).read_bytes())

    host_inc = [x for i in [gen, *incs] for x in ("-I", i)]
    rpc = f"-L{SDK / 'ipc' / 'fastrpc' / 'remote' / 'ship' / 'android_aarch64'}"
    probe = OUT / "npu_probe"
    run([NDK_CC, "-O2", "-Wall", *host_inc, HERE / "host" / "npu_probe.c", HERE / "host" / "probe.c", HERE / "host" / "npu_api.c",
         gen / "geode_npu_stub.c", rpc, "-lcdsprpc", "-lm", "-o", probe])
    hook = OUT / "libgeode_npu.so"
    run([NDK_CC, "-O2", "-Wall", "-fPIC", "-shared", *host_inc, HERE / "host" / "npu_app.c", HERE / "host" / "probe.c",
         HERE / "host" / "npu_api.c", HERE / "host" / "report.c", gen / "geode_npu_stub.c", rpc, "-lcdsprpc", "-lm",
         "-Wl,-soname,libgeode_npu.so", "-o", hook])
    print(", ".join(f"{f} ({f.stat().st_size} B)" for f in (*skels, probe, hook)))
    if a.app:
        dst = HERE.parent / "android" / "app" / "src" / "main" / "jniLibs" / "arm64-v8a"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (*skels, hook):
            (dst / f.name).write_bytes(f.read_bytes())
        print(f"copied into {dst}")
    if a.push:
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        adb = ["adb", "-s", a.serial]
        subprocess.run(adb + ["shell", "mkdir -p /data/local/tmp/npu"], env=env, check=True)
        for f in (skel, *skels, probe, hook):
            subprocess.run(adb + ["push", str(f), f"/data/local/tmp/npu/{f.name}"], env=env, check=True)
        subprocess.run(adb + ["shell", "chmod 755 /data/local/tmp/npu/npu_probe"], env=env, check=True)


if __name__ == "__main__":
    main()
