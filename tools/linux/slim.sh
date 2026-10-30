#!/bin/bash
# Trims DeepSeek Harness in the image to what its phone use runs - measured, not guessed: a scripted model calls every
# tool in a headless run and in an ACP session (resumed in a second process, as the app does after a restart), with
# trace.mjs listing each file loaded, resolved or read; prune.py drops the rest (the GUI, the web and desktop apps,
# office, speech...). Then the same runs without the trace must still pass - else the build stops - and their second
# round warms the compile cache the image ships with. Run inside the app's Linux after build.sh's npm install, with
# /workspace/trace holding trace.mjs, script_model.py, acp_drive.py, prune.py and /workspace/dedupe.py.
set -euo pipefail
D=/opt/node/lib/node_modules/@deepseek-ai/dsh
T=/workspace/trace
W=/tmp/slim
DSH_ARGS="--patch /etc/geode/dsh-geode.yml"
export GEODE_BASE_URL=http://127.0.0.1:18099/v1 GEODE_KEY=x GEODE_MODEL=geode DSH_PERMISSION_MODE=danger-full-access
BASE_NODE="--max-old-space-size=256 --max-semi-space-size=1"

python3 $T/script_model.py 18099 >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 2

# both modes once: [trace prefix]; fails when either does not finish its work
both() {
  rm -rf $W && mkdir -p $W && cd $W
  local opts="$BASE_NODE"
  [ -n "${1:-}" ] && opts="$opts --import=file://$T/trace.mjs"
  NODE_OPTIONS="$opts" TRACE_OUT="${1:-/dev/null}-headless" DSH_HOME=$W/.dsh \
      dsh --profile headless $DSH_ARGS --json "Use every tool once." > $W/headless.jsonl 2> $W/headless.err
  grep -q '"type":"final"' $W/headless.jsonl || { cat $W/headless.err; exit 1; }
  NODE_OPTIONS="$opts" TRACE_OUT="${1:-/dev/null}-acp" DSH_HOME=$W/.dsh \
      python3 $T/acp_drive.py -- dsh --profile acp $DSH_ARGS --patch /etc/geode/dsh-geode-acp.yml | tail -3
  cd /
}

echo "before: $(du -sh $D | cut -f1), $(find $D -type f | wc -l) files"
both /tmp/trace
python3 $T/prune.py $D /tmp/trace-headless /tmp/trace-acp
# what is never run: types, TypeScript sources, maps, docs, tests; node-pty's builds for other systems
find $D -type f \( -name "*.d.ts" -o -name "*.d.mts" -o -name "*.d.cts" -o -name "*.map" -o -name "*.ts" \
    -o -name "*.mts" -o -name "*.cts" -o -iname "README*" -o -iname "CHANGELOG*" -o -iname "HISTORY*" \) -delete
find $D -type d \( -name test -o -name tests -o -name __tests__ -o -name docs -o -name example -o -name examples \) \
    -prune -exec rm -rf {} + 2>/dev/null || true
find $D -path "*node-pty/prebuilds/*" -maxdepth 8 -type d ! -name "linux-arm64" ! -name prebuilds -prune \
    -exec rm -rf {} + 2>/dev/null || true
python3 /workspace/dedupe.py $D/node_modules
echo "after: $(du -sh $D | cut -f1), $(find $D -type f | wc -l) files"

# it still works - and the compile cache the image ships is warm from these runs
export NODE_COMPILE_CACHE=/var/cache/node-compile
rm -rf $NODE_COMPILE_CACHE && mkdir -p $NODE_COMPILE_CACHE
both
echo -n "warm start: "; both 2>&1 | grep -E "initialize|resumed" | tr '\n' ' '; echo
rm -rf $W /tmp/trace-* /workspace/a.txt /workspace/b.py /workspace/img.png
