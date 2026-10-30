#!/system/bin/sh
# npu/host/apptest.sh - the probe in the app's own domain (run-as): does an app process reach the cDSP?
# Copies the probe and the skel into files/npu_test of the app (a directory of its own) and runs it there.
P=xyz.tifu.geode
run-as $P sh -c 'mkdir -p files/npu_test; id; cat /proc/self/attr/current; echo'
cat /data/local/tmp/npu/npu_probe | run-as $P sh -c 'cat > files/npu_test/npu_probe; chmod 755 files/npu_test/npu_probe'
cat /data/local/tmp/npu/libgeode_npu_skel.so | run-as $P sh -c 'cat > files/npu_test/libgeode_npu_skel.so'
run-as $P sh -c 'cd files/npu_test && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH="$PWD;/vendor/dsp/cdsp" timeout 30 ./npu_probe 1 2>&1 | head -6'
run-as $P sh -c 'rm -rf files/npu_test'
