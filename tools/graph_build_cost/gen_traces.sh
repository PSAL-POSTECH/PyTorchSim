#!/bin/bash
# Produce trace.so + trace_cycles.tsv for 4 kernels: a conv2d and the matmul with the
# IDENTICAL GEMM shape, at two sizes. The TOGSim binary is not needed here -- the
# compile step fails at the simulate stage (that is expected) but trace.so is written.
set -u
: "${TORCHSIM_DIR:?source the PyTorchSim .envrc first}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="$HERE/config_8x8_nofunc.yml"
sed 's/^pytorchsim_functional_mode: 1/pytorchsim_functional_mode: 0/' \
    "$TORCHSIM_DIR/configs/systolic_ws_8x8_c1_simple_noc_tpuv3.yml" > "$CFG"
export TOGSIM_CONFIG="$CFG"
mkdir -p "$HERE/traces"
gen() {  # name  script  args...
  local name=$1; shift
  bash "$TORCHSIM_DIR/scripts/clear_codegen_cache.sh" >/dev/null 2>&1
  rm -rf /tmp/torchinductor_* "$TORCHSIM_DIR"/outputs/* 2>/dev/null
  python "$@" >/dev/null 2>&1
  local d; d=$(find "$TORCHSIM_DIR/outputs" -name trace.so -exec dirname {} \; | head -1)
  if [ -z "$d" ]; then echo "FAIL: no trace.so produced for $name"; exit 1; fi
  rm -rf "$HERE/traces/$name"; cp -r "$d" "$HERE/traces/$name"; echo "saved $name"
}
# conv2d(x[2,128,14,14], w[512,128,7,7], pad=3)  ==  GEMM(M=392, N=512, K=6272)  <- the OOM case
gen CI36 "$HERE/compile_conv.py" 2 128 512 14 7 1 3
gen MM36 "$HERE/compile_mm.py"   392 6272 512
# conv2d(x[1,128,16,16], w[64,128,7,7], pad=3)   ==  GEMM(M=256, N=64,  K=6272)  <- completes
gen CM16 "$HERE/compile_conv.py" 1 128 64 16 7 1 3
gen MM16 "$HERE/compile_mm.py"   256 6272 64
