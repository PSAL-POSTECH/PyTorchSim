#!/usr/bin/env bash
# What the MICRO 2026 tutorial's slides run, in the image: Hands-on 1 (the start builds; the answer passes
# tests/xlane_reduce/run.sh), Hands-on 2's four vcix cases, and the appendix's PyTorch op on its machine file.
set -euo pipefail
cd /workspace/vcix-accelerator
git checkout -q tutorial/xlane-reduce-solution
scripts/build.sh -j "$(nproc)" build
tests/xlane_reduce/run.sh build
cd /workspace/pytorchsim-triton-compiler
for c in vcix_vexp_lane_layout vcix_xlu_broadcast_lane0 vcix_xlu_broadcast_big_tile vcix_layout_the_machine_cannot_take; do
  python3 tests/torchsim-compile "tests/cases/mechanism/vcix/$c.py"
done
cp /workspace/vcix-accelerator/build/libxlane_tpuv6e.so "$TORCHSIM_PREFIX/vcix-build/"
cd /workspace/PyTorchSim
TORCHSIM_TARGET=/workspace/vcix-accelerator/tests/xlane_reduce/pytorch/xlane_tpuv6e.yml \
  python3 /workspace/vcix-accelerator/tests/xlane_reduce/pytorch/xlane_op.py
