#!/usr/bin/env bash
# What the MICRO 2026 tutorial's slides run, in the image: Hands-on 1 (with the answer copied over,
# tests/xlane_reduce/run.sh passes), Hands-on 2's four vcix cases, Hands-on 3's PyTorch op on its machine file (its
# values and, the machine file turning timing mode on, its cycles), and the JupyterLab the hub starts the image with. Every model comes from
# /workspace/Coprocessor/build.
set -euo pipefail
cd /workspace/Coprocessor
cp examples/xlane_reduce/answer/xlane_reduce.hpp examples/xlane_reduce/xlane_reduce.hpp
scripts/build.sh -j "$(nproc)" build
tests/xlane_reduce/run.sh build
cd /workspace/pytorchsim-triton-compiler
for c in vcix_vexp_lane_layout vcix_xlu_broadcast_lane0 vcix_xlu_broadcast_big_tile vcix_layout_the_machine_cannot_take; do
  python3 tests/pytorchsim-compile "tests/cases/mechanism/vcix/$c.py"
done
cd /workspace/PyTorchSim
TORCHSIM_TARGET=/workspace/Coprocessor/tests/xlane_reduce/pytorch/xlane.yml \
  python3 /workspace/Coprocessor/tests/xlane_reduce/pytorch/xlane_op.py
jupyterhub-singleuser --version
