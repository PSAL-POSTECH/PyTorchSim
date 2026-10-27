#!/usr/bin/env bash
# Build the compiler and its simulators (RISC-V toolchain, pk, Spike, gem5, TOGSim) at the PyTorchSim-Triton-Backend
# commit third_party/pytorchsim-triton-backend.json pins; togsim wants conan 1.56 and cmake on PATH. What it prints
# last (TORCHSIM_COMPILE_DIR, and $PREFIX/env.sh for COPROCESSOR_ENV_ROOT and PYTORCHSIM_PREFIX) is PyTorchSim's environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="${ROOT}/third_party/pytorchsim-triton-backend.json"
PREFIX="${PYTORCHSIM_PREFIX:-/opt/pytorchsim}"
DEST="${TORCHSIM_COMPILE_DIR:-${PREFIX}/pytorchsim-triton-compiler}"

REPO=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pytorchsim_triton_backend']['repository'])" "$PIN")
REF=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pytorchsim_triton_backend']['ref'])" "$PIN")
echo "PyTorchSim-Triton-Backend = ${REPO} @ ${REF} -> ${DEST}"

[ -d "$DEST/.git" ] || git clone "https://github.com/${REPO}.git" "$DEST"
git -C "$DEST" fetch -q origin "$REF"
git -C "$DEST" checkout -q "$REF"
git -C "$DEST" submodule update -q --init third_party/Coprocessor third_party/TOGSim

PYTORCHSIM_PREFIX="$PREFIX" COPROCESSOR_ENV_ROOT="${COPROCESSOR_ENV_ROOT:-${PREFIX}/coprocessor-env}" \
  env -u CC -u CXX "$DEST/setup/toolchain.sh" -j "$(nproc)" llvm triton vcixenv togsim vcix dialect ext env
echo "export TORCHSIM_COMPILE_DIR=${DEST}; source ${PREFIX}/env.sh"
