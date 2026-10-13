#!/usr/bin/env bash
# Build the compiler and the simulators it runs (RISC-V toolchain, pk, Spike, gem5, TOGSim) from source,
# at the PyTorchSim-Triton-Backend commit pinned in third_party/pytorchsim-triton-backend.json -- the
# pin CI's base image is built FROM. The togsim step wants conan 1.56 and cmake on PATH (its setup/system.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="${ROOT}/third_party/pytorchsim-triton-backend.json"
PREFIX="${TORCHSIM_PREFIX:-/workspace}"
DEST="${TORCHSIM_COMPILE_DIR:-${PREFIX}/pytorchsim-triton-compiler}"

REPO=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pytorchsim_triton_backend']['repository'])" "$PIN")
REF=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pytorchsim_triton_backend']['ref'])" "$PIN")
echo "PyTorchSim-Triton-Backend = ${REPO} @ ${REF} -> ${DEST}"

[ -d "$DEST/.git" ] || git clone "https://github.com/${REPO}.git" "$DEST"
git -C "$DEST" fetch -q origin "$REF"
git -C "$DEST" checkout -q "$REF"
git -C "$DEST" submodule update -q --init third_party/vcix-accelerator third_party/TOGSim

TORCHSIM_PREFIX="$PREFIX" VCIX_ENV_ROOT="${VCIX_ENV_ROOT:-${PREFIX}/vcix-env}" \
  env -u CC -u CXX "$DEST/setup/toolchain.sh" -j "$(nproc)" llvm triton vcixenv togsim vcix dialect ext
