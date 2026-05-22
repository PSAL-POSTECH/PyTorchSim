#!/usr/bin/env bash
# Build Gem5 / LLVM+MLIR / Spike from source.
#
# Versions are pinned in thirdparty/github-releases.json - the same manifest
# the CI docker image (ghcr.io/psal-postech/torchsim-ci) is built against.
# Cloning untagged HEADs has caused mlir-opt option-name drift in the past
# (e.g. test-tile-operation-graph's `sample-mode` <-> `tls_mode` rename), so
# always honor the pinned release_tag for a known-good Python<->mlir-opt pair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT}/thirdparty/github-releases.json"
home="/workspace"

if [ ! -f "$MANIFEST" ]; then
    echo "error: pin manifest not found at $MANIFEST" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found, installing..."
    apt -y update && apt -y install jq
fi

read_pin() {
    # $1 = key (gem5 / llvm_project / spike), echoes "<repo> <tag>"
    jq -r --arg k "$1" '.[$k] | "\(.repository) \(.release_tag)"' "$MANIFEST"
}

read GEM5_REPO  GEM5_TAG  <<< "$(read_pin gem5)"
read LLVM_REPO  LLVM_TAG  <<< "$(read_pin llvm_project)"
read SPIKE_REPO SPIKE_TAG <<< "$(read_pin spike)"

echo "Building from source using pins in $MANIFEST:"
echo "  gem5  = ${GEM5_REPO} @ ${GEM5_TAG}"
echo "  llvm  = ${LLVM_REPO} @ ${LLVM_TAG}"
echo "  spike = ${SPIKE_REPO} @ ${SPIKE_TAG}"

cd "$home"

# Gem5
apt -y update && apt -y upgrade && apt -y install scons
git clone --depth 1 --branch "$GEM5_TAG" "https://github.com/${GEM5_REPO}.git"
cd gem5 && scons build/RISCV/gem5.opt -j "$(nproc)"
export GEM5_PATH="$home/gem5/build/RISCV/gem5.opt"
cd "$home"

# LLVM + MLIR (RISCV target)
git clone --depth 1 --branch "$LLVM_TAG" "https://github.com/${LLVM_REPO}.git"
cd llvm-project && mkdir -p build && cd build && \
  cmake -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/riscv-llvm -DLLVM_TARGETS_TO_BUILD=RISCV \
        -G "Unix Makefiles" ../llvm && \
  make -j && make install
cd "$home"

# Spike Simulator
git clone --depth 1 --branch "$SPIKE_TAG" "https://github.com/${SPIKE_REPO}.git"
cd riscv-isa-sim && mkdir -p build && cd build && \
    ../configure --prefix="$RISCV" && make -j && make install
cd "$home"
