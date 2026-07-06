#!/usr/bin/env bash
# Create a sibling git worktree for parallel work and wire up per-worktree env.
#
# Container-dedicated: assumes the ghcr.io/psal-postech/torchsim-ci container
# layout. Shared binaries (gem5, LLVM, riscv toolchain) live at known paths
# and are NOT duplicated per worktree.
#
# Usage:
#   scripts/setup_worktree.sh <purpose> [base-ref]
#
# Examples:
#   scripts/setup_worktree.sh feature                  # off origin/develop, branch feature/scratch
#   scripts/setup_worktree.sh bugfix/issue-198         # off origin/develop, branch bugfix/issue-198
#   scripts/setup_worktree.sh review origin/master     # off origin/master,  branch review/scratch
#
# Result:
#   /workspace/PyTorchSim-<basename(purpose)>           new worktree
#   /workspace/PyTorchSim-<...>/.envrc                  per-worktree env (source it)
#
# After creation:
#   cd /workspace/PyTorchSim-<...>
#   source .envrc
#   (cd PyTorchSimDevice && python setup.py build_ext --inplace)   # build the .so once
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    sed -n '3,20p' "$0"
    exit 1
fi

PURPOSE="$1"
BASE_REF="${2:-origin/develop}"

# Branch name: use the purpose as-is if it already has a slash, else append /scratch.
if [[ "$PURPOSE" == */* ]]; then
    BRANCH="$PURPOSE"
    SUFFIX="${PURPOSE%%/*}"           # before first slash, used in dir name
else
    BRANCH="${PURPOSE}/scratch"
    SUFFIX="$PURPOSE"
fi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
WT_DIR="${PARENT_DIR}/$(basename "$REPO_ROOT")-${SUFFIX}"

if [[ -e "$WT_DIR" ]]; then
    echo "error: $WT_DIR already exists" >&2
    exit 1
fi

# Make sure base-ref is up to date if it's a remote ref.
if [[ "$BASE_REF" == origin/* ]]; then
    git -C "$REPO_ROOT" fetch origin "${BASE_REF#origin/}" --depth=1
fi

git -C "$REPO_ROOT" worktree add "$WT_DIR" -b "$BRANCH" "$BASE_REF"

# Default `worktree add` from a remote ref sets upstream to that remote ref,
# which means `git push` would target develop/master. Unset so the new branch
# pushes to its own name on first `git push -u origin <branch>`.
git -C "$WT_DIR" branch --unset-upstream || true

# Share the TOGSim binary from the worktree this script was run from. TOGSim
# is a standalone C++ simulator that rarely changes alongside Python frontend
# work, so symlinking saves a ~10-minute rebuild per worktree. If you do
# modify TOGSim C++ in the new worktree, run `cd TOGSim/build && make` after
# wiping the link target -- the symlink will be replaced by the local build
# output.
TOGSIM_BIN_SRC="$REPO_ROOT/TOGSim/build/bin/Simulator"
TOGSIM_BIN_DST="$WT_DIR/TOGSim/build/bin/Simulator"
if [[ -x "$TOGSIM_BIN_SRC" ]]; then
    # Resolve so we point at the real binary, not a chain of worktree symlinks.
    TOGSIM_BIN_REAL="$(readlink -f "$TOGSIM_BIN_SRC")"
    mkdir -p "$(dirname "$TOGSIM_BIN_DST")"
    ln -sfn "$TOGSIM_BIN_REAL" "$TOGSIM_BIN_DST"
    TOGSIM_LINK_MSG="Symlinked TOGSim binary from $TOGSIM_BIN_REAL"
else
    TOGSIM_LINK_MSG="TOGSim binary not found at $TOGSIM_BIN_SRC; build it once with 'cd TOGSim/build && conan install .. --build=missing && cmake .. && make -j' or symlink from another worktree."
fi

# Per-worktree env. Container-dedicated paths for shared binaries.
cat > "$WT_DIR/.envrc" <<'ENVRC'
#!/usr/bin/env bash
# Source this from the worktree root:  source .envrc
_self="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Worktree-scoped: override defaults from PyTorchSimFrontend/extension_config.py
export TORCHSIM_DIR="$_self"
export TORCHSIM_DUMP_PATH="$_self/outputs"
export TORCHSIM_LOG_PATH="$_self/togsim_results"
export TOGSIM_CONFIG="$_self/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml"

# Make `import torch_openreg` resolve to THIS worktree's .so first,
# overriding the conda-wide editable install that points at the main worktree.
export PYTHONPATH="$_self/PyTorchSimDevice:$_self:${PYTHONPATH:-}"

# Container-dedicated shared binaries.
export GEM5_PATH="/gem5/release/gem5.opt"
export TORCHSIM_LLVM_PATH="/riscv-llvm/bin"
export RISCV="/workspace/riscv"

# Prompt hint so you do not lose track of which worktree this shell is on.
export PS1="[torchsim:$(basename "$_self")] ${PS1:-\\w\\$ }"

unset _self
echo "Activated worktree: $TORCHSIM_DIR"
ENVRC

echo
echo "Created worktree: $WT_DIR"
echo "Branch:           $BRANCH (base: $BASE_REF)"
echo "$TOGSIM_LINK_MSG"
echo
echo "Next:"
echo "  cd $WT_DIR"
echo "  source .envrc"
echo "  (cd PyTorchSimDevice && python setup.py build_ext --inplace)   # build the .so once"
echo
echo "When iterating on PyTorchSimFrontend/mlir/* (or any codegen) in this worktree,"
echo "run scripts/clear_codegen_cache.sh between runs so the cached compile does not"
echo "shadow your changes."
