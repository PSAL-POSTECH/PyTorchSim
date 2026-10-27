#!/usr/bin/env bash
# Create a sibling git worktree for parallel work and wire up per-worktree env.
#
# Container-dedicated: assumes the ghcr.io/psal-postech/torchsim-test (or torchsim_base)
# layout. Shared binaries (Spike, gem5 and the riscv toolchain under COPROCESSOR_ENV_ROOT; LLVM,
# and TOGSim in the compiler checkout) live where the image puts them and are NOT duplicated.
#
# Usage:
#   scripts/setup_worktree.sh <purpose> [base-ref]
#
# Examples:
#   scripts/setup_worktree.sh feature                  # off origin/pytorchsim-triton-backend, branch feature/scratch
#   scripts/setup_worktree.sh bugfix/issue-198         # off origin/pytorchsim-triton-backend, branch bugfix/issue-198
#   scripts/setup_worktree.sh review origin/master     # off origin/master,  branch review/scratch
#
# Result:
#   /workspace/PyTorchSim-<basename(purpose)>           new worktree
#   /workspace/PyTorchSim-<...>/.envrc                  per-worktree env (source it)
#
# After creation:
#   cd /workspace/PyTorchSim-<...>
#   source .envrc
#   (cd device && python setup.py build_ext --inplace)   # build the .so once
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    sed -n '3,20p' "$0"
    exit 1
fi

PURPOSE="$1"
BASE_REF="${2:-origin/pytorchsim-triton-backend}"

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
# which means `git push` would target the base branch. Unset so the new branch
# pushes to its own name on first `git push -u origin <branch>`.
git -C "$WT_DIR" branch --unset-upstream || true

# Per-worktree env. Container-dedicated paths for shared binaries.
cat > "$WT_DIR/.envrc" <<'ENVRC'
#!/usr/bin/env bash
# Source this from the worktree root:  source .envrc
_self="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Worktree-scoped: override defaults from pytorchsim/config.py
export TORCHSIM_DIR="$_self"
export TORCHSIM_DUMP_PATH="$_self/outputs"
export TORCHSIM_LOG_PATH="$_self/togsim_results"
# The machine: a TOGSim machine file, $TOGSIM_DIR/configs/tpuv6e.yml (pytorchsim/config.py's default).
export TORCHSIM_TARGET=tpuv6e

# Make `import torch_openreg` resolve to THIS worktree's .so first,
# overriding the conda-wide editable install that points at the main worktree.
export PYTHONPATH="$_self/device:$_self:${PYTHONPATH:-}"

# Prompt hint so you do not lose track of which worktree this shell is on.
export PS1="[torchsim:$(basename "$_self")] ${PS1:-\\w\\$ }"

unset _self
echo "Activated worktree: $TORCHSIM_DIR"
ENVRC

echo
echo "Created worktree: $WT_DIR"
echo "Branch:           $BRANCH (base: $BASE_REF)"
echo
echo "Next:"
echo "  cd $WT_DIR"
echo "  source .envrc"
echo "  (cd device && python setup.py build_ext --inplace)   # build the .so once"
echo
echo "When iterating on pytorchsim/* in this worktree,"
echo "run scripts/clear_codegen_cache.sh between runs so the cached compile does not"
echo "shadow your changes."
