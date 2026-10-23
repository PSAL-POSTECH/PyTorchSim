# Parallel worktrees

Quick setup for working on multiple branches at the same time (e.g. one for
feature work, one for code review, one for a bug fix) without thrashing a
single checkout. Container-dedicated: paths assume the
`ghcr.io/psal-postech/torchsim-ci` layout.

## Why a script

Three things have to line up for parallel worktrees to actually work in this
repo:

1. **Worktree-scoped env vars.** `pytorchsim/config.py`
   anchors output / log / config paths on `TORCHSIM_DIR`. Without an override
   every worktree dumps into the same `outputs/` and `togsim_results/`.
2. **`PYTHONPATH` override.** `pip install -e device` writes a
   single editable record into conda's site-packages that points at one
   worktree. The override forces `import torch_openreg` to resolve to the
   active worktree's code and `.so` first.
3. **Branch tracking.** `git worktree add` from a remote ref sets upstream to
   that ref, so `git push` would target the base branch. The script unsets it so
   first `git push -u origin <branch>` creates the right remote branch.

The script bundles all three.

## Create a worktree

```bash
scripts/setup_worktree.sh <purpose> [base-ref]
```

`<purpose>` becomes the branch suffix and the dir suffix:

| Command | Worktree dir | Branch |
|---|---|---|
| `setup_worktree.sh feature` | `/workspace/PyTorchSim-feature` | `feature/scratch` |
| `setup_worktree.sh review` | `/workspace/PyTorchSim-review` | `refactor/scratch` (rename after) |
| `setup_worktree.sh bugfix/issue-198` | `/workspace/PyTorchSim-bugfix` | `bugfix/issue-198` |
| `setup_worktree.sh feature origin/master` | `/workspace/PyTorchSim-feature` | `feature/scratch` (off master) |

Default base is `origin/pytorchsim-triton-backend`.

## Activate

```bash
cd /workspace/PyTorchSim-bugfix
source .envrc
# → Activated worktree: /workspace/PyTorchSim-bugfix
# → prompt: [torchsim:PyTorchSim-bugfix] ...
```

`.envrc` is local to each worktree and not committed. Re-source it whenever
you open a new shell in that worktree.

## Build once per worktree

The compiled `_C.cpython-*.so` lives under `device/torch_openreg/`
and is not shared across worktrees. After activation:

```bash
(cd device && python setup.py build_ext --inplace)
```

Use `build_ext --inplace` instead of `pip install -e` so the editable
record in `/opt/conda/lib/python3.11/site-packages` keeps pointing at
whichever worktree it already pointed at — `PYTHONPATH` from `.envrc` does
the per-worktree routing. (Running `pip install -e` again rewrites that
record and will pin "default" Python to the new worktree.)

## TOGSim is shared

TOGSim is not in this repository. Every worktree runs the one the compiler
checkout carries, `$TORCHSIM_COMPILE_DIR/third_party/TOGSim` (header in
`include/`, binary at `build/bin/Simulator`, machine configs in `configs/`),
so there is nothing to link or build per worktree. To run a TOGSim you are
changing, build it in its own checkout and `export TOGSIM_DIR=<that checkout>`
in the shell: the header digest, the compiler's TOG build and the Simulator
all follow it.

## Iterating on codegen inside a worktree

`.envrc` gives each worktree its own `$TORCHSIM_DUMP_PATH=$_self/outputs`,
so parallel worktrees do not share caches. But within a worktree, after
editing anything that affects emitted MLIR or wrapper code
(`pytorchsim/*`), the next
`torch.compile` will replay the previously cached compile from
`outputs/` and your change silently does not take. Run:

```bash
scripts/clear_codegen_cache.sh
```

between iterations. It wipes `outputs/.torchinductor` (Inductor's compile
cache, set via `TORCHINDUCTOR_CACHE_DIR` inside `config.get_dump_path()`) and
the per-kernel dirs (`outputs/triton_<hash>/`). `togsim_results/` (run logs) is left
alone.

Diagnostic for the other common gotcha: if a traceback mentions a path
under `/workspace/PyTorchSim/...` while you are editing in a different
worktree, you forgot to `source .envrc` in that shell — Python imported the
canonical worktree's `pytorchsim` instead of yours.

## What the env looks like

Worktree-scoped (auto-set by `.envrc`):

| Var | Value |
|---|---|
| `TORCHSIM_DIR` | `$PWD` of the worktree |
| `TORCHSIM_DUMP_PATH` | `$PWD/outputs` |
| `TORCHSIM_LOG_PATH` | `$PWD/togsim_results` |
| `TORCHSIM_TARGET` | `tpuv6e` (the machine file `tpuv6e.yml` in the TOGSim checkout's `configs/`) |
| `PYTHONPATH` | `$PWD/device:$PWD:$PYTHONPATH` |

Shared (container-dedicated, set the same in every `.envrc`):

| Var | Value |
|---|---|
| `GEM5_PATH` | `/workspace/vcix-env/gem5/build/RISCV/gem5.opt` |
| `RISCV` | `/workspace/vcix-env/riscv` |

## Cleanup

```bash
git worktree remove /workspace/PyTorchSim-feature
git branch -D feature/scratch         # if you do not want to keep the branch
```

`git worktree list` shows the current set.

## Gotchas

- **Do not commit `.envrc`.** It is per-worktree state. Add to your
  personal global gitignore if needed.
- **Editable install conflict.** If you run `pip install -e device`
  in worktree A, then again in worktree B, Python's default `import
  torch_openreg` flips to B. With `PYTHONPATH` from `.envrc` this still
  resolves correctly in either worktree's shell, but a shell with no
  `.envrc` sourced will see whichever was installed last.
- **TOGSim FIFO files.** `/tmp/togsim_fifo_<pid>` is keyed on PID, not on
  worktree — concurrent runs from different worktrees do not collide.
- **`_C.cpython-*.so` rebuild on PyTorch update.** If you `pip install`
  a different torch version in the conda env, every worktree's `.so` is
  stale; rebuild each.
