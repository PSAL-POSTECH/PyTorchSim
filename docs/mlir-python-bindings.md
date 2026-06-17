# Enabling MLIR Python bindings

Goal: ship the MLIR Python bindings (`import mlir`, `mlir.ir`, `mlir.dialects`)
so we can write MLIR passes in Python (imperative IR rewriting via the bindings)
instead of only C++ passes in the `PSAL-POSTECH/llvm-project` fork. See
`dma-transfer-lowering.md` for the first intended use (a Python decompose pass).

## How LLVM reaches the runtime (why this touches 3 places)

```
PSAL-POSTECH/llvm-project  (fork, tag vX.Y.Z)
   .github/workflows/build-torchsim.yaml   -- CI builds + releases riscv-llvm-release.tar.gz
        |  (release asset)
        v
thirdparty/github-releases.json            -- pins llvm_project.release_tag + asset
        |
        v
Dockerfile.base                            -- downloads asset, extracts to /riscv-llvm,
                                              sets TORCHSIM_LLVM_PATH (+ now PYTHONPATH)
```

`scripts/build_from_source.sh` is the alternative source-build path (not the
normal flow, but kept consistent).

## The one real blocker: Python ABI must match

The bindings are a native CPython extension (`_mlir.cpython-3XX-*.so`). They only
import under the **same Python minor version** they were built against. The
runtime base image uses **conda Python 3.11**. So the artifact must be built with
**Python 3.11**. Building with the build container's default (ubuntu-22.04 ->
3.10) produces bindings that fail to import at runtime with a confusing error
much later -- hence the fail-fast guard in the CI step.

Patch version (3.11.x) does not matter; minor version (3.11 vs 3.10) does.

## What was changed

- **`scripts/build_from_source.sh`**: cmake gets
  `-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=$(command -v python3)`;
  build deps (nanobind/pybind11/numpy/PyYAML) pip-installed; after `make install`
  the build-tree `tools/mlir/python_packages` is copied into `/riscv-llvm`
  (install does not place it there). PYTHONPATH exported for the current shell.
- **`Dockerfile.base`**: `ENV PYTHONPATH=/riscv-llvm/python_packages/mlir_core:$PYTHONPATH`
  after the LLVM artifact is extracted.
- **`llvm-project/.github/workflows/build-torchsim.yaml`** (fork): same cmake
  flags + deps; copies `python_packages` into the `riscv-llvm` tree so the
  existing `tar` includes it; fail-fast guard requiring `python3.11`.

## Rollout sequence (must be done in order)

1. **python3.11 in the build container: done, non-root.** The CI step keeps the
   original `-u $(id -u):$(id -g)` (no root assumed) and fetches a standalone
   CPython 3.11 with `uv` (`uv venv --python 3.11`), then points
   `Python3_EXECUTABLE` at that venv. No apt / no root needed. ubuntu-22.04's
   default 3.10 is not used for the bindings.
   - ABI note: extensions built against a uv/python-build-standalone CPython 3.11
     are expected to import under the runtime conda CPython 3.11 (same minor
     version, standard builds are C-ABI compatible). The verify step below is the
     check; if it ever fails, build instead in the runtime image (`python:3.11` or
     the pytorch base) so build Python == runtime Python by construction.
2. **Push the fork changes** to `PSAL-POSTECH/llvm-project` and cut a new tag
   (e.g. `v1.0.9`). CI builds `riscv-llvm-release.tar.gz` now containing
   `python_packages/`.
3. **Bump `thirdparty/github-releases.json`** -> `llvm_project.release_tag` to the
   new tag (and `asset_name` unchanged). This triggers a new base image build.
4. **Rebuild the base image** (the fork CI already dispatches `build_base`; or run
   the PyTorchSim docker-image workflow) so `Dockerfile.base` produces an image
   with the bindings + PYTHONPATH.

## Verify

Inside the rebuilt container (or after `build_from_source.sh`):

```bash
python -c "import mlir; print(mlir.__file__)"   # -> /riscv-llvm/python_packages/mlir_core/mlir/__init__.py
python -c "from mlir.ir import Context; c=Context(); c.allow_unregistered_dialects=True; print('ok')"
python -c "from mlir.dialects import scf, affine, arith; print('dialects ok')"
```

`allow_unregistered_dialects=True` is what lets us read/write the custom ops
(`togsim.transfer`, the customized `memref.dma_start`) generically without
registering a dialect in the bindings.

## Notes / gotchas

- Keep the bindings statically linked (default, i.e. do NOT add
  `-DBUILD_SHARED_LIBS=ON` / `-DLLVM_BUILD_LLVM_DYLIB=ON`); otherwise the `.so`
  needs libMLIR/libLLVM at runtime and the artifact + LD_LIBRARY_PATH grow.
- Worktrees: add the same `PYTHONPATH` line to the worktree `.envrc` (see
  `docs/worktrees.md`) if a worktree overrides paths.
- The bindings are an additive, optional dependency: text emission + C++ passes
  keep working unchanged. Only new Python passes require the bindings present.
- This LLVM fork's MLIR bindings use **pybind11** (not nanobind) and require
  **pybind11 <= 2.10.3**: newer pybind11 (3.x) fails to compile `IRCore.cpp` with
  `def_property family does not currently support keep_alive`. Pin it
  (`pybind11>=2.9.0,<=2.10.3`). See `mlir/python/requirements.txt` for the fork's
  pins. pybind11 is build-time only; the runtime needs just the built `.so` + numpy.
- numpy: the fork's requirements pin `<=1.26`, but a local build against numpy 2.x
  compiled and imported fine, so we keep numpy at the runtime version (2.x) to
  avoid a numpy-1-built / numpy-2-runtime ABI mismatch. (Validated locally:
  conda 3.11 + pybind11 2.10.3 + numpy 2.x -> `import mlir` and parsing a custom
  `togsim.transfer` op with floordiv/mod affine maps both work.)
