# Removed files

Files taken out while moving PyTorchSim onto PyTorchSim-Triton-Backend. To restore one:

```bash
c=$(git log -1 --format=%h --diff-filter=D -- <path>)   # the commit that removed it
git checkout "$c^" -- <path>
```

## PSTO toolchain layer (replaced by PTB) -- `6db93e2`

| Path | Why |
|---|---|
| `Dockerfile.psto` | replaced by `Dockerfile.ptb` |
| `thirdparty/pytorchsim-triton-opt.json` | replaced by `thirdparty/pytorchsim-triton-backend.json` |
| `scripts/ci/psto_base_pin.sh` | replaced by `scripts/ci/ptb_base_pin.sh` |
| `scripts/ci/thirdparty_github_asset_env.sh` | resolved the old gem5/Spike release assets; those come from the vcix-accelerator environment now |
| `docs/mlir-python-bindings.md` | documented the llvm_project pin, which is gone |
| `.github/workflows/pytorchsim_triton_opt.yml` | renamed to `pytorchsim_triton_backend.yml` |

## MLIR-era and one-off leftovers -- `f9f5cd7`

| Path | Why |
|---|---|
| `debug/` | gem5 debug run against the old gem5 fork |
| `tutorial/`, `.github/workflows/docker-tutorial-image.yml` | ISPASS 2026 tutorial on the old Docker / Spike / MLIR stack (it lives on the `ispass2026` branch upstream) |
| `tests/system/test_tog_structure.py`, `tests/system/test_triton_codegen.py` | written for the old compiler, broken |
| `yolov5s.pt` | 14.8 MB model weights, read by nothing |
| `validation/gemm_tpuv3_cheatsheet.json` | read by nothing |
| `scripts/CompilerOpt_experiment/`, `ILS_experiment/`, `batch_experiment/`, `sparsity_experiment/`, `stonne_experiment/`, `stonne_experiment2/` | MLIR-era experiment runners |
| `scripts/end2end.sh`, `sparsity.sh`, `sim_time.sh`, `get_tog_result.sh` | MLIR-era experiment helpers |
| `docs/tpu_layout_padding_report.md`, `docs/triton-route-coverage*.md` | one-off report and committed CI coverage snapshots |

`gem5_script/` and `scripts/chiplet*` were removed in the same commit and restored in `34d4364`.

## Experiments moved out of the repository

| Path | Why |
|---|---|
| `experiments/` | paper artifact (cycle validation, speedup) on the tpuv3/v4 128x128 configs of the MLIR route |
| `timing_mode_validation/` | TPU v6e timing validation harness and its measured reference (`ref_v6e.csv`); kept locally, not in the repository |

The accuracy/speedup job in `pytorchsim_test.yml` ran `experiments/artifact/` and went with it.
