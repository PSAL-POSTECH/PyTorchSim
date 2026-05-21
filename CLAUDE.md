# CLAUDE.md — PyTorchSim quick reference

Reference notes for working in this repo. The canonical user-facing docs live in `README.md`; this file is a short, opinionated map for development sessions.

## What this repo is

PyTorchSim is a cycle-accurate NPU simulation framework. It plugs into the PyTorch 2 `torch.compile` stack via a custom `npu:0` device (PrivateUse1 backend) and runs three coupled simulators per compiled kernel:

1. **Gem5** (RISC-V) — produces compute-latency tables for the TOG
2. **Spike** — functional simulator that validates generated code correctness
3. **TOGSim** — the project's own cycle-accurate Tile-Operation-Graph simulator that models DRAM (Ramulator2), NoC (BookSim2), L2, systolic arrays, VPU lanes

The pipeline runs in that order on every `torch.compile` invocation; you'll see the three banners (`[Gem5]`, `[Spike]`, `[TOGSim]`) in the log when something is right.

## Repo layout (the parts that actually matter)

| Path | Purpose |
|---|---|
| `PyTorchSimFrontend/` | Python compiler stack (Inductor backend). `extension_config.py` is the central settings reader; `mlir/` contains MLIR templates per op (gemm, conv, bmm, sdpa, sort, cat, maxpool, …) |
| `PyTorchSimDevice/` | C++ PyTorch backend registering the `npu` device. Built as a pip-installed package via `setup.py`. Based on `torch_openreg` (PrivateUse1 example). Produces `_C.cpython-*.so` |
| `Simulator/simulator.py` | Python drivers: `FunctionalSimulator` (Spike), `CycleSimulator` (Gem5), `TOGSimulator` (the cycle-accurate one + multi-tenant context manager) |
| `Scheduler/scheduler.py` | Poisson arrival generator + scheduling utilities for multi-tenant runs |
| `TOGSim/` | C++ TOGSim source. `src/Simulator.cc`, `Core.cc`, `Dram.cc`, `Interconnect.cc`, `L2Cache.cc`, `Tile.cc`, `TileGraph.cc` are the core models. Externals: ramulator2, booksim, stonneCore, onnx, protobuf, spdlog, yaml-cpp |
| `AsmParser/` | `tog_generator.py`, `onnx_utility.py` — TOG generation from ONNX/ASM |
| `configs/` | TOGSim hardware configs (YAML). The default is `systolic_ws_128x128_c1_simple_noc_tpuv3.yml`. Naming pattern: `systolic_ws_<size>_c<cores>_<noc>_<target>.yml` |
| `tests/` | ~36 op- and model-level tests. Subdirs `DeepSeek/`, `Diffusion/`, `Llama/`, `MLP/`, `Mixtral_8x7B/`, `MoE/`, `Yolov5/`, `Fusion/` for whole-model workloads |
| `experiments/artifact/` | Paper reproduction scripts (`cycle_validation/run_cycle.sh`, `speedup/run_speedup.sh`) |
| `scripts/` | One-off experiment runners (CompilerOpt, ILS, batch, chiplet, sparsity, stonne, end2end). `build_from_source.sh` builds gem5/llvm/spike |
| `gem5_script/` | gem5 wrapper scripts called by `CycleSimulator` |
| `tpuv4/` | Example SRAM/L2 buffer plans for TPUv4-style persistent cache |
| `togsim_results/` | TOGSim log + trace dump directory (per-run) |
| `outputs/` | Per-run hashed output dirs |

## Running tests

Most tests follow the same pattern: build CPU reference, compile via `torch.compile` on `npu:0`, compare with `torch.allclose` (rtol=atol=1e-4). They all have `if __name__ == "__main__"` blocks.

```bash
python tests/test_add.py        # vector add (smoke test, fastest)
python tests/test_matmul.py     # GEMM
python tests/test_mlp.py        # MLP forward + backward (training path)
python tests/test_scheduler.py  # multi-tenant launch_model
python tests/test_eager.py      # eager-fallback registration
```

Run a model from `tests/Llama/`, `tests/DeepSeek/`, etc. similarly.

**For fast iteration** (skip functional check):
```bash
export pytorchsim_functional_mode=False   # skips Spike
```

**To dump intermediate IR while debugging:**
```bash
export TORCHSIM_DUMP_MLIR_IR=1
export TORCHSIM_DUMP_LLVM_IR=1
```

## Key environment variables

Read in `PyTorchSimFrontend/extension_config.py`:

| Var | Default | Purpose |
|---|---|---|
| `TORCHSIM_DIR` | `/workspace/PyTorchSim` | repo root |
| `TOGSIM_CONFIG` | `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml` | TOGSim hardware YAML |
| `GEM5_PATH` | `/workspace/gem5/build/RISCV/gem5.opt` | gem5 binary |
| `TORCHSIM_LLVM_PATH` | `/usr/bin` | LLVM tool dir |
| `TORCHSIM_LOG_PATH` | `$TORCHSIM_DIR/togsim_results` | where TOGSim logs go |
| `TORCHSIM_DUMP_PATH` | `$TORCHSIM_DIR` | misc dumps |
| `TORCHSIM_TLS_MODE` | `1` | TLS vs ILS mode |
| `TORCHSIM_USE_TIMING_POOLING` | `0` | lightweight pooling timing |
| `TORCHSIM_DEBUG_MODE` | `0` | extra debug |
| `TORCHSIM_DUMP_MLIR_IR` | `0` | dump MLIR |
| `TORCHSIM_DUMP_LLVM_IR` | `0` | dump LLVM IR |
| `SRAM_BUFFER_PLAN_PATH` | unset | L2/CMEM persistent-cache tensor plan (Python file with `plan = {...}`) |
| `TOGSIM_DEBUG_LEVEL` | unset | passed to TOGSim `--log_level` |

Note: `TOGSIM_CONFIG` is **overwritten** while inside a `with TOGSimulator(config_path=...)` block (and restored on exit). Compilation reads the same YAML as TOGSim that way.

## TOGSim YAML knobs (the ones I edit most)

Located under `configs/*.yml`:

- `num_cores`, `core_freq_mhz`, `num_systolic_array_per_core`
- `vpu_num_lanes`, `vpu_spad_size_kb_per_lane`, `vpu_vector_length_bits`
- `dram_type` (`ramulator2` | `simple`), `dram_channels`, `dram_freq_mhz`, `ramulator_config_path`
- `icnt_type` (`simple` | `booksim`), `icnt_latency_cycles`, `icnt_freq_mhz`, `icnt_config_path`
- `l2d_type` (e.g., `datacache`), `l2d_config` (AccelSim-format cache config string)
- `pytorchsim_functional_mode` (Spike on/off), `pytorchsim_timing_mode`
- `codegen_mapping_strategy`: `heuristic` | `autotune` | `external-then-heuristic` | `external-then-autotune`
- `codegen_external_mapping_file` (key `"M_N_K"` → `{TILE_M, TILE_K, TILE_N}` JSON)
- `codegen_compiler_optimization`: `"all"` | `"none"` | a list from `{fusion, reduction_epilogue, reduction_reduction, prologue, single_batch_conv, multi_tile_conv, subtile}`
- `num_partition` + `partition: {core_0: 0, core_1: 1}` for multi-tenant `stream_index` mapping

## Multi-tenant API (Simulator/simulator.py + scheduler)

```python
from Simulator.simulator import TOGSimulator
from Scheduler.scheduler import poisson_request_generator

with TOGSimulator(config_path=...):
    torch.npu.launch_model(opt_model, x, stream_index=0, timestamp=0)  # timestamp in ns
    torch.npu.synchronize()  # barrier
```

`stream_index` must be a valid queue id from the YAML's `partition` map. `timestamp` is nanoseconds; pass Poisson millisecond times × 1e6.

## Build

- **Docker (recommended):** `docker run -it --ipc=host --name torchsim -w /workspace/PyTorchSim ghcr.io/psal-postech/torchsim-ci:v1.0.1 bash`
- **TOGSim from source:** `cd TOGSim && mkdir -p build && cd build && conan install .. --build=missing && cmake .. && make -j$(nproc)`
- **PyTorchSimDevice (Python package):** `cd PyTorchSimDevice && python -m pip install --no-build-isolation -e .`
- **gem5 / LLVM+MLIR / Spike from source:** `bash scripts/build_from_source.sh` (clones to `/workspace/{gem5,llvm-project,riscv-isa-sim}`)

Conan deps for TOGSim: `boost/1.79.0`, `robin-hood-hashing/3.11.5`, `spdlog/1.11.0`, `yaml-cpp/0.8.0`.

## Where to look for X

- **Adding a new op (Inductor lowering):** `PyTorchSimFrontend/mlir/mlir_ops.py`, `mlir_lowering.py`, plus a new `mlir_<op>_template.py` if it needs its own MLIR template. Decomposition rules: `mlir_decomposition.py`. Scheduling: `mlir_scheduling.py`. Autotune: `mlir_autotune.py`.
- **Adding a PyTorch device op:** `PyTorchSimDevice/csrc/aten/native/*` (Minimal/Extra split mirrors `torch_openreg`).
- **TOGSim hardware model changes:** `TOGSim/src/{Core,Dram,Interconnect,L2Cache,Tile,TileGraph}.cc` + matching `include/*.h`.
- **TOG generation:** `AsmParser/tog_generator.py` builds the raw graph and serializes it via `AsmParser/onnx_utility.py` to **ONNX, which is the on-disk TOG format** consumed by TOGSim.
- **Eager fallback registration:** `torch.npu.register_eager_to_compile([...])` — see `tests/test_eager.py`.
- **Per-run results:** `togsim_results/<YYYYMMDD_HHMMSS_<hash>>.log` (stats) and `.trace` (instruction trace). The path is also printed at the end of every run.
- **Wrapper codegen path:** printed as `Wrapper Codegen Path = /tmp/torchinductor_<user>/<hash>/...py` — useful for inspecting generated kernel code and tensor names for `SRAM_BUFFER_PLAN_PATH`.

## Gotchas / things I've already learned

- The repo expects `python` to be a Python 3.10+ binary with `torch==2.8.0`. The frontend extends the PyTorch 2 Inductor stack — pin to this version.
- The default Gem5 path is hard-coded to `/workspace/gem5/build/RISCV/gem5.opt`. Override with `GEM5_PATH` if you build elsewhere.
- `_C.cpython-311-*.so` and `torch_openreg/lib/` are build artifacts — already in `.gitignore`, don't commit.
- TOGSim creates a per-PID FIFO under `/tmp/togsim_fifo_<pid>` for command/event comm; if a previous run crashed and left stale FIFOs, they get cleaned up on the next start, but watch for orphaned processes if you Ctrl-C mid-run.
- Multi-tenant runs **must** use the `with TOGSimulator(...)` context manager — otherwise compile-time `TOGSIM_CONFIG` and runtime config can diverge.
- `pytorchsim_functional_mode` exists as both an **env var** and a **YAML key**; the env var path is via `extension_config.py` while the YAML key is read inside the same module. They should agree.
- "No CUDA runtime is found" warnings on `import torch` are expected — this is a CPU + simulated-NPU environment, not real CUDA.

## Git workflow (per CONTRIBUTING.md)

- Fork → branch (`feature/<name>`) → PR against **`develop`**, not `main`.
- Commit prefix style observed: `[Frontend] ...`, `[TOGSim] ...`, etc.
