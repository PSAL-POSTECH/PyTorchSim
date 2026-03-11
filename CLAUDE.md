# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorchSim is a cycle-accurate NPU (Neural Processing Unit) simulation framework. It integrates a PyTorch 2.8+ compiler backend that generates NPU machine code and **Tile-Operation Graphs (TOG)**, which are then executed by the C++ **TOGSim** simulator for performance analysis and validation of neural network workloads.

## Build & Installation

### Recommended: Docker

```bash
# Standard run (ephemeral)
timestamp=$(date +%Y%m%d%H%M%S)
docker run -it --ipc=host --rm \
  --name "torchsim_${USER}_${timestamp}" \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.0.1 bash

# With local repo mounted (persist changes)
docker run -it --ipc=host --rm \
  --name "torchsim_${USER}_${timestamp}" \
  -v "$PWD:/workspace/PyTorchSim" \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.0.1 bash
```

### Manual Build (inside Docker or pre-configured host)

```bash
# 1. Build TOGSim C++ simulator
cd TOGSim && mkdir build && cd build
conan install .. --build=missing
cmake .. && make -j$(nproc)

# 2. Install the PyTorch device backend
cd PyTorchSimDevice
python -m pip install --no-build-isolation -e .
```

**Dependencies**: cmake >= 3.15, conan, LLVM/mlir-opt, Python >= 3.10, PyTorch >= 2.8, RISC-V GCC toolchain, Spike ISA simulator, Gem5.

## Running Tests

```bash
# Run a single test
python tests/test_matmul.py
python tests/test_resnet.py
python tests/test_transformer.py

# Common environment setup before running tests
export TORCHSIM_CONFIG=/path/to/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml
export pytorchsim_functional_mode=1   # Use Spike for functional verification (0 = disable for speed)
export vpu_num_lanes=128
export vpu_spad_size_kb_per_lane=128
```

CI runs 40+ tests defined in `.github/workflows/pytorchsim_test.yml`.

## Architecture

PyTorchSim has two major components:

### 1. Compiler Frontend (`PyTorchSimFrontend/`)
Hooks into the PyTorch 2.x compiler stack to intercept operator calls and compile them to NPU machine code.

- **`extension_config.py`** — Loads YAML hardware configs (via `TORCHSIM_CONFIG` env var), provides all hardware parameters, sets up logging.
- **`extension_codecache.py`** — Manages the compilation pipeline: hash-based caching → MLIR → LLVM → RISC-V assembly. Calls Gem5 for compute latency and Spike for functional verification.
- **`extension_op.py`** — Defines supported operations and dispatches them to code generators.
- **`mlir/`** — MLIR-based code generators for each op type:
  - `mlir_codegen_backend.py` — Orchestrates code generation
  - `mlir_gemm_template.py`, `mlir_conv_*.py` — Per-op templates
  - `mlir_autotune.py` — Searches tile shapes and vector strides
  - `mlir_scheduling.py` — Instruction scheduling

### 2. PyTorch Device Backend (`PyTorchSimDevice/`)
Registers a custom PyTorch device (`npu:0`) using the PrivateUse1 mechanism (torch_openreg pattern). C++ runtime in `csrc/`; Python interface in `torch_openreg/`.

### 3. TOGSim C++ Simulator (`TOGSim/`)
Cycle-accurate simulator that consumes TOG (Tile-Operation Graphs). Integrates:
- **Ramulator2** — DRAM simulation (HBM2, DDR4, LPDDR5X)
- **BookSim2** — Network-on-Chip simulation
- **stonneCore** — Systolic array modeling

### 4. Simulation Orchestration
- **`Simulator/simulator.py`** — `FunctionalSimulator`, `CycleSimulator`, `TOGSimulator` context manager
- **`Scheduler/scheduler.py`** — Multi-tenancy request scheduling across cores
- **`AsmParser/tog_generator.py`** — Generates TOG from compiled assembly

## Key Configuration

Hardware configs are YAML files in `configs/`. Key parameters:
```yaml
num_cores: 1
vpu_num_lanes: 128          # Systolic array width
vpu_spad_size_kb_per_lane: 128
dram_type: ramulator2
icnt_type: simple           # or "booksim"
codegen_mapping_strategy: heuristic  # or "autotune", "external-*"
codegen_compiler_optimization: all   # or "none", or list e.g. ["fusion", "prologue"]
pytorchsim_functional_mode: 1
pytorchsim_timing_mode: 1
```

## Important Environment Variables

| Variable | Purpose |
|---|---|
| `TORCHSIM_CONFIG` | Path to YAML hardware config (required) |
| `TORCHSIM_LOG_PATH` | Output log directory (default: `togsim_results/`) |
| `pytorchsim_functional_mode` | `1` = enable Spike verification, `0` = skip for speed |
| `TORCHSIM_TLS_MODE` | `1` = Tile-Level Sim, `0` = Instruction-Level Sim |
| `vpu_num_lanes` | Override lane count from config |
| `GEM5_PATH` | Path to Gem5 binary |
| `TORCHSIM_DUMP_MLIR_IR` | `1` = dump MLIR IR to disk for debugging |
| `TORCHSIM_DEBUG_MODE` | `1` = enable debug logging |

## Test Pattern

All tests follow a consistent pattern:
```python
device = torch.device("npu:0")
model = MyModel().to(device)
opt_fn = torch.compile(dynamic=False)(model)
result = opt_fn(input.to(device))
reference = model.to("cpu")(input)
# Compare with torch.allclose() or custom test_result()
```

Always use `torch.compile(dynamic=False)` for deterministic tile shapes.

## Output Artifacts

Write run artifacts to `output/`, `outputs/`, or `togsim_results/` — these are git-ignored. Do not commit generated logs, result folders, or notebook checkpoints.

## Git Workflow

- `master` — stable releases
- `develop` — active development; open PRs against this branch
- Feature branches: `feature/my-feature`
