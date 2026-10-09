# CLAUDE.md — PyTorchSim quick reference

Reference notes for working in this repo. The canonical user-facing docs live in `README.md`; this file is a short, opinionated map for development sessions.

## What this repo is

PyTorchSim is a cycle-accurate NPU simulation framework. It plugs into the PyTorch 2 `torch.compile` stack via a custom `npu:0` device (PrivateUse1 backend) and runs three coupled simulators per compiled kernel:

1. **Gem5** (RISC-V) — produces compute-latency tables for the TOG
2. **Spike** — functional simulator that validates generated code correctness
3. **TOGSim** — the project's own cycle-accurate Tile-Operation-Graph simulator that models DRAM (Ramulator2), NoC (BookSim2), L2, systolic arrays, VPU lanes

The pipeline runs in that order on every `torch.compile` invocation; you'll see the three banners (`[Gem5]`, `[Spike]`, `[TOGSim]`) in the log when something is right.

## Every operation runs on the NPU — a CPU fallback is a defect

An op with no `npu` kernel does not fail. It **falls back to the CPU**, quietly, and
the model still produces a number. That is the worst shape a defect can have here,
because the run looks like a pass:

- **It is not the thing we are measuring.** Work that ran on the host is absent
  from the TOG, so Gem5/TOGSim price it at zero and Spike never validates it.
  A model whose expert matmuls ran on the CPU has not been run on this NPU, and
  reporting it as supported is reporting someone else's arithmetic.
- **It hides which half is wrong.** When the numbers come out wrong, a graph
  split between device kernels and host calls gives no answer to "where did it
  diverge" — the host half is always right, so every bisect points at us whether
  or not we are at fault.
- **It is silent.** Nothing in the log says "this ran on the CPU". You have to go
  and look.

**HOW TO LOOK.** The generated wrapper names every call it does not compile. Take
the path from `Wrapper Codegen Path` in the log — there is one per graph, so read
them all, not the last one — and count the assignment lines that are not kernels:

```bash
grep -E '^\s+buf[0-9]+ = (torch\.ops|extern_kernels)' "$WRAPPER"   # must be empty
```

**MEASURED, and it is not hypothetical.** Qwen2-MoE, `--preset 2-moe`, same tree
and same device, two transformers:

```
4.51.3   12 graphs   48 npu kernels   0 cpu fallbacks     Test Passed
5.15.0    1 graph    23 npu kernels   4 cpu fallbacks     Max abs diff 1.370
                                      aten.topk, aten.histc,
                                      transformers.grouped_mm_fallback x2
```

5.x rewrote the MoE block to remove the graph breaks (12 graphs -> 1), and paid
for it in fallbacks. The wrong number came with them.

**WHAT TO DO INSTEAD.** A rewrite in `PyTorchSimFrontend/rewrite_fx_graph.py`,
written in aten ops the backend already lowers, so the work stays in the graph.
That file exists for exactly this and says so. Note that casting an operand until
the CPU kernel accepts it is only half the job: the histc decomposition there
still leaves `aten.histc` as an extern call, and its own docstring names the
device-side form it should have been (`(x == bin).sum()` against `arange(bins)` —
one elementwise and one reduce).

**IF YOU CANNOT REMOVE ONE,** say so in those words and name it. A fallback left
in place is a known hole in what "this model runs" means, and it belongs in the
report next to the number it produced — never folded into a pass.

## Repo layout (the parts that actually matter)

| Path | Purpose |
|---|---|
| `PyTorchSimFrontend/` | The graph side: `config.py` (the central settings reader) and `rewrite_fx_graph.py` (our FX graph rewrites, so ops with no `npu` kernel stay on the device) |
| `PyTorchSimCodegen/` | The `npu` codegen route: Inductor's Triton codegen for `npu` (`scheduling.py`, `wrapper_codegen.py`, `inductor_templates.py`), the kernel spec and source fixups, and the bridge that runs PyTorchSim-Triton-Backend and caches its kernel objects (`compiler_bridge.py`, `codecache.py`) |
| `PyTorchSimDevice/` | C++ PyTorch backend registering the `npu` device. Built as a pip-installed package via `setup.py`. Based on `torch_openreg` (PrivateUse1 example). Produces `_C.cpython-*.so` |
| `Simulator/` | What runs a kernel object: Spike (`functional.py`, `spike_run.py`, and `functional_verify.py`, the per-kernel CPU cross-check), gem5 (`gem5.py`, `trace_build.py`), TOGSim (`timing.py`, and `simulator.py`'s `TOGSimulator`: the process driver and multi-tenant context manager) |
| `Scheduler/scheduler.py` | Poisson arrival generator + scheduling utilities for multi-tenant runs |
| `TOGSim/` | C++ TOGSim source. `src/Simulator.cc`, `Core.cc`, `Dram.cc`, `Interconnect.cc`, `L2Cache.cc`, `Tile.cc`, `TileGraph.cc` are the core models. Externals: ramulator2, booksim, spdlog, yaml-cpp |
| `configs/` | TOGSim hardware configs (YAML). The default is `systolic_ws_256x256_c1_simple_noc_tpuv6e_functional_only.yml` -- the same machine as `..._tpuv6e.yml` and one line apart, `pytorchsim_timing_mode: 0`. Naming pattern: `systolic_ws_<size>_c<cores>_<noc>_<target>.yml` |
| `tests/` | Op- and model-level tests organized under `ops/<family>/` (elementwise, reduce, gemm, conv, attention, view, sort, sparsity, misc, fusion), `models/<name>/` (Llama, Mixtral8x7B, DeepSeek, Diffusion, MoE, MLP, MobileNet, Yolov5) plus single-file model tests (test_resnet, test_transformer, test_vit, test_mlp, test_single_perceptron), and `system/` (scheduler, eager, vectorops). Shared helper: `tests/_utils.py`. **Which of them pass is `scripts/ci/triton_route_passing.txt`**; the rest are known gaps, swept and reported by `scripts/ci/triton_route_sweep.py --all` |
| `experiments/artifact/` | Paper reproduction scripts (`cycle_validation/run_cycle.sh`, `speedup/run_speedup.sh`) |
| `scripts/` | Dev tools (`util_viewer.py`, `trace_timeline.py`, `op_coverage.py`, `clear_codegen_cache.sh`, `setup_worktree.sh`) and CI helpers (`ci/`). `build_from_source.sh` builds the compiler and its simulators through PTB |
| `togsim_results/` | TOGSim log + trace dump directory (per-run) |
| `outputs/` | Per-run hashed output dirs |

## Running tests

Most tests follow the same pattern: build CPU reference, compile via `torch.compile` on `npu:0`, compare with `torch.allclose` (rtol=atol=1e-4). They all have `if __name__ == "__main__"` blocks.

```bash
python tests/ops/elementwise/test_add.py        # vector add (smoke test, fastest)
python tests/ops/gemm/test_matmul.py     # GEMM
python tests/models/test_mlp.py        # MLP forward + backward (training path)
python tests/system/test_scheduler.py  # multi-tenant launch_model
python tests/system/test_eager.py      # eager-fallback registration
```

Run a model from `tests/models/Llama/`, `tests/models/DeepSeek/`, etc. similarly.

**CI coverage:** `.github/workflows/docker-image.yml` is the PR gate. It builds the base image, then the PyTorchSim-Triton-Backend layer (`Dockerfile.ptb`, pinned by `thirdparty/pytorchsim-triton-backend.json`: the compiler plus the vcix-accelerator RISC-V toolchain, pk, Spike and gem5), then the app image, and calls `.github/workflows/pytorchsim_test.yml` once per hardware config. That workflow is a **matrix over `scripts/ci/triton_route_passing.txt`**, one Docker container per test — so adding a test file gates PRs only once it is in that allowlist. Regenerate the allowlist with `python scripts/ci/triton_route_sweep.py --all --update-allowlist` and mirror it into the workflow.

`.github/workflows/pytorchsim_triton_backend.yml` runs when the compiler pin itself moves: the compiler's own smoke kernels end to end, plus the full coverage sweep (allowlist *and* the rest, reported not gated). Both workflows need `secrets.TNPU_TOKEN` — `PSAL-POSTECH/PyTorchSim-Triton-Backend` is private.

**For fast iteration** (skip functional check):
```bash
export pytorchsim_functional_mode=False   # skips Spike
```

**To read the IR a kernel went through:** every stage is left in
`$TORCHSIM_DUMP_PATH/triton_<hash>/` — `01-ttir.mlir`, `07-regularized.mlir`,
`08-transformed.mlir`, `09-custom.mlir`, then the ELF (`11-<kernel>.elf`). The stage a failure did not
reach is the one that owns it.

**To find which op a wrong result first diverges at** (per-kernel CPU cross-check;
sub-option of functional mode, `Simulator/functional_verify.py`). Set `pytorchsim_functional_verify_per_kernel: 1`
in the config YAML, clear the codegen cache, and re-run: each compiled kernel's
output is compared to a CPU golden and the run stops at the first divergent
kernel, naming the op and offending indices.

## Key environment variables

Read in `PyTorchSimFrontend/config.py`:

| Var | Default | Purpose |
|---|---|---|
| `TORCHSIM_DIR` | `/workspace/PyTorchSim` | repo root |
| `TOGSIM_CONFIG` | `configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_functional_only.yml` | TOGSim hardware YAML |
| `TORCHSIM_TIMING_MODE` | unset | overrides the config's `pytorchsim_timing_mode` for one run (`1` to time, `0` not to) |
| `GEM5_PATH` | `$VCIX_ENV_ROOT/gem5/build/RISCV/gem5.opt` | gem5 binary (vcix-accelerator environment) |
| `TORCHSIM_LOG_PATH` | `$TORCHSIM_DIR/togsim_results` | where TOGSim logs go |
| `TORCHSIM_DUMP_PATH` | `$TORCHSIM_DIR` | misc dumps |
| `TORCHSIM_DEBUG_MODE` | `0` | extra debug |
| `TORCHSIM_BREAKDOWN` | `0` | `1` prints where the run's wall clock went (compile per stage/pass, Spike, gem5, TOGSim) at exit, and writes `breakdown_<YYYYMMDD_HHMMSS>_<hash>.json` into the dump path — stamped like `togsim_results/`, so parallel runs sharing a dump path each keep their own |
| `TORCHSIM_COMPILE_DIR` | `$TORCHSIM_DIR/pytorchsim-triton-compiler` | compiler checkout (stages 1-6) |
| `TORCHSIM_COMPILE_PYTHON` | `sys.executable` | interpreter the compiler runs under |
| `TOGSIM_DEBUG_LEVEL` | unset | passed to TOGSim `--log_level` |

Note: `TOGSIM_CONFIG` is **overwritten** while inside a `with TOGSimulator(config_path=...)` block (and restored on exit). Compilation reads the same YAML as TOGSim that way.

## TOGSim YAML knobs (the ones I edit most)

Located under `configs/*.yml`:

- `num_cores`, `core_freq_mhz`, `num_systolic_array_per_core`
- `sa_weight_buffer_depth` (per-SA resident weight slots; **must be > 0** — the simulator errors on 0. Raise it to effectively disable the preload run-ahead throttle. Defaults to 2 if the key is absent.)
- `vpu_num_lanes`, `vpu_spad_size_kb_per_lane`, `vpu_vector_length_bits`
- `dram_type` (`ramulator2` | `simple`), `dram_channels`, `dram_freq_mhz`, `ramulator_config_path`
- `icnt_type` (`simple` | `booksim`), `icnt_latency_cycles`, `icnt_freq_mhz`, `icnt_config_path`
- `l2d_type` (e.g., `datacache`), `l2d_config` (AccelSim-format cache config string)
- `pytorchsim_functional_mode` (Spike on/off), `pytorchsim_timing_mode`
- `pytorchsim_functional_verify_per_kernel` (debug: per-kernel CPU cross-check)
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
- **Compiler + Spike / gem5 from source:** `bash scripts/build_from_source.sh` (checks out PyTorchSim-Triton-Backend at the commit pinned in `thirdparty/pytorchsim-triton-backend.json`, the same pin as the CI docker image, and runs its `setup/toolchain.sh`; Spike, pk and gem5 land in `$VCIX_ENV_ROOT`).

Conan deps for TOGSim: `boost/1.79.0`, `robin-hood-hashing/3.11.5`, `spdlog/1.11.0`, `yaml-cpp/0.8.0`.

## Where to look for X

- **Adding a new op:** nothing here, usually — Inductor's own Triton lowerings emit the kernel and PyTorchSim-Triton-Backend lowers it. What lives here is our FX graph rewrites, `rewrite_fx_graph.py`: decompositions and post-grad passes for ops with no `npu` kernel (histc, polar, topk, transformers' grouped_mm, complex arithmetic). GEMM/BMM tile selection: `PyTorchSimCodegen/inductor_templates.py` + `PyTorchSimCodegen/hardware.py`. Kernel source fixups before the compiler sees them: `PyTorchSimCodegen/source_rewrite.py`.
- **Adding a PyTorch device op:** `PyTorchSimDevice/csrc/aten/native/*` (Minimal/Extra split mirrors `torch_openreg`).
- **TOGSim hardware model changes:** `TOGSim/src/{Core,Dram,Interconnect,L2Cache,Tile,TileGraph}.cc` + matching `include/*.h`.
- **TOG generation:** the compiler emits the trace producer as C++ and the compute type of each tile (`pytorchsim_triton_compiler/trace/`, `trace_cpp` and `tile_types` in `kernel.json`); this repo measures the tiles under gem5, compiles the C++ to `trace.so` and writes `trace_cycles.tsv` (`Simulator/trace_build.py`, driven from `Simulator/timing.py`); TOGSim turns them into a TileGraph via `trace_to_tilegraph`.
- **Eager fallback registration:** `torch.npu.register_eager_to_compile([...])` — see `tests/system/test_eager.py`.
- **Per-run results:** `togsim_results/<YYYYMMDD_HHMMSS_<hash>>.log` (stats) and `.trace` (instruction trace). The path is also printed at the end of every run.
- **Utilization of a run:** `python scripts/util_viewer.py togsim_results -o util.html` builds a self-contained page from the info-level logs already on disk (systolic array / vector unit / DMA / DRAM over cycles, per kernel, one lane each so overlap is visible). Add `--timing <dump path>` for the compile clock (`timing.json`, joined by `triton_<hash>`) and `--breakdown <dump path>` for the whole-run split across compile / Spike / gem5 / TOGSim (`breakdown.json`, written by `TORCHSIM_BREAKDOWN=1`). For per-instruction Gantt detail instead, re-run with `TOGSIM_DEBUG_LEVEL=trace` and pipe through `scripts/trace_timeline.py` into Perfetto.
- **Wrapper codegen path:** printed as `Wrapper Codegen Path = /tmp/torchinductor_<user>/<hash>/...py` — useful for inspecting generated kernel code.

## Gotchas / things I've already learned

- The repo expects `python` to be a Python 3.10+ binary with `torch==2.10.0` (torchvision `0.25.0`, triton `3.6.0`). The frontend extends the PyTorch 2 Inductor stack — pin to this version. 2.10 specifically: it is the first release whose Inductor targets triton 3.6, the version PyTorchSim-Triton-Backend is built against. The pins live in `Dockerfile.base`, and editing that file changes the base-image tag automatically (the tag is `thirdparty-<sha256 of thirdparty/github-releases.json + Dockerfile.base>`, see `scripts/ci/thirdparty_base_pin.sh`).
- gem5 and Spike are the vcix-accelerator environment's, found through `VCIX_ENV_ROOT`; both load the accelerator model `TORCHSIM_COMPILE_VCIX_MODEL` (default `$TORCHSIM_PREFIX/vcix-build/libtpu.so`). Override with `GEM5_PATH` / `TORCHSIM_SPIKE` if you build elsewhere.
- `_C.cpython-311-*.so` and `torch_openreg/lib/` are build artifacts — already in `.gitignore`, don't commit.
- **Parallel runs sharing a `TORCHSIM_DUMP_PATH`** share every per-kernel workdir, and that is deliberate: the workdir holds what compiling produced, which is a function of the source and the machine. A **launch** is not, so each process gets `<workdir>/launch-<pid>/` (`Simulator/session.py`) holding its own `runtime/*.raw`, its own `trace_shape.txt` and symlinks to the shared trace — TOGSim resolves the cycle table and the shape from the directory of the trace it is handed, so linking is enough. What is left locked is only *building* the shared thing, once: `.compile.lock` (compile) and `.timing.lock` (trace.so + its gem5 cycle table). Launches take no lock. Measured on a warm cache, two `test_add.py` in parallel: 40s when launches were locked, **31s** now, against 30s for one run alone.
- TOGSim creates a per-PID FIFO under `/tmp/togsim_fifo_<pid>` for command/event comm; if a previous run crashed and left stale FIFOs, they get cleaned up on the next start, but watch for orphaned processes if you Ctrl-C mid-run.
- Multi-tenant runs **must** use the `with TOGSimulator(...)` context manager — otherwise compile-time `TOGSIM_CONFIG` and runtime config can diverge.
- `pytorchsim_functional_mode` exists as both an **env var** and a **YAML key**; the env var path is via `config.py` while the YAML key is read inside the same module. They should agree.
- "No CUDA runtime is found" warnings on `import torch` are expected — this is a CPU + simulated-NPU environment, not real CUDA.
- **Codegen changes are sticky across runs because of caches.** When iterating on `PyTorchSimCodegen/*`, `Simulator/*` or the compiler, clear `$TORCHSIM_DUMP_PATH` (default `$TORCHSIM_DIR/outputs/`) before re-running — it holds Inductor's compile cache (`.torchinductor/`, set via `TORCHINDUCTOR_CACHE_DIR` inside `config.get_dump_path()`), the per-source wrapper dirs (`<hash>/`) keyed by `config.get_write_path(src_code)`, and the per-kernel artifacts (`triton_<hash>/`). A compiler-side fix does NOT move the Inductor hash, so `triton_*` is the one that most often needs deleting; `scripts/clear_codegen_cache.sh` does all three. Otherwise a buggy graph compiled before your fix is replayed verbatim. `togsim_results/` (TOGSim run logs) is cosmetic and not part of the codegen replay path. For parallel worktrees see `docs/worktrees.md`.

## Comments: a three-line docstring per function, and nothing else

The only comment this repo accepts is a docstring of **at most three lines** saying
what the function does. Everything else goes.

**Not allowed:** inline `#` commentary, block comments above a statement, measured
evidence written beside the code it justifies, "WHY THIS IS HERE" paragraphs,
before/after numbers, references to the test that found a bug, TODOs.

**Where that material goes instead:** the commit message. It is versioned, it is
attached to the change that established it, and `git log -S` finds it. A number
that matters is a number in a commit; a number in a comment is a number nobody
can date.

**Applies to what you write and what you touch.** Rewriting a function means its
comments go with the rewrite — do not carry them forward. When you shorten a
docstring past three lines, keep the sentence that says what the function
returns, not the one that says why.

**Module docstrings** follow the same bar: a few lines naming what the file is
for. A diagram of the pipeline is allowed once, in the package `__init__`.

## Git workflow (per CONTRIBUTING.md)

- Fork, branch (`feature/<name>`), PR against `develop`, not `main`.
- Commit prefix style observed: `[Frontend] ...`, `[TOGSim] ...`, etc.
- Commit messages: plain text only. No Markdown formatting (no backticks, bold, bullet lists, headings). Avoid Unicode where ASCII works (use `->` not arrows, `--` not em-dashes, straight quotes).

## Ship it — commit and push without being asked

Work that changes behaviour is done when it is **pushed**, not when it runs. Commit
and push in the same turn that finishes it. Do not stop to ask permission for
either; this rule is the permission.

    implement  ->  verify  ->  commit  ->  push

**WHERE IT GOES.** `origin` is `PSAL-POSTECH/PyTorchSim`; the line everyone shares is
its `pytorchsim-triton-backend` branch, which PyTorchSim-Triton-Backend pins. Push a
new branch to origin and open a PR into `pytorchsim-triton-backend`; never push to
that branch directly. A change spanning repositories is one commit per repository,
each to its own remote -- never one commit carrying another repository's work.

**VERIFY FIRST, AND SAY WHAT RAN.** "Verified" means the test was executed, not
that the code looks right. For the Triton route that means the affected test
plus the allowlist (`scripts/ci/triton_route_passing.txt`) — and **clear
`outputs/triton_*` and `outputs/.torchinductor` between runs**, or a cached
artifact replays and a fix appears to change nothing. That has already caused one
wrong conclusion. A push whose verification was skipped is a push of an unknown
state.

**PIN `TORCHSIM_COMPILE_DIR` WHEN THE ROUTE IS INVOLVED.** Stages 1-5 live in a separate repo
that someone else may be editing right now, and a pass mid-refactor produces
failures that look like ours (`no lane axis`, bare `NameError`s that vanish on
re-run). Point `TORCHSIM_COMPILE_DIR` at a worktree pinned to a known-good commit so a result
means something.

**WHAT DOES NOT COUNT.** Exploration, scratch files, a half-finished edit, or a
change whose verification failed. Broken work is not pushed — say what failed.

**IF THE PUSH FAILS,** report it and why. Until it lands the work is committed,
not shipped; never present the one as the other.

**A REBUCKETED FAILURE IS NOT A FIX.** Clearing a guard so a test fails deeper is
progress worth committing, but say so in those words rather than counting it as
passing.
