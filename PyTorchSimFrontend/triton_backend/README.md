# Triton codegen route (WIP)

Replaces the hand-written MLIR emission in `PyTorchSimFrontend/mlir/` with
**Inductor's own Triton codegen**, lowered to this NPU by the **PyTorchSim
lowering pass** (owned by 이정민; the code lives in the `triton-npu` repo, so
paths and module names read `tnpu`). Opt-in and off by default; the MLIR route
is untouched and stays the production path.

The modules here are the PORT: they drive that lowering pass and wire its output
into the existing TOGSim / gem5 / Spike stack. The pass itself is not ours.

```bash
TORCHSIM_TRITON_CODEGEN=1 python tests/system/test_triton_codegen.py
```

This file is the working reference for the modules here. For how the route
compares with the MLIR one and what the numbers are, see
[`../triton-codegen-route.md`](../triton-codegen-route.md).

## Why

The MLIR route does not just emit loops — it hand-implements the whole hardware
mapping (tiling, vectorization, DMA, scratchpad, lane distribution) as ~5,500
lines of Python string emission, which entangles *what to compute* with *how to
map it*. See `docs/linalg-codegen-migration.md` for the long form.

This route keeps Inductor for the first and triton-npu for the second:

| | owns |
|---|---|
| Inductor (upstream) | fusion, index expressions, masking, reductions, the kernel source |
| triton-shared | Triton IR -> `linalg` / `tts` pointer descriptors |
| tnpu passes | `tts` -> `togsim.transfer` DMA, scratchpad, lane-banked vectors, systolic array |

## Flow

```
torch.compile
  └ TritonNPUScheduling.define_kernel            scheduling.py
        │  Inductor's triton kernel SOURCE TEXT + collected metadata
        ▼
    triton_npu_compile(src, meta, name)          codecache.py
        │  a tnpu kernel file (KernelSpec)       kernel_spec.py
        ▼
    run.py <spec> --to binary   (subprocess)     tnpu_bridge.py
        │  01-ttir → 02-ttshared → 03-adapted → 04-custom → 05-*.elf
        ▼
    TritonNPULauncher.__call__                   codecache.py
        ├ functional  tensors → runtime/*.raw → Spike → tensors   functional.py
        └ timing      04-custom.mlir → trace.so + trace_cycles.tsv → TOGSim
                       cycles measured by gem5 on a one-tile binary
```

The timing half reuses PyTorchSim's trace pipeline unchanged. The one structural
difference is that a Triton kernel body is a single program instance, so the grid
that enumerates instances is supplied by `lower_to_emitc.WorkItem` instead of
being read out of the kernel -- see "The grid is not in the kernel" below.

Artifacts land in one directory per source hash under the dump path
(`outputs/triton_<hash>/`), alongside the unmodified Inductor source
(`kernel.py`) so the rewrite is diffable.

## What works today (measured)

`x + y`, 1024 elements, on `npu:0`:

- Inductor generates the Triton kernel and our `define_kernel` intercepts it
- `kernel_spec` pins `XBLOCK` = lane count, computes `grid = (8,)`, writes the spec
- tnpu runs stages 1–5 and links **`05-triton_npu_fused_add_0.elf`** (20 B/lane spad)
- the lowering is correct in shape: `tl.load/store` became three
  `togsim.transfer` ops, and Inductor's `xmask` came through as a **masked DMA**
  (`masked_axes = [0]`, `masked_fill`), which tnpu already supports
- the trace producer comes out in the shape the design calls for: a
  `togsim_kernel_tile` computing `offset = iv[0]*128` around three `togsim_dma`
  and one `togsim_compute`, and a `togsim_kernel` looping `p < 8` over
  `togsim_dispatch`
- **TOGSim runs it: 650 cycles**, with channel-0 DRAM traffic of 16 reads x 32 B
  x 16 channels = 8192 B, exactly the 8 work-items x 2 loads x 512 B the kernel
  should move. The MLIR route on the same `x + y` reports 251 cycles -- the same
  order, and higher here because tnpu emits synchronous DMA, so nothing overlaps
  (gap 2)
- the tile's compute cost is a real measurement: gem5 samples **19 cycles** for
  the vector-add tile, via `timing.measure_tile_cycles`
- **values are correct**: the launch writes the caller's tensors from Spike and
  `torch.allclose` holds over all 1024 elements, for the fused
  `(x + y) * 2 - x` kernel too

## Shape specialisation

The functional binary is compiled for ONE shape: the spec bakes the grid, the
scalar values and the memref extents in. A dynamic-shape graph reuses that ELF,
so `functional.ShapeMismatch` rejects the launch instead of running against the
wrong bounds. The timing path has no such limit -- it takes the grid at run time
-- so `pytorchsim_functional_mode: False` studies cycles across shapes.

## Gap list, in order

1. **Shape-specialised functional launch.** Recompile per launch shape, or teach
   the tnpu wrapper to take the grid and the extents as arguments the way the
   trace producer already does.
2. **Double buffering.** tnpu emits synchronous DMA (`is_async=false`, no
   `togsim.wait`), so load → compute → store serialize inside every work-item and
   TOGSim has no overlap to model. This is the main remaining gap between the two
   routes' cycle counts.
3. **`triton_helpers`.** Any kernel using `triton_helpers.*` (reductions,
   clamps, `maximum`/`minimum`) cannot compile: the module lives in torch and
   the tnpu venv has none. `strip_for_tnpu` raises and names the helper. Needs a
   minimal vendored copy.
4. **Reductions.** Independently blocked in tnpu itself — no lane-aware
   reduction path; see `triton-npu/kernels/reduce.py`.
   Matmul is also still open on the timing side: `build_tog` finds compute nodes
   by the `vcix.iv` op name, and tnpu emits `llvm.riscv.sf.vc.*` intrinsics.
5. **Block-size policy.** `fixed_config_for` pins `XBLOCK` to the lane count and
   deliberately leaves reduction blocks unset. Real tile selection (the MLIR
   route's autotuner / `codegen_mapping_strategy`) has no equivalent here yet.
6. **Dynamic shapes.** `collect_meta` resolves numels through `size_hint`; a
   genuinely dynamic dim gives `None` and `_grid` raises.

## Three design decisions

**Block sizes are fixed at codegen time.** Inductor defers the grid to
`triton_heuristics` at runtime (`grid = cdiv(xnumel, XBLOCK)` after autotuning).
tnpu compiles one binary ahead of time and walks the grid as a sequential loop in
generated C, so there is nothing to autotune later and no runtime `grid=`
callable. Pinning the config is what makes the launch shape statically
describable — the premise of this route, not a shortcut. (`kernel_spec.fixed_config_for`)

**tnpu runs in its own process.** Its passes need LLVM 23's MLIR bindings while
this process holds LLVM 20's, and `mlir` is a namespace package, so two LLVMs in
one interpreter silently merge. The seam between them is a file, and that is
measured to work: LLVM 23 prints IR that LLVM 20's bindings parse without
complaint. (`tnpu_bridge`)

**The torch pin is what makes triton 3.6 work.** triton-npu pins triton 3.6
because 3.6 pins LLVM 23, and both sides of its textual IR seam must be the same
LLVM. torch 2.10 is the first release whose Inductor targets 3.6, so the two
simply agree -- on 2.8 the frontend had to be shimmed onto a triton it did not
expect. What remains in `_triton_compat` is not a version shim: on a box with no
GPU, `triton_hash_with_backend()` raises "0 active drivers" because it asks the
triton runtime for the current target. We never launch through that runtime, so
the value is short-circuited to a deterministic cache key.

**The grid is not in the kernel.** PyTorchSim's codegen puts the tile loops
inside the kernel; a Triton kernel describes one program instance and leaves the
grid to the launch. The trace producer wants that same split already --
`togsim_kernel_tile` per work-item, enumerated by `togsim_kernel` (design sec
9.3) -- so the models agree and only the enumeration was missing.
`_materialize_grid_loop` supplies it, on the trace artifact only: it wraps the
body in a loop tagged `outer_loop` with each program-id argument replaced by the
induction variable, and everything downstream is unchanged. It runs before
`_rewrite_signature`, which erases the kernel arguments and first asserts none
are still used -- that ordering is what decides where this can live.

## CI

`.github/workflows/triton_npu.yml`, separate from the main CI: this route is WIP,
and its toolchain layer is ~1.8 GiB that no other job needs.

```
preflight              TNPU_TOKEN set? repo readable? release present?
ensure-tnpu-base       torchsim_base + tnpu toolchain -> torchsim_tnpu_base:<pins>
build-app              ./Dockerfile on that base
tnpu-baselines         run.py doctor + add/mul/relu/gemm/bmm through Spike   (gates)
triton-route           tests/system/test_triton_codegen.py    (reports, does not gate)
mlir-route-regression  tests/ops/elementwise/test_add.py      (gates)
```

The toolchain image is pinned the same way `torchsim_base` is — the tag carries
`sha256(thirdparty/triton-npu.json + Dockerfile.tnpu)`, so it is rebuilt only when
one of those moves, and its tag also carries the base pin it was built on.
`mlir-route-regression` is there because this layer adds a *second* LLVM and a
*second* triton to the image; it checks the production path did not notice.

**Needs `secrets.TNPU_TOKEN`** — a PAT that can read `PSAL-POSTECH/triton-npu`
and its `toolchain-llvm23` release. That repo is private and the default Actions
token is scoped to this repository. `preflight` checks it before the build.

`Dockerfile.tnpu` clones the harness and runs its own `setup/restore.sh
--prebuilt`; the pins all live in that repo's `setup/versions.env`. `ref` in the
manifest is a commit, so an upstream change there moves this image's tag too.
