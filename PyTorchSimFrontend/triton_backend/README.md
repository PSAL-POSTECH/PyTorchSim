# Triton codegen route (WIP)

Replaces the hand-written MLIR emission in `PyTorchSimFrontend/mlir/` with
**Inductor's own Triton codegen**, lowered to this NPU by the **triton-npu**
(`tnpu`) pass pipeline. Opt-in and off by default; the MLIR route is untouched
and stays the production path.

```bash
TORCHSIM_TRITON_CODEGEN=1 python tests/system/test_triton_codegen.py
```

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
        │  01-ttir → 02-ttshared → 03-adapted → 04-lowered → 05-*.elf
        ▼
    TritonNPULauncher.__call__                   ← NOT WIRED YET
```

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
- the run stops in `TritonNPULauncher.__call__`, by design

## Gap list, in order

1. **Launch (functional).** Marshal the caller's tensors into
   `runtime/*.raw`, run Spike on the ELF, read outputs back. tnpu's stage 6 does
   this for its own kernels but generates inputs from the spec; here the tensors
   come from the caller.
2. **Launch (timing).** Emit `trace.so` + `trace_cycles.tsv` and hand them to
   TOGSim. Blocked on the `build_tog` adapters — the tnpu IR is structurally
   invisible to it today (no top-level `affine.for`, `scf.for` instead of
   `affine.for`, vcix as LLVM intrinsics rather than dialect ops, DMA addresses
   as `arith` chains rather than `affine.apply`, grid outside the IR).
3. **`triton_helpers`.** Any kernel using `triton_helpers.*` (reductions,
   clamps, `maximum`/`minimum`) cannot compile: the module lives in torch and
   the tnpu venv has none. `strip_for_tnpu` raises and names the helper. Needs a
   minimal vendored copy.
4. **Reductions.** Independently blocked in tnpu itself — no lane-aware
   reduction path; see `triton-npu/kernels/reduce.py`.
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

**torch 2.8's Inductor is shimmed onto triton 3.6.** Inductor targets the
triton ~3.3 API; tnpu pins 3.6 because 3.6 pins LLVM 23, and both sides of the
IR seam must be the same LLVM. So the frontend bends: `triton_key` is injected
back into `triton.compiler.compiler`, and `triton_hash_with_backend` is
short-circuited because it asks a GPU driver for the current target and there is
no GPU. The alternative — a second, torch-compatible triton in the driver
interpreter purely for codegen — stays open if the API drift widens beyond this
one symbol. (`_triton_compat`)
