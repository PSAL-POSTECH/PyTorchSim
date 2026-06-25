# Per-kernel functional verify

A debugging tool that pinpoints the **first compiled kernel** whose simulated
(Spike) output diverges from a CPU reference. Use it when a whole-model run gives
a wrong final result and you need to know *which operation* introduced the error,
instead of only seeing that the final `torch.allclose` failed.

## TL;DR

```yaml
# in your TOGSim config YAML (e.g. configs/systolic_ws_8x8_c1_simple_noc_tpuv3.yml)
pytorchsim_functional_mode: 1               # required (this is its parent)
pytorchsim_functional_verify_per_kernel: 1  # turn the verify on
```

```bash
# checks are baked into the wrapper at compile time, so clear the codegen cache
# whenever you toggle this option (see the "Gotchas" note in CLAUDE.md)
bash scripts/clear_codegen_cache.sh
python tests/models/Yolov5/test_yolov5.py
```

On the first divergent kernel you get, and the run stops there:

```
================= PER-KERNEL FUNCTIONAL VERIFY: DIVERGENCE =================
 first divergent buffer : buf1
 originating fx op      : aten.relu.default   (node 'relu')
 shape                  : (16, 16)
 elements over tol      : 90 / 256
 max abs diff           : 1.43051e-06   (rtol=1e-4 atol=1e-4)
 first bad index        : [0, 0]
 buffers verified OK    : 0
 sample mismatches (npu vs cpu):
      (0, 0): npu=0.323295  cpu=0.323294
      ...
===========================================================================
```

## Options

| Option | Where | Default | Meaning |
|---|---|---|---|
| `pytorchsim_functional_verify_per_kernel` | config YAML | `0` (off) | enable the per-kernel CPU cross-check |
| `TORCHSIM_FUNCTIONAL_VERIFY_RTOL` | env | `1e-4` | relative tolerance for the compare |
| `TORCHSIM_FUNCTIONAL_VERIFY_ATOL` | env | `1e-4` | absolute tolerance for the compare |

It is a **sub-option of `pytorchsim_functional_mode`** and is auto-disabled when
functional mode is off (there are no Spike values to verify otherwise). The
config accessor AND-gates the two, so setting the key alone with functional mode
off does nothing.

## How it works

The value of every tensor in a compiled graph comes from Spike running the
generated kernels (the timing path / TOGSim cannot change tensor values). This
tool compares those values against a CPU "golden" at the boundary of each
compiled kernel.

1. **Golden (once per `call(args)`).** At the top of the generated wrapper,
   `verify_init(gid, [inputs])` looks up the original aten graph
   (`V.graph.module`, registered at codegen time), **copies the inputs to CPU**,
   and runs the whole graph on CPU with an `fx.Interpreter` that records every
   node's output tensor by name. These are the absolute-correct reference values
   (computed from the original inputs, with no accumulated npu error).

2. **Check (after each kernel).** After a kernel writes its output buffer,
   `verify_check(buf, "buf", node_name, op)` copies that npu buffer to CPU and
   `torch.allclose`-compares it to `golden[node_name]`. The buffer is mapped to
   its originating fx node via `V.graph.get_buffer(name).origin_node`, which is
   how the report can name the offending op.

3. **Stop at first.** Each buffer is checked once, on its first appearance as a
   kernel argument (the producer kernel precedes any consumer in topological
   order). The first buffer that exceeds tolerance is the injection point -- its
   inputs were still within tolerance -- so the check logs the report and raises
   `FunctionalVerifyMismatch` immediately.

Comparison granularity is the **fused-cluster output**: Inductor fuses several
aten ops into one kernel and only the cluster's realized output buffer is
observable. The check verifies that buffer and names the cluster's output op. For
example `relu(a @ b + bias)` is one kernel writing one buffer, reported as
`aten.relu.default`.

## Limitations

- **Granularity is the fused kernel, not the individual aten op.** If the bug is
  in an op that was fused into the middle of a cluster (e.g. the matmul inside
  `relu(a@b+bias)`), it is reported at the cluster's output buffer/op (`relu`),
  not the internal culprit. Ops that get their own kernel (matmul, conv, cat,
  sdpa, sort, ...) are named precisely.
- **Requires a codegen-cache clear when toggling**, because the `verify_*` calls
  are emitted into the wrapper only when the option is on at compile time. A
  cached wrapper compiled without it will be replayed without checks.
- **Coverage, not correctness, degrades gracefully.** A buffer with no
  `origin_node`, a non-tensor, or a name the golden did not capture is silently
  skipped (fewer checks, never a false alarm). The golden run is wrapped in
  try/except: if the aten graph fails to run on CPU, checks are disabled for that
  graph rather than crashing the real run.
- **Overhead.** The full aten graph is run on CPU once per `call(args)`, and each
  buffer is copied to CPU for the compare. This is a debugging mode; leave it off
  for normal runs.

## Code

- `PyTorchSimFrontend/extension_functional_verify.py` -- graph registry, golden
  interpreter, compare/report (`register_graph`, `verify_init`, `verify_check`).
- `PyTorchSimFrontend/mlir/mlir_codegen_backend.py` -- injects the calls into the
  wrapper (`write_prefix`, `generate`, `_fverify_emit_checks`).
- `PyTorchSimFrontend/extension_config.py` -- reads the YAML key and AND-gates it
  with `pytorchsim_functional_mode`.
