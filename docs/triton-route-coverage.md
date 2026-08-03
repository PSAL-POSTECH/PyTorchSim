# Triton codegen route — test suite coverage

First measurement of PyTorchSim's existing test suite running through the Triton
codegen route (Inductor's Triton backend + the triton-npu lowering passes)
instead of the MLIR route.

| | |
|---|---|
| Date | 2026-08-03 |
| Branch | `feature/triton-codegen` @ `8e17519` |
| tnpu pin | `5d84caf` |
| torch | 2.10.0, triton 3.6.0 |
| Tests | 69 (everything under `tests/`) |
| Runtime | 5 min at `-j 10` (~50 min serial) |

Reproduce:

```bash
python scripts/ci/triton_route_sweep.py --all -j 10 \
    --markdown coverage.md --artifacts failures
```

Every claim below is backed by a file in `failures/`. Paths are given so each
one can be checked.

---

## 1. Headline

```
69 tests
├── 11  pass THROUGH the route          ← this is the coverage number
├──  5  pass without using the route    ← no kernel emitted at all
└── 53  fail
    ├── 17  missing test deps (local venv only; present in the CI image)
    └── 36  real blockers
```

**11/69, not 16/69.** Five tests pass while emitting no Triton kernel at all:
`test_matmul`, `test_bmm`, `test_topk`, `test_moe_cpu`, `test_mlir_bindings`.
Inductor sends `mm`/`bmm` to an extern call rather than generating a kernel, so
those tests never exercise the thing under test. The sweep records that
separately (`exercised` in the JSON) and keeps them out of the gate — counting
them would overstate coverage by 45%.

### What passes

| Test | Time |
|---|---|
| `tests/ops/elementwise/test_add.py` | 77.5s |
| `tests/ops/fusion/test_addmm_residual.py` | 33.0s |
| `tests/ops/fusion/test_matmul_scalar.py` | 11.5s |
| `tests/ops/fusion/test_matmul_vector.py` | 17.4s |
| `tests/ops/fusion/test_prologue_fusion.py` | 41.4s |
| `tests/ops/misc/test_expert_mask.py` | 11.0s |
| `tests/ops/reduce/test_batchnorm.py` | 37.7s |
| `tests/ops/view/test_view3D_2D.py` | 36.4s |
| `tests/system/test_eager.py` | 15.0s |
| `tests/system/test_stonne.py` | 9.7s |
| `tests/system/test_triton_codegen.py` | 10.0s |

Four of the eleven are fusion tests. Inductor's fusion is the half of this
migration we get for free, and it is already producing kernels tnpu accepts.

---

## 2. Worked example — how one failure is diagnosed

`tests/ops/reduce/test_softmax.py`. The sweep leaves this behind:

```
failures/tests_ops_reduce_test_softmax/
  kernel.py        the Inductor Triton kernel, unmodified
  error.txt        bucket, stage, last 60 lines
```

`error.txt` opens with the routing header:

```
test:   tests/ops/reduce/test_softmax.py
bucket: triton_helpers
stage:  0 kernel generated, not accepted
```

And `kernel.py` is the whole reason, in 28 lines:

```python
@triton.jit
def triton_npu_fused__softmax_0(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_1 = r0_index
    x0 = xindex
    tmp0  = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1  = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3  = tl.where(xmask, tmp1, float("-inf"))
    tmp4  = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)   # <-- blocker 1
    tmp5  = tmp0 - tmp4
    tmp6  = libdevice.exp(tmp5)                                    # <-- blocker 2
    tmp7  = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9  = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = (tmp6 / tmp10)
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp11, xmask)
```

Two blockers, visible without running anything:

- `triton_helpers.max2` — lives in `torch._inductor.runtime`, and the tnpu venv
  deliberately has no torch.
- `libdevice.exp` — an `@core.extern` intrinsic with no triton_shared
  implementation.

Note what *is* fine: `tl.load` with a mask, `tl.where`, `tl.sum`, `tl.store`,
the 2-D `[XBLOCK, R0_BLOCK]` broadcast. Softmax is not blocked on anything
structural. It is blocked on two function calls.

This is the whole bug report, and it needed no rerun to produce.

---

## 3. Where kernels stop

Each test is placed at the furthest stage any of its kernels produced an
artifact for. The stage a kernel *fails to reach* owns the failure.

| Stage | Count | |
|---|---|---|
| — no kernel generated | 26 | died in torch/Inductor before codegen |
| 0 generated, rejected | 16 | `kernel_spec` refused to describe it |
| 1 triton → ttir | 1 | |
| 2 ttir → tts/linalg | 1 | triton-shared |
| 4 tnpu lower (DMA, lanes, spad) | 6 | |
| 5 trace producer | 3 | |

**The lowering passes are not the bottleneck yet.** Only 2 of 53 failures are a
tnpu pass rejecting IR. The other 34 real blockers stop earlier — in the port
that feeds tnpu, or in torch itself. The next round of work is mostly on our
side of the seam.

---

## 4. Failures by cause, with evidence

### `spec_incomplete` — 13 · owner: `triton_backend/kernel_spec.py`

**libdevice intrinsics (5).** `@core.extern` members with no triton_shared
implementation; a call returns `None`.

| Test | Symbol |
|---|---|
| `ops/elementwise/test_exponent.py` | `libdevice.exp` |
| `ops/elementwise/test_pointwise.py` | `libdevice.isnan` |
| `ops/elementwise/test_transcendental.py` | `libdevice.tanh` |
| `ops/reduce/test_layernorm.py` | `libdevice.rsqrt` |
| `ops/view/test_floormod_axis_split.py` | `libdevice.rsqrt` |

Without the diagnostic added this session, these failed as a bare
`NameError('libdevice is not defined')` inside tnpu's stage-1 worker — which is
what made them look like six separate lowering bugs. They now say:

```
SpecIncomplete: kernel calls libdevice.{exp}: those are extern math intrinsics
with no implementation on the triton_shared backend. They need lowering to a
VPU op (or a scalar fallback) before this kernel can compile.
```

**Multi-axis grid (4).** `fixed_config_for` pins only the outermost axis, so
`YBLOCK` is `None` and the grid cannot be computed. This is the known
block-size policy gap.

| Test | Diagnostic |
|---|---|
| `ops/view/test_transpose2D.py` | axis `y`: ynumel=156, YBLOCK=None |
| `ops/view/test_transpose3D.py` | axis `y`: ynumel=2728, YBLOCK=None |
| `ops/fusion/test_conv_fusion.py` | axis `y`: ynumel=192, YBLOCK=None |
| `ops/conv/test_conv_view_input.py` | axis `y`: ynumel=512, YBLOCK=None |

**Reduction blocks unset (2)** — `R0_BLOCK` is left unset on purpose:
`ops/fusion/test_bmm_reduction.py`, `ops/fusion/test_matmul_reduction.py`.

**Genuine metadata hole (1)** — `ops/misc/test_widen_dtype.py`: no dtype/numel
for `out_ptr0`; `collect_meta` could not resolve it from `V.graph`.

### `triton_helpers` — 7 · owner: `triton_backend`

| Test | Helper |
|---|---|
| `ops/reduce/test_softmax.py` | `max2` |
| `ops/sort/test_sort.py` | `sort_with_index` |
| `ops/elementwise/test_activation.py` | `maximum` |
| `ops/conv/test_cnn.py` | `maximum` |
| `ops/fusion/test_matmul_activation.py` | `maximum` |
| `ops/sparsity/test_sparsity.py` | `maximum` |
| `models/test_mlp.py` | `maximum` |

Four of seven want only `maximum`. The fix is a small vendored file, not a pass
change.

### `wrapper_gap` — 6 · owner: `triton_backend`

Every one, identically:

```
AttributeError: 'TritonNPUWrapperCodegen' object has no attribute 'estimate_peak'
```

`ops/attention/test_gqa.py`, `test_gqa_decode.py`,
`ops/fusion/test_attention_fusion.py`, `test_transformer_fusion.py`,
`models/Mixtral8x7B/test_attention.py`, `models/test_transformer.py`

Every attention and transformer test in the suite, blocked on one unimplemented
method.

### `device_op` — 3 · owner: `PyTorchSimDevice`

Predates this route — the MLIR route intercepts these before the dispatcher.

| Test | Error |
|---|---|
| `ops/conv/test_conv2d.py` | `convolution_overrideable not implemented` |
| `ops/conv/test_group_conv.py` | `convolution_overrideable not implemented` |
| `ops/attention/test_sdpa.py` | `_scaled_dot_product_fused_attention_overrideable not implemented` |

### `tnpu_stage` — 2 · owner: triton-npu lowering passes

The only failures that are genuinely a pass rejecting IR — and the artifacts
show the full chain from Python to the rejected op.

**`ops/conv/test_pool.py`** — stage 1.

`kernel.py` ends with an innocuous line Inductor emits after a reduction:

```python
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = 49.0
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()          # <-- this
```

`01-ttir.mlir:46` is what that becomes:

```mlir
%tmp4_17 = tt.expand_dims %tmp4 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
%tmp6_18 = arith.divf %tmp4_17, %tmp6 : tensor<128x1xf32>
ttg.barrier all                 # <-- GPU dialect op
%0 = tt.splat %in_out_ptr0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
```

and `triton-shared-opt` cannot parse it:

```
01-ttir.mlir:46:5: error: Dialect `ttg' not found for custom op 'ttg.barrier'
```

`ttg` is the GPU dialect. A `tl.debug_barrier()` in a pointwise-after-reduction
kernel is meaningless on this target, but it is in the IR and the parser stops
on it.

**`ops/reduce/test_reduce.py`** — stage 2. A plain `(a + b).sum(dim=1)`:

```python
tmp0 = tl.load(in_ptr0 + (r0_1 + 47*x0), r0_mask & xmask, other=0.0)
tmp1 = tl.load(in_ptr1 + (r0_1 + 47*x0), r0_mask & xmask, other=0.0)
tmp2 = tmp0 + tmp1
tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
```

survives `01-ttir.mlir`, then fails converting to linalg:

```
error: "-":101:11: 'linalg.index' op expected dim (2) to be lower than
       the number of loops (2) of the enclosing LinalgOp
```

Both artifacts carry the diagnostic and the offending `.mlir`, so they can go
upstream as-is.

### `togsim` / `other` — 5

| Test | Stage | Detail |
|---|---|---|
| `ops/view/test_cat.py` | 4 | `[Spike] triton_npu_fused_cat_0 failed`, exit 255 |
| `ops/misc/test_masked_nondividing.py` | 4 | `[Spike] triton_npu_fused_constant_pad_nd_0 failed` |
| `ops/misc/test_indirect_access.py` | 5 | TOGSim returned `inf` cycles for `index_put` |
| `system/test_hetro.py` | — | `KeyError: 'vpu_num_lanes'` (hetero config lacks the key) |
| `ops/sparsity/test_sparse_core.py` | — | `TypeError: '>' between Tensor and torch.device` (test-side bug) |

The two Spike failures are the most interesting in the sweep: the only cases
that compile all the way to a working RISC-V binary and then fail at run time.
`test_cat`'s `04-custom.mlir` shows the lowering did its job —

```mlir
"togsim.transfer"(%reinterpret_cast_5, %c0, %2, %c0, %7, %c0, %c2, %c1, %6)
    {vlane_split_axis = 0, ...}
"togsim.transfer"(%reinterpret_cast_4, %c0, %1, %c0, %7, %c0, %c2, %c1, %6)
    {vlane_split_axis = 0, ...}
"togsim.transfer"(%reinterpret_cast,   %9, %0, %c0, %7, %c0, %c3, %c1, %c0, %13)
    {vlane_split_axis = 0, ...}
```

— two input DMAs and one output DMA, lane-split on axis 0, exactly the shape
`cat` should produce. It is the execution that goes wrong, not the lowering
structure.

**Caveat on these two.** `tnpu.spike` reports only `StageError: command failed
with exit code 255`; spike's own stderr does not survive. Running the recorded
spike command by hand on the same workdir exits 0, because `write_inputs`
rewrites `runtime/*.raw` per launch and a by-hand run replays stale inputs. So
the failing input is not currently reproducible outside the pipeline. Surfacing
spike's stderr the way `TnpuError` now surfaces tnpu's is the prerequisite for
diagnosing these, and is not yet done.

### `missing_dep` — 17 · not a route problem

`transformers` (5), `torchvision` (4), `matplotlib` (4), `pytest` (2),
`diffusers`, `requests`, `sklearn`. Local venv only — these run for real in the
CI image, which is why the sweep belongs in CI.

---

## 5. Infrastructure

### The runner

`scripts/ci/triton_route_sweep.py`. `TORCHSIM_TRITON_CODEGEN` is read once at
device registration (`PyTorchSimDevice/torch_openreg/__init__.py:30`), so **no
test file needed to change** — all 69 were already tests of this route. Only a
runner was missing.

Three outputs:

1. **Gate** — `scripts/ci/triton_route_passing.txt` lists what passes today; CI
   fails if any regresses. Coverage grows by regenerating the file
   (`--update-allowlist`), so it cannot silently shrink.
2. **Report** — bucketed by owning layer and by pipeline stage.
3. **Artifacts** — one directory per failing test (section 2).

A deeper failure keeps more. `tests_ops_conv_test_cnn/` holds
`01-ttir.mlir 02-ttshared.mlir 03-adapted.mlir 04-custom.mlir kernel.py
stage.log error.txt` — the complete lowering chain up to the point it stopped.

### Parallelism

Tests are independent subprocesses with their own dump dir, Inductor cache
(`TORCHINDUCTOR_CACHE_DIR` follows `TORCHSIM_DUMP_PATH`) and TOGSim FIFO (keyed
by pid), so `-j` needs no coordination. Threads, not processes: `run_one` only
waits on a subprocess. **69 tests: ~50 min → 5 min at `-j 10`.**

### CI

`.github/workflows/triton_npu.yml`, job `triton-route-suite`:

- **Allowlisted tests** — gates.
- **Full sweep** — `continue-on-error`; writes `coverage.md` into the step
  summary and uploads `triton-route-coverage` (results.json + failures/).

Jobs run on the PSAL Slurm runner farm (`PSAL-POSTECH/slurm-ghr`): `runs-on`
carries the `slurm` label, image builds and the sweep on `big` (16c/64G/2h),
the rest on small. Do not add `docker/setup-buildx-action` — the runner
registers its own builder.

---

## 6. Three diagnostics fixed while measuring

The reporting infrastructure could not be built until these were fixed, because
each was destroying the evidence.

**`kernel.py` was written after the check that rejects it.** `write_spec_file`
raises for exactly the kernels worth keeping (`triton_helpers`,
`SpecIncomplete`) and ran *before* the source was saved — so the interesting
sources were the ones being thrown away. Reordered; the dump now exists for all
16 rejected kernels, including the softmax example in section 2.

**tnpu reported "exit 1" and nothing else.** `run.py` prints a stage table to
stdout and the real diagnostic only to `stage.log`. Before:

```
torch._inductor.exc.InductorError: TnpuError: tnpu pipeline failed (exit 1)
```

After (`TnpuError` now reads `stage.log`):

```
torch._inductor.exc.InductorError: TnpuError: tnpu pipeline failed (exit 1)
  triton.compiler.errors.CompilationError: at 8:11:
  NameError('tl_math is not defined')
```

That single change resolved six failures into one bug:

**`libdevice` and `tl_math` were collateral damage.** `strip_for_tnpu` drops
`from torch...`, and Inductor imports both names from
`torch._inductor.runtime.triton_helpers` — but they are re-exports of *triton's
own* symbols, not torch code. Six kernels died as a bare `NameError` inside
stage 1.

- `tl_math` is now rebound from `triton.language`. `test_pointwise` gets
  through fourteen ops and as far as the trace producer instead of failing on
  the first.
- `libdevice` cannot be rebound (its members are `@core.extern` with no
  triton_shared implementation, so a call returns `None`) and is now named
  explicitly, the way `triton_helpers` is.

Net effect: `tnpu_stage` 8 → 2, `spec_incomplete` 7 → 13. The same 53 tests
fail; six of them now say something true.

**Separately:** the local TOGSim build was from 07-20 and predated
`trace_shape.txt` support, so `togsim_kernel` was called with `shape_args =
nullptr` and every Triton-route test died with SIGSEGV in `trace_to_tilegraph`.
A rebuild fixed it — not a code problem, and CI builds from source so it was
never affected. Worth knowing if anyone else has a stale `TOGSim/build`.

---

## 7. Next, ranked by tests unblocked per fix

Counts are measured, not estimated — though a test unblocked at one stage may
simply fail at the next.

| # | Fix | Unblocks | Owner |
|---|---|---|---|
| 1 | Implement `TritonNPUWrapperCodegen.estimate_peak` | 6 | triton_backend |
| 2 | Vendor a torch-free `triton_helpers` into the tnpu venv | 7 | triton_backend |
| 3 | Lower `libdevice` intrinsics (`exp`, `tanh`, `rsqrt`, `isnan`) to VPU ops | 5 | tnpu or triton_backend |
| 4 | Multi-axis block policy in `fixed_config_for` | 4 | triton_backend |
| 5 | Hand `ttg.barrier` + `linalg.index` rank error upstream | 2 | tnpu |
| 6 | Surface spike's stderr, then diagnose `cat` / `constant_pad_nd` | 2 | triton_backend, then investigate |

**1 is the cheapest by a wide margin** — one method, six tests, and it opens the
entire attention/transformer family.

**2 and 3 unblock softmax together.** Section 2 shows softmax needs both; either
alone leaves it failing on the other.

**3 needs a decision before work starts:** lower in a tnpu pass, or substitute a
triton-level polyfill in `strip_for_tnpu`. The former is correct; the latter is
cheap and would unblock measurement sooner.

**6 has a prerequisite.** These are the only wrong-answer failures in the suite
and the most likely to be a real lowering bug, but they cannot be diagnosed
until spike's stderr survives the subprocess — the same fix already applied to
`TnpuError` in section 6.

---

## 8. Caveats

- The 17 `missing_dep` failures are local-venv artifacts. In the CI image those
  tests run for real and the buckets will shift — probably toward `wrapper_gap`
  and `triton_helpers`, since most are transformer and CNN models.
- Unblocking a bucket moves its tests to the *next* failure, not necessarily to
  passing.
- These numbers were taken with the section-6 fixes already applied, so they are
  not comparable to a run from before them.
