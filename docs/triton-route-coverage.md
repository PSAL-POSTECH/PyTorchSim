# Triton codegen route — test suite coverage

First measurement of PyTorchSim's existing test suite running through the Triton
codegen route (Inductor's Triton backend + the triton-npu lowering passes),
instead of the MLIR route.

| | |
|---|---|
| Date | 2026-08-03 |
| Branch | `feature/triton-codegen` @ `d356e35` |
| tnpu pin | `5d84caf` |
| torch | 2.10.0, triton 3.6.0 |
| Tests | 69 (everything under `tests/`) |
| Runtime | 5 min at `-j 10` (~50 min serial) |

Reproduce:

```bash
python scripts/ci/triton_route_sweep.py --all -j 10 \
    --markdown coverage.md --artifacts failures
```

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

**11/69, not 16/69.** Five tests pass while emitting no Triton kernel
whatsoever: `test_matmul`, `test_bmm`, `test_topk`, `test_moe_cpu`,
`test_mlir_bindings`. Inductor sends `mm`/`bmm` to an extern call rather than
generating a kernel, so those tests never exercise the thing under test. The
sweep records that separately (`exercised` in the JSON) and keeps them out of
the gate — otherwise the number would overstate coverage by 45%.

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

Four of the eleven are fusion tests. That is the encouraging part: Inductor's
fusion is the half of this migration we get for free, and it is already
producing kernels tnpu accepts.

---

## 2. Where kernels stop

Each test is placed at the furthest stage any of its kernels produced an
artifact for. The stage a kernel *fails to reach* is the one that owns the
failure.

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
side of the seam, not upstream's.

---

## 3. Failures by cause

### `spec_incomplete` — 13 · owner: `triton_backend/kernel_spec.py`

Three distinct sub-causes:

**libdevice intrinsics (5).** `@core.extern` members with no triton_shared
implementation; a call returns `None`.

| Test | Symbol |
|---|---|
| `ops/elementwise/test_exponent.py` | `libdevice.exp` |
| `ops/elementwise/test_pointwise.py` | `libdevice.isnan` |
| `ops/elementwise/test_transcendental.py` | `libdevice.tanh` |
| `ops/reduce/test_layernorm.py` | `libdevice.rsqrt` |
| `ops/view/test_floormod_axis_split.py` | `libdevice.rsqrt` |

**Multi-axis grid (4).** `fixed_config_for` pins only the outermost axis, so
`YBLOCK` is `None` and the grid cannot be computed. This is the known
block-size policy gap.

| Test | Detail |
|---|---|
| `ops/view/test_transpose2D.py` | axis `y`: ynumel=156, YBLOCK=None |
| `ops/view/test_transpose3D.py` | axis `y`: ynumel=2728, YBLOCK=None |
| `ops/fusion/test_conv_fusion.py` | axis `y`: ynumel=192, YBLOCK=None |
| `ops/conv/test_conv_view_input.py` | axis `y`: ynumel=512, YBLOCK=None |

**Reduction blocks unset (2)** — `R0_BLOCK` is left unset on purpose:
`ops/fusion/test_bmm_reduction.py`, `ops/fusion/test_matmul_reduction.py`.

**Genuine metadata hole (1)** — `ops/misc/test_widen_dtype.py`: no dtype/numel
for `out_ptr0`, `collect_meta` could not resolve it from `V.graph`.

### `triton_helpers` — 7 · owner: `triton_backend`

The module lives in torch; the tnpu venv deliberately has none.

| Test | Helper |
|---|---|
| `ops/reduce/test_softmax.py` | `max2` |
| `ops/sort/test_sort.py` | `sort_with_index` |
| `ops/elementwise/test_activation.py` | `maximum` |
| `ops/conv/test_cnn.py` | `maximum` |
| `ops/fusion/test_matmul_activation.py` | `maximum` |
| `ops/sparsity/test_sparsity.py` | `maximum` |
| `models/test_mlp.py` | `maximum` |

Four of seven want only `maximum`. Fix is a small vendored file, not a pass
change.

### `wrapper_gap` — 6 · owner: `triton_backend`

Every one: `'TritonNPUWrapperCodegen' object has no attribute 'estimate_peak'`.

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

The only failures that are genuinely a pass rejecting IR.

| Test | Stage | Diagnostic |
|---|---|---|
| `ops/conv/test_pool.py` | 1 | ``Dialect `ttg' not found for custom op 'ttg.barrier'`` |
| `ops/reduce/test_reduce.py` | 2 | `'linalg.index' op expected dim (2) to be lower than the number of loops (2) of the enclosing LinalgOp` |

Both artifacts carry the MLIR diagnostic and the offending `.mlir`, so they can
go upstream as-is.

### `togsim` / `other` — 5

| Test | Stage | Detail |
|---|---|---|
| `ops/view/test_cat.py` | 4 | `[Spike] triton_npu_fused_cat_0 failed` |
| `ops/misc/test_masked_nondividing.py` | 4 | `[Spike] triton_npu_fused_constant_pad_nd_0 failed` |
| `ops/misc/test_indirect_access.py` | 5 | TOGSim returned `inf` cycles for `index_put` |
| `system/test_hetro.py` | — | `KeyError: 'vpu_num_lanes'` (hetero config lacks the key) |
| `ops/sparsity/test_sparse_core.py` | — | `TypeError: '>' between Tensor and torch.device` (test-side bug) |

The two Spike failures are the most interesting in the sweep: the only cases
that compile all the way to a working RISC-V binary and then produce the wrong
thing. Everything else fails before it can be wrong.

### `missing_dep` — 17 · not a route problem

`transformers` (5), `torchvision` (4), `matplotlib` (4), `pytest` (2),
`diffusers`, `requests`, `sklearn`. Local venv only — these run for real in the
CI image, which is why the sweep belongs in CI.

---

## 4. Infrastructure

### The runner

`scripts/ci/triton_route_sweep.py`. `TORCHSIM_TRITON_CODEGEN` is read once at
device registration (`PyTorchSimDevice/torch_openreg/__init__.py:30`), so **no
test file needed to change** — all 69 were already tests of this route. Only a
runner was missing.

Three outputs:

1. **Gate** — `scripts/ci/triton_route_passing.txt` lists what passes today.
   CI fails if any regresses. Coverage grows by regenerating the file
   (`--update-allowlist`), so it cannot silently shrink.
2. **Report** — bucketed by owning layer and by pipeline stage.
3. **Artifacts** — one directory per failing test.

### Per-failure artifacts

```
failures/tests_ops_reduce_test_softmax/
  kernel.py        the Inductor Triton kernel, unmodified
  error.txt        bucket, stage, last 60 lines
  01-ttir.mlir     whatever stage IR it reached
  stage.log
```

For `test_softmax` the kernel names its own blocker:

```python
def triton_npu_fused__softmax_0(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    ...
    tmp0  = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp4  = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)   # <- lives in torch
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp11, xmask)
```

### Parallelism

Tests are independent subprocesses with their own dump dir, Inductor cache
(`TORCHINDUCTOR_CACHE_DIR` follows `TORCHSIM_DUMP_PATH`) and TOGSim FIFO (keyed
by pid), so `-j` needs no coordination. Threads, not processes: `run_one` only
waits on a subprocess. **69 tests: ~50 min → 5 min at `-j 10`.**

### CI

`.github/workflows/triton_npu.yml`, job `triton-route-suite`:

- **Allowlisted tests** — gates.
- **Full sweep** — `continue-on-error`, writes `coverage.md` to the step
  summary and uploads `triton-route-coverage` (results.json + failures/).

Jobs run on the PSAL Slurm runner farm (`PSAL-POSTECH/slurm-ghr`): `runs-on`
carries the `slurm` label, image builds and the sweep on `big` (16c/64G/2h),
the rest on small. Do not add `docker/setup-buildx-action` — the runner
registers its own builder.

---

## 5. Three diagnostics fixed while measuring

The reporting infrastructure could not be built until these were fixed, because
each one was destroying the evidence.

**`kernel.py` was written after the check that rejects it.**
`write_spec_file` raises for exactly the kernels worth keeping
(`triton_helpers`, `SpecIncomplete`) and ran *before* the source was saved — so
the interesting sources were the ones being thrown away. Reordered; the dump
now exists for all 16 rejected kernels.

**tnpu reported "exit 1" and nothing else.** `run.py` prints a stage table to
stdout and the real diagnostic only to `stage.log`. `TnpuError` now reads that
file and carries the failing line. That single change resolved six failures
into one bug:

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
`trace_shape.txt` support, so every Triton-route test died with SIGSEGV in
`trace_to_tilegraph`. A rebuild fixed it — not a code problem, and CI builds
from source so it was never affected. Worth knowing if anyone else has a stale
`TOGSim/build`.

---

## 6. Next, ranked by tests unblocked per fix

Counts are measured, not estimated — though a test unblocked at one stage may
simply fail at the next.

| # | Fix | Unblocks | Owner |
|---|---|---|---|
| 1 | Implement `TritonNPUWrapperCodegen.estimate_peak` | 6 | triton_backend |
| 2 | Vendor a torch-free `triton_helpers` into the tnpu venv | 7 | triton_backend |
| 3 | Lower `libdevice` intrinsics (`exp`, `tanh`, `rsqrt`, `isnan`) to VPU ops | 5 | tnpu or triton_backend |
| 4 | Multi-axis block policy in `fixed_config_for` | 4 | triton_backend |
| 5 | Hand `ttg.barrier` + `linalg.index` rank error upstream | 2 | tnpu |
| 6 | Investigate the two Spike failures (`cat`, `constant_pad_nd`) | 2 | investigate |

**1 is the cheapest by a wide margin** — one method, six tests, and it opens the
entire attention/transformer family.

**3 needs a decision before work starts:** lower in a tnpu pass, or substitute a
triton-level polyfill in `strip_for_tnpu`. The former is correct; the latter is
cheap and would unblock measurement sooner.

**6 is the most likely to be a real bug.** Everything else is a missing feature;
these two compile to a working binary and produce the wrong answer.

---

## 7. Caveats

- The 17 `missing_dep` failures are local-venv artifacts. In the CI image those
  tests run for real and the buckets will shift — probably toward `wrapper_gap`
  and `triton_helpers`, since most are transformer and CNN models.
- Unblocking a bucket moves its tests to the *next* failure, not necessarily to
  passing.
- These numbers were taken with the section-5 fixes already applied, so they are
  not comparable to a run from before them.
