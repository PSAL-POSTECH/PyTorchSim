# Triton codegen route — test suite coverage

The existing test suite run through the Triton codegen route (Inductor's Triton
backend + the triton-npu lowering passes) instead of the MLIR route. Korean
version: [`triton-route-coverage.ko.md`](triton-route-coverage.ko.md).

| | |
|---|---|
| Date | 2026-08-04 |
| Branch | `feature/triton-helpers` @ `7899a17` |
| tnpu pin | `d46995f` |
| torch | 2.10.0, triton 3.6.0 |
| Tests | 69 (everything under `tests/`) |

Reproduce:

```bash
python scripts/ci/triton_route_sweep.py --all -j 8 \
    --markdown coverage.md --artifacts failures
```

Every claim below is backed by a file in `failures/`.

---

## 1. Headline

```
69 tests
├── 13  pass THROUGH the route          ← this is the coverage number
├──  2  pass without using the route
└── 54  fail
    ├── 17  missing test deps (local venv only; present in the CI image)
    └── 37  real blockers
```

**26 of the 37 are two tnpu bugs.** Not thirty-seven problems; two, reported with
reproducers at `PSAL-POSTECH/triton-npu#2`.

### What passes

| Test | Time |
|---|---|
| `ops/elementwise/test_activation.py` | 83.7s |
| `ops/elementwise/test_add.py` | 79.7s |
| `ops/elementwise/test_exponent.py` | 10.4s |
| `ops/elementwise/test_transcendental.py` | 33.7s |
| `ops/misc/test_expert_mask.py` | 10.0s |
| `ops/reduce/test_batchnorm.py` | 37.6s |
| `ops/sparsity/test_sparse_core.py` | 15.2s |
| `ops/view/test_transpose2D.py` | 32.3s |
| `ops/view/test_transpose3D.py` | 182.6s |
| `ops/view/test_view3D_2D.py` | 38.6s |
| `system/test_eager.py` | 15.8s |
| `system/test_stonne.py` | 10.0s |
| `system/test_triton_codegen.py` | 10.4s |

`test_topk` and `test_mlir_bindings` pass without emitting a kernel; they are
recorded separately and kept out of the gate, since counting them would be
counting nothing.

---

## 2. Failures by cause

| bucket | count | owner |
|---|---|---|
| `tnpu_stage` | 30 | triton-npu |
| `missing_dep` | 17 | test environment (CI image has them) |
| `wrong_values` | 3 | numerics — investigate |
| `spec_incomplete` | 2 | triton_backend |
| `device_op` | 1 | PyTorchSimDevice |
| `other` | 1 | unclassified |

### `tnpu_stage` — 30, and 26 of them are two bugs

**`tl.assume` → `llvm.intr.assume` (16).** Inductor's mm template emits
`tl.assume(pid_m >= 0)` as a hint; it lowers to an op in a dialect
`triton-shared-opt` does not load. Deleting only those lines from the ttir makes
the same stage-2 command succeed and produce a `linalg.matmul`.

```
models/test_mlp.py                     ops/fusion/test_matmul_activation.py
models/test_transformer.py             ops/fusion/test_matmul_reduction.py
ops/attention/test_gqa.py              ops/fusion/test_matmul_scalar.py
ops/conv/test_conv2d.py                ops/fusion/test_matmul_vector.py
ops/fusion/test_addmm_residual.py      ops/fusion/test_prologue_fusion.py
ops/fusion/test_attention_fusion.py    ops/fusion/test_transformer_fusion.py
ops/fusion/test_bmm_reduction.py       ops/gemm/test_bmm.py
ops/sparsity/test_sparsity.py          ops/gemm/test_matmul.py
```

**`select_lane_axis` on a matmul result (10).**

```
lane_axis.Fatal: the demand 'linalg.matmul' has to move for the lane axes to
agree, but it is about a value 'linalg.matmul' PRODUCES -- a relayout makes a
new value, so there is no edge to put one on
```

Case 4 in that pass's own docstring. Unlike the first, this looks like an
unsupported dataflow rather than a stray hint.

```
models/Mixtral8x7B/test_attention.py   ops/elementwise/test_pointwise.py
ops/attention/test_gqa_decode.py       ops/fusion/test_conv_fusion.py
ops/conv/test_cnn.py                   ops/misc/test_indirect_access.py
ops/conv/test_conv_view_input.py       ops/misc/test_masked_nondividing.py
ops/conv/test_group_conv.py            ops/view/test_cat.py
```

**`linalg.index` rank (3)** — `ops/conv/test_pool.py`, `ops/reduce/test_reduce.py`,
`system/test_vectorops.py`. The original report on that PR.

**`linalg.generic` shape (1)** — `ops/sort/test_sort.py`.

### `wrong_values` — 3

`ops/reduce/test_softmax.py`, `ops/reduce/test_layernorm.py`,
`ops/view/test_floormod_axis_split.py`. These compile, run, and return the wrong
answer, which makes them the most dangerous category here even though it is the
smallest. Both reductions.

### The rest — 4

| test | |
|---|---|
| `ops/misc/test_widen_dtype.py` | `collect_meta` cannot resolve dtype/numel for `out_ptr0` |
| `system/test_hetro.py` | the stonne config has no `vpu_num_lanes`, which every block size is pinned to |
| `ops/attention/test_sdpa.py` | `_scaled_dot_product_fused_attention_overrideable` not registered |
| `models/MoE/test_moe_cpu.py` | unclassified |

---

## 3. Where kernels stop

| Stage | Count |
|---|---|
| — no kernel generated | 20 |
| 0 generated, rejected | 1 |
| 1 triton → ttir | 15 |
| 2 ttir → tts/linalg | 6 |
| 4 tnpu lower | 5 |
| 5 trace producer | 7 |

Fifteen at stage 1 is the `tl.assume` group: the kernel is generated and
survives Inductor, and the first tnpu stage is where the hint bites.

---

## 4. What changed since the first measurement

The first run of this sweep read 11/69. The number is 13 now, but the two are not
comparable — most of the movement is in what got measured, not what works.

**Coverage got stricter twice.** A test that emits no kernel was already
excluded; now a test that emits a kernel and still hands *its own op* to
`extern_kernels` is excluded too. Six tests moved out that way —
`test_matmul_scalar` generated the `mul` and called `extern_kernels.mm`.

**And the route stopped taking that exit.** `inductor_templates` puts npu in
`GPU_TYPES` so `use_triton_template` considers it, registers heuristics for
`mm`/`bmm`/`addmm`/`baddbmm`, and replaces autotuning with a fixed choice —
there is no device to benchmark on. `mm` and `addmm` now go through Inductor's
template instead of aten, which is why sixteen tests newly reach `tl.assume`:
they were not passing before, they were not simulating.

Fixed along the way: `libdevice`/`tl_math` were being stripped with the torch
import that named them; the multi-axis grid was built from a key that never
matched and emitted in the wrong order; tensor strides were not carried into the
launch; `device_guard` returned `"pass"`, which the caller writes as
`with pass:`.

---

## 5. What to do next

1. **`tl.assume`** — 16 tests. A hint with no semantics, dropped in
   `normalize_upstream` rather than in every frontend that feeds tnpu.
2. **`select_lane_axis` on a matmul result** — 10 tests. Heavier: an unsupported
   dataflow, not a stray op.
3. **The three wrong answers** — smallest and most dangerous. Both reductions,
   so likely one cause.
4. `linalg.index` (3), `linalg.generic` (1) — already on that PR.

1 and 2 are 26 of the 37 real blockers and both sit upstream. On our side the
list is four tests long.
