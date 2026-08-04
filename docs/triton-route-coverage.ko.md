# Triton codegen route 커버리지

기존 테스트 스위트를 MLIR 경로가 아니라 **Triton 경로**(Inductor의 Triton 백엔드
+ triton-npu lowering pass)로 돌린 결과입니다.

| | |
|---|---|
| 측정일 | 2026-08-04 |
| 브랜치 | `feature/triton-helpers` @ `7899a17` |
| tnpu 핀 | `d46995f` |
| 환경 | torch 2.10.0, triton 3.6.0 |
| 대상 | 69개 (`tests/` 전체) |

재현:

```bash
python scripts/ci/triton_route_sweep.py --all -j 8 \
    --markdown coverage.md --artifacts failures
```

아래 모든 주장은 `failures/` 아래 파일로 뒷받침됩니다.

---

## 1. 결론

```
69개
├── 13  경로를 타고 통과        ← 커버리지 수치
├──  2  통과하지만 경로 미사용
└── 54  실패
    ├── 17  로컬 패키지 부재 (CI 이미지에는 있음)
    └── 37  실제 블로커
```

**37개 중 26개가 tnpu 버그 두 건입니다.** 서른일곱 개의 문제가 아니라 두 개이고,
재현 커널과 함께 `PSAL-POSTECH/triton-npu#2`에 보고돼 있습니다.

### 통과한 13개

| 테스트 | 시간 |
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

`test_topk`과 `test_mlir_bindings`는 커널을 안 만들고 통과합니다. 따로 기록하고
gate에서 뺐습니다 — 세면 아무것도 아닌 것을 세는 셈입니다.

---

## 2. 원인별

| 버킷 | 수 | 담당 |
|---|---|---|
| `tnpu_stage` | 30 | triton-npu |
| `missing_dep` | 17 | 테스트 환경 (CI 이미지엔 있음) |
| `wrong_values` | 3 | 수치 — 조사 필요 |
| `spec_incomplete` | 2 | triton_backend |
| `device_op` | 1 | PyTorchSimDevice |
| `other` | 1 | 미분류 |

### `tnpu_stage` 30건 — 그중 26건이 두 버그

**`tl.assume` → `llvm.intr.assume` (16건).** Inductor의 mm 템플릿이
`tl.assume(pid_m >= 0)`를 힌트로 냅니다. 이게 `triton-shared-opt`가 로드하지 않는
다이얼렉트의 op가 됩니다. ttir에서 그 줄만 지우면 같은 stage 2 명령이 성공하고
`linalg.matmul`이 나옵니다.

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

**matmul 결과에 대한 `select_lane_axis` (10건).**

```
lane_axis.Fatal: the demand 'linalg.matmul' has to move for the lane axes to
agree, but it is about a value 'linalg.matmul' PRODUCES -- a relayout makes a
new value, so there is no edge to put one on
```

그 패스 자체 문서의 case 4입니다. 앞의 것과 달리 흘러들어온 힌트가 아니라
**미지원 데이터플로**로 보입니다.

```
models/Mixtral8x7B/test_attention.py   ops/elementwise/test_pointwise.py
ops/attention/test_gqa_decode.py       ops/fusion/test_conv_fusion.py
ops/conv/test_cnn.py                   ops/misc/test_indirect_access.py
ops/conv/test_conv_view_input.py       ops/misc/test_masked_nondividing.py
ops/conv/test_group_conv.py            ops/view/test_cat.py
```

**`linalg.index` rank (3건)** — `ops/conv/test_pool.py`,
`ops/reduce/test_reduce.py`, `system/test_vectorops.py`. 그 PR의 원 리포트입니다.

**`linalg.generic` shape (1건)** — `ops/sort/test_sort.py`.

### `wrong_values` 3건

`ops/reduce/test_softmax.py`, `ops/reduce/test_layernorm.py`,
`ops/view/test_floormod_axis_split.py`. 컴파일되고 실행되는데 **답이 틀립니다.**
가장 작은 범주이면서 가장 위험합니다. 앞의 둘은 reduction입니다.

### 나머지 4건

| 테스트 | |
|---|---|
| `ops/misc/test_widen_dtype.py` | `collect_meta`가 `out_ptr0`의 dtype/numel을 못 구함 |
| `system/test_hetro.py` | stonne config에 `vpu_num_lanes`가 없음 — 모든 블록 크기가 그걸 기준으로 함 |
| `ops/attention/test_sdpa.py` | `_scaled_dot_product_fused_attention_overrideable` 미등록 |
| `models/MoE/test_moe_cpu.py` | 미분류 |

---

## 3. 커널이 어디서 멈추는가

| 단계 | 수 |
|---|---|
| — 커널 생성 전 | 20 |
| 0 생성 후 거절 | 1 |
| 1 triton → ttir | 15 |
| 2 ttir → tts/linalg | 6 |
| 4 tnpu lower | 5 |
| 5 trace producer | 7 |

stage 1의 15건이 `tl.assume` 무리입니다. 커널이 만들어지고 Inductor를 통과한 뒤,
tnpu 첫 단계에서 힌트에 걸립니다.

---

## 4. 첫 측정 이후 무엇이 바뀌었나

이 스윕의 첫 실행은 11/69였습니다. 지금은 13이지만 **둘을 비교하면 안 됩니다.**
움직인 것 대부분이 "무엇이 되는가"가 아니라 "무엇을 셌는가"입니다.

**커버리지 기준이 두 번 엄격해졌습니다.** 커널을 안 내는 테스트는 원래 뺐고, 이제는
커널을 내면서도 **자기 이름의 연산**을 `extern_kernels`에 넘기는 것도 뺍니다. 그렇게
6개가 빠졌습니다 — `test_matmul_scalar`은 `mul`만 만들고 `extern_kernels.mm`을
불렀습니다.

**그리고 경로가 그 비상구를 안 쓰게 됐습니다.** `inductor_templates`가 npu를
`GPU_TYPES`에 넣어 `use_triton_template`이 고려하게 하고,
`mm`/`bmm`/`addmm`/`baddbmm` heuristic을 등록하고, autotune을 고정 선택으로
대체합니다 — 잴 하드웨어가 없으니까요. 이제 `mm`과 `addmm`이 aten이 아니라 Inductor
템플릿을 탑니다. 16개가 새로 `tl.assume`에 닿는 이유가 이것입니다. **그 전에는
통과한 게 아니라 시뮬레이션을 안 했던 것입니다.**

그 과정에서 고친 것: `libdevice`/`tl_math`가 그 이름을 만들던 torch import와 함께
지워지고 있었고, 다축 grid가 한 번도 매칭되지 않는 키로 만들어지고 순서도 뒤집혀
있었으며, 텐서 stride가 launch로 전달되지 않았고, `device_guard`가 `"pass"`를
반환해 호출부가 `with pass:`를 쓰고 있었습니다.

---

## 5. 다음 순서

1. **`tl.assume`** — 16개. 의미 없는 힌트이고, tnpu를 쓰는 프론트엔드마다 벗기는
   것보다 `normalize_upstream`에서 한 번 버리는 게 맞습니다.
2. **matmul 결과의 `select_lane_axis`** — 10개. 흘러든 op가 아니라 미지원
   데이터플로라 더 무겁습니다.
3. **오답 3건** — 가장 작고 가장 위험합니다. 둘 다 reduction이라 원인이 하나일
   가능성이 높습니다.
4. `linalg.index`(3), `linalg.generic`(1) — 이미 그 PR에 있습니다.

1번과 2번이 실제 블로커 37개 중 26개이고 둘 다 업스트림입니다. 우리 쪽 목록은
네 개뿐입니다.
