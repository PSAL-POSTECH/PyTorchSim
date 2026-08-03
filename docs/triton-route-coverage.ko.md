# Triton codegen route 커버리지 측정 보고서

기존 PyTorchSim 테스트 스위트를 MLIR 경로가 아니라 **Triton 경로**(Inductor의
Triton 백엔드 + triton-npu lowering pass)로 돌린 첫 측정 결과입니다.

| | |
|---|---|
| 측정일 | 2026-08-03 |
| 브랜치 | `feature/triton-codegen` @ `6e3bd7e` |
| tnpu 핀 | `5d84caf` |
| 환경 | torch 2.10.0, triton 3.6.0 |
| 대상 | 69개 (`tests/` 전체) |
| 소요 | `-j 10` 기준 5분 (직렬 약 50분) |

재현:

```bash
python scripts/ci/triton_route_sweep.py --all -j 10 \
    --markdown coverage.md --artifacts failures
```

아래 모든 주장은 `failures/` 아래 실제 파일로 뒷받침됩니다. 확인할 수 있도록
경로를 함께 적었습니다.

---

## 1. 결론부터

```
69개 테스트
├── 11  경로를 타고 통과        ← 이것이 커버리지 수치
├──  5  통과하지만 경로 미사용    ← 커널을 아예 안 만듦
└── 53  실패
    ├── 17  로컬 venv 패키지 없음 (CI 이미지에는 있음)
    └── 36  실제 블로커
```

**16/69가 아니라 11/69입니다.** 다섯 개(`test_matmul`, `test_bmm`, `test_topk`,
`test_moe_cpu`, `test_mlir_bindings`)는 통과하지만 Triton 커널을 하나도 만들지
않습니다. Inductor가 `mm`/`bmm`을 커널 생성 대신 extern call로 내리기 때문에,
이 테스트들은 정작 검증 대상을 한 번도 거치지 않습니다. 스윕은 이를 별도로
기록(JSON의 `exercised`)하고 gate에서 제외합니다. 포함시키면 커버리지가 45%
부풀려집니다.

### 통과한 11개

| 테스트 | 시간 |
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

11개 중 **4개가 fusion 테스트**입니다. 이 마이그레이션에서 Inductor의 fusion은
공짜로 얻는 절반인데, 이미 tnpu가 받아들이는 커널을 만들어내고 있다는 뜻입니다.

---

## 2. 실패 하나가 어떻게 진단되는가 — softmax 전 과정

`tests/ops/reduce/test_softmax.py`. 스윕이 남기는 것:

```
failures/tests_ops_reduce_test_softmax/
  kernel.py        Inductor 가 만든 Triton 커널 원본
  error.txt        버킷, 단계, 로그 마지막 60줄
```

`error.txt` 첫 줄이 담당자를 지정합니다:

```
test:   tests/ops/reduce/test_softmax.py
bucket: triton_helpers
stage:  0 kernel generated, not accepted
```

그리고 `kernel.py`가 이유 전부를 보여줍니다:

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
    tmp4  = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)   # <-- 블로커 1
    tmp5  = tmp0 - tmp4
    tmp6  = libdevice.exp(tmp5)                                    # <-- 블로커 2
    tmp7  = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9  = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = (tmp6 / tmp10)
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp11, xmask)
```

막힌 곳은 두 군데뿐이고, 아무것도 실행하지 않고 눈으로 확인됩니다.

- `triton_helpers.max2` — `torch._inductor.runtime`에 있는 모듈인데, tnpu venv는
  의도적으로 torch가 없습니다.
- `libdevice.exp` — `@core.extern` 인트린식으로 triton_shared 백엔드에 구현이
  없습니다.

**주목할 점은 나머지가 전부 멀쩡하다는 것입니다.** 마스크가 붙은 `tl.load`,
`tl.where`, `tl.sum`, `tl.store`, 2차원 `[XBLOCK, R0_BLOCK]` 브로드캐스트가 모두
문제없이 통과합니다. softmax는 구조적으로 막힌 게 아니라 **함수 호출 두 개**에
막혔습니다.

이것이 버그 리포트 전체이고, 재실행 없이 만들어집니다.

---

## 3. 커널이 어디서 멈추는가

각 테스트를, 그 테스트의 커널이 산출물을 남긴 가장 깊은 단계에 배치했습니다.
커널이 **도달하지 못한** 단계가 그 실패의 담당자입니다.

| 단계 | 수 | |
|---|---|---|
| — 커널 생성 전 | 26 | codegen 이전에 torch/Inductor에서 사망 |
| 0 생성 후 거절 | 16 | `kernel_spec`이 기술을 거부 |
| 1 triton → ttir | 1 | |
| 2 ttir → tts/linalg | 1 | triton-shared |
| 4 tnpu lower (DMA, lane, spad) | 6 | |
| 5 trace producer | 3 | |

**lowering pass는 아직 병목이 아닙니다.** 53개 실패 중 tnpu pass가 IR을 거절한
것은 **단 2건**입니다. 나머지 34개 실제 블로커는 그보다 앞 — tnpu에 넘겨주는
우리 쪽 포트, 또는 torch 자체 — 에서 멈춥니다. 다음 작업은 대부분 seam의 우리
쪽에 있습니다.

---

## 4. 원인별 상세 (근거 포함)

### `spec_incomplete` — 13개 · 담당: `triton_backend/kernel_spec.py`

**libdevice 인트린식 (5개).** `@core.extern` 멤버로 triton_shared 구현이 없어
호출하면 `None`이 반환됩니다.

| 테스트 | 심볼 |
|---|---|
| `ops/elementwise/test_exponent.py` | `libdevice.exp` |
| `ops/elementwise/test_pointwise.py` | `libdevice.isnan` |
| `ops/elementwise/test_transcendental.py` | `libdevice.tanh` |
| `ops/reduce/test_layernorm.py` | `libdevice.rsqrt` |
| `ops/view/test_floormod_axis_split.py` | `libdevice.rsqrt` |

이번 세션에서 진단을 고치기 전에는 이것들이 tnpu stage-1 워커 안에서
`NameError('libdevice is not defined')` 로만 죽었습니다. 그래서 서로 다른 여섯
개의 lowering 버그처럼 보였습니다. 지금은 이렇게 말합니다:

```
SpecIncomplete: kernel calls libdevice.{exp}: those are extern math intrinsics
with no implementation on the triton_shared backend. They need lowering to a
VPU op (or a scalar fallback) before this kernel can compile.
```

**다축 grid (4개).** `fixed_config_for`가 가장 바깥 축만 고정하기 때문에
`YBLOCK`이 `None`이 되고 grid를 계산할 수 없습니다. 알려진 block-size 정책
공백입니다.

| 테스트 | 진단 |
|---|---|
| `ops/view/test_transpose2D.py` | axis `y`: ynumel=156, YBLOCK=None |
| `ops/view/test_transpose3D.py` | axis `y`: ynumel=2728, YBLOCK=None |
| `ops/fusion/test_conv_fusion.py` | axis `y`: ynumel=192, YBLOCK=None |
| `ops/conv/test_conv_view_input.py` | axis `y`: ynumel=512, YBLOCK=None |

**reduction block 미설정 (2개)** — `R0_BLOCK`을 의도적으로 비워둡니다:
`ops/fusion/test_bmm_reduction.py`, `ops/fusion/test_matmul_reduction.py`.

**진짜 메타데이터 구멍 (1개)** — `ops/misc/test_widen_dtype.py`: `out_ptr0`의
dtype/numel을 `collect_meta`가 `V.graph`에서 해결하지 못했습니다.

### `triton_helpers` — 7개 · 담당: `triton_backend`

| 테스트 | 헬퍼 |
|---|---|
| `ops/reduce/test_softmax.py` | `max2` |
| `ops/sort/test_sort.py` | `sort_with_index` |
| `ops/elementwise/test_activation.py` | `maximum` |
| `ops/conv/test_cnn.py` | `maximum` |
| `ops/fusion/test_matmul_activation.py` | `maximum` |
| `ops/sparsity/test_sparsity.py` | `maximum` |
| `models/test_mlp.py` | `maximum` |

7개 중 4개가 `maximum` 하나만 필요합니다. pass 수정이 아니라 작은 파일 하나를
vendoring 하는 작업입니다.

### `wrapper_gap` — 6개 · 담당: `triton_backend`

전부 동일합니다:

```
AttributeError: 'TritonNPUWrapperCodegen' object has no attribute 'estimate_peak'
```

`ops/attention/test_gqa.py`, `test_gqa_decode.py`,
`ops/fusion/test_attention_fusion.py`, `test_transformer_fusion.py`,
`models/Mixtral8x7B/test_attention.py`, `models/test_transformer.py`

스위트의 **attention·transformer 테스트 전부**가 미구현 메서드 하나에 막혀
있습니다.

### `device_op` — 3개 · 담당: `PyTorchSimDevice`

이 경로 이전부터 있던 문제입니다. MLIR 경로는 이들을 dispatcher 도달 전에
가로챕니다.

| 테스트 | 오류 |
|---|---|
| `ops/conv/test_conv2d.py` | `convolution_overrideable not implemented` |
| `ops/conv/test_group_conv.py` | `convolution_overrideable not implemented` |
| `ops/attention/test_sdpa.py` | `_scaled_dot_product_fused_attention_overrideable not implemented` |

### `tnpu_stage` — 2개 · 담당: triton-npu lowering pass

진짜로 pass가 IR을 거절한 유일한 두 건입니다. 산출물에 Python부터 거절된 op까지
사슬 전체가 남아 있습니다.

**`ops/conv/test_pool.py`** — stage 1.

`kernel.py` 끝에, Inductor가 reduction 뒤에 붙이는 무해해 보이는 한 줄:

```python
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = 49.0
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()          # <-- 이것
```

`01-ttir.mlir:46`에서 이렇게 됩니다:

```mlir
%tmp4_17 = tt.expand_dims %tmp4 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
%tmp6_18 = arith.divf %tmp4_17, %tmp6 : tensor<128x1xf32>
ttg.barrier all                 # <-- GPU 다이얼렉트 op
%0 = tt.splat %in_out_ptr0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
```

그리고 `triton-shared-opt`가 파싱하지 못합니다:

```
01-ttir.mlir:46:5: error: Dialect `ttg' not found for custom op 'ttg.barrier'
```

`ttg`는 GPU 다이얼렉트입니다. reduction 뒤 pointwise 커널에서 이 타깃에는 의미
없는 배리어인데, IR에 남아 있어서 파서가 멈춥니다.

**`ops/reduce/test_reduce.py`** — stage 2. 평범한 `(a + b).sum(dim=1)`:

```python
tmp0 = tl.load(in_ptr0 + (r0_1 + 47*x0), r0_mask & xmask, other=0.0)
tmp1 = tl.load(in_ptr1 + (r0_1 + 47*x0), r0_mask & xmask, other=0.0)
tmp2 = tmp0 + tmp1
tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
```

`01-ttir.mlir`까지는 살아남고, linalg 변환에서 실패합니다:

```
error: "-":101:11: 'linalg.index' op expected dim (2) to be lower than
       the number of loops (2) of the enclosing LinalgOp
```

두 산출물 모두 진단과 문제의 `.mlir`을 함께 갖고 있어, 그대로 업스트림에 넘길
수 있습니다.

### `togsim` / `기타` — 5개

| 테스트 | 단계 | 내용 |
|---|---|---|
| `ops/view/test_cat.py` | 4 | `[Spike] triton_npu_fused_cat_0 failed`, exit 255 |
| `ops/misc/test_masked_nondividing.py` | 4 | `[Spike] triton_npu_fused_constant_pad_nd_0 failed` |
| `ops/misc/test_indirect_access.py` | 5 | TOGSim이 `index_put`에 `inf` 사이클 반환 |
| `system/test_hetro.py` | — | `KeyError: 'vpu_num_lanes'` (hetero config에 키 없음) |
| `ops/sparsity/test_sparse_core.py` | — | `TypeError: '>' between Tensor and torch.device` (테스트 쪽 버그) |

Spike 실패 두 건이 이번 스윕에서 가장 흥미롭습니다. **동작하는 RISC-V 바이너리
까지 컴파일된 뒤 런타임에 실패하는 유일한 사례**입니다. `test_cat`의
`04-custom.mlir`을 보면 lowering은 제 역할을 했습니다:

```mlir
"togsim.transfer"(%reinterpret_cast_5, %c0, %2, %c0, %7, %c0, %c2, %c1, %6)
    {vlane_split_axis = 0, ...}
"togsim.transfer"(%reinterpret_cast_4, %c0, %1, %c0, %7, %c0, %c2, %c1, %6)
    {vlane_split_axis = 0, ...}
"togsim.transfer"(%reinterpret_cast,   %9, %0, %c0, %7, %c0, %c3, %c1, %c0, %13)
    {vlane_split_axis = 0, ...}
```

입력 DMA 2개와 출력 DMA 1개, 축 0으로 lane split — `cat`이 내야 할 정확한
모양입니다. **lowering 구조가 아니라 실행이 잘못됩니다.**

**이 두 건에 대한 단서.** `tnpu.spike`는 `StageError: command failed with exit
code 255`만 보고하고 spike 자신의 stderr는 살아남지 못합니다. 기록된 spike
명령을 같은 workdir에서 손으로 돌리면 exit 0이 나오는데, `write_inputs`가 매
launch마다 `runtime/*.raw`를 새로 쓰기 때문에 손으로 돌린 실행은 **옛 입력을
재생**하기 때문입니다. 따라서 실패하는 입력은 현재 파이프라인 밖에서 재현할 수
없습니다. `TnpuError`에 적용한 것과 같은 방식으로 spike의 stderr를 노출시키는
것이 이 두 건 진단의 선결 조건이고, 아직 하지 않았습니다.

### `missing_dep` — 17개 · 경로 문제 아님

`transformers`(5), `torchvision`(4), `matplotlib`(4), `pytest`(2), `diffusers`,
`requests`, `sklearn`. 로컬 venv에만 없는 것으로, CI 이미지에서는 실제로
돌아갑니다. 스윕이 CI에 있어야 하는 이유이기도 합니다.

---

## 5. 인프라

### 러너

`scripts/ci/triton_route_sweep.py`. `TORCHSIM_TRITON_CODEGEN`은 device 등록
시점에 한 번 읽히므로(`PyTorchSimDevice/torch_openreg/__init__.py:30`),
**테스트 파일은 하나도 고칠 필요가 없었습니다.** 69개 전부가 이미 이 경로의
테스트였고, 없던 것은 러너뿐이었습니다.

세 가지 산출물:

1. **Gate** — `scripts/ci/triton_route_passing.txt`에 현재 통과 목록. 하나라도
   깨지면 CI 실패. 커버리지는 이 파일을 재생성해서만 늘어나므로
   (`--update-allowlist`) 조용히 줄어들 수 없습니다.
2. **Report** — 담당 레이어와 파이프라인 단계로 분류.
3. **Artifacts** — 실패 테스트당 디렉토리 하나 (2절 참고).

더 깊이 간 실패는 더 많이 남깁니다. `tests_ops_conv_test_cnn/`에는
`01-ttir.mlir 02-ttshared.mlir 03-adapted.mlir 04-custom.mlir kernel.py
stage.log error.txt`가 있습니다 — 멈춘 지점까지의 lowering 사슬 전체입니다.

### 병렬화

테스트는 각자 독립 서브프로세스이고 자기 덤프 디렉토리, Inductor 캐시
(`TORCHINDUCTOR_CACHE_DIR`가 `TORCHSIM_DUMP_PATH`를 따라감), TOGSim FIFO(pid
기준)를 갖습니다. 따라서 `-j`에 조율이 필요 없습니다. 프로세스가 아니라
스레드입니다 — `run_one`은 서브프로세스를 기다리기만 합니다.
**69개 기준 약 50분 → `-j 10`에서 5분.**

### CI

`.github/workflows/triton_npu.yml`의 `triton-route-suite` 잡:

- **Allowlisted tests** — gate 역할.
- **Full sweep** — `continue-on-error`. `coverage.md`를 step summary에 쓰고
  `triton-route-coverage`(results.json + failures/)를 업로드.

잡은 PSAL Slurm 러너 팜(`PSAL-POSTECH/slurm-ghr`)에서 돕니다. `runs-on`에
`slurm` 라벨이 있어야 하고, 이미지 빌드와 스윕은 `big`(16c/64G/2h), 나머지는
small 버킷입니다. `docker/setup-buildx-action`은 추가하면 안 됩니다 — 러너가
자체 빌더를 등록해 둡니다.

---

## 6. 측정하면서 고친 진단 3가지

이 셋을 고치기 전에는 보고 인프라를 만들 수 없었습니다. 각각이 증거를 파괴하고
있었기 때문입니다.

**`kernel.py`가 그것을 거절하는 검사 뒤에 저장되고 있었습니다.**
`write_spec_file`은 정확히 보존할 가치가 있는 커널(`triton_helpers`,
`SpecIncomplete`)에서 예외를 던지는데, 소스 저장보다 **먼저** 실행됐습니다.
결국 흥미로운 소스일수록 버려지고 있었습니다. 순서를 뒤집었고, 이제 거절된 16개
커널 전부의 덤프가 남습니다 — 2절의 softmax 예시가 그중 하나입니다.

**tnpu가 "exit 1"만 보고했습니다.** `run.py`는 stage 표를 stdout에 찍고 진짜
진단은 `stage.log`에만 씁니다. 이전:

```
torch._inductor.exc.InductorError: TnpuError: tnpu pipeline failed (exit 1)
```

이후 (`TnpuError`가 `stage.log`를 읽음):

```
torch._inductor.exc.InductorError: TnpuError: tnpu pipeline failed (exit 1)
  triton.compiler.errors.CompilationError: at 8:11:
  NameError('tl_math is not defined')
```

이 한 가지 변경으로 여섯 개 실패가 **하나의 버그**로 정리됐습니다:

**`libdevice`와 `tl_math`가 유탄을 맞고 있었습니다.** `strip_for_tnpu`가
`from torch...`를 지우는데, Inductor는 이 두 이름을
`torch._inductor.runtime.triton_helpers`에서 import합니다. 그런데 이들은 torch
코드가 아니라 **triton 자체 심볼의 재수출**입니다. 커널 여섯 개가 stage 1 안에서
맨 `NameError`로 죽고 있었습니다.

- `tl_math`는 `triton.language`에서 다시 바인딩했습니다. `test_pointwise`가 첫
  op에서 죽던 것이 op 14개를 지나 trace producer까지 갑니다.
- `libdevice`는 재바인딩이 불가능합니다(멤버가 `@core.extern`이고 triton_shared
  구현이 없어 호출하면 `None`). `triton_helpers`와 같은 방식으로 명시적으로
  이름을 밝히도록 했습니다.

순효과: `tnpu_stage` 8 → 2, `spec_incomplete` 7 → 13. **같은 53개가 실패하지만
그중 6개가 이제 참을 말합니다.**

**별건으로**, 로컬 TOGSim 빌드가 07-20자여서 `trace_shape.txt` 지원 이전이었고,
그래서 `togsim_kernel`이 `shape_args = nullptr`로 호출되어 모든 Triton 경로
테스트가 `trace_to_tilegraph`에서 SIGSEGV로 죽었습니다. 재빌드로 해결됐습니다 —
코드 문제가 아니고, CI는 소스에서 빌드하므로 영향이 없었습니다. 다른 사람이 낡은
`TOGSim/build`를 갖고 있다면 알아둘 만합니다.

---

## 7. 다음 작업 — 해제되는 테스트 수 기준

수치는 측정값이지 추정이 아닙니다. 다만 한 단계에서 풀린 테스트가 다음 단계에서
그냥 다시 실패할 수는 있습니다.

| # | 작업 | 해제 | 담당 |
|---|---|---|---|
| 1 | `TritonNPUWrapperCodegen.estimate_peak` 구현 | 6 | triton_backend |
| 2 | torch 없는 `triton_helpers`를 tnpu venv에 vendoring | 7 | triton_backend |
| 3 | `libdevice` 인트린식(`exp`, `tanh`, `rsqrt`, `isnan`)을 VPU op로 lowering | 5 | tnpu 또는 triton_backend |
| 4 | `fixed_config_for`에 다축 block 정책 | 4 | triton_backend |
| 5 | `ttg.barrier` + `linalg.index` rank 오류를 업스트림에 전달 | 2 | tnpu |
| 6 | spike stderr 노출 후 `cat` / `constant_pad_nd` 진단 | 2 | triton_backend → 조사 |

**1번이 압도적으로 쌉니다** — 메서드 하나로 6개, attention/transformer 계열
전체가 열립니다.

**2번과 3번은 함께 해야 softmax가 열립니다.** 2절에서 봤듯 softmax는 둘 다
필요하고, 하나만 고치면 다른 하나에서 계속 실패합니다.

**3번은 착수 전 결정이 필요합니다**: tnpu pass에서 lowering할 것인가,
`strip_for_tnpu`에서 triton 레벨 polyfill로 대체할 것인가. 전자가 옳고 후자가
싸며 측정을 더 빨리 풀어줍니다.

**6번은 선결 조건이 있습니다.** 스위트에서 유일하게 "답이 틀리는" 실패이고 진짜
lowering 버그일 가능성이 가장 높지만, spike의 stderr가 서브프로세스를 넘어오기
전에는 진단할 수 없습니다 — 6절에서 `TnpuError`에 이미 적용한 것과 같은
수정입니다.

---

## 8. 이 측정이 말해주지 않는 것

- `missing_dep` 17개는 로컬 venv 사정입니다. CI 이미지에서는 실제로 돌기 때문에
  버킷이 이동할 것입니다 — 대부분 transformer·CNN 모델이므로 아마 `wrapper_gap`
  과 `triton_helpers` 쪽으로 갑니다.
- 어떤 버킷을 풀면 그 테스트들은 **다음 실패**로 이동하는 것이지, 반드시 통과로
  가는 것이 아닙니다.
- 이 수치는 6절의 수정이 이미 적용된 상태에서 측정한 것이라, 그 이전 실행과
  직접 비교할 수 없습니다.
