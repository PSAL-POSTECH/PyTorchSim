# TPU Layout Assignment & Padding 메커니즘 조사 보고서

> **목적**: PyTorchSim에서 TPU 워크로드의 메모리 footprint / compute utilization을 정확히 모델링하기 위해, XLA/Mosaic 컴파일 파이프라인에서 (1) 어느 축이 lane/sublane으로 선택되는지, (2) 패딩이 언제·어떻게 일어나는지, (3) 그 패딩이 물리적으로 물질화되는지 마스킹으로 처리되는지를 정리한 핸드오프 문서.
>
> **수신자**: PyTorchSim 모델링/구현 담당 agent
> **작성 기준일**: 2026-06-18
> **신뢰도 표기**: [확정]=공개 문서/소스로 검증됨, [추론]=문서 기반 합리적 추론, [미확인]=공개 자료로 닿지 못함

---

## 0. 한 줄 요약

TPU에서 **lane(128) 축 선택과 패딩은 XLA의 layout assignment pass에서 동시에 결정**되며, 패딩은 두 층위로 나뉜다: **(A) 8×128 레이아웃 정렬 패딩은 주소 정렬상 강제 물질화**(실제 텐서가 HBM에서 커짐), **(B) 그보다 큰 연산 블록 크기의 경계(tail)는 masking/peeling 등으로 처리**(대체로 비물질화). 모델링 시 (A)는 footprint+traffic, (B)는 compute utilization만 반영해야 한다.

---

## 1. 컴파일 파이프라인 순서 [확정]

```
프론트엔드(JAX/PyTorch) 
  → JAXPR/FX 
  → HLO (DotGeneral 등, layout 미확정)
  → [layout assignment]   ← lane/sublane 축 + 패딩 결정
  → [fusion]              ← op들을 커널로 묶음
  → LLO (TPU-specific IR)
  → VLIW bundles
```

- 출처: JAX→VLIW 컴파일러 추적 (patricktoulme.substack.com), OpenXLA 공식 문서.
- 핵심 함의: **layout 결정이 fusion보다 먼저**다. 따라서 "어느 축이 lane인가 / 얼마나 패딩되는가"는 fusion을 몰라도 결정 가능하지만, **실제 pad/relayout op의 삽입(물질화)은 fusion이 보이는 단계(LLO)에서** 해야 minimal하게 된다.
- PyTorchSim 관점: 이 분리를 그대로 따를 것. FX/상위 단계에서는 layout **결정**(메타데이터)만, 실제 padding/relayout **물질화**는 하위 단계에서.

---

## 2. lane/sublane 축 선택 메커니즘 [확정]

### 2.1 결정 시점과 표현
layout assignment pass에서 HLO 텐서에 `{minor_to_major : T(tile)}` 어노테이션이 부착되는 순간 확정.

관측된 예시 (layout assignment 전후):
```
// Before
%dot.10 = f32[16,64]{1,0} dot(...)
%reduce.16 = f32[16]{0} reduce(...)
// After
%dot.10 = f32[16,64]{1,0:T(8,128)} dot(...)
%reduce.16 = f32[16]{0:T(128)} reduce(...)
```
- `{1,0}` = minor_to_major 순서 (row-major: 마지막 차원이 메모리 연속).
- `:T(8,128)` = TPU 타일링, VPU의 **8 sublane × 128 lane**에 대응.

### 2.2 규칙
- **minor_to_major 리스트의 첫 원소(가장 minor한 차원) = lane(128) 방향**, 그 다음 = sublane(8) 방향.
- 타일링은 **항상 most-minor 두 축에만** 적용. 나머지 major 차원은 타일링 없이 그대로 (rank 무관).
- 누가 결정하나: **matmul/dot이 anchor로 layout 강제** → elementwise는 통과 → reduce는 cross-lane 비용 때문에 lane에서 빠지려는 압력 → graph 위로 전파(propagation).

### 2.3 검증 방법 (PyTorchSim ground truth)
```bash
XLA_FLAGS="--xla_dump_to=/path --xla_dump_hlo_as_text=true" python model.py
```
덤프된 HLO의 `:T(...)` 어노테이션으로 실제 lane 축/패딩을 추측 아닌 컴파일러 출력으로 확인 가능.
(주의: `--xla_enable_hlo_passes_only=layout-assignment` 단독은 후속 buffer assignment에서 에러 가능 → 전체 덤프 권장. 출처: openxla/xla issue #12850)

---

## 3. 타일 크기 규칙 [확정]

| 조건 | 타일 | 비고 |
|---|---|---|
| f32, 일반 | `T(8,128)` | 32-bit 8×128 벡터 레지스터에 대응 |
| bf16 | `T(8,128)(2,1)` | 2단계 타일링 = BF16 packing. 짝/홀수 행 16-bit 둘을 묶어 32-bit 하나로 |
| 2nd-minor 차원 = 1 or 2 | `T(2,128)` | "Compact 2nd Minor Layout" — 메모리 절약 |
| 2nd-minor 차원 = 3 or 4 | `T(4,128)` | 동일 목적 |

- **중요 정정**: sublane 패딩이 항상 8은 아니다. 작은 2nd-minor 차원이면 2 또는 4로 줄어든다.
  - 함의: **LLM 디코딩 token=1의 sublane 패딩은 8배가 아니라 2배** (`T(2,128)`).
- bf16 packing 이유: TPU는 32-bit 네이티브. most-minor보다 2nd-minor 가로지르는 데이터 이동이 효율적이라 같은 column에서 16-bit 둘을 모음.
- 출처: OpenXLA tiled_layout 문서, gdymind 블로그.

### 3.1 MXU 크기 (세대별, 연산 단위 — 레이아웃 타일과 별개) [확정]
- v6e, TPU7x(Ironwood): **256×256**
- v6e 이전: **128×128**
- peak FLOPs 위해선 matmul 차원이 해당 세대 MXU 크기보다 커야 함.
- ⚠️ 이건 *연산* 단위. *메모리 레이아웃* 타일은 세대 무관 8×128 유지. 둘을 섞지 말 것.
- 출처: Google Cloud TPU performance guide.

---

## 4. 패딩 처리: 두 층위 [핵심 — 확정]

### 4.1 (A) 8×128 레이아웃 정렬 패딩 = 강제 물질화
- **실제 텐서가 HBM에서 패딩된 크기로 저장됨.** 회피 불가.
- 이유: 타일이 HBM에 연속으로 깔리려면 각 타일이 꽉 찬 8×128이어야 함 → 주소 정렬(address alignment) 필수.
- MLIR 코드생성 일반론에서 이는 `nofold` 패딩에 해당: "address alignment가 필수인 경우 강제로 패딩". value padding이 불필요해 보여도 정렬 때문에 fold되지 않고 물질화됨.
- Google 공식 확인: 128×8 청크를 못 채우면 XLA가 텐서를 패딩하고, 이는 "on-chip 메모리 저장량을 늘리고 OOM 유발 가능" = 물리적 공간 점유.
- 패딩량 = `⌈d/tile⌉ × tile − d` (lane/sublane 각각).
- 출처: OpenXLA, Google Cloud performance guide, MLIR codegen 논문(arxiv 2202.03293).

### 4.2 (B) 연산 블록 크기(>8×128) 경계 = tail 처리
컴파일러가 비용 보고 세 전략 중 선택 (MLIR codegen 논문 §3.2):

1. **Loop peeling / versioning** (비물질화): main loop는 정적 상수 부분, 경계는 cleanup loop(타일=1로 축소). 텐서 안 키움.
2. **실제 패딩** (물질화): 동적 타일을 정적 크기로 패딩, 값은 소비 연산의 **neutral(항등원)** — matmul이면 0. `tensor.pad` op으로 물질화, 크기 = 정적 타일 − 동적 타일. 추가 복사 비용 발생.
3. **명시적 masking** (비물질화): MXU가 0 포함 전체 계산 후 출력에서 마스킹. "MXU엔 '하지 마라' 신호가 없어 패딩 0을 실데이터와 곱하고 출력에서 마스킹 — 정확하나 느림(버려지는 work에 MXU 비용 지불)".

- 기본은 비물질화(masking/peeling)가 흔함. pad 물질화는 정렬 필수 케이스.
- 출처: MLIR codegen 논문, Pallas matmul 튜토리얼(neuropurrfectai).

### 4.3 두 층위의 관계 [추론 — 합리적]
- 연산 블록 tail이 8×128 정렬과도 안 맞으면: 이미 물질화된 레이아웃 패딩(A)을 그대로 읽고, 블록 크기와 정렬 사이의 간격(B의 순수분)만 masking/peeling.
- 즉 "레이아웃 패딩은 물리적으로 이미 존재, 그걸 넘어서는 블록 경계분만 tail 전략"으로 두 층이 포개짐.

---

## 5. TPU 특유 제약: tiled 축 경계가 비싼 이유 [확정]

- Ragged Paged Attention 논문: 논리/물리 레이아웃 불일치 + narrow dtype packing 때문에 **tiled 차원(특히 lane)에서 임의 메모리 슬라이스가 근본적으로 어렵다** (VREG blending 없이 ragged 입력을 메모리에 직접 쓰는 경우 특히).
- 실전 해법: **ragged/동적 차원을 non-tiled(major) 축에 배치**, packing을 2nd-minor에 삽입해 XLA가 최소 타일 `T(packing,128)`을 쓰도록 강제 → 임의 동적 슬라이스 가능.
- Pallas `pl.BoundedSlice` / `pl.ds`로 동적 크기 청크 처리 가능 (패딩 대신 정확 크기).
- 함의: 컴파일러/커널이 작은·동적 축을 lane에서 빼려 애쓰는 이유가 정량적으로 설명됨.
- 출처: Ragged Paged Attention (arxiv 2604.15464), JAX Pallas pipelining 문서.

---

## 6. LLM 디코딩 특이사항 [확정 + 추론]

- 디코딩 GEMV(`[1,hidden]×[hidden,out]`)에서 token=1 차원:
  - 보통 hidden이 lane(128)에, token=1은 sublane으로 → **sublane 2배 패딩**(`T(2,128)`), lane까지 가지 않음. [추론, §3 규칙 기반]
- **activation 패딩 traffic은 디코딩에서 무시 가능**: activation(`[1,hidden]`, 수 KB)은 weight/KV cache(수십~수백 GB)보다 3~5 자릿수 작음. 8배든 2배든 전체 traffic에서 미미. [확정 — 디코딩 memory-bound 특성]
- 디코딩 traffic 모델은 **weight 전체 재읽기 + KV cache 읽기에 집중**할 것. activation 패딩 오차의 영향은 작음.
- 패딩의 실질 페널티는 traffic이 아니라 MXU utilization 저하인데, memory-bound regime에선 wall-clock 비결정적.
- → self-spec decoding 연구 동기(한 번 읽은 weight당 토큰 더 뽑기)와 직결.

---

## 7. PyTorchSim 모델링 권고 [실행 항목]

### 7.1 두 비용 함수를 분리하라 (가장 중요)
- **footprint / HBM traffic 함수**: (A) 8×128 레이아웃 정렬 패딩만 물리 크기로 계산. 예: 길이 100 → 128로 저장·전송. bf16 packing, small-tile(2/4×128) 변형 반영.
- **compute utilization 함수**: (B) 연산 블록 경계 패딩 처리.
  - 기본 masking: MXU 사이클 낭비로 utilization ↓, **traffic 중복 계상 금지**.
  - pad 물질화 케이스(정렬 필수)만 추가 복사 traffic.
  - peeling 케이스는 작은 cleanup 커널로 별도.
- **함정**: 연산 경계 패딩을 traffic으로 또 더하면 대역폭 과대평가. 물리 패딩(A)만 traffic+compute, 연산 경계(B)는 compute만.

### 7.2 layout 결정 로직
- minor_to_major 축 선택 + 타일 크기 선택을 §2~§3 규칙으로 모델링.
- matmul anchor → 전파 → reduce는 lane에서 빼기 선호.
- 세대별 MXU 크기(128 vs 256)를 파라미터화.

### 7.3 비대칭 반영
- tiled 축(lane) 경계 처리 비용 > non-tiled 축. 이 비대칭을 넣으면 실제 커널이 동적 축을 major로 미는 동작 재현됨(§5).

### 7.4 검증
- §2.3의 `XLA_FLAGS` 덤프로 실제 `:T(...)` 어노테이션 떠서 모델 출력과 대조.
- 가능하면 LLO 덤프까지 떠서 경계 타일이 `tensor.pad`(물질화) vs masking/peeling 중 무엇으로 처리되는지 확인.

---

## 8. 미확인 / 후속 조사 필요 [미확인]

1. **Mosaic의 tail 전략 선택 휴리스틱**: peeling vs pad vs masking을 언제 고르는지의 내부 규칙. Mosaic이 상당 부분 비공개(Google 내부 컴파일러)라 공개 자료로 닿지 못함. → LLO 덤프 실측이 유일한 확실한 길.
2. **연산 블록 경계 0의 정확한 주입 위치**: VREG / VMEM / MXU 입력 중 어디서 0이 주입되고 VMEM 점유에 잡히는지. → Mosaic 소스 또는 LLO 덤프 필요.
3. **layout_assignment.cc의 minor_to_major 선택 휴리스틱 코드 레벨**: matmul이 정확히 어떤 layout을 강제하고 어떻게 전파하는지. OpenXLA 공개 소스(layout_assignment.cc, instruction_fusion.cc)에서 추적 가능 — 아직 코드 레벨로 파지 않음.
4. **DMA의 strided/partial transfer가 레이아웃 패딩을 정확히 어떻게 처리하는지** (세대별): v4부터 512B granularity striding 지원은 확인됨. 패딩 영역 전송 회피 가능 여부의 정밀 동작은 세대·XLA 버전 의존.

---

## 9. 출처 목록

- OpenXLA — Tiled layout: https://openxla.org/xla/tiled_layout
- OpenXLA — Shapes and layout: https://openxla.org/xla/shapes
- Google Cloud — TPU performance guide: https://docs.cloud.google.com/tpu/docs/performance-guide
- Google Cloud — Intro to Cloud TPU: https://docs.cloud.google.com/tpu/docs/intro-to-tpu
- Google Cloud — TPU v4: https://docs.cloud.google.com/tpu/docs/v4
- From JAX to VLIW (컴파일러 추적, layout assignment 전후 HLO): https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation
- Pallas matmul 튜토리얼 (MXU 마스킹, tail 패딩): https://neuropurrfectai.substack.com/p/part-2-your-first-pallas-kernel-tiled
- Composable/Modular Code Generation in MLIR (경계 타일 3전략, nofold): https://arxiv.org/pdf/2202.03293
- Ragged Paged Attention (tiled 축 슬라이스 제약): https://arxiv.org/html/2604.15464
- JAX — TPU pipelining (동적 슬라이스): https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html
- openxla/xla issue #12850 (pass 덤프 방법): https://github.com/openxla/xla/issues/12850
- gdymind 블로그 (small-tile, packing 정리): https://gdymind.com/2026/02/26/XLA02-shapes-layout-tiling/
