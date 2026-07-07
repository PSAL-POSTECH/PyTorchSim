# Masked DMA via supplementary descriptor — design plan

상태: 설계 합의 단계 (구현 전). 작성 맥락: MobileNet wrapper3(8x8) CI 버그 추적에서
출발했으나, 이건 단발 패치가 아니라 **padding/masking을 1급 transfer 속성으로 만드는
일반 primitive** 설계다.

> ⚠️ **이 문서는 §0의 수정으로 갱신됨.** §3~§9의 일부 서술(특히 "seam=ops.masked",
> "self.masks를 pre_load로 이동", "단일 pre_load/pre_store", "둘 다 divisibility 제거")은
> §0에 의해 정정/대체된다. 충돌 시 §0 우선.

---

## 0. 검토 결과 — 핵심 수정 / 미해결 (5-agent review 반영)
실제 코드 대조 리뷰에서 나온 정정. 아래 항목이 §3~§9의 해당 서술을 **대체**한다.

### D1 (결정적) — ✅ 실측 해소
우려: 정렬 타일 W=32라도 **마지막 타일 `114 mod 32 = 18`** 이 16의 배수가 아니면 ragged 재발(=W=19),
`skip+fill`로 못 고침(holes는 spad write-주소 오매핑, predicate 무관).
**실측 결과(divisibility 끈 채): tail은 이미 full-32다.** 생성 커널이 **고정 descriptor `memref<2x32x8>`** +
`affine.for W = 0 to 114 step 32`(= ceil(114/32)=4회) → 마지막 타일(96)도 **같은 full-32**(그래서 오늘 phantom
읽어 segfault). **18로 clamp 안 함.** → full-32는 lane-aligned(32=2·16, n_outerloop 깨끗)라 **ragged 없음**,
14칸 phantom만 **mask로 skip**하면 됨.

> **단 이 full-extent를 incidental(고정 descriptor 부작용)에 의존하지 말 것.** 미래에 "tail을 valid(18)로
> clamp" 같은 최적화가 들어오면 ragged가 재발한다. 그러므로 **"tail 타일 = full lane-aligned extent + mask"
> 를 codegen/tiling의 *명시적 불변식*으로 박는다** (high-level: loop bound를 lane-aligned 배수로 두고 overflow를
> mask로; 또는 codegen: tail descriptor를 full-extent로 보장). — **결정**.

### D2 (안전성). transfer-clamp는 **무조건**(operand/output shape 기반), ops.masked와 분리
divisibility는 P5에서 **전역(tiler)** 제거되는데 read-mask가 **op별 조건부(fallback 가능)**면,
fallback한 op은 MVIN(전체 비-나눔 타일) → **far-OOB read → segfault(원래 버그 재발)**. 그러므로
**MVIN/MVOUT의 bound-clamp(read=operand-shape, write=output-shape)는 ops.masked seam과 무관하게
*항상* 적용**되는 transfer 속성이어야 한다. ops.masked는 그 위의 추가 mask일 뿐. (§3.4 "seam=ops.masked"
서술 정정.)

### D3 (seam). ops.masked body는 load가 아니라 **subgraph**, 그리고 `self.masks`는 다른 버퍼
- Inductor `ops.masked(mask, body, other)`의 body는 **subgraph**(loader closure). 입력이 fuse되면
  산술이 inline → "body=순수 load" 구조적으로 거짓(avg_pool/cat/scatter도 산술 포함).
  → **변형은 "body가 순수 affine access임을 *post-fusion subgraph*에서 보수적으로 증명"될 때만**.
  이 판정기가 안전 게이트(거짓양성=산술 누락=조용한 오답).
  **✅ 실측: constant_pad는 `masked_subblock1`(=ops.masked)를 emit하고 그 body는 순수 `ops.load`** →
  기본 pad에선 seam·precondition 성립. (subgraph 위험은 **fused pad / avg_pool/cat/scatter** 등 타 케이스에 한정.)
- **`self.masks` 버퍼는 reduction-tail mask**(`get_mask`, reduction에서만)이지 ops.masked border mask가
  아니다(§9c의 "self.masks를 pre_load로 이동" 정정). WRITE-tail은 이 `get_mask`(출력경계 lt) 재사용,
  READ-border는 별도 ops.masked 경로 — **둘은 다른 메커니즘**.

### 코드젠 (최종). 고정 phase 폐기 → **step 리스트(cursor + append)** — 상세 §9c
- 글로벌 pre_load/pre_store 도, per-DMA pre/post 도 아님(둘 다 특수케이스). **일반 표현 = spad-handle 위 step 리스트.**
  `self.steps`(append 순서 emit) + `self.current_step`(기본 sink=compute loop body). **간선 불필요**(Inductor body가 topo-정렬).
- 핸들러: compute는 current_step, mvin/mvout/build는 `new_step` append. multi-stage·mask·shared·tail 모두
  append 순서로 자동(→ `comptute_depedency→dma_stores` 핵 소멸).
- ops가 다루는 건 **per-lane vector**, descriptor는 **memref**. masked load/store는 **메모리-도메인 op으로 분리**:
  `build` step `{vector mask 계산 → vector.store → @memref}` + `mvin/mvout` step `{gated transfer(@memref 소비)}`.
  **access op(indirect/masked) 반환 = memref handle**, transfer가 operand로 소비.

### indirect (정정). smuggle ↔ indirect_dims는 **한 메커니즘**, 추출은 lower_dma_to_gemmini
- `index + Symbol`(:1545)과 `indirect_dims`(:498~503)는 분리 항목이 아니라 **하나의 파이프라인**(smuggle이 indirect_dims를 먹임). 실제 추출은 `decompose_transfer`가 아니라 **`lower_dma_to_gemmini._find_indirect`(:192~205)**.

### divisibility 제거는 **단계적** (내부 모순 정정)
§4/§6의 "둘 다 제거"는 *최종 상태(P5)*를 말함. **P2는 `:1464` 유지**(표현만 정리), `:789`/`:1464` 동시 제거는 **P5**.

### HW (양호). Spike는 feasible
predicate gate는 기존 `is_used_vlane` 지점에 추가; **버그성 `d_addr!=0` 휴리스틱은 explicit predicate로 *교체*(AND 아님)**; fill(0)은 이미 됨(−inf용 fill-value 레지스터 필요); masked+indirect 한 패스 합성(mask AND를 indirect-load 앞); mvin2/3는 thin wrapper; 신규 레지스터 2~3개+CONFIG5 여유.

### 미해결 (실측/설계 필요)
1. ~~constant_pad가 `ops.masked`를 emit하나?~~ ✅ 해소 (emit함, body=순수 load — D3).
2. ~~tail DMA 타일이 full-32냐 clamp-18이냐?~~ ✅ 해소 (full-32 — D1). **단 이걸 *의도적 불변식*으로
   박아야 함**(incidental 의존 금지 — D1 결정).
3. "순수 affine access" 보수 판정기 명세 (post-fusion subgraph) — fused pad/avg_pool/cat/scatter 대비.
4. bitmask 레이아웃 ↔ lane-bank 주소 일치 + 정렬 타일 순서 의존성.
5. **[TODO — 나중]** tail full-extent 불변식을 **어디서** 강제하나 (high-level loop-bound pad vs
   codegen tail-descriptor 보장). 지금은 incidental하게 full-32라 당장 막진 않음 → 후속 작업으로 미룸.

---

## 1. 한 줄 요약
지금은 "DMA로 타일 전체를 가져온 뒤 compute에서 select로 masking out" 한다.
이걸 **"mask를 먼저 정하고, DMA가 유효 위치만 옮긴다(masked DMA)"** 로 뒤집는다.
mask는 `togsim.transfer`에 붙는 **부가 descriptor(1-bit predicate)** 로 전달한다.

---

## 2. 왜 (context)
### 직접 버그
- 8x8 config에서 `constant_pad_nd` 가 출력 내부 유효 원소 ~2%를 0으로 만든다(holes).
- divisibility를 끄면 대신 **segfault**(입력 텐서 밖 read).

### 근본 원인 — "load-all-then-mask-out"
현재 생성 IR(요지):
```
MVIN(in → spad)                 # 경계 없이 타일 전체 load
mask = cmpi(0<=idx<S) ...        # 사후
sel  = select(mask, val, 0.0)    # masking out
MVOUT(spad → out)
```
mask가 **load 다음**이라 OOB load를 못 막는다.

> **divisibility 제약의 존재 이유**: 부분 타일(partial tile)에서 생기는 **padding/OOB를
> 계산·처리할 수단이 없었기 때문**이다. 그래서 "타일이 차원을 정확히 나눠떨어져서
> *애초에 부분 타일이 안 생기게*" 강제한 **우회 제약**이다. 즉 divisibility는 원인이
> 아니라, padding을 못 다뤄서 건 목발 — 이 문서의 masked-DMA가 padding을 다루게 되면
> 그 목발(`_index_expr:789`, `convert_indirect_indexing:1464`)은 제거된다.

그 우회 제약이 다음을 연쇄로 일으킨다:
```
divisibility 강제  →  114엔 8의 배수 약수 없음  →  ragged 타일(W=19)
   →  lane-banking 오매핑  →  holes
divisibility 끄면  →  부분 타일이 텐서 밖  →  far-OOB read  →  segfault
```
divisibility 제약은 두 곳에 있다:
- `mlir_codegen_backend.py:789` `_index_expr` (마스크/index_expr 용)
- `mlir_codegen_backend.py:1464` `convert_indirect_indexing` (gather 용)

### 왜 일반 primitive인가
padding(constant/reflect/replicate), conv/pool padding, slice, partial-tile tail,
dynamic shape, masked/sparse load, gather — 전부 "유효영역만 transfer + 경계정책"
이라는 같은 모양이다. 한 번 제대로 세우면 이들이 같은 틀의 인스턴스가 된다.

---

## 3. 설계

### 3.1 핵심 원칙
1. **주소 계산은 DMA(stride)가 한다.** vector lane으로 옮기지 않는다.
2. **descriptor는 부가 정보만**: `{ indirect offset, mask flag }`.
   - offset: gather일 때만, i64.
   - mask flag: mask일 때만, **1-bit packed predicate(bitmask)**.
   - 둘은 직교 — offset만 / flag만 / 둘 다.
3. **masking은 transfer 단계**에서. compute의 사후 select를 없앤다.

### 3.2 왜 1-bit bitmask
- flag는 본질이 1비트. byte/i64로 부풀리면 **타일크기 × N(전체 작업량)** 만큼 낭비.
- 이미 `cmpi`가 `vector<i1>`(레인당 1비트)을 만든다 → 그대로 packed 저장, 변환 공짜.
- SIMD predicate(RVV `v0.t`)와 같은 자연스러운 형태. MVIN이 레인별 비트로 gate.
- byte-granular indirect 읽기(load_uint8)는 offset 채널이지 flag 채널이 아니다.

### 3.3 두 유효범위 소스
- **READ(MVIN)** = operand 경계. `ops.masked` 의 mask가 이것(= `0<=idx<S`).
  - 이 mask index는 **mask 전용**이고 폴드된 load 주소(index2)와 분리돼 있음(확인됨)
    → 폴드(−3616 분해) 문제 무관.
- **WRITE(MVOUT)** = 출력 차원(`var_ranges`). **universal, 항상 exact.**
  - 이게 partial-tile tail을 처리하고, **divisibility를 없애는 토대**.

### 3.4 흐름 (코드젠 변경)
```
# READ: mask가 load 전에 필요 → pre-pass로 hoist
[pre-pass]  mask = cmpi/and(...)         # index-only → hoist 가능
            store mask → flag_bitmask
MVIN(in → spad, mask=flag_bitmask)       # 비트=0이면 skip + fill(c)
[compute]   데이터 연산만 (select 제거)
MVOUT(spad → out, mask=write_bitmask)    # compute 뒤라 flag 자연히 있음
```
- **seam = `ops.masked` 핸들러 교체**: select 생성 대신 (a) mask predicate를
  pre-pass+flag로, (b) gated transfer.
- 신규 핵심 = **READ용 mask pre-pass(MVIN 앞으로 떼어내기)**. MVOUT는 순서상 공짜.

### 3.5 policy = descriptor 채우기 규칙
경계/마스크 밖 위치를 descriptor에 어떻게 넣느냐가 곧 정책:
| op | OOB/invalid 위치 처리 |
|---|---|
| constant_pad / mask / tail | predicate=0 → skip + fill(c) |
| maxpool-pad | fill(−inf) |
| replication_pad | clamp된 주소 (실제 load) |
| reflection_pad | reflect된 주소 |
| gather | 로드한 index로 만든 offset |
DMA는 "멍청한 소비자"(읽고 skip/load), 똑똑함은 descriptor 채우기에. **v1 = fill+skip만.**

---

## 4. 페이로드 (무엇을 해결/통일)
- `_index_expr`·`convert_indirect_indexing` **둘 다 divisibility 제거**(최종 P5; P2는 :1464 유지 — §0)
  → tiler가 레인 정렬 타일 선택 → ragged 소멸 → **holes/segfault 해결(= CI 버그)**.
  단 tail 타일은 full-extent+mask여야 함(§0 D1).
- **sparse/임의 mask** 지원(per-position predicate, box 제약 없음).
- **dynamic shape**: 경계가 symbolic이어도 predicate만 맞으면 됨.
- pad 모드·mask·indirect·tail을 **한 메커니즘**으로 통일.

---

## 5. Soundness
- skip은 **항상 안전**(안 건드리고 fill). "어디가 invalid냐"는 `cmpi`(exact)에서 옴 → 근사 없음.
- mask는 load 마스크라 **항상 index-only → hoist 안전**.
- 변형 불가/불확실하면 **기존 compute-select fallback**(정확, 비최적).
- 단 **WRITE-tail(출력경계)은 항상 적용 가능**(fallback 불필요, 늘 exact).

---

## 6. 단계 (phasing)
> 이 coarse 버전은 **§9b의 P1~P6**(검증 게이트 포함)로 대체됨. 순서: indirect 개편(P2) →
> mask(P3·P4) → divisibility 제거(P5) → policy 확장(P6). divisibility는 P5에서 양 경로 제거(§0).

---

## 7. 결정된 것 vs 열린 것
**결정**
- 주소는 DMA, descriptor는 부가({offset i64, mask 1-bit bitmask}).
- mask = 1-bit packed predicate(cmpi의 vector<i1> 재사용).
- seam = `ops.masked` lowering.
- 유효원: read=mask, write=출력shape.
- box-attr/per-axis 경계 레지스터는 **안 씀**(predicate가 직접 유효성).

**열린 (다음 고민)**
- (a) flag bitmask **packing 포맷 + MVIN의 레인별 소비** 방식(predicate 채널 신설/재사용).
- (b) `ops.masked`에서 **mask predicate를 떼어 pre-pass loop로** 만드는 코드젠 — *핵심 난도*.
- (c) flag 버퍼/predicate 레이아웃이 **DMA per-lane 인덱싱(banking)** 과 일치하도록.
- (d) policy 확장(clamp/reflect)을 이 primitive에 흡수 vs Inductor index 변환에 둘지.
- (e) compute-select fallback 판정 조건(정확 명세).

---

## 8. 검증 계획
1. **pad repro** (`tests/scratch_pad_repro.py`): 8x8에서 W=32(레인정렬, 114 안 나눔)으로
   **holes=0 + segfault 없음** (현재는 둘 다 터짐 = 성공 기준).
2. **mobilenet wrapper3** 풀모델 `allclose`.
3. **dynamic shape** pad 케이스(symbolic 경계) 정확성.
4. **sparse mask** (where 등) 정확성 — predicate 일반성 확인.
5. **bounds-masked gather** (`ops.masked(mask, indirect_load)`) — offset+mask **두 채널 합성** 정확성.
6. pointwise/indirect 전 op CI (divisibility 제거가 광범위 영향).

---

## 9. 리스크
- 코드젠 흐름 재배치(pre-pass)와 `ops.masked` lowering 교체 — frontend 큰 변경.
- Spike MVIN/MVOUT에 predicate 소비 추가 — HW 모델 변경.
- divisibility 제거가 **모든 pointwise/indirect 커널**에 영향 → 광범위 회귀 검증 필수.
- 단계적으로: WRITE-tail(가장 안전·universal)부터 → READ-mask → 나머지.

---

## 9b. 구체 구현 계획 (검토용)

### 인터페이스 계약 (먼저 못박을 것)
**A. IR — `togsim.transfer` 의 descriptor 필드** (둘 다 optional, 없으면 기존과 동일)
```
togsim.transfer(dram, base_idx, spad, sram_idx, tag, dma_type, vst,
    # 신규 descriptor (없으면 기존 동작)
    mask   = { buf: <spad bitmask ref>, fill: <f32/i const> }   # masked 일 때
    offset = { buf: <spad ref>, elem_size: i, stride: i }       # indirect 일 때
)
```
- mask 없으면 = 전체 유효(기존). offset 없으면 = base 주소(기존).

**B. Spike — descriptor 소비 (ad-hoc indirect 모드 대체)**
- 통합 config 1개: 어떤 채널 있나(mask?/offset?) + 각 buf addr/elem_size/stride.
- MVIN/MVOUT per-position: `valid = mask_bit(pos)` (mask 채널 있으면, 없으면 1);
  `addr = base + (offset 채널 ? offset(pos) : 0)`; `valid==0 → skip + fill`.
- 기존 `dma_indirect_mode`/`config4`/`mvin2,mvin3` 를 이 한 경로로 수렴.

**B-2. masked + indirect 합성 (1급 케이스 — 놓치면 안 됨)**
두 채널은 **독립**이므로 네 조합 모두 같은 경로로 성립:
| offset | mask | 의미 |
|---|---|---|
| - | - | 일반 affine transfer |
| O | - | gather/scatter |
| - | O | masked/padded |
| **O** | **O** | **bounds-masked gather** (예: `ops.masked(mask, indirect_load(idx))`) |
- 소비 규칙이 이미 둘을 따로 검사(`addr=base+offset`, `gate by mask`)하므로 합성은 자동.
- 코드젠: pre-pass가 **offset 버퍼(기존 indirect 머신)** 와 **mask bitmask(신규)** 를 *둘 다* 채우고, 같은 gated transfer가 둘 다 소비.
- 순서: gather 인덱스(offset)도 mask도 데이터/index-only 선계산 → gated MVIN 전에 둘 다 준비.

### 단계 (각 단계 = 검증 게이트)
**순서 결정: indirect 개편(substrate 확립) → mask 확장 → divisibility 제거.**
indirect refactor가 descriptor/pre-buffer substrate를 먼저 깔고(회귀 0, 동작 불변),
mask는 그 위에 flag 채널을 얹는다.
**중요: read-border 와 write-tail 은 divisibility 제거에 *함께* 묶인다** — 정렬 타일(W=32)의
마지막 부분 타일이 읽기도 쓰기도 텐서 밖 → 둘 다 막아야 divisibility 끌 수 있음.

- **P1. 플러밍 (동작 불변)**: (i) codegen에 **`pre_load`/`pre_store` buffer 신설 + splice**(§9c),
  (ii) `togsim.transfer`에 mask/offset descriptor 필드(기본 off), (iii) Spike에 통합 descriptor 소비
  경로(기본 off). 빈 버퍼·off라 기존 테스트 그대로 통과 = 게이트.
- **P2. indirect 개편 (clean refactor, 동작 불변)**: ad-hoc 제거 →
  - 트리거: `indirect_indexing` 핸들러(:786, 지금 이름만 반환)에서 **심볼을 indirect로 기록** +
    offset을 `pre_load`에 materialize. load/store는 **symbol-set membership**으로 indirect 판정.
  - 기존 `tmp` 문자열(:603,1201,1211,1460,1471,1508)·`index+Symbol`(:1545)·`indirect_dims`·
    `comptute_depedency`(:554) 전부 제거.
  - Spike `dma_indirect_mode`/config4/mvin2,3 를 통합 descriptor로 수렴.
  - **divisibility(:1464)는 아직 유지** (표현만 정리). 게이트 = **기존 gather 회귀 0**.
- **P3. READ-border masked MVIN**: `ops.masked` lowering 교체 → **단, body가 순수 access(load)일 때만**
  (아니면 compute-select fallback) → mask cmpi→bitmask를 `pre_load`에 + gated MVIN, select 제거.
  **divisibility 켠 채** 결과 동일 = 게이트(회귀 0).
- **P4. WRITE-tail masked MVOUT**: 출력경계(var_ranges) predicate → `pre_store`에 → MVOUT gate.
  divisibility 켠 채 회귀 0 = 게이트.
- **P5. divisibility 제거**: `_index_expr:789`·`convert_indirect_indexing:1464` 재컴파일 제거(affine 한정)
  → tiler가 레인 정렬 타일 선택. **pad repro: W=32로 holes=0 + segfault 없음** + mobilenet wrapper3
  allclose + **bounds-masked gather**(offset+mask 합성) = 게이트.
- **P6. policy 확장**: clamp/reflect → replication/reflection pad.

### 검토 필요 결정점 (구현 중 실측 필요 — 단정 금지)
1. **mask pre-pass 생성**: body의 mask sub-block을 그대로 hoist 재사용 vs cmpi를 별도 loop로 재방출. (어느 게 깔끔한지 ops.masked 처리 코드 보고 결정.)
2. **bitmask 레이아웃 ↔ DMA per-lane banking 일치**: flag가 데이터 타일과 같은 (lane,slot) 매핑이어야. 정렬 타일에서만 성립(자기충족이지만 순서 주의).
3. **fill 값 출처**: masked op별(0 / −inf). ops.masked의 `other` 인자에서.
4. **fallback 조건**: mask가 index-only 아님/표현 불가 시 compute-select 유지. WRITE-tail은 항상 적용(예외).
5. **divisibility 제거의 광범위 영향**: affine 아닌(floor/mod view) 경로는 제약 유지 — 면제 경계 정확히.

## 9c. Codegen 전략 (phase 스케줄)

### 현재 구조 — 이미 phase-순서 buffer 조립
`codegen_loops` (mlir_codegen_backend.py:936, splice 순서 968~979) 가 buffer를 순서대로 조립:
```
applys → indexed_buffer → dma_loads(MVIN)
   → [compute_body: masks → loads → compute → stores]
   → dma_stores(MVOUT)
```
- buffer 초기화: mlir_codegen_backend.py:324~328 (`applys/masks/dma_loads/dma_stores/indexed_buffer`).
- **`self.masks` 버퍼가 이미 있으나 dma_loads *다음*(compute_body 안)** → 지금은 "load 후 mask".

### 결정(최종) — 고정 phase를 버리고 **spad-handle DAG = step 리스트**로
> pre_load/pre_store(글로벌) 도, per-DMA pre/post 도 아님. 둘 다 특수케이스라 폐기.
> **일반 표현 = spad handle 위의 outer-level step DAG, topological emit.** (논의 합의.)

**handle**: spad에 사는 named 버퍼. **tile(데이터)·descriptor(offset/mask) 동일 취급.**
**step**: outer-level MLIR 텍스트 한 블록 (`mvin` 하나 / `mvout` 하나 / `build` descriptor mini-loop / `compute` inner 루프 통째).
```
Step = { kind, code }              # 간선 없음. kind는 디버그/가독용
self.steps        : list[Step]     # append 순서대로 emit
self.current_step : Step           # 기본 write 대상 (= 현재 loop body = compute step)
```
**cursor + append (간선 불필요):**
- **간선(produces/consumes) 안 씀.** Inductor body가 이미 topo-정렬(def<use)이라 **처리하며 append하면 그게 곧 valid 순서**. 간선은 prefetch 재정렬 같은 *최적화* 때나 필요 — 지금 안 함.
- 핸들러는 기본적으로 `current_step.code`에 emit. 별도 step 필요하면 `new_step(kind)` append.
  `mvin/mvout`은 transfer 한 줄. `build`는 내부 ops 동안만 current_step을 그 step으로 잠깐 돌렸다 복원
  (= 기존 **`override_buffer_cse`** 가 하는 일 그대로 → 큰 rewrite 아님).
- **현재 loop body = current_step**(compute, 기본 sink). loop level 2개 유지: mvin/mvout/build는 outer, compute는 inner(`compute_idx`) 루프 한 step.

append 예 (순서가 곧 정답):
```
current_step = compute(loop body)
ops.load(idx)        → new_step('mvin')                              # MVIN idx
ops.load(x, gather)  → new_step('build'); current→build; offset emit; current→compute 복원
                       new_step('mvin')                              # MVIN gather
ops.add(...)         → current_step(compute)에 누적
ops.store(...)       → new_step('mvout')
⇒ steps = [mvin idx][build off][mvin gather][compute][mvout]   # comptute_depedency 핵 소멸
```

모든 케이스가 한 규칙의 인스턴스:
```
mask(padding):     [build @msk (index-only)] → [mvin tile (mask=@msk)]
multi-stage gather:[mvin @idx] → [build @off (consumes @idx)] → [mvin @x (off=@off)]
shared:            [build @off] → [mvin @a (off=@off)], [mvin @b (off=@off)]
write-tail:        [compute] → [build @wmsk] → [mvout (mask=@wmsk)]
현재 3-phase:       [mvin...] → [compute] → [mvout...]   ← degenerate case
```
→ 영역/tier/carve-out/per-DMA 같은 특수처리 **불필요**. readiness = DAG 위치(emergent).

**구현 증분(회귀 0 출발):**
1. `self.steps` + `self.current_step` 도입 + `codegen_loops` 고정 splice(968~979)를 `for s in self.steps: splice(s.code)` 로.
   기존 buffer(dma_loads/compute_body/dma_stores)를 **현재 순서대로 step으로 래핑** → 동작 불변 = 게이트.
2. load/store 핸들러가 고정 버퍼 대신 **`new_step` append**(per-mvin 입자), compute는 current_step → interleave 가능해짐.
3. masked/indirect가 **build step** append(`override_buffer_cse`로 current 잠깐 전환), transfer step이 그 descriptor handle 소비.
   기존 `comptute_depedency→dma_stores` 핵(:554)은 append 순서로 자연 대체.
- prologue(const/alloc/spad 글로벌, applys/indexed_buffer 인덱스 셋업)는 step 앞 **선언부**로 분리 유지.
- `self.masks`(reduction-tail, `get_mask`)는 **그대로** compute step 내부 — ops.masked border와 무관(§0).

## 9d. P2 상세 — indirect 개편 (reuse / discard / new)

### 현재 indirect 흐름 (end-to-end)
1. `indirect_indexing`(:786): 로드값 var 이름만 반환.
2. 그 `tmp` 심볼이 이후 index sympy 식에 섞임.
3. `load`(:549)/`store`(:603): `"tmp" in str(index)` 로 감지 → `convert_indirect_indexing`.
4. `convert_indirect_indexing`(:1459): 인덱스값 spad에 materialize → stride·sum → 버퍼 store
   → `return index + sympy.Symbol(out)` (심볼 밀어넣기). `comptute_depedency`로 `dma_stores` 라우팅(:552).
5. `get_dma_info → parse_indices → affine_apply`(:498~503): 인덱스(+심볼)가 affine indirect operand로.
6. `emit_transfer`(:1330): togsim.transfer (indirect 모름 — 그냥 index operand에 묻어감).
7. `decompose_transfer` → `lower_dma_to_gemmini`(:144~161): affine operand에서 `indirect_memref`
   추출 → CONFIG의 indirect bit(:149) + **CONFIG4**(인덱스버퍼 base/elem_size/stride) emit. MVIN(mvin2/3) 소비.

### REUSE (그대로 — 실제 머신)
- **offset 값 materialize**: `convert_indirect_indexing` 1478~1525 (인덱스 로드→stride→sum→버퍼 store)
  = offset descriptor 생성기. 단 **`pre_load` 버퍼로 라우팅**.
- **Spike 하드웨어 소비**: CONFIG4 + `dma_indirect_mode` + 인덱스버퍼 읽기(가변 elem_size). 인터페이스만 통합.
- **`lower_dma_to_gemmini`의 CONFIG4 emit**(:156~161): `indirect_memref` 받아 emit — 대부분 재사용.
  (mask 채널용 CONFIG 추가는 P3.)
- `get_scratchpad_buffer`.

### DISCARD / REPLACE (ad-hoc)
- `"tmp" in str`(:603,1201,1211,1460,1471,1508) → **explicit `set_indirectN`/`indirect_symbols`
  set membership**. 근거(실측): indirect는 Inductor LoopBody에서 **`set_indirectN(value)`**
  특수 노드로 표현되며(`reduction`/`scan` 과 같은 급의 special node — torch
  `_inductor/codegen/common.py:2468` 가 `("set_indirect","reduction","scan")` 로 처리), index 식엔
  명시 심볼 `indirect0` 로 등장한다(`index1 = 16*indirect0 + p1`). 즉 `tmp` 문자열이 아니라
  이 **명시 심볼**을 `indirect_indexing`(:786)에서 기록하면 깔끔.
- `index + Symbol(out)`(:1545) 심볼 밀어넣기 → **transfer의 별도 offset descriptor operand**. index는 base affine 유지.
- `indirect_dims` affine 배선(:498~503) → 불필요(offset 분리).
- `comptute_depedency → dma_stores` 라우팅(:552~560) → **phase 배치**.

### NEW
- `togsim.transfer`에 **explicit offset descriptor operand**(offset 버퍼 memref) → index 오염 안 함.
- `decompose_transfer`/`lower_dma_to_gemmini`: affine operand 추출 대신 **explicit operand** 읽기.

### SUBTLE (주의)
- `index + Symbol`(:1545)과 `indirect_dims`(:498~503)는 **분리 항목이 아니라 하나의 파이프라인**
  (smuggle이 indirect_dims를 먹임). explicit operand가 이 체인 전체를 대체.
- **multi-stage**(인덱스 load → gather): offset materialize를 인덱스 load **다음** · gather MVIN **앞**에
  배치 — 단일 pre_load로는 표현 불가, **per-DMA pre 버퍼**(§0)로 해소. 현재 `comptute_depedency=dma_stores`가 그 ad-hoc 버전.
- indirect 추출은 `decompose_transfer`가 아니라 **`lower_dma_to_gemmini._find_indirect`(:192~205)** — explicit operand 읽기로 교체.

### 예제 — gather `x[idx] + 1` (실측, x:[64,16] idx:[32]i64 → out:[32,16])

**Stage A. Loop-level IR (Inductor LoopBody, `body.debug_str()` verbatim; `#`주석만 설명용)**
```
var_ranges = {p0: 32, p1: 16}
index0 = p0                  # idx[p0]
index1 = 16*indirect0 + p1   # x[16*indirect0 + p1]   (indirect0 = 로드된 idx 값)
index2 = 16*p0 + p1          # out
def body(self, ops):
    load   = ops.load('arg1_1', index0)   # idx[p0]
    set_indirect0(load)                    # indirect0 := load   (= ops.indirect_indexing, 특수노드)
    load_1 = ops.load('arg0_1', index1)    # gather
    add    = ops.add(load_1, 1.0)
    ops.store('buf0', index2, add)
```

**Stage B. 현재 생성 MLIR (ad-hoc, 실측)**
```mlir
memref.global @buf0_spad : memref<16x16xi64>           # idx 값 버퍼
togsim.transfer MVIN(idx, %index0, @buf0_spad)          # 1) idx MVIN
%tmp7 = index_cast(... vector_load @buf0_spad ...)      # 2) idx값 → index
%apply0 = affine.apply #map0(%index1)[%tmp7] {indirect_access}   # 3) 심볼 밀어넣기 + 태그
togsim.transfer MVIN(x, %apply0, @buf1_spad)            # 4) gather (dram_stride=[0,1]: row=0)
# 5) lower_dma_to_gemmini: {indirect_access}+index_cast(load(@buf0)) 추적 → CONFIG4(@buf0,esize=8,stride)
```

**Stage C. 새 descriptor 모델 (제안)**  — `index1 = 16*indirect0 + p1` 분리: base=`p1`, offset=`16*indirect0`
```mlir
togsim.transfer MVIN(idx, %index0, @buf0_spad)          # 1) 동일
@off = indirect_access(@buf0_spad, stride=16)          # 2) 오프셋 materialize → memref handle 반환
togsim.transfer MVIN(x, base=#map(%index1), offset=@off)  # 3) offset = explicit operand, index는 base만
# 4) lower: offset operand 직접 읽어 CONFIG (태그/체인 추적 X)
```

| | 현재 | 새 모델 |
|---|---|---|
| 트리거 | `"tmp" in str` | `set_indirect0` → indirect 심볼 set |
| 전달 | `affine.apply[...%tmp7]{indirect_access}` | transfer **explicit offset operand `@off`** |
| `@off` 정체 | lowering이 역추적 | **`indirect_access(...)` 반환 = memref handle** |
| index | indirect 섞임 | base affine(`p1`)만 |
| +mask 시 | (없음) | `(base, offset=@off, mask=@msk)` 자리 추가 → 합성 자동 |

### P2 게이트
기존 gather 테스트 회귀 0 (동작 불변). divisibility(:1464)는 유지(표현만 정리).

## 10. 코드 맵
- divisibility 제약: `mlir_codegen_backend.py:789`(_index_expr), `:1464`(convert_indirect_indexing).
- indirect 버퍼 머신(재사용 대상): `mlir_codegen_backend.py:1459` `convert_indirect_indexing`.
- 타일링: `mlir_common.py` `pad_vlane_tile`(:441), `adjust_tile_to_divisible`(:349),
  `is_dim_dividable`(:337).
- transfer emit: `mlir_codegen_backend.py` `emit_transfer` / `togsim.transfer`.
- indirect 핸들러: `mlir_codegen_backend.py:786` `indirect_indexing`(지금 이름만 반환),
  `:1459` `convert_indirect_indexing`(materialize+심볼 밀어넣기), `:549/603` load/store의 `tmp` 감지.
- indirect 전송 경로: `emit_transfer:1330` → `passes/decompose_transfer.py`(affine indirect operand 추출)
  → `passes/lower_dma_to_gemmini.py:144~161`(CONFIG indirect bit + **CONFIG4** emit).
- Spike DMA: `riscv-isa-sim/riscv/insns/torchsim_mvin_common.h`
  (dma_buffer fill, `d_addr!=0` skip, `indirect_mode`, 가변 `indirect_element_size`);
  config: `torchsim_config4.h`(dma_indirect_addr/stride/element_size); `processor.h:561~564`.
- codegen phase buffer: `mlir_codegen_backend.py:324~328` 초기화, `codegen_loops:936`(splice 968~979).
- Inductor body의 mask: `ops.masked`/`masked_subblock`, `cmpi/and` → `vector<i1>`.
