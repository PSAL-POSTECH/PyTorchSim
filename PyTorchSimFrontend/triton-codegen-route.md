# Triton 코드젠 경로를 PyTorchSim에 이식

`torch.compile`이 `npu:0`에서 PyTorchSim의 자체 MLIR 코드젠 대신 **Inductor의 Triton 백엔드**를 쓰고, **NPU lowering pass**가 이를 RISC-V로 낮추는 두 번째 코드젠 경로.

## 작업 경계

이 경로는 두 부분으로 나뉩니다. **이 문서가 보고하는 것은 아래쪽입니다.**

| 부분 | 하는 일 | 소관 |
|---|---|---|
| **NPU lowering pass** | Triton IR → linalg/memref → tts 레벨 백엔드 패스 → vcix/gemmini → RISC-V ELF | **이정민** — 범위 밖 |
| **기존 PyTorchSim으로의 이식** | 그 lowering pass를 기존 시뮬레이션 스택에 얹는 일 | 이 문서 |

lowering pass 자체는 만들지 않았습니다. **이미 있는 것을 PyTorchSim이 쓸 수 있는 형태로 이식하고, 기존 TOGSim / gem5 / Spike 스택에 물린 것**이 여기서 한 일입니다.

용어: 이 lowering 계층을 문서 전체에서 **NPU lowering pass**로 부릅니다. 다만 코드가 `triton-npu` 저장소에 있어서 **실제 식별자는 `tnpu`로 남아 있고**(`tnpu_bridge.py`, `tnpu/passes/`, `tnpu.spike`, `strip_for_tnpu`), 문서에서 코드를 찾아갈 수 있도록 그 이름들은 그대로 인용합니다.

## 현재 도달점

| | |
|---|---|
| functional | 연결됨. `x + y`, `(x+y)*2 - x` **max abs error 0.0** (1024 elements, Spike) |
| timing | 연결됨. TOGSim **650 cycles**, 타일 compute는 gem5 **19 cycles 실측** |
| 동적 shape | 처리됨. 트레이스 하나가 모든 shape을 섬김 — n=1024 → grid 8, n=4096 → grid 32 |
| **커버리지** | **elementwise와 그 융합까지.** 그 밖은 5절 실측 참고 |
| CI | 전 잡 green |

---

## 1. 기존 MLIR 경로와의 차이

### 갈라지는 곳과 합쳐지는 곳

```
                    torch.compile / Inductor 스케줄
                              │
              ┌───────────────┴───────────────┐
              │                               │
        [기존] MLIR 경로                 [신규] Triton 경로
              │                               │
   Inductor 스케줄 → 손으로 쓴          Inductor 의 Triton 코드젠이
   op별 MLIR 템플릿                     낸 커널 소스를 가로챔
   (gemm, conv, sdpa, sort,             (op별 템플릿 없음)
    cat, maxpool, bmm …)
              │                               │
   PyTorchSim mlir/ 패스               NPU lowering pass
   PSAL LLVM 20  (in-process)          stock LLVM 23  (subprocess)
                                       └ 담당 이정민 — 범위 밖.
                                         여기서 한 일은 이 블록을
                                         아래 합류점까지 잇는 배선
              │                               │
              └───────────────┬───────────────┘
                              │
                    ▼  여기서 다시 합류  ▼
                   trace.so + trace_cycles.tsv
                              → TOGSim
                     (트레이스 계약은 완전히 동일)
```

핵심은 **TOGSim이 두 경로를 구분하지 못한다**는 점입니다. 트레이스 생산자의 형태가 같으므로 하드웨어 모델·DRAM·NoC·L2는 한 줄도 손대지 않았습니다.

### 항목별 대조

| | 기존 MLIR 경로 | 신규 Triton 경로 |
|---|---|---|
| 커널을 만드는 주체 | op별 MLIR 템플릿 (직접 작성) | Inductor의 Triton 코드젠 |
| **커널 하나의 의미** | **루프 네스트 전체** | **타일 하나** |
| grid | 루프 네스트에서 읽어냄 | 커널 밖에 있음 → `WorkItem`이 합성 |
| lowering | `PyTorchSimFrontend/mlir/` | NPU lowering pass (subprocess) |
| 융합 | 템플릿 + `codegen_compiler_optimization` | Inductor가 이미 한 것을 물려받음 |
| op 커버리지 | gemm, conv×4, sdpa, sort, cat, maxpool, bmm | elementwise + 그 융합 |
| functional | `FunctionalSimulator.run_spike` | lowering pass 의 stage 6 (`tnpu.spike`) |
| timing | `trace.so` + `trace_cycles.tsv` → TOGSim | **동일** |
| 타일 cycle 실측 | gem5 | **동일** (`build_tog` sample 모드 공유) |
| DMA | 비동기 + `togsim.wait` 배리어 | **동기만** |
| 동적 shape | 트레이스 경로 미지원 (PR #269 진행 중) | 동작 |

### 이 대조가 말해주는 것

**Triton 경로가 앞선 곳 — 동적 shape.** 기존 경로의 C++ 트레이스는 `trace_to_tilegraph(..., nullptr, 0)`으로 shape 인자를 아예 넘기지 않아 shape마다 트레이스를 다시 만들어야 하고, 그걸 푸는 작업이 PR #269로 아직 열려 있습니다. Triton 경로는 `shape_args`를 통해 트레이스 하나가 모든 shape을 섬깁니다.

**기존 경로가 앞선 곳 — op 커버리지와 DMA 겹침.** 템플릿 9종 대 elementwise, 그리고 비동기 DMA 유무. 후자가 4절 사이클 격차의 원인입니다.

**바뀌지 않은 것 — TOGSim 전체.** 하드웨어 설정, gem5 샘플링 방식, 트레이스 계약. 두 경로는 같은 시뮬레이터를 먹입니다.

---

## 2. 파이프라인

```
torch.compile
  └ TritonNPUScheduling.define_kernel                     scheduling.py
       │  Inductor 가 만든 triton 소스 텍스트 + 수집한 메타데이터
       ▼
    triton_npu_compile(src, meta, name)                   codecache.py
       │  KernelSpec 생성                                 kernel_spec.py
       │    - 블록 크기를 constexpr 로 고정
       │    - 인자 역할(in/out/inout) · dtype · numel
       │    - grid, 사용자 스칼라 값
       ▼
    NPU lowering pass   (subprocess)                      tnpu_bridge.py
       │   담당 이정민 — 범위 밖. 여기서 한 일은 이 단계를
       │   호출하고 그 결과를 아래 스택에 물린 부분.
       │
       │  1 ttir      triton 커널 → Triton IR
       │  2 ttshared  → linalg / memref / scf.for      (triton-shared)
       │  3 adapt     tts 레벨 백엔드 6개 패스          (tnpu/passes/)
       │  4 lower     vcix → gemmini DMA → LLVM
       │  5 binary    mlir-translate → llc → RISC-V ELF
       ▼
    TritonNPULauncher.__call__                            codecache.py
       │
       ├ functional   텐서 → runtime/*.raw → Spike → 텐서  functional.py
       │                lowering pass 의 stage 6 (tnpu.spike) 재사용
       │
       └ timing       04-custom.mlir                       timing.py
                        ├ build_tog sample → gem5 → 타일 cycle 실측
                        └ build_skeleton → trace.so + trace_cycles.tsv
                                              → TOGSim
```

**LLVM 이음매.** lowering pass는 stock LLVM 23을, PyTorchSim은 PSAL LLVM 20을 씁니다. `mlir`이 namespace 패키지라 한 인터프리터에 공존할 수 없어, 두 쪽은 **텍스트 MLIR을 주고받는 subprocess**로 갈라져 있습니다.

**lowering pass에 필요했던 진입점 3개.** 패스 로직을 고치는 것이 아니라 바깥에서 호출할 수 있게 여는 변경입니다 ([triton-npu#1](https://github.com/PSAL-POSTECH/triton-npu/pull/1)).

| 훅 | 왜 필요했나 |
|---|---|
| `tnpu.cycle` | 타일 하나만 gem5로 재려면 DMA를 지운 1-program 바이너리가 필요 |
| `dram_arg` | TOG 빌더가 DMA의 DRAM 쪽이 어느 커널 인자인지 알아야 함 |
| `tnpu.spike` | stage 6이 자체 생성 입력 대신 **호출자의 텐서**로 돌아야 함 |

---

## 3. 핵심 설계 문제: 커널 하나가 무엇을 뜻하는가

```
MLIR 경로   커널 = 루프 네스트 전체.  TOG 가 루프에서 work-item 을 읽어냄
Triton      커널 = 타일 하나.         grid 는 커널 밖, launch 가 쥐고 있음
```

이식의 본질적 어려움은 여기 하나로 모입니다. 그런데 TOGSim의 트레이스 계약(`docs/design/togsim_cpp_trace.md` §9.1/§9.3)이 이미 둘을 구분하고 있었습니다:

- `togsim_kernel_tile(ctx, iv, n)` — work-item 하나
- `togsim_kernel(ctx, shape_args, n)` — 병렬 영역의 열거

Triton 커널 본문은 전자에 대응하므로 **후자를 합성해서 씌우면** 계약을 그대로 만족합니다.

### grid 를 outer loop 으로 되세우기

Triton 쪽에서 grid 는 커널이 아니라 **KernelSpec 에 붙어 있습니다.** `kernel_spec.grid_of(meta)` 가 Inductor 의 numel 과 고정한 블록 크기로부터 축별 ceil-div 를 계산해 `grid=(8,)` 같은 값을 spec 에 적고, 커널 본문은 그 중 자기 몫이 몇 번째인지를 `pidX/Y/Z` 인자로 받을 뿐입니다.

기존 PyTorchSim 은 정반대를 기대합니다. `build_tog` 는 **역할 속성이 붙은 최상위 루프**를 TOG 의 루트로 잡습니다:

```python
_LOOP_ROLE_ATTRS = ("outer_loop", "accumulation_loop", "inner_loop")
roots = [op for op in block.operations
         if op.operation.name == "affine.for" and _has_loop_role(op)]
```

즉 work-item 을 **루프에서 읽어냅니다.** 그런데 Triton 커널에는 그 루프가 없습니다 — grid 로 흩어져 있으니까요. 루프가 없으면 루트도 없고, TOG 가 비게 됩니다.

그래서 spec 의 grid 를 다시 루프로 세웁니다. `_materialize_grid_loop` 이 하는 일입니다:

```
들어올 때   func @k(..., %pidX: i32, ...)          <- 타일 하나. 루프 없음
              body(%pidX)

나갈 때     scf.for %p = 0 to G {                  {outer_loop = true}
              body(<%pidX 를 index_cast %p 로 치환>)
            }
```

`WorkItem(parallel_args, grid)` 이 **어느 인자가 program id 인지**와 **축이 몇 개인지**를 들고 있습니다. 패스는 축마다 루프를 하나씩 중첩하고, 본문을 그 안으로 옮기고, pid 인자의 모든 사용처를 루프 유도변수로 바꿉니다. 마지막에 `outer_loop` 속성을 답니다 — **`build_tog` 가 찾는 바로 그 표식**이고, 이게 붙어야 합성한 루프가 TOG 의 루트가 됩니다.

(축별 범위는 `WorkItem` 이 들고 있을 수도, 런타임으로 미룰 수도 있습니다. timing 경로는 후자를 씁니다 — 바로 아래.)

결과적으로 Triton 이 grid 로 표현한 것과 PyTorchSim 이 outer loop 로 표현한 것이 같은 것을 가리키게 되고, 그 뒤 파이프라인(`build_skeleton` → `trace.so` → TOGSim)은 MLIR 경로와 한 글자도 다르지 않게 흘러갑니다.

### 동적 shape이 여기서 나옵니다

`_materialize_grid_loop`은 축 **개수**만 컴파일에 박고 **범위**는 `shape_args`에서 읽습니다.

```
컴파일 시   축이 몇 개인지만 안다   →  루프 네스트 골격 생성
런타임      실제 numel 로 grid 계산 →  trace_shape.txt 로 전달
            TOGSim 이 build_trace_tilegraph 에서 읽어 shape_args 로 주입
```

측정: `dynamic=True`로 n=1024 → grid 8, n=4096 → grid 32. 트레이스 재생성 없음.

다차원 grid(Triton 제약상 최대 3D)도 지원합니다. 구현 중 두 번 틀렸고 둘 다 rank ≥ 2에서만 드러났습니다 — 종료자가 있는 블록 끝에 삽입하는 문제, bound를 루프 뒤에 만들어 dominance를 깨는 문제. 그래서 테스트가 생성된 C++가 아니라 **MLIR 모듈 자체를 verify**합니다.

---

## 4. 측정 결과

### functional

| 커널 | 원소 | max abs error |
|---|---:|---:|
| `x + y` | 1024 | 0.0 |
| `(x + y) * 2 - x` (Inductor가 단일 커널로 융합) | 1024 | 0.0 |

### timing

| 항목 | 값 | 확인 내용 |
|---|---:|---|
| 타일 compute (gem5) | 19–21 | 마커 사이 `numCycles` 실측. placeholder 아님 |
| TOGSim 총계 | 650 | DRAM 트래픽 8192 B = 8 work-item × 2 load × 512 B, 정확히 일치 |
| 기존 MLIR 경로 (동일 연산) | 251 | 같은 자릿수 |

650 대 251은 모델 오류가 아닙니다. **lowering pass가 동기 DMA만 내보내기 때문**입니다 — 생성된 IR에 `togsim.transfer` 3개, `togsim.wait` **0개**. work-item 안에서 load → compute → store가 직렬화되어 TOGSim이 겹칠 것이 없습니다. 기존 경로는 `togsim.wait` → `togsim.memory_barrier` 태그 슬롯 기계를 갖추고 있습니다.

### 도중에 찾은 버그: 인자 한 칸 밀림

functional 배선은 배관 작업일 줄 알았는데 첫 실행에서 **1024개 중 896개가 틀렸습니다.** `pid_x=0` 블록만 맞고 나머지 7개는 전부 0.

```
lowered MLIR   @k(%arg0..2: memref<*xf32>   in_ptr0, in_ptr1, out_ptr0
                  %arg3: i32                xnumel      <- 사용자 스칼라
                  %arg4,5,6: i32            gridX,Y,Z
                  %arg7,8,9: i32            pidX,Y,Z )

wrapper 호출   k(1,&d_in_ptr0, 1,&d_in_ptr1, 1,&d_out_ptr0, 8,1,1, pid_x,pid_y,pid_z)
                                                           +---- i32 6개뿐 ----+
                                                                xnumel 누락
```

triton-shared는 사용자 스칼라를 자기 grid/pid 인자 **앞에** 둡니다. wrapper는 이를 `spec.extra["scalar_args"]`에서 읽는데 우리가 생성하는 spec에는 `extra`가 없었습니다. 인자가 밀려 `pidX`가 `pid_y`(grid 루프가 절대 바꾸지 않는 값)를 받았고, program 0이 8번 돈 셈이 됐습니다.

**틀린 값이 쓰레기가 아니라 0으로 나온 점**이 고약합니다. 쓰레기값이면 즉시 눈에 띄지만 0은 그럴듯해 보입니다. timing 경로는 인자 위치를 lowered MLIR 시그니처에서 직접 읽어 애초에 정확했고, 그래서 functional을 붙이기 전까지 드러나지 않았습니다.

---

## 5. 다음 작업 — 기존 경로 커버리지까지 검증

확인된 것은 elementwise와 그 융합뿐입니다. 다음 작업은 기능을 더 얹는 것이 아니라, **기존 MLIR 경로를 지탱하는 op 테스트 스위트를 Triton 경로로 그대로 돌려 어디까지 가는지 확인하는 것**입니다. 대상은 `tests/ops/` 아래 이미 있는 것들이고(`elementwise`, `reduce`, `gemm`, `conv`, `attention`, `view`, `sort`, `fusion`, `misc`), MLIR 경로가 통과하는 범위가 목표선입니다.

### 1차 실측

대표 케이스를 `TORCHSIM_TRITON_CODEGEN=1`로 돌린 결과입니다. **경로 진입** 열은 Triton 경로를 실제로 탔는지(작업 디렉터리 생성 여부)를 뜻합니다 — 이걸 보지 않으면 Inductor가 extern으로 뺀 것을 통과로 착각합니다.

| 케이스 | 경로 진입 | 결과 |
|---|---|---|
| `x + y` | 예 | 값 일치 |
| `(x+y)*2 - x` (융합) | 예 | 값 일치 |
| `x.t() + 1` | 예 | **값 틀림 — 4030/4096** |
| `relu` | 예 | 중단: `SpecIncomplete: triton_helpers.maximum` |
| `softmax` | 예 | 중단: `SpecIncomplete: triton_helpers.max2` |
| `exp` | 예 | 중단: lowering pass 실패 |
| `sum(dim=1)` | 예 | 중단: lowering pass 실패 (`bank_vectorize`) |
| `cat` | 예 | 중단: lowering pass 실패 |
| `a @ b` | **아니오** | Inductor가 `aten.mm` extern으로 처리 — 경로에 도달하지 않음 |

### 우선순위

**1. non-contiguous 텐서.** 값이 조용히 틀리는 유일한 항목이라 최우선입니다. `x.t() + 1`의 출력이 정확히 `x + 1`이고 — transpose가 통째로 무시되고 — 아무 예외도 나지 않습니다. 원인은 lowering pass가 아니라 **이식 쪽**입니다:

```
Inductor 가 낸 커널   tmp0 = tl.load(in_ptr0 + x0)     <- 인덱스가 항등.
                     transpose 를 인덱스 식이 아니라
                     출력 버퍼의 stride (1,64) 로 접었음

functional.py        write_inputs  t.contiguous()...tofile()  <- 저장 순서를 재배열
                     read_outputs  t.copy_(flat.view_as(t))   <- 논리 순서로 되씀
```

둘 다 **논리 순서와 저장 순서가 같다**고 가정합니다. contiguous 텐서에서만 참이고, 지금까지 통과한 것이 정확히 그 집합입니다. 저장 순서 기준으로 읽고 쓰도록 고치고 비-contiguous 케이스를 테스트에 넣어야 합니다.

**2. `triton_helpers` 벤더링.** `relu`, `softmax`, `clamp`, `max`, `min` 등 상당수가 여기서 막힙니다. 모듈이 torch 안에 있고 lowering pass 쪽 venv에는 없습니다. `strip_for_tnpu`가 어떤 헬퍼인지 이름을 대고 멈추므로, 필요한 것만 최소로 벤더링하면 커버리지가 한 번에 크게 늘어납니다.

**3. lowering pass 쪽 실패 (`exp`, `cat`, reduction).** 담당(이정민)과 나눌 부분입니다. reduction만 원인이 파악돼 있습니다 — `tt.reduce(axis=1)`이 `linalg.transpose` + `linalg.reduce`가 되고, 스크래치패드가 레인 뱅킹되어 축소 축이 레인 안에 머물러야 하는데 그렇지 못해 `bank_vectorize`가 거부합니다. `exp`와 `cat`은 미확인.

**4. matmul 경로 진입.** 지금은 Inductor가 `aten.mm` extern으로 빼서 Triton 경로를 타지 않습니다. 값은 맞게 나오지만 시뮬레이터를 거치지 않은 값입니다. Triton 템플릿을 쓰게 하려면 `max_autotune` 계열 설정이 필요하고, 그래야 systolic array 경로를 볼 수 있습니다.

**5. double buffering.** 커버리지가 아니라 정확도 항목 — 4절의 251 대 650 격차. lowering pass가 비동기 DMA와 `togsim.wait`를 내야 하고, 기존 경로에 이미 있는 기계를 옮기는 일입니다.

### 회귀 방지

`tests/system/test_triton_codegen.py`가 현재 경계를 못박고 있습니다. reduction은 **거부되는 동안 통과**하도록 되어 있어서 컴파일에 성공하면 테스트가 실패합니다 — 레인 경로가 생겼거나(그럼 체크를 지우면 됨), 하드웨어가 하지 않을 연산을 시뮬레이션하고 있다는 뜻이기 때문입니다. 위 항목이 하나씩 풀릴 때마다 이 방식으로 경계를 옮겨 적으면 됩니다.

---

모듈별 동작과 사용법은 [`triton_backend/README.md`](triton_backend/README.md)에, 이식 작업 본체는 [PyTorchSim#305](https://github.com/PSAL-POSTECH/PyTorchSim/pull/305)에 있습니다.

측정 환경: torch 2.10.0+cpu / triton 3.6.0, `systolic_ws_128x128_c1_simple_noc_tpuv3.yml`, `vpu_num_lanes` 128. 기존 MLIR 경로는 `tests/ops/elementwise/test_add.py` 통과로 회귀 없음 확인.
