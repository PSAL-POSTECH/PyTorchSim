# Inductor Triton 코드젠 경로를 PyTorchSim에 연결

`torch.compile`이 `npu:0`에서 PyTorchSim의 자체 MLIR 코드젠 대신 **Inductor의 Triton 백엔드**를 쓰고, **triton-npu(tnpu)**가 이를 RISC-V로 낮추는 두 번째 코드젠 경로.

**functional과 timing 양쪽이 연결되어 있고, 동적 shape도 처리됩니다.**

모듈별 동작과 사용법은 [`triton_backend/README.md`](triton_backend/README.md)에 있습니다.
이 문서는 기존 경로와의 대조, 설계 판단, 측정 결과를 다룹니다.

| | |
|---|---|
| functional | `x + y`, `(x+y)*2 - x` 모두 **max abs error 0.0** (1024 elements, Spike) |
| timing | TOGSim **650 cycles**, 타일 compute는 gem5 **19 cycles 실측** |
| 동적 shape | 트레이스 하나가 모든 shape을 섬김 — n=1024 → grid 8, n=4096 → grid 32, 재컴파일 없음 |
| 변경량 | PyTorchSim 18 commits / 23 files / +2209−39, tnpu 3 commits / +177−4 |
| CI | 전 잡 green (툴체인 빌드 + 값 검증 + 기존 경로 회귀 확인) |

---

## 1. 파이프라인

```
torch.compile
  └ TritonNPUScheduling.define_kernel                     scheduling.py
       │  Inductor 가 만든 triton 소스 텍스트 + 수집한 메타데이터
       ▼
    triton_npu_compile(src, meta, name)                   codecache.py
       │  tnpu KernelSpec 생성                            kernel_spec.py
       │    - 블록 크기를 constexpr 로 고정
       │    - 인자 역할(in/out/inout) · dtype · numel
       │    - grid, 사용자 스칼라 값
       ▼
    tnpu 파이프라인   (별도 인터프리터, subprocess)        tnpu_bridge.py
       │  1 ttir      triton 커널 → Triton IR
       │  2 ttshared  → linalg / memref / scf.for   (triton-shared)
       │  3 adapt     tts 레벨 백엔드 6개 패스        (tnpu/passes/)
       │  4 lower     vcix → gemmini DMA → LLVM
       │  5 binary    mlir-translate → llc → RISC-V ELF
       ▼
    TritonNPULauncher.__call__                            codecache.py
       │
       ├ functional   텐서 → runtime/*.raw → Spike → 텐서  functional.py
       │                tnpu stage 6 (tnpu.spike) 재사용
       │
       └ timing       04-custom.mlir                       timing.py
                        ├ build_tog sample → gem5 → 타일 cycle 실측
                        └ build_skeleton → trace.so + trace_cycles.tsv
                                              → TOGSim
```

**LLVM 이음매.** tnpu는 stock LLVM 23을, PyTorchSim은 PSAL LLVM 20을 씁니다. `mlir`이 namespace 패키지라 한 인터프리터에 공존할 수 없어, 두 쪽은 **텍스트 MLIR을 주고받는 subprocess**로 갈라져 있습니다.

---

## 2. 기존 MLIR 경로와의 차이

### 갈라지는 지점과 합쳐지는 지점

```
                    torch.compile / Inductor 스케줄
                              │
              ┌───────────────┴───────────────┐
              │                               │
        [기존] MLIR 경로                 [신규] Triton 경로
              │                               │
   Inductor 스케줄 → 손으로 쓴          Inductor 의 Triton 코드젠
   op별 MLIR 템플릿                     이 낸 커널 소스를 가로챔
   (gemm, conv, sdpa, sort,             (op별 템플릿 없음)
    cat, maxpool, bmm …)
              │                               │
   PyTorchSim mlir/ 패스               tnpu 패스 (subprocess)
   PSAL LLVM 20                        stock LLVM 23
              │                               │
              └───────────────┬───────────────┘
                              │
                    ▼ 여기서 다시 합류 ▼
              trace.so + trace_cycles.tsv
                      → TOGSim
              (트레이스 계약은 완전히 동일)
```

핵심은 **TOGSim이 두 경로를 구분하지 못한다**는 점입니다. 트레이스 생산자의 형태가 같으므로 하드웨어 모델·DRAM·NoC·L2는 손대지 않았습니다.

### 항목별 대조

| | 기존 MLIR 경로 | 신규 Triton 경로 |
|---|---|---|
| 커널을 만드는 주체 | PyTorchSim의 op별 MLIR 템플릿 | Inductor의 Triton 코드젠 |
| **커널 하나의 의미** | **루프 네스트 전체** | **타일 하나** |
| grid | 루프 네스트에서 읽어냄 | 커널 밖에 있음 → `WorkItem`이 합성 |
| lowering | `PyTorchSimFrontend/mlir/` (in-process) | tnpu (subprocess, LLVM 23) |
| 융합 | 템플릿과 `codegen_compiler_optimization` | Inductor가 이미 한 것을 물려받음 |
| op 커버리지 | gemm, conv×4, sdpa, sort, cat, maxpool, bmm | elementwise + 그 융합 |
| functional | `FunctionalSimulator.run_spike` | tnpu stage 6 (`tnpu.spike`) |
| timing | `trace.so` + `trace_cycles.tsv` → TOGSim | **동일** |
| 타일 cycle 실측 | gem5 | **동일** (`build_tog` sample 모드 공유) |
| DMA | 비동기 + `togsim.wait` 배리어 | **동기만** (`togsim.wait` 0개) |
| 동적 shape | 트레이스 경로는 아직 미지원 (PR #269 진행 중) | timing 경로에서 동작 |

### 이 대조가 말해주는 것

**Triton 경로가 앞선 곳** — 동적 shape. 기존 경로의 C++ 트레이스는 `trace_to_tilegraph(..., nullptr, 0)`으로 shape 인자를 아예 넘기지 않아 shape마다 트레이스를 다시 만들어야 하고, 그걸 푸는 작업이 PR #269로 아직 열려 있습니다. Triton 경로는 `shape_args`를 통해 **트레이스 하나가 모든 shape을 섬깁니다.**

**기존 경로가 앞선 곳** — op 커버리지와 DMA 겹침. 템플릿 9종 대 elementwise, 그리고 비동기 DMA 유무. 후자가 아래 사이클 격차의 원인입니다.

**바뀌지 않은 것** — TOGSim, 하드웨어 설정, gem5 샘플링 방식, 트레이스 계약. 두 경로는 같은 시뮬레이터를 먹입니다.

---

## 3. 핵심 설계 문제: 커널 하나가 무엇을 뜻하는가

```
MLIR 경로   커널 = 루프 네스트 전체.  TOG 가 루프에서 work-item 을 읽어냄
Triton      커널 = 타일 하나.         grid 는 커널 밖, launch 가 쥐고 있음
```

TOGSim의 트레이스 계약(`docs/design/togsim_cpp_trace.md` §9.1/§9.3)이 이미 이 둘을 구분합니다:

- `togsim_kernel_tile(ctx, iv, n)` — work-item 하나
- `togsim_kernel(ctx, shape_args, n)` — 병렬 영역의 열거

Triton 커널 본문은 전자에 대응하므로, **후자를 합성해서 씌우면** 계약을 그대로 만족합니다. 그 합성이 `lower_to_emitc.WorkItem` + `_materialize_grid_loop`입니다.

### 동적 shape이 여기서 나옵니다

`_materialize_grid_loop`은 축 **개수**만 컴파일에 박고, **범위**는 `shape_args`에서 읽습니다.

```
컴파일 시   축이 몇 개인지만 안다  →  루프 네스트 골격 생성
런타임      실제 numel 로 grid 계산 →  trace_shape.txt 로 전달
            TOGSim 이 build_trace_tilegraph 에서 읽어 shape_args 로 주입
```

측정: `dynamic=True`로 n=1024 → grid 8, n=4096 → grid 32. 트레이스 재생성 없음.

다차원 grid(Triton 제약상 최대 3D)도 지원합니다. 구현 중 두 번 틀렸고 둘 다 rank ≥ 2에서만 드러났습니다 — 종료자가 있는 블록 끝에 삽입하는 문제, 그리고 bound를 루프 뒤에 만들어 dominance를 깨는 문제. 그래서 테스트가 생성된 C++가 아니라 **MLIR 모듈 자체를 verify**합니다.

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

650 대 251은 모델 오류가 아닙니다. **tnpu가 동기 DMA만 내보내기 때문**입니다 — 생성된 IR에 `togsim.transfer` 3개, `togsim.wait` **0개**. work-item 안에서 load → compute → store가 직렬화되어 TOGSim이 겹칠 것이 없습니다. 기존 경로는 `togsim.wait` → `togsim.memory_barrier` 태그 슬롯 기계를 갖추고 있습니다.

---

## 5. 도중에 찾은 실제 버그

functional 배선은 배관 작업일 줄 알았는데, 첫 실행에서 **1024개 중 896개가 틀렸습니다.** `pid_x=0` 블록만 맞고 나머지 7개는 전부 0.

```
MLIR  func.func @k(%arg0..2: memref<*xf32>,   in_ptr0, in_ptr1, out_ptr0
                   %arg3: i32                 xnumel      <- 사용자 스칼라
                   %arg4,5,6: i32             gridX,Y,Z
                   %arg7,8,9: i32             pidX,Y,Z )

wrapper  k(1,&d_in_ptr0, 1,&d_in_ptr1, 1,&d_out_ptr0, 8, 1, 1, pid_x, pid_y, pid_z);
                                                      +------ i32 6개뿐 ------+
                                                             xnumel 누락
```

triton-shared는 사용자 스칼라를 자기 grid/pid 인자 **앞에** 둡니다. tnpu wrapper는 이를 `spec.extra["scalar_args"]`에서 읽는데, PyTorchSim이 생성하는 spec에는 `extra`가 아예 없었습니다. 인자가 한 칸씩 밀려 `pidX`가 `pid_y`(grid 루프가 절대 바꾸지 않는 값)를 받았고, program 0이 8번 돈 셈이 됐습니다.

**틀린 값이 쓰레기가 아니라 0으로 나온 점**이 고약합니다. 쓰레기값이면 즉시 눈에 띄지만 0은 그럴듯해 보입니다. timing 경로는 인자 위치를 lowered MLIR 시그니처에서 직접 읽어 애초에 정확했고, 그래서 functional을 붙이기 전까지 드러나지 않았습니다.

---

## 6. 일반성을 위해 되돌린 설계 둘

**DMA가 어느 인자에 속하는지 — 추론에서 선언으로.** 처음에는 TOG 빌더가 memref view 연산을 거꾸로 걸어 올라가 인자 인덱스를 추론했습니다. 아는 view 연산에 대해서만 맞는 방식이라, 생산자(tnpu)가 `dram_arg`를 직접 적어 내려보내도록 바꾸고 추론 코드를 삭제했습니다.

**grid — 컴파일 타임 상수에서 런타임 인자로.** 위 3절.

---

## 7. 현재 상태

| 기능 | 상태 | 내용 |
|---|---|---|
| elementwise + 융합 | 동작 | 값·사이클 모두 통과, CI 포함 |
| 동적 shape (timing) | 동작 | 트레이스 하나가 모든 shape |
| 다차원 grid | 동작 | 테스트가 IR과 dispatch 양쪽 검증 |
| 동적 shape (functional) | 제약 | 바이너리가 shape 특수화 → `ShapeMismatch`로 거부 |
| double buffering | 미착수 | tnpu가 동기 DMA만 발행. 251 vs 650의 주원인 |
| matmul timing | 미착수 | `build_tog`는 `vcix.iv` 이름으로 compute 노드를 찾는데 tnpu는 `llvm.riscv.sf.vc.*` 인트린식을 냄 |
| `triton_helpers` | 차단 | 모듈이 torch 안에 있고 tnpu venv에는 없음 |
| reduction | 차단 | tnpu 자체 문제 — 아래 |

**동적 shape의 한 가지 단서.** timing은 완전히 동작합니다. functional 바이너리는 tnpu가 grid·스칼라 값·memref extent를 전부 구워 넣어 shape 특수화되어 있어서, shape이 다른 launch를 `ShapeMismatch`로 **거부합니다** — 틀린 경계로 실행하는 대신. 사이클만 볼 때는 `pytorchsim_functional_mode: False`로 모든 shape을 돌릴 수 있습니다.

**reduction이 막힌 지점.** `tt.reduce(axis=1)`이 triton-shared를 지나면 `linalg.transpose permutation=[1,0]` + `linalg.reduce dimensions=[0]`가 됩니다. transpose는 `transpose-reduce-to-rank0` 여부와 무관하게 삽입됩니다(rank 2에서 동일함을 측정). stage 3의 다섯 패스는 통과하고 `bank_vectorize`가 거부합니다 — 스크래치패드가 **레인 뱅킹**되어 있어 축소되는 축이 레인 안에 머물러야 하는데, identity-elementwise가 아니고 스칼라 폴백은 뱅킹된 스크래치패드를 읽게 되기 때문입니다.

테스트(`check_reduction_is_refused`)가 이 경계를 못박습니다. **reduction이 컴파일에 성공하면 테스트가 실패합니다** — 레인 경로가 생겼거나(그럼 체크를 지우면 됨), 하드웨어가 하지 않을 연산을 시뮬레이션하고 있다는 뜻이기 때문입니다.

---

## 8. PR과 검증

| PR | 범위 | 상태 |
|---|---|---|
| [PyTorchSim #305](https://github.com/PSAL-POSTECH/PyTorchSim/pull/305) | 18 commits · 23 files · +2209/−39 | draft, mergeable, CI green |
| [triton-npu #1](https://github.com/PSAL-POSTECH/triton-npu/pull/1) | 3 commits · 5 files · +177/−4 | open |

CI(`.github/workflows/triton_npu.yml`)는 툴체인 레이어가 ~1.8 GiB라 본 CI와 분리:

```
Check tnpu access             success
Build tnpu toolchain image    success
Build app image on tnpu base  success
Inductor Triton route         success   <- test_triton_codegen.py (값 검증 포함)
MLIR route still passes       success   <- 기존 경로 회귀 없음
triton-npu baselines          success   <- doctor + add/mul/relu/gemm/bmm
```

**머지 순서.** `thirdparty/triton-npu.json`이 tnpu 커밋 `22df065`를 핀하는데, 이는 `feature/timing-form`에만 있고 `main`에는 없습니다. sha라 CI fetch는 되지만 #1이 리베이스 머지되면 뜹니다 — **#1 머지 → 핀을 main 커밋으로 재조정 → #305** 순서가 안전합니다.

---

## 9. 다음 우선순위

1. **double buffering** — tnpu가 비동기 DMA + `togsim.wait`를 내도록. 두 경로의 사이클 격차를 실제로 좁히는 유일한 항목이고, 기존 경로에 이미 있는 기계를 tnpu 쪽에 만드는 일입니다.
2. **shape 특수화 해소** — launch shape마다 재컴파일하거나, tnpu wrapper도 트레이스 생산자처럼 grid와 extent를 인자로 받게. 후자가 근본적.
3. **matmul timing** — `build_tog`가 vcix 인트린식을 인식하도록. systolic array 경로가 열립니다.
4. **reduction 레인 경로** — tnpu의 `bank_vectorize`에 reduction 추가, transpose를 `vlane_split_axis`로 흡수. 가장 큰 작업.

---

측정 환경: torch 2.10.0+cpu / triton 3.6.0, `systolic_ws_128x128_c1_simple_noc_tpuv3.yml`, `vpu_num_lanes` 128. 기존 MLIR 경로는 `tests/ops/elementwise/test_add.py` 통과로 회귀 없음 확인.
