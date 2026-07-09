# 조사 과제: conv2d가 같은 크기 matmul보다 메모리를 훨씬 많이 쓰고 느린 이유

## 0. 배경과 용어 (먼저 읽을 것)

TOGSim은 PyTorchSim의 사이클 시뮬레이터다. 커널 하나를 시뮬레이션하는 과정은 두 단계다.

1. **그래프 구축 단계**: 컴파일된 producer 라이브러리(`trace.so`)를 실행한다. 이 라이브러리가
   `togsim_dma()`, `togsim_compute()` 같은 콜백을 순서대로 호출해 명령어 스트림을 흘려보내고,
   TOGSim은 그것을 받아 `TileGraph`를 메모리에 만든다.
   `TileGraph`는 `Instruction` 객체들과 그 사이 의존성 간선으로 이루어진 그래프다.
2. **시뮬레이션 단계**: 만들어진 `TileGraph`를 사이클 단위로 실행해 총 사이클 수를 낸다.

용어 정의:
- **producer 콜백 호출 1건** = 명령어 후보 1개. 코드에서는 `TraceRec` 구조체다.
- **`Instruction`** = `TileGraph`에 실제로 들어가는 명령어 객체. `sizeof(Instruction) == 520` 바이트.
- **의존성 간선(dep edge)** = "명령어 A가 끝나야 B가 시작한다"는 관계. 구현이
  `std::set<std::shared_ptr<Instruction>>`이라 간선 하나가 `std::set` 노드 = 48바이트를 쓴다.
- **dispatch 단위** = producer가 `togsim_dispatch()`를 한 번 호출할 때마다 생기는 독립 실행 단위.
  `TileGraph`에서는 `Tile` 하나에 대응한다. 커널 하나가 dispatch를 여러 개 낸다.

이 조사에서 쓰는 커널 4개 (전부 8x8 systolic array 설정, `vpu_num_lanes=8`):

| 별칭 | 실제 연산 | 대응하는 GEMM (M, N, K) | 상태 |
|------|-----------|------------------------|------|
| `CI36` | `F.conv2d(x[2,128,14,14], w[512,128,7,7], stride=1, padding=3)` | M=392, N=512, K=6272 | **메모리 초과로 죽음** |
| `MM36` | `torch.mm(a[392,6272], b[6272,512])` | 동일 | 정상 완료 |
| `CM16` | `F.conv2d(x[1,128,16,16], w[64,128,7,7], stride=1, padding=3)` | M=256, N=64, K=6272 | 정상 완료 |
| `MM16` | `torch.mm(a[256,6272], b[6272,64])` | 동일 | 정상 완료 |

conv를 GEMM으로 보는 방식: `M = batch * H_out * W_out`, `N = C_out`, `K = C_in * kh * kw`.
즉 `CI36`과 `MM36`은 **수학적으로 완전히 같은 GEMM**이다.

**문제**: CI(GitHub Actions)의 `tests/ops/conv/test_conv2d.py` 중 `CI36`에 해당하는 케이스가
8x8 설정에서 TOGSim 프로세스가 SIGKILL(메모리 초과)로 죽는다. 죽기까지 5시간 40분 걸렸다.

## 1. 이미 측정으로 확정된 사실 (다시 검증할 필요 없음)

이 브랜치에 이미 들어 있는 개선 2개 (둘 다 사이클 수 불변 확인):
- producer 스트림을 통째로 `std::vector`에 담아두던 것을 없애고 흘려보내며 바로 `TileGraph`를
  만들도록 변경 → 최대 메모리 28.5% 감소
- 누적(accumulator) 버퍼에 의존성을 걸 때 매번 전체를 재탐색하던 O(K^2) 루프를 O(K)로 수정
  → 한 케이스에서 탐색 횟수 1,258,840,576 → 50,176

**메모리 공식** (여러 케이스에서 오차 3% 이내로 검증됨):

    최대 메모리 ≈ (Instruction 개수 × 520바이트) + (의존성 간선 개수 × 48바이트) + 기본값(약 23.5MB)

`Instruction`이 520바이트인 이유: 종류와 무관하게 `std::vector` 7개 + `std::string` 2개 +
의존성 `std::set` 2개를 항상 들고 있다. 그런데 전체 명령어의 92~97%는 연산(COMPUTE) 명령어이고,
그 필드들은 DMA 명령어에만 쓰여서 대부분 빈 껍데기다.

그리고:
- **최대 메모리는 "그래프 구축 단계"에서 이미 다 찍힌다.** 시뮬레이션 단계에서는 거의 늘지 않는다.
- **그래프 구축 시간은 producer 콜백 호출 수에 정비례한다.** conv, matmul 둘 다 콜백 50만 건당
  0.2~0.3초로 일정하다. (즉 구축 자체에 제곱 복잡도 같은 것은 없다.)
- **사이클 수는 정상이다.** 이론값 `MAC수 / (8*8*2 arrays)`와 일치한다.

## 2. 문제의 핵심 관측

같은 GEMM(M=392, N=512, K=6272)인데:

| | 구축 시간 | 최대 메모리 | Instruction 수 | 의존성 간선 수 | dispatch 수 |
|---|---|---|---|---|---|
| `MM36` (matmul) | 2.8초 | 1,752MB | 2,521,360 | 7,476,232 | 8 (완료) |
| `CI36` (conv) | 400초에도 미완 | 2.9GB 넘고 증가 중 | 5,491,592 넘고 증가 중 | 10,650,895 넘음 | 16 넘고 증가 중 |

그런데 **작은 쌍에서는 conv가 오히려 가볍다** (같은 GEMM M=256, N=64, K=6272):

| | 최대 메모리 | Instruction 수 |
|---|---|---|
| `CM16` (conv) | 192MB | 232,072 |
| `MM16` (matmul) | 232MB | 302,112 |

**dispatch 하나당 Instruction 수는 conv 약 343,000 / matmul 약 315,000으로 거의 같다.**
다른 것은 오직 **dispatch 개수**다. matmul은 8개에서 끝나는데 conv는 16개를 넘겨도 계속 늘어난다.

`Instruction 총수 = dispatch 개수 × (dispatch당 Instruction 수)`이고, `Instruction 총수 × 520바이트`가
곧 메모리다. 따라서:

> conv의 메모리 초과와 느린 구축은 뿌리가 하나로 보인다.
> **conv가 `togsim_dispatch()`를 훨씬 많이 호출하고, 호출할 때마다 명령어 뭉치를 다시 만들어낸다.**

(단, 이건 아직 **추론**이다. `CI36`의 최종 숫자를 못 재서 확정하지 못했다.)

## 3. 해야 할 일

### (1) `CI36`의 최종 숫자를 측정한다
메모리 큰 장비(32~64GB 권장)에서 그래프 구축을 **끝까지** 돌려서 보고할 것:
최종 dispatch 개수 / Instruction 개수 / 의존성 간선 개수 / 최대 메모리 / 구축에 걸린 시간.
(원래 개발 박스는 3~6GB 지점에서 커널 OOM killer에 죽었다.)

### (2) conv는 왜 dispatch를 그렇게 많이 만드는가
dispatch 개수를 정하는 것은 TOGSim이 아니라 `trace.so`를 찍어내는 **프론트엔드 코드 생성기**다.
어떤 루프를 `togsim_dispatch()`로 감싸는지 확인할 것 (`build_skeleton` 부근, 그리고 conv 템플릿과
matmul 템플릿의 차이). conv 쪽이 왜 더 잘게 감싸는가?

### (3) CPU 70초 멈춤이 실제 현상인가
한 번은 콜백 50만 건 구간 하나에서 CPU 시간이 70초 튀었다(양옆 구간은 0.3초). 재실행에서는
재현되지 않았고, 그 직전 15분 평균 부하(load average)가 10이었다. 즉 **다른 워크로드와의 경합이나
커널 메모리 회수** 때문일 가능성이 크고 코드 문제가 아닐 수 있다. 계측이 wall / user / sys 시간을
모두 찍으므로 이것으로 판별할 것:
- user 시간 급증 → 우리 코드의 계산
- sys 시간 급증 → 커널의 메모리 회수 (메모리 압박)
- wall 시간만 급증, CPU 시간 평탄 → 다른 프로세스와의 경합 (우리 문제 아님)

## 4. 실행 방법

전제: PyTorchSim 개발 환경(파이썬 + torch + 컴파일 툴체인)이 갖춰진 서버.
CI가 쓰는 도커 이미지를 쓰면 가장 확실하다.

```bash
git clone https://github.com/PSAL-POSTECH/PyTorchSim.git
cd PyTorchSim
git checkout togsim/graph-build-cost-instrumented   # 계측이 이미 들어있는 브랜치
git submodule update --init --recursive             # TOGSim/extern/* 필요

# 1) TOGSim 빌드.  빌드 디렉토리는 반드시 로컬 디스크에.
#    (NFS 위에서 빌드하지 말 것. /tmp이 tmpfs(RAM)인 서버면 /tmp도 피할 것.)
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}
mkdir -p /local/tgbuild && cd /local/tgbuild
cmake /path/to/PyTorchSim/TOGSim -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j8 Simulator
# 참고: 개발 박스에서는 conan이 깨져 있어서 기존 conanbuildinfo.cmake를 TOGSim/build/ 에
#      복사해 재사용해야 했다. 새 서버에서 conan이 정상이면 conan install 로 만들면 된다.

# 2) trace.so 4개 생성 (TOGSim 바이너리 없이 파이썬 환경만 있으면 된다)
cd /path/to/PyTorchSim
source .envrc
bash tools/graph_build_cost/gen_traces.sh

# 3) 본 실험: 그래프 구축만 측정 (시뮬레이션은 건너뜀)
SIM=/local/tgbuild/bin/Simulator bash tools/graph_build_cost/run.sh MM36 CI36
# 작은 쌍(빠르게 끝남, 위 표의 값을 재현해 계측이 정상인지 확인용):
SIM=/local/tgbuild/bin/Simulator bash tools/graph_build_cost/run.sh CM16 MM16
```

`TOGSIM_BUILD_ONLY=1`(run.sh가 자동으로 설정)이면 `TileGraph` 구축이 끝나는 즉시 프로세스가
종료하므로, 시뮬레이션 없이 구축 비용만 잰다.

계측이 stderr에 찍는 것:
- `[PROG]` : 콜백 50만 건마다 → `rec`(누적 콜백 수), `wall`/`user`/`sys` 초,
  `inst`(Instruction 수), `edges`(간선 수), `maxw`(한 버퍼에 쌓인 생산자 최대 개수),
  `wi`(지금까지 본 dispatch 수)
- `[WI]`   : 구축이 끝났을 때의 최종 합계

## 5. 보고할 것

- `CI36`의 최종 dispatch 개수 / Instruction 개수 / 간선 개수 / 최대 메모리 / 구축 시간
- 콜백 50만 건당 wall·user·sys 시간이 끝까지 일정한지 (즉 코드 레벨 멈춤이 있는지 없는지)
- conv와 matmul의 `togsim_dispatch()` 호출 규칙 차이, 그리고 conv 쪽이 왜 더 잘게 나뉘는지

## 6. 주의사항

- 측정한 것만 보고할 것. 사실과 추론을 명시적으로 구분할 것.
- **작은 케이스로 큰 케이스를 예단하지 말 것.** conv는 작을 때는 matmul보다 가볍고,
  M/N이 커질 때만 역전된다.
- 아래 가설들은 이미 측정으로 **반박됐다**. 반복하지 말 것:
  - "conv의 그래프 구축이 제곱 복잡도다" → 구축 시간은 콜백 수에 정비례함이 측정됨
  - "conv가 weight preload 명령어를 dispatch마다 중복 생성한다" → matmul이 오히려 4배 더 냄
    (`MM16` 100,368개 vs `CM16` 25,092개)
  - "구축 단계가 전체 시간을 지배한다" → 완주 가능한 케이스에서 구축은 0.2~0.4초이고
    전체 시간의 99%는 시뮬레이션 단계다
- 이 브랜치의 계측 커밋은 **머지하면 안 된다**. 계측 없는 깨끗한 수정본은 `togsim/graph-build-cost`.
