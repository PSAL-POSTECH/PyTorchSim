# fused kernel 추가하기

[English](fused_kernel.md)

fused kernel 은 Inductor 가 만들어 줄 커널 대신, 직접 작성한 Triton 커널로 npu 에서 실행하는
연산입니다. topk, grouped_mm, SDPA 의 fused prefill/decode 가 그렇습니다. 각각은
`pytorchsim/frontend/dispatcher/` 안의 `CustomFusedKernel` 하위 클래스입니다. 등록은 기반 클래스가
모두 같은 방식으로 해 주므로, 작성자는 커널과 그 커널이 언제 적용되는지만 적으면 됩니다.

## 먼저: 정말 필요한지 확인

| 연산 | 할 일 |
|---|---|
| Inductor 가 lowering 하고, 생성한 Triton 을 PyTorchSim-Triton-Backend 가 컴파일한다 | 없음. 대부분의 연산이 여기에 해당합니다. |
| npu 커널이 없지만 다른 aten 연산으로 표현할 수 있다 (histc, polar, 복소수 연산) | `pytorchsim/frontend/inductor/post_grad_pass.py` 의 재작성 |
| Inductor 가 만드는 커널이 하드웨어가 실제로 돌리는 커널과 다르다 (하나로 합친 SDPA, MoE 의 grouped GEMM, 레지스터 안에서 끝나는 topk) | fused kernel. 이 문서의 대상입니다. |

## 호출이 가는 길

`install()` 은 클래스를 `takes` 에 적은 각 연산의 npu 커널로 등록합니다. 등록 위치는
`AutogradPrivateUse1` 과 `PrivateUse1` 두 곳입니다. `inference_mode` 에서는 autograd 를 건너뛰므로
호출이 `PrivateUse1` 로 들어오기 때문입니다. 그 뒤 모든 호출은 아래 경로 중 하나로 갑니다.

```
npu 위의 aten 연산
 ├─ torch.compile 밖                            → fallback()      "fallback" 으로 셈
 ├─ 컴파일 중, plan() 이 Decline 을 돌려줌        → fallback()      이유를 로그로 남김
 │  또는 기울기가 필요한데 backward 를 안 적음
 └─ 컴파일 중, plan() 이 블록을 돌려줌            → triton_op `name` → launch()   launch 마다 셈
```

npu 에서 Triton 커널은 컴파일된 그래프 안에서만 launch 됩니다. 그래서 eager 호출은 항상 fallback
으로 갑니다. 컴파일된 그래프 안에서 도는 fallback 은 Inductor 의 커널이 되며, 로그에는 남지만
횟수에는 들어가지 않습니다.

## 클래스

```python
class MyOp(CustomFusedKernel):
    name = "torchsim::my_op"                  # 컴파일된 그래프가 부르는 triton_op
    takes = (torch.ops.aten.my_op.default,)   # 이 클래스가 npu 커널이 되는 연산

    def kernels(self): ...                    # launch() 가 launch 할 수 있는 @triton.jit 커널들
    def plan(self, *args): ...                # 블록 (dict), 또는 Decline("이유")
    def launch(self, plan, *args): ...        # 출력을 만들고 wrap_triton 으로 launch
    def fallback(self, *args): ...            # 같은 결과를 aten 연산으로
```

| 멤버 | 필수 | 설명 |
|---|---|---|
| `name` | 예 | 새로 만드는 연산의 qualified name, `"torchsim::<op>"`. 새 연산의 이름을 짓는 자리라서 문자열입니다. |
| `takes` | 예 | 대신 받을 연산들. `OpOverload` 객체로 적어서, 이름이 틀리면 import 시점에 바로 실패하게 합니다. install 뒤에야 정의되는 연산만 qualified name 문자열로 적습니다 (예: `"transformers::grouped_mm_fallback"`. transformers 를 import 해야 생깁니다). |
| `kernels()` | 예 | `@triton.jit` 커널들의 튜플. launch 마다 `calls` 에 그 커널 이름으로 셉니다. 커널을 클래스 아래에 정의할 수 있도록 메서드로 둡니다. |
| `plan(*args)` | 예 | 커널이 이 호출을 맡을지 정하는 유일한 곳. `launch` 가 읽을 블록 dict 를 돌려주거나, `Decline("이유")` 를 돌려줍니다. |
| `launch(plan, *args)` | 예 | 출력을 할당하고 커널을 launch 합니다. `plan` 뒤 인자들의 타입 표기가 그대로 triton_op 의 스키마가 됩니다. |
| `fallback(*args)` | 예 | `launch` 가 돌려주는 것과 같은 결과를 aten 연산으로 계산합니다. 미분 가능해야 합니다. |
| `arguments(op, *args, **kwargs)` | 아니오 | `op` 의 호출을 triton_op 인자로 바꿉니다. 기본값을 확정하고 값을 정규화합니다 (예: 음수 `dim`). 기본 동작은 위치 인자를 그대로 넘기는 것입니다. |
| `outputs(op, result, *args)` | 아니오 | 연산이 커널보다 더 많은 결과를 돌려줄 때, `launch`(또는 `fallback`)의 결과를 연산의 결과로 바꿉니다. SDPA 의 overrideable 연산은 9개를 돌려줍니다. |
| `backward(ctx, *grads)`, `setup_context(ctx, inputs, output)` | 아니오 | 없으면 기울기가 필요한 호출은 커널을 거절하고 fallback 으로 갑니다. 있으면 커널로 갑니다. |

`plan`, `launch`, `fallback` 은 모두 `arguments` 가 돌려준 인자를 같은 순서로 받습니다.

## 단계별로

### 1. 파일 하나: `pytorchsim/frontend/dispatcher/<op>.py`

클래스를 맨 위에, 커널과 helper 는 그 아래에 둡니다. 실제 예시 중 가장 짧은 것은
[topk.py](../pytorchsim/frontend/dispatcher/topk.py) 입니다. 아래는 행 단위 연산
`mylib::row_scale(Tensor x, float alpha) -> Tensor` 로 보인 전체 모양입니다. 이 연산은 여러분의
라이브러리가 정의한다고 가정합니다.

```python
import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.fused_kernel import (CustomFusedKernel, Decline,
                                                         fits_scratchpad, next_pow2)


class RowScale(CustomFusedKernel):
    name = "torchsim::row_scale"
    takes = (torch.ops.mylib.row_scale.default,)

    def kernels(self):
        return (_row_scale_kernel,)

    def plan(self, x, alpha):
        if not x.is_floating_point():
            return Decline(f"{x.dtype} is not a floating dtype")
        ep = next_pow2(x.shape[-1])
        xblock = int(config.CONFIG_MACHINE.vector_lanes)
        while xblock > 1 and not fits_scratchpad(_tiles(xblock, ep, x.element_size())):
            xblock //= 2
        if not fits_scratchpad(_tiles(xblock, ep, x.element_size())):
            return Decline(f"a row of {x.shape[-1]} overruns the scratchpad budget")
        return {"XBLOCK": xblock, "EP": ep}

    def launch(self, plan, x: torch.Tensor, alpha: float) -> torch.Tensor:
        E = x.shape[-1]
        rows = x.numel() // E
        o = torch.empty_like(x)
        wrap_triton(_row_scale_kernel)[(triton.cdiv(rows, plan["XBLOCK"]), 1, 1)](
            x.contiguous(), o, rows, E, plan["EP"], plan["XBLOCK"], alpha)
        return o

    def fallback(self, x, alpha):
        return x * alpha


ROW_SCALE = RowScale()


def install():
    ROW_SCALE.install()


@triton.jit
def _row_scale_kernel(x_ptr, o_ptr, rows, E: tl.constexpr, EP: tl.constexpr,
                      XBLOCK: tl.constexpr, ALPHA: tl.constexpr):
    r = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    c = tl.arange(0, EP)[None, :]
    live = (r < rows) & (c < E)
    x = tl.load(x_ptr + r * E + c, mask=live, other=0.0)
    tl.store(o_ptr + r * E + c, x * ALPHA, mask=live)


def _tiles(xblock, ep, esize):
    """The [rows, cols, bytes] tiles one program holds: the row and its scaled copy."""
    return [(xblock, ep, esize), (xblock, ep, esize)]
```

### 2. 등록

[dispatcher/\_\_init\_\_.py](../pytorchsim/frontend/dispatcher/__init__.py) 의 `install()` 에 더합니다.

```python
from . import attention, grouped_mm, row_scale, topk
...
row_scale.install()
```

### 3. 값만이 아니라 경로를 테스트

fallback 은 커널과 같은 답을 냅니다. 그래서 값만 비교하는 테스트는, `plan` 이 조용히 모든 호출을
fallback 으로 보내도 통과합니다. `calls` 도 같이 확인하세요.
[test_topk.py](../tests/ops/sort/test_topk.py) 의 형태를 따르고, 케이스마다 가야 할 경로를 적습니다.

| 케이스 | 기대하는 `calls` |
|---|---|
| 커널이 맡는 컴파일 호출 | `{"_row_scale_kernel": 1}` |
| `plan` 이 거절하는 컴파일 호출 | `{}` |
| eager 호출 | `{"fallback": 1}` |
| 입력이 requires_grad 인 컴파일 호출 | backward 가 없으면 `{}`, 있으면 커널 횟수. 기울기도 확인합니다. |
| `inference_mode` 에서 컴파일 | `{"_row_scale_kernel": 1}` |
| 같은 컴파일 호출을 두 번 | `{"_row_scale_kernel": 2}` (launch 를 세고, trace 는 세지 않습니다) |

케이스마다 먼저 `ROW_SCALE.calls` 를 비우고 `torch._dynamo.reset()` 을 부릅니다. 테스트에서 커널에
`codecache.observe_launches` 를 걸지 마세요. 커널 하나에는 launch 감시자가 하나만 걸리므로,
테스트가 건 감시자가 `calls` 를 세는 감시자를 덮어씁니다.

npu 에서 스크립트로 실행합니다: `python tests/ops/<family>/test_<op>.py [case ...]`.

## 규칙

이 저장소 메인테이너의 리뷰 기준을 따른 것입니다.

- **판단은 `plan` 에서만.** 같은 판단을 `launch` 나 다른 곳에 반복하지 말고, 특정 shape 에 거는
  분기를 두지 마세요. 거절할 때는 이유를 적습니다. 그 이유가 로그 한 줄이 됩니다.
- **블록은 머신에서 구합니다.** `config.CONFIG_MACHINE` (`vector_lanes`, `vlen_bits`, `spad_budget`) 과
  `fits_scratchpad(tiles)` 를 씁니다. `BLOCK = 128` 같은 하드코딩은 하지 않습니다.
- **커널 인자는 위치로 넘깁니다** (`wrap_triton(kernel)[grid](...)`). Inductor 는 키워드 인자를
  backend 옵션일 수도 있는 것으로 읽습니다.
- **커널은 특정 grid 에 기대지 않습니다.** 같은 컴파일된 커널 객체가 여러 grid 로 launch 됩니다.
- **레인마다 다른 값은 벡터로 둡니다.** 스칼라나 `vector<1xT>` 로 줄이지 마세요. 모든 레인이 스칼라
  레지스터 파일 하나를 공유하므로 레인 0 의 값만 남습니다.
- **`fallback` 은 `launch` 와 같아야 합니다.** 출력, dtype, 인덱스 규칙 (예: 같은 값이면 어느 인덱스가
  이기는지) 이 같고, 미분 가능해야 합니다.
- **PyTorch 공개 API 만 씁니다.** `torch.library`, `triton_op`, `wrap_triton` 이면 충분합니다.
  `torch._...` 내부 함수를 쓰거나 전역 decomposition 테이블을 바꾸지 마세요.

## 추가 패턴

- **커널이 처리하지 못하는 인자** (mask, dropout, bias, 다른 출력 dtype): triton_op 인자로 두고
  `plan` 이 거절하게 합니다. 그러면 모든 호출을 `fallback` 이 맡을 수 있습니다.
  [attention.py](../pytorchsim/frontend/dispatcher/attention.py) 와
  [grouped_mm.py](../pytorchsim/frontend/dispatcher/grouped_mm.py) 를 보세요.
- **시그니처가 맞는 여러 연산을 한 클래스로**: 모두 `takes` 에 적습니다. `GroupedMM` 은
  `aten._grouped_mm` 과 transformers 의 `grouped_mm_fallback` 을 함께 받습니다.
- **한 연산에 커널 여러 개**: 모두 `kernels()` 에 적습니다. 고른 커널은 plan 에 담고, `launch` 에서
  `plan["kernel"] is _decode_kernel` 처럼 객체로 고릅니다. 이름 문자열로 고르지 않습니다.
- **커널이 계산하는 것보다 출력이 많을 때**: 나머지는 `outputs()` 에서 만듭니다. 예를 들어 SDPA 의
  `lse` 와 backward 에서만 쓰는 출력들은 빈 텐서입니다.
