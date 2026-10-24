# post-grad 재작성 추가하기

[English](post_grad_rewrite.md)

post-grad 재작성은 npu 그래프 안의 연산 하나를 Inductor 가 lowering 할 수 있는 다른 aten 연산들로
바꿔서, 그 일이 장치에 남게 하는 것입니다. 재작성이 없으면 npu 커널이 없는 연산은 CPU 의 extern
aten 호출로 돌고, 시뮬레이터 밖으로 빠집니다. histc, 1 x K 행과 행렬의 곱, transposed convolution,
길이가 0 인 행렬곱, 복소수 연산이 재작성으로 처리됩니다. 모두 `pytorchsim/frontend/inductor/post_grad_pass/` 에 op 계열마다 모듈 하나씩(`histc.py`, `gemm.py`, `convolution.py`, `complex.py`) 있습니다. 연산 하나의 재작성은
데코레이터를 붙인 함수 하나이므로, 작성자는 패턴과 적용 조건과 대체 코드만 적으면 됩니다.

## 먼저: 정말 필요한지 확인

| 연산 | 할 일 |
|---|---|
| Inductor 가 lowering 하고, 생성한 Triton 을 PyTorchSim-Triton-Backend 가 컴파일한다 | 없음. 대부분의 연산이 여기에 해당합니다. |
| npu 커널이 없지만 다른 aten 연산으로 표현할 수 있다 (histc, polar, 복소수 연산) | 재작성. 이 문서의 대상입니다. |
| Inductor 가 만드는 커널이 하드웨어가 실제로 돌리는 커널과 다르다 (하나로 합친 SDPA, MoE 의 grouped GEMM, 레지스터 안에서 끝나는 topk) | fused kernel: [fused_kernel.ko.md](fused_kernel.ko.md) |

먼저 연산을 trace 해서 무엇이 pass 까지 오는지 보세요. pass 는 Inductor 의 decomposition 테이블을
거친 뒤의 그래프를 보므로, 아예 오지 않는 연산도 있습니다. 예를 들어 `torch.special.ndtr` 은 그때
이미 `erf` 와 산술 연산입니다.

## 노드가 처리되는 길

`registration.py` 는 `pytorchsim_post_grad_pass` 를 npu 의 `device_custom_pass` 로 등록합니다
(`register_backend_for_device`). 그래서 Inductor 는 이 pass 를 npu 그래프에서만 돌립니다. pass 는
그래프 패턴(`PATTERNS.apply(gm)`)을 먼저 돌리고, 그다음 `complex_real_pairs` 를 돌립니다.
모든 노드는 아래 경로 중 하나로 갑니다.

```
npu 그래프, Inductor 의 decomposition 뒤
 ├─ PATTERNS.apply(gm), 패턴이 가리키는 target 의 노드마다
 │   ├─ 입력이나 출력이 복소수                       → Inductor 의 matcher 가 건너뜀
 │   ├─ 패턴이 맞지 않음,                            → 그대로 둠             세지 않음
 │   │  또는 extra_check(match) 가 False
 │   └─ fn(node, *args, **kwargs) 가 돌려준 것이
 │       ├─ (replacement, example_args)             → match.replace_by_example   "rewritten" 으로 셈
 │       └─ Decline("이유")                          → 노드가 남음               로그를 남기고 "fallback" 으로 셈
 └─ complex_real_pairs, 복소수 값으로 이어진 복소수 노드 묶음마다
     ├─ 모든 연산에 _REAL_PAIR 규칙이 있음           → 실수 [..., 2] 쌍       "complex_real_pairs": "rewritten"
     └─ 그렇지 않음                                  → 묶음이 남음            로그를 남기고 "complex_real_pairs": "fallback"
```

남은 노드는 `fallback_route(target)` 가 말하는 곳에서 돌고, 로그 한 줄에 그 위치가 적힙니다.
Inductor 자체 커널이거나, 시뮬레이터가 보지 못하는 CPU 의 extern aten 호출입니다. pass 는 대체 코드를
그래프 자신의 fake mode 에서 trace 하므로, 테스트가 그래프를 직접 넘겨도 돕니다.

## 구성 요소

```python
@register_rewrite(CallFunction(aten.my_op.default, Arg(), Arg()), extra_check)  # 이 재작성이 맡는 노드
def my_op(node, x, dim):                                 # 맞은 노드와 그 인자
    def compute(x, *, dim): ...                          # 대체 코드, 평범한 torch 코드
    if fake_value(x) is None:
        return Decline("its input has no fake value")    # 노드는 남고, 이유가 로그에 남음
    return functools.partial(compute, dim=dim), [x]      # 대체 코드와 그 텐서 인자
```

| 구성 요소 | 설명 |
|---|---|
| `register_rewrite(pattern, extra_check)` | 데코레이터. 함수를 `PATTERNS`(Inductor 의 `PatternMatcherPass`)에 등록하고, 결과를 `CALLS` 에 함수 이름으로 셉니다. `extra_check` 의 기본값은 모든 match 를 받는 것입니다. 평범한 함수로 부르면 함수 하나를 여러 패턴에 등록합니다 (`gemm_zero_length`). |
| `pattern` | `torch._inductor.pattern_matcher` 의 Inductor 패턴 언어. `CallFunction(op, Arg(), ...)` 는 연산과 인자마다 `Arg()` 하나를 적습니다. `CallFunctionVarArgs(op)` 는 인자 개수와 상관없이 받으므로, 선택 인자가 빠질 수 있는 연산에 씁니다 (histc). |
| `extra_check(match)` | 재작성이 적용될 노드를 고릅니다. `False` 는 이 재작성이 맡을 노드가 아니라는 뜻이라, 로그도 횟수도 남지 않습니다. `match.args` 가 잡힌 인자이고, fake 값은 `fake_value` 로 읽습니다. |
| `fn(node, *args, **kwargs)` | 맞은 노드와 그 인자를 받습니다. `(replacement, example_args)` 를 돌려줍니다. `replacement` 는 Inductor 가 노드 자리에 trace 해 넣는 평범한 torch 코드이고 (`match.replace_by_example`), `example_args` 는 그 코드가 받는 그래프 노드입니다. 상수는 `functools.partial` 로 묶어서, 텐서 노드만 example 인자가 되게 합니다. |
| `Decline("이유")` | 다른 반환값. 재작성이 이 노드를 맡을 수 없다는 뜻입니다. 노드는 남고 (fallback), 로그에 이유와 노드가 대신 도는 곳이 적힙니다. fused kernel 의 `plan` 이 돌려주는 것과 같은 `Decline` 입니다. |
| `CALLS` | `{재작성 이름: {"rewritten": n, "fallback": n}}`, 노드마다 셉니다. 복소수 값은 `"complex_real_pairs"` 아래에 묶음마다 한 번 셉니다. |
| `_REAL_PAIR` | `{복소수 연산: 규칙}`. 규칙은 복소수 연산 하나를 실수 `[..., 2]` 쌍으로 적습니다. `aten.polar`, `view_as_complex`, `unsqueeze`, `mul` (`Tensor` 와 `Scalar`) 의 규칙이 있습니다. |

복소수 연산은 패턴으로 다루지 않습니다. Inductor 의 matcher 는 입력이나 출력이 복소수인 노드를
건너뛰므로 (`fallback_node_due_to_unsupported_type`), 복소수 연산에 건 패턴은 맞지 않습니다. 실수 쌍
pass 는 복소수 노드 묶음을 통째로 바꾸거나 전혀 바꾸지 않습니다. 규칙이 없는 연산이 하나라도 있으면
묶음은 그대로 남습니다.

## 단계별로

### 1. op 계열 모듈 안의 함수 하나

`post_grad_pass/` 안에서 그 op 계열의 모듈에 둡니다. 새 계열이면 모듈을 새로 만들고,
`post_grad_pass/__init__.py` 끝에서 다른 모듈들과 함께 등록 순서대로, `complex` 보다 앞에 import 합니다.
그 위에 섹션 주석 한 줄(`# logcumsumexp: ...`)을 달고, 패키지 docstring 의 목록에도 같은 순서로 한 줄을
더합니다. 모듈은 `register_rewrite` 와 `fake_value` 를 `pytorchsim.frontend.inductor.post_grad_pass` 에서
import 합니다. 둘 중 하나라도 빠지면
`tests/system/test_module_form.py` 가 실패합니다. 대체 코드는 `histc` 의 `counts` 처럼 재작성 함수 안에 두고,
확인과 반환은 그 뒤에 둡니다. 실제 예시 중 가장 짧은 것은 `histc` 입니다. 아래는 새로 만드는
예시인 `aten.logcumsumexp` 입니다. Inductor 는 이 연산을 fallback 으로만 lowering 하므로 npu 에서는
CPU 로 갑니다. 출력 하나하나는 prefix 에 대한 logsumexp 이고, `logsumexp` 는 lowering 됩니다.

```python
@register_rewrite(CallFunction(aten.logcumsumexp.default, Arg(), Arg()))
def logcumsumexp(node, x, dim):
    """aten.logcumsumexp as one logsumexp per prefix (prefix_logsumexp)."""

    def prefix_logsumexp(x, *, dim):
        """Output i is the logsumexp of x[..., :i+1] along dim: x against the prefix mask, the rest -inf, so
        each prefix takes its own max (one shift by the row max sends early, small terms to 0)."""
        n = x.shape[dim]
        x = x.movedim(dim, -1)
        i = torch.arange(n, device=x.device)
        prefix = i.reshape(-1, 1) >= i.reshape(1, -1)
        y = torch.where(prefix, x.unsqueeze(-2), float("-inf")).logsumexp(-1)
        return y.movedim(-1, dim)

    v = fake_value(x)
    if v is None:
        return Decline("its input has no fake value")
    if not v.is_floating_point():
        return Decline(f"{v.dtype} is not a floating dtype")
    return functools.partial(prefix_logsumexp, dim=dim), [x]
```

정수 입력은 aten 에 정수 logcumsumexp 가 없으므로 거절합니다. 노드는 남고 aten 과 똑같이 에러를
냅니다. 대체 코드는 aten 이 행마다 n 만큼 하는 일을 n² 만큼 합니다. 그 대신 시뮬레이터가 보지 못하는
CPU 호출이 아니라 장치의 벡터 연산이 되므로, PR 에 그렇게 적습니다.

### 2. 복소수 연산: `_REAL_PAIR` 에 규칙 하나

규칙은 `call` (복소수 노드 앞에 노드를 넣고 그 fake 값을 정합니다), 복소수 노드 `n`, `pair`
(지금까지 만든 각 복소수 입력의 실수 쌍) 를 받습니다. `n` 의 쌍을 담은 노드를 돌려줍니다. `-z` 는
이렇습니다.

```python
def _pair_of_neg(call, n, pair):
    """-z: the pair negated."""
    return call(aten.neg.default, pair[n.args[0]])


_REAL_PAIR = {
    ...
    aten.neg.default: _pair_of_neg,
}
```

`_REAL_PAIR` 에 있는 target 은 그 모든 호출을 처리할 수 있는 것으로 보므로, 규칙은 그 연산이 받는
모든 형태를 처리해야 합니다. `_pair_of_mul` 은 복소수 곱하기 복소수, 복소수 곱하기 실수, Python
복소수 스칼라를 모두 처리합니다. 규칙이 넣는 노드는 다시 decomposition 되지 않으므로, 각 노드에
그대로 Inductor lowering 이 있어야 합니다.

### 3. 값만이 아니라 그래프와 경로를 테스트

재작성이 없어도 노드는 fallback 으로 가서 CPU 에서 맞는 답을 냅니다. 그래서 값만 비교하는 테스트는
재작성이 한 번도 적용되지 않아도 통과합니다. 그래프와 `CALLS` 도 같이 확인하세요.
[test_complex_to_real.py](../tests/ops/misc/test_complex_to_real.py) 의 형태를 따릅니다. 함수를
Inductor 가 post-grad pass 에 넘기는 모양으로 trace 하고, pass 를 돌린 뒤 무엇이 남았는지 봅니다.

```python
gm = make_fx(f, decomposition_table=select_decomp_table(), tracing_mode="fake")(x)
CALLS.clear()
pytorchsim_post_grad_pass(gm)
assert not [n for n in gm.graph.nodes if n.target is aten.logcumsumexp.default]
assert not _unlowered(gm), f"no Inductor lowering for {_unlowered(gm)}"
check_result("logcumsumexp (rewritten graph)", gm(x), f(x))
assert CALLS == {"logcumsumexp": {"rewritten": 1}}, CALLS
```

`_unlowered(gm)` 는 Inductor 에 lowering 이 없는 연산을 나열합니다. 두 테스트 파일 어느 쪽에서든
복사해 쓰세요. `make_fx` 가 CPU 에서 그 연산을 trace 하지 못하면,
[test_histc.py](../tests/ops/misc/test_histc.py) 의 `_histc_graph` 처럼 fake 값을 넣은 노드 하나짜리
그래프를 직접 만듭니다. 케이스마다 가야 할 경로를 적습니다.

| 케이스 | 기대하는 `CALLS` |
|---|---|
| 호출 하나를 trace 한 그래프 | `{"logcumsumexp": {"rewritten": 1}}`. 연산이 사라지고, `_unlowered(gm)` 가 비어 있고, 값이 맞습니다. |
| 한 그래프에 호출 두 개 | `{"logcumsumexp": {"rewritten": 2}}` (노드를 셉니다) |
| 입력에 fake 값이 없는, 직접 만든 노드 | `{"logcumsumexp": {"fallback": 1}}`. 노드가 남습니다. |
| 정수 입력 | `{"logcumsumexp": {"fallback": 1}}`. 노드가 남습니다. |
| 경계 케이스: `-inf` prefix, float32 의 `[-200, 0]`, 양수와 음수 `dim` | 재작성되고, 값이 aten 과 같습니다 |
| 새 규칙이 있는 복소수 묶음 | `{"complex_real_pairs": {"rewritten": 1}}`, 복소수 노드가 남지 않습니다 |
| 규칙이 없는 연산이 있는 복소수 묶음 | `{"complex_real_pairs": {"fallback": 1}}`, 묶음이 그대로입니다 |
| `npu:0` 에서 `torch.compile(f)` | 값이 CPU 와 같습니다 |

케이스마다 먼저 `CALLS` 를 비우고, 컴파일하는 케이스 앞에서는 `torch._dynamo.reset()` 을 부릅니다.
pass 는 컴파일된 그래프마다 한 번 돌므로, `CALLS` 는 실행 횟수가 아니라 컴파일 횟수를 셉니다. 끝에서
끝까지 도는 케이스는 실제 커널이 계산할 때도 재작성이 맞는 답을 내는지 확인합니다. 재작성 자체를
고정하는 것은 그래프 케이스들입니다.

npu 에서 스크립트로 실행합니다: `python tests/ops/misc/test_<op>.py`.

## 규칙

이 저장소 메인테이너의 리뷰 기준을 따른 것입니다.

- **노드는 target 으로 판정합니다.** `n.target is aten.x.default` 로 비교하거나
  `callable(n.target)` 을 확인합니다. pass 에서도 테스트에서도 `node.op == "call_function"` 같은 op
  문자열 비교를 하지 마세요.
- **텍스트를 파싱하지 않습니다.** fake 값 (`fake_value`), target, 인자를 읽습니다. 출력한 그래프나 커널
  소스를 읽지 마세요.
- **조용한 fallback 을 두지 않습니다.** `fn` 이 노드를 맡을 수 없으면 이유를 담아 `Decline` 을
  돌려줍니다. 그 이유가 로그 한 줄이 됩니다. `extra_check` 는 이 재작성이 맡을 노드가 아닌 경우에만
  씁니다.
- **shape 특수 케이스를 넣지 않습니다.** 특정 shape 나 행 수로 노드를 고르지 말고, 그것을 되돌리는
  후속 예외도 만들지 마세요. `_is_row` 는 일반적인 형태 (모든 1 x K 행) 를 받습니다.
- **상수는 머신에서 구합니다.** 크기는 fake 값이나 `config.CONFIG_MACHINE` 에서 얻습니다.
  하드코딩하지 않습니다.
- **대체 코드는 연산과 같아야 합니다.** 값, dtype, 경계 규칙이 같아야 합니다. `histc` 는 aten 처럼
  위쪽 끝 값을 마지막 bin 에 넣고, `[min, max]` 밖의 값은 버리고, `min == max` 면 범위를 넓히고,
  입력의 dtype 으로 돌려줍니다. 이런 규칙마다 테스트를 둡니다.
- **공개 경로가 있으면 PyTorch 공개 API 를 씁니다.** `torch._inductor.pattern_matcher` 는 Inductor
  자체의 post-grad pass 들이 쓰는 방식이고, pass 는 trace 할 때 그래프의 fake mode 를
  `torch._inductor.virtualized.V.set_fake_mode` 로 정합니다. 둘 다 공개된 대응물이 없습니다. 그러므로 재작성을 더하는 PR 에는
  그렇게 적어야 합니다.
- **재작성은 `post_grad_pass/` 안에 둡니다.** pass 의 `uuid` 가 그 안의 모듈을 모두 해시하므로, 어느
  모듈을 고쳐도 캐시 키가 바뀝니다. 패키지 밖에 둔 재작성은 해시에 그 파일을 더해야 합니다. 그러지 않으면
  재작성 없이 만든 그래프를 컴파일이 재사용할 수 있습니다.

## 추가 패턴

- **선택 인자**: `CallFunctionVarArgs(op)` 를 쓰고 기본값은 `fn` 에서 채웁니다. `histc` 가 `bins`,
  `min`, `max` 를 이렇게 다룹니다.
- **여러 연산에 같은 조건**: `mm_row` (`aten.mm`) 와 `addmm_row`
  (`aten.addmm`) 는 `_is_row` 를 함께 씁니다. 이 함수는 두 연산 모두의 마지막 인자 두 개를
  읽습니다.
- **여러 연산에 같은 재작성**: `register_rewrite(pattern, check)(fn)` 를 반복문 안에서 부릅니다.
  `gemm_zero_length` 은 이렇게 `mm`, `bmm`, `addmm`, `baddbmm` 에 등록되고, 조건은 연산마다
  만들며 (`_is_zero_length(bias)`), `CALLS` 에는 한 이름으로 셉니다.
- **상수 인자에 거는 조건**: `convolution_transposed` 는 `transposed` 인자가 `True` 일 때만
  `aten.convolution` 을 받습니다 (`match.args[6] is True`).
- **인자마다 `Arg()` 하나씩.** matcher 는 같은 패턴 객체가 두 번 쓰이면 "같은 값이 두 번"으로
  읽습니다. 그래서 `CallFunction(op, *[Arg()] * 9)` 는 아무것도 잡지 못합니다 ("repeated pattern
  differs"). `convolution_transposed` 처럼 `CallFunction(op, *(Arg() for _ in range(9)))` 로 씁니다.
- **대체 코드 안의 상수**: `functools.partial` 로 묶습니다. `example_args` 에는 그래프 노드만 넣습니다.
