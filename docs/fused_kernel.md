# Adding a fused kernel

[한국어](fused_kernel.ko.md)

A fused kernel is an op the npu runs as a Triton kernel you write yourself, in place of the kernels
Inductor would generate for it. topk, grouped_mm and SDPA's fused prefill and decode are fused
kernels. Each one is a `CustomFusedKernel` subclass in `pytorchsim/frontend/dispatcher/`. The base
class registers every one the same way, so you write the kernel and when it applies, nothing else.

## First, check that you need one

| The op | What to do |
|---|---|
| Inductor lowers it, and PyTorchSim-Triton-Backend compiles the Triton it generates | Nothing. Most ops are this. |
| It has no npu kernel, and other aten ops can express it (histc, polar, complex arithmetic) | A rewrite in `pytorchsim/frontend/inductor/postgrad/` |
| Inductor's kernels for it are not the kernel the hardware runs (SDPA as one fused kernel, MoE's grouped GEMM, topk in registers) | A fused kernel: this guide |

## How a call is routed

`install()` registers your class as the npu kernel of each op in `takes`, at `AutogradPrivateUse1`
and at `PrivateUse1` (a call made under `inference_mode` skips autograd and lands at the second).
Every call then takes one of these routes:

```
aten op on the npu
 ├─ outside torch.compile                      → fallback()      counted "fallback"
 ├─ compiling, plan() returns a Decline,       → fallback()      logged with the reason
 │  or it needs a gradient and you wrote no backward
 └─ compiling, plan() returns blocks           → triton_op `name` → launch()   each launch counted
```

A Triton kernel only launches on the npu inside a compiled graph, so an eager call always runs the
fallback. A fallback that runs inside a compiled graph becomes Inductor's own kernels. It is
logged, but not counted.

## The class

```python
class MyOp(CustomFusedKernel):
    name = "torchsim::my_op"                  # the triton_op a compiled graph calls
    takes = (torch.ops.aten.my_op.default,)   # the ops whose npu kernel this is

    def kernels(self): ...                    # the @triton.jit kernels launch() may launch
    def plan(self, *args): ...                # blocks (a dict), or Decline("why")
    def launch(self, plan, *args): ...        # allocate outputs, launch through wrap_triton
    def fallback(self, *args): ...            # the same outputs, as aten ops
```

| Member | Required | What it is |
|---|---|---|
| `name` | yes | The new op's qualified name, `"torchsim::<op>"`. It is a string because you are naming a new op. |
| `takes` | yes | The ops it replaces, as `OpOverload` objects, so a misspelt op fails at import. Use a qualified-name string only for an op that is defined after install (for example `"transformers::grouped_mm_fallback"`, which exists only once transformers is imported). |
| `kernels()` | yes | A tuple of the `@triton.jit` kernels. Each launch counts in `calls` under the kernel's own name. It is a method so the kernels can be defined below the class. |
| `plan(*args)` | yes | The one place that decides whether the kernel takes the call. It returns a dict of blocks that `launch` reads, or `Decline("reason")`. |
| `launch(plan, *args)` | yes | It allocates the outputs and launches the kernel. Its type annotations after `plan` are the triton_op's schema. |
| `fallback(*args)` | yes | It returns what `launch` returns, but computed with aten ops. It must be differentiable. |
| `arguments(op, *args, **kwargs)` | no | Turns a call of `op` into the triton_op's arguments: it fixes defaults and normalizes values (for example a negative `dim`). By default the positional arguments pass through unchanged. |
| `outputs(op, result, *args)` | no | Turns `launch`'s (or `fallback`'s) result into the op's outputs, when the op returns more than the kernel. SDPA's overrideable op returns nine. |
| `backward(ctx, *grads)`, `setup_context(ctx, inputs, output)` | no | Without them, a call that needs a gradient declines the kernel and runs the fallback. With them, it runs the kernel. |

`plan`, `launch` and `fallback` all take what `arguments` returns, in the same order.

## Step by step

### 1. One file: `pytorchsim/frontend/dispatcher/<op>.py`

Put the class first and the kernels and helpers below it. [topk.py](../pytorchsim/frontend/dispatcher/topk.py)
is the shortest real example. Here is the whole shape on a row-wise op, `mylib::row_scale(Tensor x,
float alpha) -> Tensor`, which a library of yours defines:

```python
import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import (
    CustomFusedKernel, Decline, fits_scratchpad, next_pow2)


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

### 2. Register it

Add it to `install()` in [dispatcher/\_\_init\_\_.py](../pytorchsim/frontend/dispatcher/__init__.py):

```python
from . import attention, grouped_mm, row_scale, topk
...
row_scale.install()
```

### 3. Test the routes, not only the values

The fallback gives the same answer as the kernel. So a test that only compares values still passes
when `plan` quietly sends every call to the fallback. Check `calls` as well. Copy the shape of
[test_topk.py](../tests/ops/sort/test_topk.py), and give each case the route it must take:

| Case | Expected `calls` |
|---|---|
| a compiled call the kernel takes | `{"_row_scale_kernel": 1}` |
| a compiled call `plan` declines | `{}` |
| an eager call | `{"fallback": 1}` |
| a compiled call whose input requires grad | `{}` with no backward; the kernel's count with one. Check the gradient too. |
| compiled under `inference_mode` | `{"_row_scale_kernel": 1}` |
| the same compiled call run twice | `{"_row_scale_kernel": 2}` (launches are counted, traces are not) |

Clear `ROW_SCALE.calls` and call `torch._dynamo.reset()` before each case. Do not call
`codecache.observe_launches` on your kernel in a test. Each kernel has one launch observer, so a
test's own observer replaces the one that counts `calls`.

Run it on the npu as a script: `python tests/ops/<family>/test_<op>.py [case ...]`.

## Rules

These follow the maintainer's review rules for this repo.

- **`plan` is the only gate.** Do not repeat the decision in `launch` or anywhere else, and do not
  key it on particular shapes. When it declines, say why: the reason is the log line.
- **Derive blocks from the machine.** Use `config.CONFIG_MACHINE` (`vector_lanes`, `vlen_bits`,
  `spad_budget`) and `fits_scratchpad(tiles)`. Do not hard-code `BLOCK = 128`.
- **Pass the kernel's arguments by position** in `wrap_triton(kernel)[grid](...)`. Inductor reads
  a keyword argument as a possible backend option.
- **The kernel must not depend on one grid.** The same compiled kernel object is launched with
  different grids.
- **Lane-varying values stay in vectors.** Do not reduce them to a scalar or a `vector<1xT>`: all
  lanes share one scalar register file, so only lane 0's value survives.
- **`fallback` must match `launch`**: the same outputs, dtypes and index rules (for example, which
  index wins on ties), and differentiable.
- **Use public PyTorch only.** `torch.library`, `triton_op` and `wrap_triton` are all you need. Do
  not use `torch._...` internals or edit global decomposition tables.

## Extra patterns

- **Arguments the kernel cannot handle** (a mask, dropout, a bias, another output dtype): make them
  triton_op arguments and have `plan` decline them. `fallback` then covers every call. See
  [attention.py](../pytorchsim/frontend/dispatcher/attention.py) and
  [grouped_mm.py](../pytorchsim/frontend/dispatcher/grouped_mm.py).
- **One class for several ops** with compatible signatures: list them all in `takes`.
  `GroupedMM` takes both `aten._grouped_mm` and transformers' `grouped_mm_fallback`.
- **Several kernels for one op**: list them all in `kernels()`. Put the chosen kernel in the plan
  and pick it in `launch` by identity, as in `plan["kernel"] is _decode_kernel`, not by a name
  string.
- **More outputs than the kernel computes**: build the rest in `outputs()`. For example, SDPA's
  `lse` and its other backward-only outputs are empty tensors.
