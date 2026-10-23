"""CustomFusedKernel's routing, on a toy op `torchsim_test::twice` (x * 2) whose plan takes rows of
a whole block and has no backward. Each case checks the value against x * 2 and the counts in
`calls`: a "_twice_kernel" count per launch, a "fallback" count per eager call.

  eager       outside torch.compile the fallback runs
  compiled    a compiled call the plan takes launches the kernel
  declined    a compiled call the plan declines runs the fallback compiled into the graph
  grad        a compiled call needing a gradient declines the kernel (no backward), and the
              gradient is the fallback's
  inference   a compiled call under inference_mode (PrivateUse1) launches the kernel

Usage: test_custom_fused_kernel.py [case ...]   (all cases when none is named)
"""
import sys

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from pytorchsim.frontend.dispatcher.custom_fused_kernel import CustomFusedKernel, Decline

device = torch.device("npu:0")
BLOCK = 128

torch.library.define("torchsim_test::twice", "(Tensor x) -> Tensor")


class Twice(CustomFusedKernel):
    name = "torchsim_test::twice_fused"
    takes = (torch.ops.torchsim_test.twice.default,)

    def kernels(self):
        return (_twice_kernel,)

    def plan(self, x):
        if x.numel() % BLOCK:
            return Decline(f"{x.numel()} is not a whole number of {BLOCK}")
        return {"BLOCK": BLOCK}

    def launch(self, plan, x: torch.Tensor) -> torch.Tensor:
        o = torch.empty_like(x)
        wrap_triton(_twice_kernel)[(x.numel() // plan["BLOCK"], 1, 1)](x.contiguous(), o,
                                                                       plan["BLOCK"])
        return o

    def fallback(self, x):
        return x * 2


@triton.jit
def _twice_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) * 2)


TWICE = Twice()
TWICE.install()

#  name: (want, numel, compiled, grad, inference_mode)
CASES = {
    "eager": ({"fallback": 1}, 4 * BLOCK, False, False, False),
    "compiled": ({"_twice_kernel": 1}, 4 * BLOCK, True, False, False),
    "declined": ({}, 4 * BLOCK + 3, True, False, False),
    "grad": ({}, 4 * BLOCK, True, True, False),
    "inference": ({"_twice_kernel": 1}, 4 * BLOCK, True, False, True),
}


def run_case(name):
    want, numel, compiled, grad, inference = CASES[name]
    x = torch.randn(numel).to(device).requires_grad_(grad)
    f = torch.ops.torchsim_test.twice
    if compiled:
        torch._dynamo.reset()
        f = torch.compile(f, dynamic=False)
    TWICE.calls.clear()
    with torch.inference_mode(inference):
        out = f(x)
    took = dict(TWICE.calls)
    ok = torch.equal(out.detach().cpu(), x.detach().cpu() * 2) and took == want
    if grad:
        out.sum().backward()
        ok = ok and torch.equal(x.grad.cpu(), torch.full((numel,), 2.0))
    print(f"{name}: calls={took} want={want} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    if not all([run_case(n) for n in names]):
        sys.exit(1)
    print("PASS")
