import os
import sys
import torch
import torch._dynamo
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

# REGISTERED ON PURPOSE, and this file exists because of what that used to do.
#
# `eager_to_compile` gives an op an npu kernel whose body torch.compiles the op.
# That is safe exactly while Inductor can generate code for it. When it cannot,
# it emits the aten call as a fallback, and an ATen implementation that reaches
# the same overload comes straight back into the wrapper:
#
#     aten::mul.out on npu  -> wrapper -> torch.compile
#     Inductor              -> cannot codegen complex, emits aten.mul.Tensor
#     ATen mul.Tensor       -> allocates an out, calls aten::mul.out
#     aten::mul.out on npu  -> wrapper again
#
#     RecursionError: maximum recursion depth exceeded
#
# DeepSeek-V2's RoPE closed that loop, because it multiplies complex tensors.
# It does not close any more: rewrite_fx_graph.py rewrites the complex
# multiply into real arithmetic, so Inductor generates a kernel and no fallback
# is emitted. This test is that sentence, executable -- it recurses on a tree
# without that pass and passes with it.
torch.npu.register_eager_to_compile(["aten::mul.out"])


def _rope(x, a):
    freqs = a.transpose(1, 2)
    fc = torch.polar(torch.ones_like(freqs), freqs)
    xc = torch.view_as_complex(x.reshape(1, 32, 32, 2))
    return torch.view_as_real(xc * fc).flatten(-2)


def test_registered_mul_out_does_not_recurse(device):
    x, a = torch.randn(1, 32, 64), torch.randn(1, 32, 32)
    cpu_out = _rope(x, a)

    # The failure this pins is a RecursionError, which torch.compile surfaces
    # wrapped; catching it explicitly says what happened rather than letting a
    # 1000-frame traceback speak.
    try:
        npu_out = torch.compile(_rope, dynamic=False)(x.to(device=device),
                                                      a.to(device=device))
    except RecursionError:
        raise AssertionError(
            "aten::mul.out recursed through its own eager_to_compile wrapper -- "
            "the complex multiply became a fallback again")

    test_result("eager_to_compile mul.out + complex", npu_out, cpu_out)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_registered_mul_out_does_not_recurse(device)
