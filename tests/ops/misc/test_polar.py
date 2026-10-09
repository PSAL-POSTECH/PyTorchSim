import os
import sys
import torch
import torch._dynamo
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def _rope_like(a):
    # DeepSeek-V2's RoPE: freqs arrives as a TRANSPOSED view out of a bmm, and
    # polar is called on it. The transpose is the whole point of the test --
    # on a contiguous input nothing here can go wrong.
    freqs = a.transpose(1, 2)
    return torch.view_as_real(torch.polar(torch.ones_like(freqs), freqs))


def test_polar_on_transposed(device, n=32):
    # aten::polar is CompositeExplicitAutograd, so it is registered for this
    # backend too and beats the boxed CPU fallback: the composite runs here,
    # allocates the result ITSELF (contiguous), and only then calls
    # aten::polar.out, which is what falls back. Inductor's meta models what a
    # real backend does -- TensorIterator carries the operands' permutation
    # into the output -- so the two disagree:
    #
    #     assert_size_stride(buf14, (1, 32, 32), (1024, 1, 32),
    #                        'torch.ops.aten.polar.default')
    #     AssertionError: expected size 32==32, stride 32==1 at dim=1
    #
    # PyTorchSimFrontend/rewrite_fx_graph.py takes the extern call out
    # instead of arguing about the layout: the pair is built with ops this
    # backend compiles and view_as_complex names it, which Inductor already
    # treats as a view over real data.
    x = torch.randn(1, n, n)
    cpu_out = _rope_like(x)

    opt_fn = torch.compile(dynamic=False)(_rope_like)
    npu_out = opt_fn(x.to(device=device))

    test_result("polar on a transposed view", npu_out, cpu_out)


def test_polar_contiguous(device, n=32):
    # The ordinary case, which passed before the decomposition too: it is here
    # so a future rewrite that only handles the transposed shape is caught.
    def f(a):
        return torch.view_as_real(torch.polar(torch.ones_like(a), a))

    x = torch.randn(1, n, n)
    cpu_out = f(x)
    npu_out = torch.compile(dynamic=False)(f)(x.to(device=device))
    test_result("polar on a contiguous tile", npu_out, cpu_out)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_polar_on_transposed(device)
    test_polar_contiguous(device)
