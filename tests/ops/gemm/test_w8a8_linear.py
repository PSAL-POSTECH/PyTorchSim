"""One Linear quantized W8A8 with torchao's quantize_ (int8 weights, int8 activations, int32 accumulation) on
the npu, against the same quantized Linear on the CPU. Cases: dynamic (a scale per activation row), static
(one calibrated activation scale) and rownorm (bf16, its output read by a row norm); [case ...], all when none.
Without torchao, W8A8Linear does the same math."""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"),
                                "tests"))
from result_check import check_result

device = torch.device("npu:0")
M, K, N = 64, 512, 256


def _symmetric(t, scale):
    """`t` divided by `scale`, rounded and clamped to int8."""
    return torch.round(t / scale).clamp(-128, 127).to(torch.int8)


class W8A8Linear(nn.Module):
    """torchao's W8A8 Linear without torchao: an int8 weight with a scale per output channel, the input int8 per
    row (act_scale None) or by act_scale, the int8 matmul in int32 as torchao emits it off cuda, then rescaled."""

    def __init__(self, linear, act_scale=None):
        super().__init__()
        w = linear.weight.detach()
        w_scale = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer("w_q", _symmetric(w, w_scale))
        self.register_buffer("w_scale", w_scale.reshape(1, -1))
        self.register_buffer("bias", linear.bias.detach().clone())
        self.register_buffer("act_scale", act_scale)

    def forward(self, x):
        s = self.act_scale if self.act_scale is not None else x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
        acc = torch.mm(_symmetric(x, s).to(torch.int32), self.w_q.t().to(torch.int32))
        return acc.to(torch.float32) * s * self.w_scale + self.bias


def _quantized(name, linear, x):
    """`linear` quantized for case `name` (static calibrates its scale on `x`): torchao's, else W8A8Linear."""
    act_scale = (x.abs().amax() / 127).reshape(1, 1) if name == "static" else None
    try:
        from torchao.quantization import (Int8DynamicActivationInt8WeightConfig,
                                          Int8StaticActivationInt8WeightConfig, quantize_)
        from torchao.quantization.granularity import PerTensor
    except ImportError:
        return W8A8Linear(linear, act_scale), "W8A8Linear"
    config = (Int8DynamicActivationInt8WeightConfig() if act_scale is None
              else Int8StaticActivationInt8WeightConfig(act_quant_scale=act_scale, granularity=PerTensor()))
    model = nn.Sequential(linear)
    quantize_(model, config)
    return model, "torchao"


def run_case(name):
    torch.manual_seed(0)
    linear, x = nn.Linear(K, N).eval(), torch.randn(M, K)
    step = (x.abs().max() / 127 * linear.weight.detach().abs().max()).item()
    with torch.no_grad():
        fp32 = linear(x)
        model, by = _quantized(name, linear, x)
        ref = model(x)
        torch._dynamo.reset()
        out = torch.compile(model.to(device), dynamic=False)(x.to(device)).cpu()
    print(f"W8A8Linear {name} ({by}): max |npu - cpu int8| {(out - ref).abs().max().item():.3g}, "
          f"max |cpu int8 - fp32| {(ref - fp32).abs().max().item():.3g}, one step {step:.3g}")
    check_result(f"W8A8Linear {name}", out, ref, atol=step)
    return True


def run_rownorm(name):
    """A bf16 dynamic W8A8 Linear whose output a row RMS norm reads, as Qwen3's q_norm does: the npu result against
    the CPU's, and its dequantized result stored by the int8 matmul's epilogue, so the wrapper holds no int32 buffer."""
    from torch._inductor.utils import run_and_get_code

    torch.manual_seed(0)
    linear, x = nn.Linear(K, N, bias=False).to(torch.bfloat16).eval(), torch.randn(M, K, dtype=torch.bfloat16)
    inner, by = _quantized("dynamic", linear, x)
    if by != "torchao":
        print(f"W8A8Linear {name}: skipped, it needs torchao's bf16 dequantization")
        return True
    model = _ThenRowNorm(inner)
    with torch.no_grad():
        ref = model(x)
        torch._dynamo.reset()
        out, codes = run_and_get_code(torch.compile(model.to(device), dynamic=False), x.to(device))
    int32 = [l.strip() for c in codes for l in c.splitlines() if "empty_strided" in l and "torch.int32" in l]
    print(f"W8A8Linear {name}: max |npu - cpu int8| {(out.cpu().float() - ref.float()).abs().max().item():.3g}, "
          f"int32 buffers {int32}")
    check_result(f"W8A8Linear {name}", out.cpu(), ref, rtol=5e-2, atol=5e-2)
    check_result(f"W8A8Linear {name} (no int32 buffer)", torch.tensor([not int32]), torch.tensor([True]))
    return True


class _ThenRowNorm(nn.Module):
    """`inner`, then each output row divided by its root mean square taken in float32."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return y * torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(y.dtype)


CASES = ("dynamic", "static", "rownorm")


if __name__ == "__main__":
    for n in sys.argv[1:] or CASES:
        (run_rownorm if n == "rownorm" else run_case)(n)
