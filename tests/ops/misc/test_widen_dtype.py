import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def _widen(device, name, src_dtype, dst_dtype, lo, hi):
    # Widening conversions lower to vsext.vf*/vzext.vf* (VI_VV_EXT). A Spike bug
    # wrote every lane's result to lane 1 (vu_idx dropped), zeroing the rest; this
    # guards against that regression. Signed / uint8<128 only, so the sign-extension
    # is correct independent of the separate uint8->int8 dtype issue (#238).
    a = torch.randint(lo, hi, (128, 128), dtype=src_dtype)
    fn = lambda a: a.to(dst_dtype)
    res = torch.compile(dynamic=False)(fn)(a.to(device=device))
    out = fn(a)
    test_result(name, res, out)


def test_widen(device):
    _widen(device, "int8->int16",    torch.int8,  torch.int16,  -128, 128)
    _widen(device, "int8->int32",    torch.int8,  torch.int32,  -128, 128)
    _widen(device, "int16->int32",   torch.int16, torch.int32, -1000, 1000)
    _widen(device, "uint8->int32",   torch.uint8, torch.int32,     0, 128)
    _widen(device, "uint8->float32", torch.uint8, torch.float32,   0, 128)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_widen(device)
