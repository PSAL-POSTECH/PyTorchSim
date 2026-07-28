import os
import sys
import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_view3D_2D(device, size=(16, 8, 16), t_x=0, t_y=1):
    def view3D_2D(a):
        return a.transpose(t_x, t_y).contiguous().view(-1, size[0] * size[2])
    torch.manual_seed(0)
    cpu_input = torch.randn(size)
    input = cpu_input.clone().to(device=device)
    opt_fn = torch.compile(dynamic=False)(view3D_2D)
    res = opt_fn(input)
    out = view3D_2D(cpu_input)
    test_result("view 3D->2D", res, out)

def test_view2D_3D(device, size=(512, 768), h=12, d_k=64):
    def view2D_3D(a):
        return a.view(-1, h, d_k).transpose(0, 1).contiguous()
    torch.manual_seed(0)
    cpu_input = torch.randn(size)
    input = cpu_input.clone().to(device=device)
    opt_fn = torch.compile(dynamic=False)(view2D_3D)
    res = opt_fn(input)
    out = view2D_3D(cpu_input)
    test_result("view 2D->3D", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_view3D_2D(device)
    test_view3D_2D(device, [12, 512, 64])
    test_view2D_3D(device, size=(512, 1024), h=16, d_k=64)

