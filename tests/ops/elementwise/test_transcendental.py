import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_tanh(device, size=(128, 128)):
    def tanh(a):
        return torch.tanh(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(tanh)
    res = opt_fn(x)
    out = tanh(x.cpu())
    test_result("Tanh", res, out)

def test_exp(device, size=(128, 128)):
    def exp(a):
        return torch.exp(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(exp)
    res = opt_fn(x)
    out = exp(x.cpu())
    test_result("Exp", res, out)

def test_erf(device, size=(128, 128)):
    def erf(a):
        return torch.erf(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(erf)
    res = opt_fn(x)
    out = erf(x.cpu())
    test_result("Erf", res, out)

def test_sin(device, size=(128, 128)):
    def sin(a):
        return torch.sin(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(sin)
    res = opt_fn(x)
    out = sin(x.cpu())
    test_result("Sin", res, out)

def test_cos(device, size=(128, 128)):
    def cos(a):
        return torch.cos(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(cos)
    res = opt_fn(x)
    out = cos(x.cpu())
    test_result("Cos", res, out)

def test_lgamma(device, size=(128, 128)):
    def lgamma(a):
        return torch.lgamma(a)
    
    # lgamma has poles at x = 0, -1, -2, ...; randn would land near them and
    # blow up the comparison. Build one tensor that covers every code path
    # instead (on compile, one simulation run):
    #   rows    0:32 -> reflection branch, small positive x (x < 0.5)
    #   rows   32:64 -> reflection branch, negative x, away from the poles
    #   rows   64:96 -> large x, exercises th tmp/log cancellation
    #   rows   96:   -> the plain Lanczos path
    x = torch.empty(size).uniform_(0.5, 4.5)
    x[0:32].uniform_(0.1, 0.49)
    x[32:64].uniform_(-2.9, -2.1)
    x[64:96].uniform_(10.0, 100.0)

    x = x.to(device=device)
    opt_fn = torch.compile(dynamic=False)(lgamma)
    res = opt_fn(x)
    out = lgamma(x.cpu())
    test_result("Lgamma", res, out)

def test_erfinv(device, size=(128, 128)):
    def erfinv(a):
        return torch.erfinv(a)
    
    # erfinv splits at |x| = 0.996625 (w = 5); a plain uniform(-0.99, 0.99)
    # never reaches the tail branch yet still passes. Cover both explicitly:
    #   rows     0:32 -> tail branch, positive
    #   rows    32:64 -> tail branch, negative
    #   rows    64:96 -> near zero, checks p* x -> 0
    #   rows    96:   -> central branch
    x = torch.empty(size).uniform_(-0.9, 0.9)
    x[0:32].uniform_(0.997, 0.99999)
    x[32:64].uniform_(-0.99999, -0.997)
    x[64:96].uniform_(-0.01, 0.01)

    x = x.to(device=device)
    opt_fn = torch.compile(dynamic=False)(erfinv)
    res = opt_fn(x)
    out = erfinv(x.cpu())
    test_result("Erfinv", res, out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    test_tanh(device)
    test_exp(device)
    test_erf(device)
    test_sin(device)
    test_cos(device)
    test_lgamma(device)
    test_erfinv(device)