import torch
import torch._dynamo
import torch.utils.cpp_extension

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

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