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

def test_vectoradd(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorAdd", res, out)

def test_vector_scalar_add(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn([1]).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorScalarAdd", res, out)

def test_vector_tensor_add(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn(size[-1]).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorTensorAdd", res, out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    test_vectoradd(device, (1, 1))
    test_vectoradd(device, (47, 10))
    test_vectoradd(device, (128, 128))
    test_vectoradd(device, (4071, 429))
    test_vector_tensor_add(device, (128, 128))
