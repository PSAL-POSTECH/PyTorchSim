import torch

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

def test_BMM(device, batch_size=1, m=32, n=16, k=64):
    def bmm(a, b):
        return torch.bmm(a, b.transpose(1, 2))
    torch.manual_seed(0)
    a = torch.randn(batch_size, m, k).to(device=device)
    b = torch.randn(batch_size, n, k).to(device=device)
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(a, b)
    out = bmm(a.cpu(), b.cpu())
    test_result("BMM Forward", res, out)

def test_addBMM(device, batch_size=1, m=32, n=16, k=64, bias_rank=1):#TODO: Fusion should be implemented for this test
    def bmm(a, b, bias):
        return torch.bmm(a, b.transpose(1, 2)) + bias
    torch.manual_seed(0)
    a = torch.randn(batch_size, m, k).to(device=device)
    b = torch.randn(batch_size, n, k).to(device=device)
    bias = torch.randn(batch_size, n) if bias_rank == 1 else torch.randn(batch_size, m, n)
    bias = bias.to(device=device)
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(a, b, bias)
    out = bmm(a.cpu(), b.cpu(), bias.cpu())
    test_result("BMM Forward", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_BMM(device)
    test_BMM(device, 2, 256, 128, 256)
    test_BMM(device, 2, 128, 256, 256)
    test_BMM(device, 2, 256, 256, 128)
    test_BMM(device, 4, 256, 256, 256)
    test_BMM(device, 12, 512, 512, 64)
    test_BMM(device, 16, 512, 512, 64)