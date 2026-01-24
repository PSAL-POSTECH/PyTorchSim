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

def test_Transpose2D(device, size=(16, 32)):
    def transpose(a):
        return a.transpose(0, 1).contiguous()
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[0], size[1]).float().to(device=device)
    opt_fn = torch.compile(dynamic=False)(transpose)
    res = opt_fn(x)
    out = transpose(x.cpu())
    test_result("Transpose Forward", res, out)

def test_Transpose2D_2(device, size=(16, 32)):
    def transpose(a, b):
        return a.transpose(0, 1) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[0], size[1]).float().to(device=device)
    y = torch.randn(size[1], size[0]).float().to(device=device)

    opt_fn = torch.compile(dynamic=False)(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose2 Forward", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_Transpose2D(device, [64, 156])
    test_Transpose2D_2(device, [16, 64])
    test_Transpose2D(device, [640, 256])
    test_Transpose2D_2(device, [160, 264])

