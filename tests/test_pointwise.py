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

def test_abs(device, size=(128, 128)):
    def abs_fn(a):
        return torch.abs(a)

    x = (torch.randn(size) * 10).to(device=device)
    opt_fn = torch.compile(dynamic=False)(abs_fn)
    res = opt_fn(x)
    out = abs_fn(x.cpu())
    test_result("Abs", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")

    test_abs(device, size=(128, 128))
    test_abs(device, size=(1, 1))