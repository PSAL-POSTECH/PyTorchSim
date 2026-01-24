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

def test_exponent(device, size=(128, 128)):
    def exponent(a):
        return a.exp()
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(exponent)
    res = opt_fn(x)
    out = exponent(x.cpu())
    test_result("exponent", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_exponent(device, size=(32, 32))
