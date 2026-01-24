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

def test_bmm_reduce(device, batch=12, size=512):
    def bmm(a, b):
        result = torch.bmm(a, b.transpose(1,2))
        return result, result.max(dim=1).values
    torch.manual_seed(0)
    N = size
    input = torch.randn(batch, N, 64)
    weight = torch.randn(batch, N, 64)
    #input = torch.arange(1, N * N + 1, dtype=torch.float32).reshape(N, N).to(dtype=torch.float32)
    #weight = torch.eye(N, dtype=torch.float32)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(x1, w1)
    y = bmm(x2, w2)
    test_result("BMM Reduction Fusion activation", res[0], y[0])
    test_result("BMM Reduction Fusion reduction", res[1], y[1])

if __name__ == "__main__":
    device = torch.device("npu:0")
    #test_bmm_reduce(device)
    test_bmm_reduce(device, 12, 512)
    test_bmm_reduce(device, 4, 256)
    test_bmm_reduce(device, 6, 768)
    test_bmm_reduce(device, 2, 128)
