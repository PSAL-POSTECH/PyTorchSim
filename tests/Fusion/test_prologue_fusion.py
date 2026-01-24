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

def test_elem_broadcast_fusion(device):
    def matmul_fused(a, b, c):
        return torch.matmul(c * a, b)
    torch.manual_seed(0)
    input = torch.randn(128, 128)
    weight = torch.randn(128, 128)
    c = torch.randn(128, 1, dtype=torch.float32)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    c1 = c.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    c2 = c.to("cpu")
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1, c1)
    y = matmul_fused(x2, w2, c2)
    test_result("Matmul Scalar Fusion Forward", res, y)

def test_elem_fusion(device):
    def matmul_fused(a, b, c):
        return torch.matmul(c * a, b)
    torch.manual_seed(0)
    input = torch.randn(128, 128)
    weight = torch.randn(128, 128)
    c = torch.randn(128, 128, dtype=torch.float32)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    c1 = c.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    c2 = c.to("cpu")
    opt_fn = torch.compile(dynamic=False)(matmul_fused)
    res = opt_fn(x1, w1, c1)
    y = matmul_fused(x2, w2, c2)
    test_result("Matmul Element-wise Fusion Forward", res, y)

def test_elem_bmm_weight_fusion(device, batch_size=1, m=512, n=512, k=64):
    def bmm(a, b, c, d):
        return torch.bmm(a , (d+b)*c)
    torch.manual_seed(0)
    a = torch.randn(batch_size, m, k).to(device=device)
    b = torch.randn(batch_size, 1, n).to(device=device)
    c = torch.randn(batch_size, 1, n)
    c = c.to(device=device)
    d = torch.randn(batch_size, k, n).to(device=device)
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(a, b, c, d)
    out = bmm(a.cpu(), b.cpu(), c.cpu(), d.cpu())
    print(torch.max(torch.abs(res.cpu() - out)))
    test_result("BMM Element-wise Fusion Forward", res, out)

def test_elem_bmm_input_fusion(device, batch_size=1, m=512, n=512, k=64):
    def bmm(a, b, c, d):
        return torch.bmm((a+b)*c , d)
    torch.manual_seed(0)
    a = torch.randn(batch_size, m, k).to(device=device)
    b = torch.randn(batch_size, 1, k).to(device=device)
    c = torch.randn(batch_size, 1, k)
    c = c.to(device=device)
    d = torch.randn(batch_size, k, n).to(device=device)
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(a, b, c, d)
    out = bmm(a.cpu(), b.cpu(), c.cpu(), d.cpu())
    print(torch.max(torch.abs(res.cpu() - out)))
    test_result("BMM Element-wise Fusion Forward", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_elem_broadcast_fusion(device)
    test_elem_fusion(device)
    test_elem_bmm_input_fusion(device, batch_size=4, m=512, n=512, k=64)
    test_elem_bmm_weight_fusion(device, batch_size=12, m=512, n=512, k=64)