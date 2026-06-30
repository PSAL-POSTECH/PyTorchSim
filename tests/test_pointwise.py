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

    # Print input and output values (print first 5 if size is large)
    input_val = x.flatten()[:10].tolist() if size != (1, 1) else x.item()
    output_val = res.flatten()[:10].tolist() if size != (1, 1) else res.item()
    print(f"[{size}] Abs Input: {input_val}")
    print(f"[{size}] Abs Output: {output_val}")

    test_result("Abs", res, out)

def test_sign(device, size=(128, 128)):
    def sign_fn(a):
        return torch.sign(a)

    x = torch.randn(size).to(device=device)
    x[x.abs() < 0.3] = 0.0

    opt_fn = torch.compile(dynamic=False)(sign_fn)
    res = opt_fn(x)
    out = sign_fn(x.cpu())

    # Print input and output values (print first 10 if size is large)
    input_val = x.flatten()[:10].tolist() if size != (1, 1) else x.item()
    output_val = res.flatten()[:10].tolist() if size != (1, 1) else res.item()
    print(f"[{size}] Sign Input: {input_val}")
    print(f"[{size}] Sign Output: {output_val}")

    test_result("Sign", res, out)

def test_isnan(device, size=(128, 128)):
    def isnan_fn(a):
        return torch.isnan(a)

    # Generate random floats on CPU and inject NaNs, then move to NPU
    x_cpu = torch.randn(size)
    x_cpu[x_cpu.abs() < 0.3] = float('nan')
    x = x_cpu.to(device=device)

    opt_fn = torch.compile(dynamic=False)(isnan_fn)
    res = opt_fn(x)
    out = isnan_fn(x.cpu())

    input_val = x.flatten()[:10].tolist() if size != (1, 1) else x.item()
    output_val = res.flatten()[:10].tolist() if size != (1, 1) else res.item()
    print(f"[{size}] IsNaN Input: {input_val}")
    print(f"[{size}] IsNaN Output: {output_val}")

    test_result("IsNaN", res, out)

def test_isinf(device, size=(128, 128)):
    def isinf_fn(a):
        return torch.isinf(a)

    # Generate random floats and inject positive/negative infinities
    x_cpu = torch.randn(size)
    x_cpu[x_cpu > 1.0] = float('inf')
    x_cpu[x_cpu < -1.0] = float('-inf')
    x = x_cpu.to(device=device)

    opt_fn = torch.compile(dynamic=False)(isinf_fn)
    res = opt_fn(x)
    out = isinf_fn(x.cpu())

    input_val = x.flatten()[:10].tolist() if size != (1, 1) else x.item()
    output_val = res.flatten()[:10].tolist() if size != (1, 1) else res.item()
    print(f"[{size}] IsInf Input: {input_val}")
    print(f"[{size}] IsInf Output: {output_val}")

    test_result("IsInf", res, out)

def test_fmod(device, size=(128, 128)):
    def fmod_fn(a, b):
        return torch.fmod(a, b)

    x = (torch.randn(size) * 10).to(device=device)
    y = (torch.randn(size) * 3 + 1).to(device=device)  # Avoid dividing by zero

    opt_fn = torch.compile(dynamic=False)(fmod_fn)
    res = opt_fn(x, y)
    out = fmod_fn(x.cpu(), y.cpu())

    input_val_x = x.flatten()[:10].tolist() if size != (1, 1) else x.item()
    input_val_y = y.flatten()[:10].tolist() if size != (1, 1) else y.item()
    output_val = res.flatten()[:10].tolist() if size != (1, 1) else res.item()
    print(f"[{size}] Fmod Input X: {input_val_x}")
    print(f"[{size}] Fmod Input Y: {input_val_y}")
    print(f"[{size}] Fmod Output: {output_val}")

    test_result("Fmod", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")

    test_abs(device, size=(128, 128))
    test_abs(device, size=(1, 1))

    test_sign(device, size=(128, 128))
    test_sign(device, size=(1, 1))

    test_isnan(device, size=(128, 128))
    test_isnan(device, size=(1, 1))

    test_isinf(device, size=(128, 128))
    test_isinf(device, size=(1, 1))

    test_fmod(device, size=(128, 128))
    test_fmod(device, size=(1, 1))