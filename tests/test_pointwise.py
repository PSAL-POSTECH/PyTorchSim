import torch
import os

def clear_caches():
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
    from torch._inductor.codecache import FxGraphCache
    AOTAutogradCache.clear()
    torch._dynamo.reset()
    os.environ["TORCHINDUCTOR_CACHE"] = "0"
    FxGraphCache.clear()
    
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

def run_test(name, device, fn, inputs, size_desc, rtol=1e-4, atol=1e-4):
    """
    Harness function to compile, execute on NPU, compare with CPU, and print details.
    inputs: single tensor or tuple/list of tensors (on CPU)
    """
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]
        
    npu_inputs = [x.to(device=device) for x in inputs]
    cpu_inputs = [x.clone() for x in inputs]

    clear_caches()
    
    opt_fn = torch.compile(dynamic=False)(fn)
    res = opt_fn(*npu_inputs)
    out = fn(*cpu_inputs)

    # Print input / output slices (up to 10 elements)
    for idx, x in enumerate(inputs):
        label = f"X" if len(inputs) == 1 else f"X{idx+1}"
        val = x.flatten()[:10].tolist() if x.numel() > 1 else x.item()
        print(f"[{size_desc}] {name} Input {label}: {val}")
    
    out_val = res.flatten()[:10].tolist() if res.numel() > 1 else res.item()
    print(f"[{size_desc}] {name} Output: {out_val}")

    test_result(f"{name}_{size_desc}", res, out, rtol=rtol, atol=atol)

def test_abs(device):
    def abs_fn(a):
        return torch.abs(a)

    # 1. Float Vector (Aligned)
    run_test("Abs_Float", device, abs_fn, torch.randn(128, 128) * 10, "128x128")
    # 2. Float Scalar
    run_test("Abs_Float", device, abs_fn, torch.randn(1, 1) * 10, "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Abs_Float", device, abs_fn, torch.randn(15, 15) * 10, "15x15")
    run_test("Abs_Float", device, abs_fn, torch.randn(129, 129) * 10, "129x129")
    # 4. Int Vector (Aligned)
    run_test("Abs_Int", device, abs_fn, torch.randint(-100, 100, (128, 128), dtype=torch.int32), "128x128_int")
    # 5. Non-contiguous (Transposed & Strided Sliced)
    run_test("Abs_Float_Strided", device, abs_fn, (torch.randn(128, 128) * 10).t(), "128x128_strided_transposed")
    run_test("Abs_Float_Strided", device, abs_fn, (torch.randn(128, 256) * 10)[:, ::2], "128x128_strided_sliced")

def test_sign(device):
    def sign_fn(a):
        return torch.sign(a)

    # 1. Float Vector (Aligned)
    x_float = torch.randn(128, 128)
    x_float[x_float.abs() < 0.3] = 0.0
    run_test("Sign_Float", device, sign_fn, x_float, "128x128")
    # 2. Float Scalar (includes zero and nonzero)
    x_scalar = torch.tensor([[0.0]])
    run_test("Sign_Float", device, sign_fn, x_scalar, "1x1_zero")
    x_scalar_non_zero = torch.tensor([[-4.5]])
    run_test("Sign_Float", device, sign_fn, x_scalar_non_zero, "1x1_nonzero")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    x_tail = torch.randn(15, 15)
    x_tail[x_tail.abs() < 0.3] = 0.0
    run_test("Sign_Float", device, sign_fn, x_tail, "15x15")
    x_tail_large = torch.randn(129, 129)
    x_tail_large[x_tail_large.abs() < 0.3] = 0.0
    run_test("Sign_Float", device, sign_fn, x_tail_large, "129x129")
    # 4. Int Vector (Aligned)
    x_int = torch.randint(-5, 5, (128, 128), dtype=torch.int32)
    run_test("Sign_Int", device, sign_fn, x_int, "128x128_int")

def test_isnan(device):
    def isnan_fn(a):
        return torch.isnan(a)

    # 1. Float Vector with NaNs (Aligned)
    x = torch.randn(128, 128)
    x[x.abs() < 0.3] = float('nan')
    run_test("IsNaN", device, isnan_fn, x, "128x128")
    # 2. Float Scalar
    x_scalar = torch.tensor([[float('nan')]])
    run_test("IsNaN", device, isnan_fn, x_scalar, "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    x_tail = torch.randn(15, 15)
    x_tail[x_tail.abs() < 0.3] = float('nan')
    run_test("IsNaN", device, isnan_fn, x_tail, "15x15")
    x_tail_large = torch.randn(129, 129)
    x_tail_large[x_tail_large.abs() < 0.3] = float('nan')
    run_test("IsNaN", device, isnan_fn, x_tail_large, "129x129")

def test_isinf(device):
    def isinf_fn(a):
        return torch.isinf(a)

    # 1. Float Vector with Infs (Aligned)
    x = torch.randn(128, 128)
    x[x > 1.0] = float('inf')
    x[x < -1.0] = float('-inf')
    run_test("IsInf", device, isinf_fn, x, "128x128")
    # 2. Float Scalar
    x_scalar = torch.tensor([[float('-inf')]])
    run_test("IsInf", device, isinf_fn, x_scalar, "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    x_tail = torch.randn(15, 15)
    x_tail[x_tail > 1.0] = float('inf')
    x_tail[x_tail < -1.0] = float('-inf')
    run_test("IsInf", device, isinf_fn, x_tail, "15x15")
    x_tail_large = torch.randn(129, 129)
    x_tail_large[x_tail_large > 1.0] = float('inf')
    x_tail_large[x_tail_large < -1.0] = float('-inf')
    run_test("IsInf", device, isinf_fn, x_tail_large, "129x129")

def test_fmod(device):
    def fmod_fn(a, b):
        return torch.fmod(a, b)

    # 1. Float Vector (Aligned)
    x = torch.randn(128, 128) * 10
    y = torch.randn(128, 128) * 3 + 1
    run_test("Fmod", device, fmod_fn, (x, y), "128x128")
    #2. Float Scalar
    run_test("Fmod", device, fmod_fn, (torch.tensor([[5.5102357203957]]), torch.tensor([[2.0235825235]])), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Fmod", device, fmod_fn, (torch.randn(15, 15) * 10, torch.randn(15, 15) * 3 + 1), "15x15")
    run_test("Fmod", device, fmod_fn, (torch.randn(129, 129) * 10, torch.randn(129, 129) * 3 + 1), "129x129")
    # 4. Broadcasting (128x1 vs 1x128)
    run_test("Fmod_Broadcast", device, fmod_fn, (torch.randn(128, 1) * 10, torch.randn(1, 128) * 3 + 1), "broadcast")

def test_lshift(device):
    def lshift_fn(a, b):
        return torch.bitwise_left_shift(a, b)

    # 1. Int Vector (Aligned)
    run_test("LShift", device, lshift_fn, (torch.randint(1, 100, (128, 128), dtype=torch.int32), torch.randint(1, 5, (128, 128), dtype=torch.int32)), "128x128")
    # 2. Int Scalar
    run_test("LShift", device, lshift_fn, (torch.randint(1, 100, (1, 1), dtype=torch.int32), torch.randint(1, 5, (1, 1), dtype=torch.int32)), "1x1")
    # 3. Int Vector (Tail / Remainder - Small & Large)
    run_test("LShift", device, lshift_fn, (torch.randint(1, 100, (15, 15), dtype=torch.int32), torch.randint(1, 5, (15, 15), dtype=torch.int32)), "15x15")
    run_test("LShift", device, lshift_fn, (torch.randint(1, 100, (129, 129), dtype=torch.int32), torch.randint(1, 5, (129, 129), dtype=torch.int32)), "129x129")
    # 4. Broadcasting (128x1 vs 1x128)
    run_test("LShift_Broadcast", device, lshift_fn, (torch.randint(1, 100, (128, 1), dtype=torch.int32), torch.randint(1, 5, (1, 128), dtype=torch.int32)), "broadcast")

def test_rshift(device):
    def rshift_fn(a, b):
        return torch.bitwise_right_shift(a, b)

    # 1. Int Vector (Aligned)
    run_test("RShift", device, rshift_fn, (torch.randint(10, 1000, (128, 128), dtype=torch.int32), torch.randint(1, 5, (128, 128), dtype=torch.int32)), "128x128")
    # 2. Int Scalar
    run_test("RShift", device, rshift_fn, (torch.randint(10, 1000, (1, 1), dtype=torch.int32), torch.randint(1, 5, (1, 1), dtype=torch.int32)), "1x1")
    # 3. Int Vector (Tail / Remainder - Small & Large)
    run_test("RShift", device, rshift_fn, (torch.randint(10, 1000, (15, 15), dtype=torch.int32), torch.randint(1, 5, (15, 15), dtype=torch.int32)), "15x15")
    run_test("RShift", device, rshift_fn, (torch.randint(10, 1000, (129, 129), dtype=torch.int32), torch.randint(1, 5, (129, 129), dtype=torch.int32)), "129x129")
    # 4. Broadcasting (128x1 vs 1x128)
    run_test("RShift_Broadcast", device, rshift_fn, (torch.randint(10, 1000, (128, 1), dtype=torch.int32), torch.randint(1, 5, (1, 128), dtype=torch.int32)), "broadcast")

def test_copysign(device):
    def copysign_fn(a, b):
        return torch.copysign(a, b)

    # 1. Float Vector (Aligned)
    run_test("Copysign", device, copysign_fn, (torch.randn(128, 128) * 10, torch.randn(128, 128)), "128x128")
    # 2. Float Scalar (test sign of zero case specifically: negative zero)
    run_test("Copysign", device, copysign_fn, (torch.tensor([[3.0]]), torch.tensor([[-0.0]])), "1x1_negzero")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Copysign", device, copysign_fn, (torch.randn(15, 15) * 10, torch.randn(15, 15)), "15x15")
    run_test("Copysign", device, copysign_fn, (torch.randn(129, 129) * 10, torch.randn(129, 129)), "129x129")
    # 4. Broadcasting (128x1 vs 1x128)
    run_test("Copysign_Broadcast", device, copysign_fn, (torch.randn(128, 1) * 10, torch.randn(1, 128)), "broadcast")

def test_erfc(device):
    def erfc_fn(a):
        return torch.erfc(a)

    # 1. Float Vector (Aligned)
    run_test("Erfc", device, erfc_fn, torch.randn(128, 128), "128x128")
    # 2. Float Scalar (test large positive float)
    run_test("Erfc", device, erfc_fn, torch.tensor([[4.5]]), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Erfc", device, erfc_fn, torch.randn(15, 15), "15x15")
    run_test("Erfc", device, erfc_fn, torch.randn(129, 129), "129x129")

def test_hypot(device):
    def hypot_fn(a, b):
        return torch.hypot(a, b)

    # 1. Float Vector (Aligned)
    run_test("Hypot", device, hypot_fn, (torch.randn(128, 128), torch.randn(128, 128)), "128x128")
    # 2. Float Scalar (Result should be exactly 5.0)
    run_test("Hypot", device, hypot_fn, (torch.tensor([[3.0]]), torch.tensor([[4.0]])), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Hypot", device, hypot_fn, (torch.randn(15, 15), torch.randn(15, 15)), "15x15")
    run_test("Hypot", device, hypot_fn, (torch.randn(129, 129), torch.randn(129, 129)), "129x129")
    # 4. Broadcasting (128x1 vs 1x128)
    run_test("Hypot_Broadcast", device, hypot_fn, (torch.randn(128, 1), torch.randn(1, 128)), "broadcast")

def test_cosh(device):
    def cosh_fn(a):
        return torch.cosh(a)

    # 1. Float Vector (Aligned)
    run_test("Cosh", device, cosh_fn, torch.randn(128, 128), "128x128")
    # 2. Float Scalar (test large positive float)
    run_test("Cosh", device, cosh_fn, torch.tensor([[4.5]]), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Cosh", device, cosh_fn, torch.randn(15, 15), "15x15")
    run_test("Cosh", device, cosh_fn, torch.randn(129, 129), "129x129")

def test_sinh(device):
    def sinh_fn(a):
        return torch.sinh(a)

    # 1. Float Vector (Aligned)
    run_test("Sinh", device, sinh_fn, torch.randn(128, 128), "128x128")
    # 2. Float Scalar (test large positive float)
    run_test("Sinh", device, sinh_fn, torch.tensor([[4.5]]), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Sinh", device, sinh_fn, torch.randn(15, 15), "15x15")
    run_test("Sinh", device, sinh_fn, torch.randn(129, 129), "129x129")
    
def test_acosh(device):
    def acosh_fn(a):
        return torch.acosh(a)

    # 1. Float Vector (Aligned)
    run_test("Acosh", device, acosh_fn, torch.rand(128, 128) + 1, "128x128")  # Values in [1, 2]
    # 2. Float Scalar
    run_test("Acosh", device, acosh_fn, torch.tensor([[1.5]]), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Acosh", device, acosh_fn, torch.rand(15, 15) + 1, "15x15")
    run_test("Acosh", device, acosh_fn, torch.rand(129, 129) + 1, "129x129")

def test_asinh(device):
    def asinh_fn(a):
        return torch.asinh(a)

    # 1. Float Vector (Aligned)
    run_test("Asinh", device, asinh_fn, torch.randn(128, 128), "128x128")
    # 2. Float Scalar
    run_test("Asinh", device, asinh_fn, torch.tensor([[1.5]]), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Asinh", device, asinh_fn, torch.randn(15, 15), "15x15")
    run_test("Asinh", device, asinh_fn, torch.randn(129, 129), "129x129")

def test_atanh(device):
    def atanh_fn(a):
        return torch.atanh(a)

    # 1. Float Vector (Aligned)
    run_test("Atanh", device, atanh_fn, torch.rand(128, 128) * 2 - 1, "128x128")  # Values in (-1, 1)
    # 2. Float Scalar
    run_test("Atanh", device, atanh_fn, torch.tensor([[0.5]]), "1x1")
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Atanh", device, atanh_fn, torch.rand(15, 15) * 2 - 1, "15x15")
    run_test("Atanh", device, atanh_fn, torch.rand(129, 129) * 2 - 1, "129x129")

def test_log(device):
    def log_fn(a):
        return torch.log(a)

    # 1. Float Vector (Aligned)
    run_test("Log", device, log_fn, torch.rand(128, 128) + 1e-5, "128x128")  # Avoid log(0)
    # 2. Float Scalar
    run_test("Log", device, log_fn, torch.tensor([[2.71828]]), "1x1")  # log(e) = 1
    # 3. Float Vector (Tail / Remainder - Small & Large)
    run_test("Log", device, log_fn, torch.rand(15, 15) + 1e-5, "15x15")
    run_test("Log", device, log_fn, torch.rand(129, 129) + 1e-5, "129x129")
    
if __name__ == "__main__":
    device = torch.device("npu:0")

    # test_abs(device)
    # test_sign(device)
    # test_isnan(device)
    # test_isinf(device)
    # test_fmod(device)
    # test_lshift(device)
    # test_rshift(device)
    # test_copysign(device)
    # test_erfc(device)
    # test_hypot(device)
    # test_cosh(device)
    # test_sinh(device)
    # test_acos(device)
    # test_acosh(device)
    # test_asinh(device)
    # test_atanh(device)
    test_log(device)
    