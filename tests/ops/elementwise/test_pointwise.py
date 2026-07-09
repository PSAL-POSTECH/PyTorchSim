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


# aligned, scalar, small tail, large tail
STD_SIZES = [(128, 128), (1, 1), (15, 15), (129, 129)]


def run_op(name, device, fn, make_input, cases=None, sizes=STD_SIZES, rtol=1e-4, atol=1e-4):
    torch.manual_seed(0)
    for rows, cols in sizes:
        run_test(name, device, fn, make_input(rows, cols), f"{rows}x{cols}", rtol=rtol, atol=atol)
    for desc, inputs in (cases or []):
        run_test(name, device, fn, inputs, desc, rtol=rtol, atol=atol)


def test_abs(device):
    run_op("Abs", device, torch.abs, lambda r, c: torch.randn(r, c) * 10,
           cases=[
               ("int", torch.randint(-100, 100, (128, 128), dtype=torch.int32)),
               ("strided_transposed", (torch.randn(128, 128) * 10).t()),
               ("strided_sliced", (torch.randn(128, 256) * 10)[:, ::2]),
           ])

def test_sign(device):
    def make(r, c):
        x = torch.randn(r, c)
        x[x.abs() < 0.3] = 0.0
        return x
    run_op("Sign", device, torch.sign, make,
           cases=[
               ("zero", torch.tensor([[0.0]])),
               ("nonzero", torch.tensor([[-4.5]])),
               ("int", torch.randint(-5, 5, (128, 128), dtype=torch.int32)),
           ])

def test_isnan(device):
    def make(r, c):
        x = torch.randn(r, c)
        x[x.abs() < 0.3] = float('nan')
        return x
    run_op("IsNaN", device, torch.isnan, make,
           cases=[("scalar_nan", torch.tensor([[float('nan')]]))])

def test_isinf(device):
    def make(r, c):
        x = torch.randn(r, c)
        x[x > 1.0] = float('inf')
        x[x < -1.0] = float('-inf')
        return x
    run_op("IsInf", device, torch.isinf, make,
           cases=[("scalar_inf", torch.tensor([[float('-inf')]]))])

def test_fmod(device):
    run_op("Fmod", device, torch.fmod,
           lambda r, c: (torch.randn(r, c) * 10, torch.randn(r, c) * 3 + 1),
           cases=[("broadcast", (torch.randn(128, 1) * 10, torch.randn(1, 128) * 3 + 1))])

def test_lshift(device):
    run_op("LShift", device, torch.bitwise_left_shift,
           lambda r, c: (torch.randint(1, 100, (r, c), dtype=torch.int32),
                         torch.randint(1, 5, (r, c), dtype=torch.int32)),
           cases=[("broadcast", (torch.randint(1, 100, (128, 1), dtype=torch.int32),
                                 torch.randint(1, 5, (1, 128), dtype=torch.int32)))])

def test_rshift(device):
    run_op("RShift", device, torch.bitwise_right_shift,
           lambda r, c: (torch.randint(10, 1000, (r, c), dtype=torch.int32),
                         torch.randint(1, 5, (r, c), dtype=torch.int32)),
           cases=[("broadcast", (torch.randint(10, 1000, (128, 1), dtype=torch.int32),
                                 torch.randint(1, 5, (1, 128), dtype=torch.int32)))])

def test_copysign(device):
    run_op("Copysign", device, torch.copysign,
           lambda r, c: (torch.randn(r, c) * 10, torch.randn(r, c)),
           cases=[
               ("negzero", (torch.tensor([[3.0]]), torch.tensor([[-0.0]]))),
               ("broadcast", (torch.randn(128, 1) * 10, torch.randn(1, 128))),
           ])

def test_erfc(device):
    run_op("Erfc", device, torch.erfc, lambda r, c: torch.randn(r, c),
           cases=[("large", torch.tensor([[4.5]]))])

def test_hypot(device):
    run_op("Hypot", device, torch.hypot,
           lambda r, c: (torch.randn(r, c), torch.randn(r, c)),
           cases=[
               ("3-4-5", (torch.tensor([[3.0]]), torch.tensor([[4.0]]))),
               ("broadcast", (torch.randn(128, 1), torch.randn(1, 128))),
           ])

def test_cosh(device):
    run_op("Cosh", device, torch.cosh, lambda r, c: torch.randn(r, c),
           cases=[("large", torch.tensor([[4.5]]))])

def test_sinh(device):
    run_op("Sinh", device, torch.sinh, lambda r, c: torch.randn(r, c),
           cases=[("large", torch.tensor([[4.5]]))])

def test_log(device):
    # domain: x > 0
    run_op("Log", device, torch.log, lambda r, c: torch.rand(r, c) + 1e-5,
           cases=[("e", torch.tensor([[2.71828]]))])

def test_acosh(device):
    # domain: x >= 1
    run_op("Acosh", device, torch.acosh, lambda r, c: torch.rand(r, c) + 1,
           cases=[("one", torch.tensor([[1.0]]))])

def test_asinh(device):
    run_op("Asinh", device, torch.asinh, lambda r, c: torch.randn(r, c))

def test_atanh(device):
    # domain: (-1, 1)
    run_op("Atanh", device, torch.atanh, lambda r, c: torch.rand(r, c) * 2 - 1)

def test_atan(device):
    # domain: all reals; boundary: zero, unit, large |x| (range-reduction path)
    run_op("Atan", device, torch.atan, lambda r, c: torch.randn(r, c) * 5,
           cases=[("boundary", torch.tensor([[0.0, 1.0, -1.0, 1e3, -1e3, 1e-4]]))])

def test_asin(device):
    # domain: [-1, 1]; boundary includes the +/-1 endpoints
    run_op("Asin", device, torch.asin, lambda r, c: torch.rand(r, c) * 2 - 1,
           cases=[("boundary", torch.tensor([[0.0, 1.0, -1.0, 0.5, -0.5]]))])

def test_acos(device):
    # domain: [-1, 1]; boundary includes the +/-1 endpoints
    run_op("Acos", device, torch.acos, lambda r, c: torch.rand(r, c) * 2 - 1,
           cases=[("boundary", torch.tensor([[0.0, 1.0, -1.0, 0.5, -0.5]]))])

def test_atan2(device):
    # atan2(y, x); boundary: axes and diagonals across all four quadrants
    # (0, 0) excluded: composition yields atan(0/0)=nan while torch returns 0
    y = torch.tensor([[1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0]])
    x = torch.tensor([[0.0, 1.0, 0.0, -1.0, 1.0, -1.0, -1.0, 1.0]])
    run_op("Atan2", device, torch.atan2, lambda r, c: (torch.randn(r, c), torch.randn(r, c)),
           cases=[("boundary", (y, x))])
 
if __name__ == "__main__":
    device = torch.device("npu:0")

    test_abs(device)
    test_sign(device)
    test_isnan(device)
    test_isinf(device)
    test_fmod(device)
    test_lshift(device)
    test_rshift(device)
    test_copysign(device)
    test_erfc(device)
    test_hypot(device)
    test_cosh(device)
    test_sinh(device)
    test_log(device)
    test_acosh(device)
    test_asinh(device)
    test_atanh(device)
    test_atan(device)
    test_asin(device)
    test_acos(device)
    test_atan2(device)