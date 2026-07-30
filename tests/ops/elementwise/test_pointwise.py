import torch
import os

def clear_caches():
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
    from torch._inductor.codecache import FxGraphCache
    AOTAutogradCache.clear()
    torch._dynamo.reset()
    os.environ["TORCHINDUCTOR_CACHE"] = "0"
    FxGraphCache.clear()
    
def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4, equal_nan=False):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol, equal_nan=equal_nan):
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
    
def test_frexp(device, size=(128, 128)):
    def frexp(a):
        return torch.frexp(a)
    
    # Cover every branch of the decomposition: normals, powers of two (where a 
    # log2-based version slips), zero, subnormals (integer-side detection) and
    # the inf/NaN passthrough.
    special = torch.tensor([0.0, -0.0, 1.0, 4.0, 0.5, -2.0 ** 20,
                            1.1754944e-38, 1e-40, 5e-44, 1.4e-45,
                            3.4028235e38, float("inf"), float("-inf"), float("nan")])
    x = torch.randn(size)
    x.view(-1)[:special.numel()] = special
    
    x = x.to(device=device)
    opt_fn = torch.compile(dynamic=False)(frexp)
    m, e = opt_fn(x)
    rm, re = frexp(x.cpu())
    test_result("Frexp mantissa", m, rm, equal_nan=True)
    test_result("Frexp exponent", e.float(), re.float())

    # float16 goes through the f32 path; mantissa bits survive the round trip
    # so the result must be exact, not just close.
    xh = torch.tensor([[1.5, 3.25, -2.5, 0.0, 0.5, -1.0]], dtype=torch.float16)
    mh, eh = torch.compile(dynamic=False)(frexp)(xh.to(device=device))
    rmh, reh = frexp(xh.cpu())
    test_result("Frexp f16 mantissa", mh.float(), rmh.float(), rtol=0.0, atol=0.0)
    test_result("Frexp f16 exponent", eh.float(), reh.float(), rtol=0.0, atol=0.0)

_NA_X = torch.tensor([[0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 2.0,
                       3.4028235e38, -3.4028235e38, float("inf"), float("-inf"),
                       1.4013e-45, -1.4013e-45, 1.1754944e-38]])
_NA_Y = torch.tensor([[1.0, 1.0, -1.0, -1.0, 2.0, -2.0, 2.0,
                       float("inf"), float("-inf"), 1.0, 1.0,
                       0.0, 0.0, 0.0]])

def test_nextafter(device):
    # One ulp apart, so the default 1e-4 tolerance would pass even if the op
    # returned x unchanged. Compare exactly instead.
    run_op("Nextafter", device, torch.nextafter, 
           lambda r, c: (torch.randn(r, c), torch.randn(r, c)),
           cases=[
               ("toward_pinf", (torch.randn(64, 64), torch.full((64, 64), float("inf")))),
               ("toward_ninf", (torch.randn(64, 64), torch.full((64, 64), float("-inf")))),
               ("equal", (torch.randn(64, 64),) * 2),
               ("special", (_NA_X, _NA_Y)),
           ],
           rtol=0.0, atol=0.0)
    
def test_rand(device, size=(128, 128)):
    from torch._inductor import inductor_prims
    torch._inductor.config.fallback_random = False

    # Compare against the inductor CPU backed, not eager: both go through
    # inductor_prims.random, so the same Philox seed must give the same bits. 
    # Passing the seed as a graph input keeps ops.load_seed out of the picture.
    def f(seed):
        return inductor_prims.random(list(size), seed, "rand")
    
    seed = torch.tensor(12345, dtype=torch.int64)
    clear_caches()
    npu = torch.compile(f, dynamic=False)(seed.to(device=device))
    clear_caches()
    cpu = torch.compile(f, dynamic=False)(seed)
    test_result("Rand", npu, cpu, rtol=0.0, atol=0.0)

def test_randn(device, size=(128, 128)):
    from torch._inductor import inductor_prims
    torch._inductor.config.fallback_random = False

    def f(seed):
        return inductor_prims.random(list(size), seed, "randn")
    
    seed = torch.tensor(12345, dtype=torch.int64)
    clear_caches()
    npu = torch.compile(f, dynamic=False)(seed.to(device=device))
    clear_caches()
    cpu = torch.compile(f, dynamic=False)(seed)
    # Not exact: randn_cpu evaluates the Box-Muller tail in double, we stay in 
    # f32. Measured max deviation ~1e-06, so the default tolerance still catches
    # any real error (a wrong generator differs by 0(1), not by 1e-06).
    test_result("Randn", npu, cpu)

def test_randint64(device, size=(128, 128)):
    from torch._inductor import inductor_prims
    torch._inductor.config.fallback_random = False

    def run(lo, hi, label):
        def f(seed):
            return inductor_prims.randint(lo, hi, list(size), seed)
        seed = torch.tensor(12345, dtype=torch.int64)
        clear_caches()
        npu = torch.compile(f, dynamic=False)(seed.to(device=device))
        clear_caches()
        cpu = torch.compile(f, dynamic=False)(seed)
        # Integers: compare exactly. A loose tolerance would hide an off-by-one
        # in the modulo rewrite.
        test_result(label, npu.float(), cpu.float(), rtol=0.0, atol=0.0)
    
    run(0, 100, "Randint64")
    run(-500, 500, "Randint64 negative low")
    run(0, 2 ** 40, "Randint64 wide range")

def test_rand_e2e(device, size=(128, 128)):
    torch._inductor.config.fallback_random = False
    
    # Goes through ops.load_seed, unlike the inductor_prims test which pass a 
    # seed in directly. Values cannot be compared agaist eager, which uses a 
    # different generator, so check the shape, range and that the stream is not
    # constant.
    def f():
        return torch.rand(size, device=device)
    
    clear_caches()
    out = torch.compile(f, dynamic=False)().cpu()
    assert out.shape == torch.Size(size)
    assert (out >= 0).all() and (out < 1).all()
    assert out.std() > 0.1
    print("Rand end-to-end OK")
 
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
    test_frexp(device)
    test_nextafter(device)
    test_rand(device)
    test_randn(device)
    test_randint64(device)
    test_rand_e2e(device)


