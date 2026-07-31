import os
import sys
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_tanh(device, size=(128, 128)):
    def tanh(a):
        return torch.tanh(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(tanh)
    res = opt_fn(x)
    out = tanh(x.cpu())
    test_result("Tanh", res, out)

def test_exp(device, size=(128, 128)):
    def exp(a):
        return torch.exp(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(exp)
    res = opt_fn(x)
    out = exp(x.cpu())
    test_result("Exp", res, out)

def test_erf(device, size=(128, 128)):
    def erf(a):
        return torch.erf(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(erf)
    res = opt_fn(x)
    out = erf(x.cpu())
    test_result("Erf", res, out)

def test_sin(device, size=(128, 128)):
    def sin(a):
        return torch.sin(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(sin)
    res = opt_fn(x)
    out = sin(x.cpu())
    test_result("Sin", res, out)

def test_cos(device, size=(128, 128)):
    def cos(a):
        return torch.cos(a)
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(cos)
    res = opt_fn(x)
    out = cos(x.cpu())
    test_result("Cos", res, out)

def test_lgamma(device, size=(128, 128)):
    def lgamma(a):
        return torch.lgamma(a)
    
    # lgamma has poles at x = 0, -1, -2, ...; randn would land near them and
    # blow up the comparison. Build one tensor that covers every code path
    # instead (one compile, one simulation run):
    #   rows    0:32 -> reflection branch, small positive x (x < 0.5)
    #   rows   32:64 -> reflection branch, negative x, away from the poles
    #   rows   64:96 -> large x, exercises the tmp/log cancellation
    #   rows  96:112 -> reflection at large |x|, where folding pi*x matters
    #   rows   112:  -> the plain Lanczos path
    x = torch.empty(size).uniform_(0.5, 4.5)
    x[0:32].uniform_(0.1, 0.49)
    x[32:64].uniform_(-2.9, -2.1)
    x[64:96].uniform_(10.0, 100.0)
    x[96:112].uniform_(-50.9, -50.1)   # 반사 경로, 큰 |x|

    x = x.to(device=device)
    opt_fn = torch.compile(dynamic=False)(lgamma)
    res = opt_fn(x)
    out = lgamma(x.cpu())
    test_result("Lgamma", res, out)

    xh = torch.empty(size).uniform_(0.5, 4.5).half()
    test_result("Lgamma f16", 
                torch.compile(dynamic=False)(lgamma)(xh.to(device)).float(),
                lgamma(xh.float()), rtol=1e-2, atol=1e-2)

    # Poles: torch gives inf at 0, -1, -2, ... f32 cannot hit an exact zero of
    # sin(pi*x), so this needs an explicit branch and no random band would ever
    # catch a regression here.
    poles = torch.tensor([0.0, -1.0, -2.0, -5.0, -20.0, -100.0])
    pole_out = torch.compile(dynamic=False)(lgamma)(poles.to(device=device))
    test_result("Lgamma poles", pole_out, lgamma(poles))

    # +/-inf must return +inf; NaN must propagate.
    nonfinite = torch.tensor([float("inf"), float("-inf"), float("nan")])
    nonfinite_out = torch.compile(dynamic=False)(lgamma)(nonfinite.to(device=device))
    test_result("Lgamma nonfinite", nonfinite_out, lgamma(nonfinite), equal_nan=True)

    # Values one f32 ULP away from negative integer poles exercise the
    # reflection argument reduction. Returning a plausible but inaccurate
    # finite value here is easy when sin(pi*x) is evaluated near +/-pi.
    neg_one = torch.tensor(-1.0, dtype=torch.float32)
    neg_two = torch.tensor(-2.0, dtype=torch.float32)
    near_poles = torch.stack([
        torch.nextafter(neg_one, torch.tensor(float("-inf"))),
        torch.nextafter(neg_one, torch.tensor(float("inf"))),
        torch.nextafter(neg_two, torch.tensor(float("-inf"))),
        torch.nextafter(neg_two, torch.tensor(float("inf"))),
    ])
    near_pole_out = torch.compile(dynamic=False)(lgamma)(near_poles.to(device=device))
    test_result("Lgamma near poles", near_pole_out, lgamma(near_poles))

    # Scalar path (tile_size == 1) takes a separate branch in the op and no
    # (128, 128) tensor ever reaches it.
    scalar = torch.tensor(2.5)
    test_result("Lgamma scalar",
                torch.compile(dynamic=False)(lgamma)(scalar.to(device=device)),
                lgamma(scalar))

def test_erfinv(device, size=(128, 128)):
    def erfinv(a):
        return torch.erfinv(a)
    
    # erfinv splits at |x| = 0.996625 (w = 5); a plain uniform(-0.99, 0.99)
    # never reaches the tail branch yet still passes. Cover both explicitly:
    #   rows     0:32 -> tail branch, positive
    #   rows    32:64 -> tail branch, negative
    #   rows    64:96 -> near zero, checks p * x -> 0
    #   rows    96:   -> central branch
    x = torch.empty(size).uniform_(-0.9, 0.9)
    x[0:32].uniform_(0.997, 0.99999)
    x[32:64].uniform_(-0.99999, -0.997)
    x[64:96].uniform_(-0.01, 0.01)

    x = x.to(device=device)
    opt_fn = torch.compile(dynamic=False)(erfinv)
    res = opt_fn(x)
    out = erfinv(x.cpu())
    test_result("Erfinv", res, out)

    # |x| == 1 and |x| > 1: the polynomial branch cannot produce these on its 
    # own, and a band stopping at 0.99999 never reaches them.
    edge = torch.tensor([1.0, -1.0, 1.5, -1.5])
    edge_out = torch.compile(dynamic=False)(erfinv)(edge.to(device=device))
    test_result("Erfinv edges", edge_out, erfinv(edge), equal_nan=True)

    scalar = torch.tensor(0.5)
    test_result("Erfinv scalar",
                torch.compile(dynamic=False)(erfinv)(scalar.to(device=device)),
                erfinv(scalar))
    
    # The implementation uses the single-precision Giles coefficients. f64 must
    # fail explicitly instead of returning a badly inaccurate value near |x|=1.
    x64 = torch.nextafter(
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    try:
        torch.compile(dynamic=False)(erfinv)(x64.to(device=device))
    except Exception as exc:
        if "PyTorchSim erfinv supports float32 and float16 only" not in str(exc):
            raise
        print("--------------------------")
        print("|Erfinv f64 reject Test Passed|")
        print("--------------------------")
    else:
        raise AssertionError("Erfinv f64 input must be rejected")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    test_tanh(device)
    test_exp(device)
    test_erf(device)
    test_sin(device)
    test_cos(device)
    test_lgamma(device)
    test_erfinv(device)