import argparse
import torch
import torch._dynamo
import torch.utils.cpp_extension

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
        print("custom out:", out.cpu())
        print("cpu out:", cpu_out)
        raise SystemExit(1)


def test_equal(name, out, cpu_out):
    if torch.equal(out.cpu(), cpu_out):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out:", out.cpu())
        print("cpu out:", cpu_out)
        raise SystemExit(1)


def _normalize_dim(dim: int, rank: int) -> int:
    d = dim if dim >= 0 else rank + dim
    if d < 0 or d >= rank:
        raise ValueError(f"dim out of range: dim={dim}, rank={rank}")
    return d


def test_sort_stable(device, size=(128, 128), dim=-1, descending=False):
    _normalize_dim(dim, len(size))

    def sort_stable_fn(x):
        return torch.sort(x, stable=True, dim=dim, descending=descending)

    x = torch.randn(size, dtype=torch.float32)
    x_npu = x.to(device=device)

    opt_sort = torch.compile(dynamic=False)(sort_stable_fn)
    out_values, out_indices = opt_sort(x_npu)

    ref_values, ref_indices = torch.sort(x, stable=True, dim=dim, descending=descending)

    test_result("Sort.stable/values", out_values, ref_values)
    test_equal("Sort.stable/indices", out_indices, ref_indices)


def test_sort_values_stable(device, size=(128, 128), dim=-1, descending=False):
    _normalize_dim(dim, len(size))

    def sort_out_fn(x):
        out_values = torch.empty_like(x, device=x.device)
        out_indices = torch.empty_like(x, dtype=torch.int64, device=x.device)
        return torch.sort(x, stable=True, dim=dim, descending=descending, out=(out_values, out_indices))

    x = torch.randn(size, dtype=torch.float32)
    x_npu = x.to(device=device)

    opt_sort = sort_out_fn# torch.compile(dynamic=False)(sort_out_fn)
    out_values, out_indices = opt_sort(x_npu)

    ref_values, ref_indices = torch.sort(x, stable=True, dim=dim, descending=descending)

    test_result("Sort.values_stable/values", out_values, ref_values)
    test_equal("Sort.values_stable/indices", out_indices, ref_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sort tests")
    parser.add_argument("--shape", type=str, default="(128,128)")
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "default", "values"],
    )
    args = parser.parse_args()

    shape = tuple(map(int, args.shape.strip("()").split(",")))

    from Scheduler.scheduler import PyTorchSimRunner

    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()

    # Register recursive-compile bridge only when values_stable path is explicitly tested.
    if args.mode in ("all", "values"):
        torch.npu.register_eager_to_compile([
            "aten::sort.values_stable",
        ])

    if args.mode in ("all", "default"):
        test_sort_stable(device, size=shape, dim=args.dim, descending=args.descending)
    if args.mode in ("all", "values"):
        test_sort_values_stable(device, size=shape, dim=args.dim, descending=args.descending)
