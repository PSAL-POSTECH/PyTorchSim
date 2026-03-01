import argparse
from pathlib import Path

import torch


def _test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        return

    message = f"|{name} Test Failed|"
    print("-" * len(message))
    print(message)
    print("-" * len(message))
    print("custom out: ", out.cpu())
    print("cpu out: ", cpu_out)
    raise RuntimeError(f"{name} mismatch")


def _togsim_log_count() -> int:
    log_dir = Path("togsim_results")
    if not log_dir.exists():
        return 0
    return len(list(log_dir.glob("*.log")))


def _assert_simulation_happened(before_count: int, case_name: str):
    after_count = _togsim_log_count()
    if after_count <= before_count:
        raise RuntimeError(
            f"{case_name}: TOGSim log count did not increase "
            f"(before={before_count}, after={after_count})"
        )
    print(f"{case_name}: TOGSim logs increased ({before_count} -> {after_count})")


def test_cat_default(device):
    def cat_default_fn(a, b):
        return torch.cat([a, b], dim=0)

    x = torch.randn(8, 16, device=device)
    y = torch.randn(6, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_default_fn)

    before = _togsim_log_count()
    out = opt_fn(x, y)
    _assert_simulation_happened(before, "cat.default")

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    _test_result("cat.default", out, cpu_out, rtol=1e-4, atol=1e-4)


def test_cat_out(device):
    def cat_out_fn(a, b, out):
        return torch.ops.aten.cat.out([a, b], 0, out=out)

    x = torch.randn(8, 16, device=device)
    y = torch.randn(6, 16, device=device)
    out_buf = torch.empty(14, 16, device=device)
    opt_fn = torch.compile(dynamic=False)(cat_out_fn)

    before = _togsim_log_count()
    out = opt_fn(x, y, out_buf)
    _assert_simulation_happened(before, "cat.out")

    cpu_out = torch.cat([x.cpu(), y.cpu()], dim=0)
    _test_result("cat.out", out, cpu_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cat simulation tests")
    parser.add_argument(
        "--case",
        choices=["default", "out", "all"],
        default="all",
        help="Which cat case to run",
    )
    args = parser.parse_args()

    device = torch.device("npu:0")

    if args.case in ("default", "all"):
        test_cat_default(device)
    if args.case in ("out", "all"):
        test_cat_out(device)
