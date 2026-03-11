import argparse
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

def test_sort(device, size=(128, 128), dim=-1, descending=False, stable=True):
    def sort_test(x):
        return torch.sort(x, dim=dim, descending=descending, stable=stable)

    x = torch.randn(size, dtype=torch.float32)
    x_npu = x.to(device=device)

    opt_sort = torch.compile(dynamic=False)(sort_test)
    out_values, out_indices = opt_sort(x_npu)
    ref_values, ref_indices = torch.sort(x, stable=stable, dim=dim, descending=descending)

    prefix = "Sort.stable" if stable else "Sort.unstable"
    test_result(f"{prefix}/values size={size}, dim={dim}, desc={descending}", out_values, ref_values)
    if stable:
        test_result(f"{prefix}/indices size={size}, dim={dim}, desc={descending}", out_indices, ref_indices)
    else:
        # Unstable sort does not guarantee tie ordering; validate index-value consistency instead.
        gathered = torch.gather(x, dim, out_indices.cpu())
        test_result(f"{prefix}/indices_gather size={size}, dim={dim}, desc={descending}", gathered, out_values.cpu())


def test_sort_stable_suite(device):
    # Keep sort-axis sizes compatible with backend constraints (vector-size multiple).
    cases = [
        {"size": (64,), "dim": 0, "descending": False},          # 1D
        {"size": (4, 64), "dim": 1, "descending": True},         # 2D, last dim
        {"size": (2, 8, 32), "dim": 2, "descending": False},     # 3D, last dim
        {"size": (2, 16, 4), "dim": 1, "descending": True},      # 3D, middle dim
        {"size": (2, 4, 8, 32), "dim": 3, "descending": False},  # 4D, last dim
        {"size": (4, 2, 32, 8), "dim": 2, "descending": True},   # 4D, inner dim
    ]
    for case in cases:
        test_sort(
            device=device,
            size=case["size"],
            dim=case["dim"],
            descending=case["descending"],
            stable=True,
        )


def test_sort_duplicate_cases(device):
    duplicate_cases = [
        {"size": (64,), "dim": 0, "descending": False},
        {"size": (4, 64), "dim": 1, "descending": True},
        {"size": (2, 8, 32), "dim": 2, "descending": False},
    ]
    for case in duplicate_cases:
        base = torch.arange(case["size"][case["dim"]], dtype=torch.int64) % 7
        view_shape = [1] * len(case["size"])
        view_shape[case["dim"]] = case["size"][case["dim"]]
        x = base.view(view_shape).expand(case["size"]).to(torch.float32)
        noise = torch.randn(case["size"], dtype=torch.float32) * 0.0
        x = x + noise

        def sort_test(inp):
            return torch.sort(inp, dim=case["dim"], descending=case["descending"], stable=True)

        out_values, out_indices = torch.compile(dynamic=False)(sort_test)(x.to(device=device))
        ref_values, ref_indices = torch.sort(
            x, dim=case["dim"], descending=case["descending"], stable=True
        )
        test_result(f"Sort.dup/stable_values {case}", out_values, ref_values)
        test_equal(f"Sort.dup/stable_indices {case}", out_indices, ref_indices)

        def sort_test_unstable(inp):
            return torch.sort(inp, dim=case["dim"], descending=case["descending"], stable=False)

        out_values_u, out_indices_u = torch.compile(dynamic=False)(sort_test_unstable)(x.to(device=device))
        ref_values_u, _ = torch.sort(x, dim=case["dim"], descending=case["descending"], stable=False)
        test_result(f"Sort.dup/unstable_values {case}", out_values_u, ref_values_u)
        gathered_u = torch.gather(x, case["dim"], out_indices_u.cpu())
        test_result(f"Sort.dup/unstable_gather {case}", gathered_u, out_values_u.cpu())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sort tests")
    parser.add_argument("--shape", type=str, default="(64, 32, 16)")
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--descending", action="store_true")
    args = parser.parse_args()

    shape = tuple(map(int, args.shape.strip("()").split(",")))

    from Scheduler.scheduler import PyTorchSimRunner

    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()

    test_sort_stable_suite(device)
    test_sort_duplicate_cases(device)