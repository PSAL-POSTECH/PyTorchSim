"""Drive the Triton codegen route as far as it currently goes.

This route is WIP (see PyTorchSimFrontend/triton_backend/README.md). The test is
written to report WHERE it stops rather than to assert success: the value right
now is a reproducible statement of the next gap, not a pass/fail gate. Register
it in .github/workflows/pytorchsim_test.yml only once the route runs end to end.

    TORCHSIM_TRITON_CODEGEN=1 python tests/system/test_triton_codegen.py
"""
import os
import sys
import traceback

# Must be set before torch_openreg registers the Inductor backend for `npu`.
os.environ.setdefault("TORCHSIM_TRITON_CODEGEN", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


N = 1024


def build():
    def fn(x, y):
        return x + y

    x = torch.randn(N)
    y = torch.randn(N)
    return fn, x, y


def main():
    from PyTorchSimFrontend import extension_config
    from PyTorchSimFrontend.triton_backend import tnpu_bridge

    print(f"TORCHSIM_TRITON_CODEGEN = {extension_config.CONFIG_TRITON_CODEGEN}")
    print(f"TNPU_DIR                = {extension_config.CONFIG_TNPU_DIR}")
    ok, _out = tnpu_bridge.doctor()
    print(f"tnpu doctor             = {'ok' if ok else 'FAILED (see run.py doctor)'}")
    print()

    fn, x, y = build()
    expected = fn(x, y)

    opt = torch.compile(fn, backend="inductor")
    try:
        got = opt(x.to("npu:0"), y.to("npu:0"))
    except Exception as e:  # noqa: BLE001 - the point is to report the stop
        print(f"STOPPED AT: {type(e).__name__}")
        print()
        traceback.print_exc()
        print()
        print("The stage reached is what this test measures; see the traceback "
              "above and README.md's gap list.")
        return 1

    err = (got.cpu() - expected).abs().max().item()
    print(f"max_abs_err = {err}")
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
