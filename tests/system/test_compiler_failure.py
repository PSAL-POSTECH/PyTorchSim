"""Why a compile failed is read from the failure.json the compiler leaves, never from what it
printed: a scratchpad over its budget is a tile too big by usage / budget, and a kernel file the
compiler cannot load stops outside every stage and is no reason to shrink anything."""
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: F401  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim.driver import autotune, compiler_bridge


def _failed(spec_path):
    with tempfile.TemporaryDirectory() as workdir:
        try:
            compiler_bridge.run_pipeline(spec_path, workdir)
        except compiler_bridge.CompilerError as e:
            return e
    raise AssertionError(f"{spec_path} compiled")


def test_spad_overflow_is_too_big():
    spec = os.path.join(compiler_bridge.compiler_dir(),
                        "tests/cases/mechanism/tile/tile_spad_over_budget.py")
    e = _failed(spec)
    usage, budget = e.failure["usage"], e.failure["budget"]
    assert e.failure["kind"] == "spad-overflow" and usage > budget > 0
    assert autotune.too_big(e) == "%d bytes/lane over a budget of %d" % (usage, budget)


def test_unloadable_kernel_is_not_too_big():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write('raise ImportError("not a kernel file")\n')
    try:
        e = _failed(f.name)
    finally:
        os.unlink(f.name)
    assert e.failure is None and autotune.too_big(e) is None
    assert "not a kernel file" in str(e)


if __name__ == "__main__":
    test_spad_overflow_is_too_big()
    test_unloadable_kernel_is_not_too_big()
    print("PASS")
