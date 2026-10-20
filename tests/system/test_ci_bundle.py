"""scripts/ci/bundle.py's workloads are the rows of the compiler's measurements: each suite's csv, its rows of the
suite's input dtype, named by their dimensions, and topk's T and E read as the vector suite's R and C."""
import csv
import importlib.util
import os
import sys
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported

from pytorchsim import config  # noqa: E402

ROWS = {
    "tpu_v6e_softmax_torchxla.csv": [{"R": 8, "C": 128, "dtype_in": "bf16"}, {"R": 8, "C": 256, "dtype_in": "f32"}],
    "tpu_v6e_load_torchxla.csv": [{"program": "max", "R": 16, "C": 128, "dtype_in": "bf16"}],
    "tpu_v6e_topk_torchxla.csv": [{"T": 32, "E": 128, "K": 8, "dtype_in": "f32"},
                                  {"T": 64, "E": 128, "K": 8, "dtype_in": "bf16"}],
}


def _bundle():
    """scripts/ci/bundle.py as a module."""
    path = os.path.join(TORCHSIM_DIR, "scripts", "ci", "bundle.py")
    spec = importlib.util.spec_from_file_location("ci_bundle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compiler_dir(root):
    """A compiler checkout holding only the vector suite's measured rows."""
    d = os.path.join(root, "validation", "vector")
    os.makedirs(d)
    for name, rows in ROWS.items():
        with open(os.path.join(d, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    return root


def test_vector_workloads_and_their_dtypes():
    """softmax and max at bf16, topk at f32 with R = T, C = E and K; a row of another dtype is not a workload."""
    bundle = _bundle()
    with tempfile.TemporaryDirectory() as root, \
         mock.patch.object(config, "CONFIG_TORCHSIM_COMPILE_DIR", _compiler_dir(root)):
        got = {name: (dims, shapes, dtype) for name, _, dims, _, shapes, dtype in bundle.workloads({"vector"})}
    assert got == {
        "vector_softmax_8x128_bf16": ({"program": "softmax", "R": 8, "C": 128}, [(8, 128)], "bf16"),
        "vector_max_16x128_bf16": ({"program": "max", "R": 16, "C": 128}, [(16, 128)], "bf16"),
        "vector_topk_32x128x8_f32": ({"program": "topk", "R": 32, "C": 128, "K": 8}, [(32, 128)], "f32"),
    }, got


def test_topk_runs_the_rows_k():
    """The topk workload is torch.topk over the last dim with the row's K."""
    bundle = _bundle()
    with tempfile.TemporaryDirectory() as root, \
         mock.patch.object(config, "CONFIG_TORCHSIM_COMPILE_DIR", _compiler_dir(root)):
        fn = next(fn for name, _, _, fn, _, _ in bundle.workloads({"vector"}) if name.startswith("vector_topk"))
    x = torch.randn(32, 128)
    values, indices = fn(x)
    ref = torch.topk(x, 8, dim=-1)
    assert torch.equal(values, ref.values) and torch.equal(indices, ref.indices)


if __name__ == "__main__":
    test_vector_workloads_and_their_dtypes()
    test_topk_runs_the_rows_k()
    print("PASS")
