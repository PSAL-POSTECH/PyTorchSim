"""Two launches of one kernel in a session (an mm on two weights, as k_proj and v_proj) get a directory each,
whose launch.json names the call's tensors by their IR buffers, an inline reinterpret_tensor(arg0_1, ...) as
arg0_1. Breaks when launches share one directory, or when names are read off strings or the caller's frame."""
import json
import os
import sys

os.environ["TORCHSIM_TIMING_MODE"] = "1"
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch.library import triton_op, wrap_triton  # noqa: E402


@triton.jit
def _add_one_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n, other=0.0) + 1.0, mask=offs < n)


@triton_op("torchsim_launch_record::add_one", mutates_args=())
def _add_one(x: torch.Tensor) -> torch.Tensor:
    o = torch.empty_like(x)
    wrap_triton(_add_one_kernel)[(x.numel() // 128, 1, 1)](x, o, x.numel(), 128)
    return o


def _launch_dirs(trace_path):
    """The launch directory of each launch in a session's trace, in order; a copy node's handle is in its
    kernel's."""
    out = []
    with open(trace_path) as f:
        for line in f:
            if line.startswith("LAUNCH_KERNEL"):
                d = os.path.dirname(line.split(",")[4])
                while d != "/" and not os.path.basename(d).startswith("launch-"):
                    d = os.path.dirname(d)
                if d not in out:
                    out.append(d)
    return out


def test_each_launch_keeps_its_own_record():
    device = torch.device("npu:0")
    g = torch.Generator().manual_seed(0)
    x, w1, w2 = (torch.randn(*s, generator=g).to(device) for s in ((64, 256), (256, 128), (256, 128)))
    f = torch.compile(lambda x, w: x @ w, dynamic=False)
    with torch.no_grad(), torch.npu.TOGSimulator() as sim:
        out1, out2 = f(x, w1), f(x, w2)
    torch.testing.assert_close(out1.cpu(), x.cpu() @ w1.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out2.cpu(), x.cpu() @ w2.cpu(), rtol=1e-3, atol=1e-3)
    dirs = _launch_dirs(sim.trace_path)
    assert len(dirs) == 2, f"two launches of one kernel share {dirs}"
    records = []
    for d in dirs:
        assert os.path.isfile(os.path.join(d, "launch.json")), f"{d} holds no launch record"
        with open(os.path.join(d, "launch.json")) as fh:
            records.append(json.load(fh))
    assert records[0]["kernel"] == records[1]["kernel"], records
    ptrs = [{a.get("data_ptr") for a in r["args"]} for r in records]
    assert w1.data_ptr() in ptrs[0] and w2.data_ptr() not in ptrs[0], records[0]
    assert w2.data_ptr() in ptrs[1] and w1.data_ptr() not in ptrs[1], records[1]
    assert all(a["name"] for r in records for a in r["args"] if "data_ptr" in a), records


def _records(trace_path):
    out = []
    for d in _launch_dirs(trace_path):
        with open(os.path.join(d, "launch.json")) as fh:
            out.append(json.load(fh))
    return out


def test_an_inline_view_is_named_by_its_buffer():
    from torch._inductor.utils import run_and_get_code
    x = torch.arange(512, dtype=torch.float32).to(torch.device("npu:0"))
    torch._dynamo.reset()
    with torch.no_grad(), torch.npu.TOGSimulator() as sim:
        out, code = run_and_get_code(torch.compile(lambda x: _add_one(x[128:384]), dynamic=False), x)
    torch.testing.assert_close(out.cpu(), x.cpu()[128:384] + 1)
    assert any("(reinterpret_tensor(arg0_1" in line for c in code for line in c.splitlines()), code
    (record,) = _records(sim.trace_path)
    first = record["args"][0]
    assert first["name"] == "arg0_1" and first["data_ptr"] == x.data_ptr() + 128 * 4, record


if __name__ == "__main__":
    test_each_launch_keeps_its_own_record()
    test_an_inline_view_is_named_by_its_buffer()
    print("PASS")
