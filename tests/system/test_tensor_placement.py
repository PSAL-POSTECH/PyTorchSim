"""In one TOGSim session kernel A writes x and kernel B reads it: both launches' tog_tensor_addrs.txt place
x at torch.npu.sim_address(x), live storages never overlap and a freed storage's range is reused. Breaks when
TOGSim gets no addresses, the allocator forgets a free, or addresses are host pointers or 0 per launch."""
import os
import sys

os.environ["TORCHSIM_TIMING_MODE"] = "1"
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402

ADDRS = "tog_tensor_addrs.txt"


def _launch_dirs(trace_path):
    """Each launch's directory in a session's trace, in order; a copy node's is its kernel's."""
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


def _addrs(launch_dir):
    """{role: [(addr, bytes)]} of a launch's tog_tensor_addrs.txt."""
    out = {}
    with open(os.path.join(launch_dir, ADDRS)) as f:
        for line in f:
            name, addr, size, align, role = line.split()
            assert int(addr) % int(align) == 0, line
            out.setdefault(role, []).append((int(addr), int(size)))
    return out


def _range(t):
    return torch.npu.sim_address(t), t.untyped_storage().nbytes()


def _disjoint(ranges):
    ranges = sorted(ranges)
    return all(a + n <= b for (a, n), (b, _) in zip(ranges, ranges[1:]))


def test_a_written_buffer_is_read_at_its_address():
    device = torch.device("npu:0")
    a = torch.randn(64, 256).to(device)
    write = torch.compile(lambda a: a + 1.0, dynamic=False)
    read = torch.compile(lambda x: x * 2.0, dynamic=False)
    with torch.no_grad(), torch.npu.TOGSimulator() as sim:
        x = write(a)
        y = read(x)
    torch.testing.assert_close(y.cpu(), (a.cpu() + 1.0) * 2.0)
    first, second = (_addrs(d) for d in _launch_dirs(sim.trace_path))
    x_at = (torch.npu.sim_address(x), x.numel() * x.element_size())
    assert first["out"] == [x_at] and second["in"] == [x_at], (first, second, x_at)
    assert first["in"] == [(torch.npu.sim_address(a), a.numel() * 4)], first
    assert _disjoint([_range(a), _range(x), _range(y)]), [_range(t) for t in (a, x, y)]


def test_live_storages_never_overlap_and_a_freed_range_is_reused():
    device = torch.device("npu:0")
    align = torch.npu.sim_address_alignment()
    live = [torch.empty(n, device=device) for n in (1, 4096, 4097, 300000, 7)]
    assert _disjoint([_range(t) for t in live]), [_range(t) for t in live]
    assert all(torch.npu.sim_address(t) % align == 0 for t in live)
    view = live[3][128:]
    assert torch.npu.sim_address(view) == torch.npu.sim_address(live[3]) + 128 * 4
    big = torch.empty(16 << 20, device=device)
    at = torch.npu.sim_address(big)
    del big
    again = torch.empty(16 << 20, device=device)
    assert torch.npu.sim_address(again) == at, (torch.npu.sim_address(again), at)
    assert _disjoint([_range(t) for t in live + [again]])


if __name__ == "__main__":
    test_a_written_buffer_is_read_at_its_address()
    test_live_storages_never_overlap_and_a_freed_range_is_reused()
    print("PASS")
