"""The resident mm (inductor/autotune/mm_resident.py): offered alone where copy_node.plan holds an operand, else
the mm template; an epilogue fuses into its store; compile, inference_mode, eager and grad give its values;
a launch alone times its copy node, then the kernel, and reports the sum."""
import json
import os
import sys

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402

_LAUNCHED = []

from pytorchsim import config, contract  # noqa: E402
from pytorchsim.frontend.inductor.autotune import mm_resident  # noqa: E402
from pytorchsim.simulator import timing  # noqa: E402


class _Node:
    def __init__(self, dtype):
        self.dtype = dtype

    def get_dtype(self):
        return self.dtype


class _Inputs:
    """What templates_for reads of Inductor's MMKernelInputs: device, dtypes, sizes."""

    def __init__(self, m, k, n, dtype=torch.bfloat16, out=None, device="npu"):
        self.device_type, self._mnk, self._dtype, self._out = device, (m, n, k), dtype, out or dtype

    def nodes(self):
        return [_Node(self._dtype), _Node(self._dtype)]

    def out_dtype(self):
        return self._out

    def mnk_symbolic(self):
        return self._mnk


def _offered(*args, **kwargs):
    from torch._inductor.kernel.mm import mm_template
    return mm_resident.templates_for(_Inputs(*args, **kwargs), ["aten", mm_template], "mm")


def test_a_held_mm_is_offered_the_resident_template_alone():
    assert _offered(512, 768, 768) == [mm_resident.mm_resident_template]


def test_declined_mms_keep_the_mm_template():
    """Breaks if a vector product, a dtype the dot does not take, an out_dtype, too large a pair or another
    device reaches the resident mm."""
    from torch._inductor.kernel.mm import mm_template
    for args, kwargs in (((1, 4096, 4096), {}), ((64, 64, 64), {"dtype": torch.float64}),
                         ((512, 768, 768), {"out": torch.float32}), ((8192, 8192, 8192), {}),
                         ((512, 768, 768), {"device": "cpu"})):
        assert _offered(*args, **kwargs) == ["aten", mm_template], (args, kwargs)


def test_its_config_is_the_plan():
    (cfg,) = mm_resident.configs(_Inputs(8192, 4096, 4096))
    assert cfg["RESIDENT"] == 0 and cfg["BLOCK_M"] == 8192 and cfg["BLOCK_K"] == 4096, cfg


def _launched(fn, *xs, **kw):
    """`fn` run on the npu over `xs` (compiled unless `eager`), with the meta of every kernel it launched."""
    from pytorchsim.backend import launch_hooks
    if not _LAUNCHED:
        launch_hooks.observe_every_launch(lambda la: _LAUNCHED and _LAUNCHED[-1].append(la.meta))
    _LAUNCHED.append([])
    torch._dynamo.reset()
    run = fn if kw.pop("eager", False) else torch.compile(fn, dynamic=False)
    out = run(*[x.to("npu:0") for x in xs], **kw)
    return out, _LAUNCHED.pop()


def _resident(metas):
    """The launches the resident mm made: the one config copy_node.plan gave (template_record.PLANNED)."""
    from pytorchsim.backend import template_record
    return [m for m in metas if m.get(template_record.SOURCE) == template_record.PLANNED]


def _close(got, want):
    assert torch.allclose(got.cpu().float(), want.float(), rtol=2e-2, atol=2e-1), (got, want)


def _ab(m=384, k=256, n=192, grad=False):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(m, k, generator=g).to(torch.bfloat16).requires_grad_(grad),
            torch.randn(k, n, generator=g).to(torch.bfloat16).requires_grad_(grad))


def test_compile_runs_the_resident_mm_with_its_epilogue_fused():
    """Breaks if relu(mm) takes a second kernel for its relu, or the mm leaves the resident template."""
    a, b = _ab()
    out, metas = _launched(lambda x, y: torch.relu(torch.mm(x, y)), a, b)
    _close(out, torch.relu(a.float() @ b.float()))
    assert len(metas) == 1 and len(_resident(metas)) == 1, metas


def test_inference_mode_compiles_through_it():
    a, b = _ab()
    with torch.inference_mode():
        out, metas = _launched(torch.mm, a, b)
    _close(out, a.float() @ b.float())
    assert _resident(metas), metas


def test_eager_compiles_its_mm_through_it():
    """Eager npu ops run through eager_to_compile: an eager mm launches the resident mm, with the right
    values."""
    a, b = _ab(m=320)
    out, metas = _launched(torch.mm, a, b, eager=True)
    _close(out, a.float() @ b.float())
    assert _resident(metas), metas


def test_grad_flows_through_it():
    """Breaks if the backward of a compiled mm loses its gradient: its mms are Inductor's too."""
    a, b = _ab(grad=True)
    torch._dynamo.reset()
    an, bn = a.detach().to("npu:0").requires_grad_(), b.detach().to("npu:0").requires_grad_()
    torch.compile(torch.mm, dynamic=False)(an, bn).float().sum().backward()
    ref_a, ref_b = a.detach().float().requires_grad_(), b.detach().float().requires_grad_()
    (ref_a @ ref_b).sum().backward()
    _close(an.grad, ref_a.grad)
    _close(bn.grad, ref_b.grad)


def test_a_launch_alone_sums_its_copy_node():
    """Breaks if simulate_standalone drops the copy node, or reports it without the kernel."""
    from pytorchsim.simulator.timing import togsim
    logs = {"copy": 300, "kernel": 1000}
    saved = (togsim.TOGSimProcess.run_standalone, togsim.TOGSimProcess.read_result, timing.copy_node_handles)
    togsim.TOGSimProcess.run_standalone = staticmethod(lambda h, a, **kw: "copy" if "copy" in h else "kernel")
    togsim.TOGSimProcess.read_result = staticmethod(lambda r: {"total_cycles": logs[r], "units": {}})
    timing.copy_node_handles = lambda wd: ["copy_node_1/tile_graph.onnx"]
    try:
        got = timing.simulate_standalone("wd", "mine", "attr")
    finally:
        togsim.TOGSimProcess.run_standalone, togsim.TOGSimProcess.read_result, timing.copy_node_handles = saved
    assert got["total_cycles"] == 1300 and got["kernel_cycles"] == 1000 and got["copy_node_cycles"] == [300], got


def test_a_timed_mm_carries_its_copy_node():
    """Breaks if the resident mm reaches TOGSim without the copy node of the operand it holds (timing mode
    only)."""
    if not config.pytorchsim_timing_mode:
        print("SKIP test_a_timed_mm_carries_its_copy_node: TORCHSIM_TIMING_MODE is off")
        return
    from pytorchsim.backend import launch_hooks
    workdirs = []
    launch_hooks.observe_every_launch(lambda la: workdirs.append(la.workdir))
    a, b = _ab(m=256, k=256, n=768)
    _launched(torch.mm, a, b)
    nodes = [(contract.kernel_object(w) or {}).get("copy_nodes") for w in workdirs]
    assert any(n and n[0]["arg"] == 1 and n[0]["bytes"] == 256 * 768 * 2 for n in nodes), json.dumps(nodes)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
