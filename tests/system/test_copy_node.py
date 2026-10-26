"""The resident mm (inductor/autotune/mm_resident.py): offered alone to the mm holding the graph's prefetch,
the mm template elsewhere; an epilogue fuses into its store; compile, inference_mode, eager and grad give
its values; a launch alone times its copy node, then the kernel, and reports the sum."""
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
    """What templates_for reads of Inductor's MMKernelInputs: device, dtypes, sizes (a bias node first for
    an addmm)."""

    def __init__(self, m, k, n, dtype=torch.bfloat16, out=None, device="npu", bias=False):
        self.device_type, self._mnk, self._dtype, self._out = device, (m, n, k), dtype, out or dtype
        self._count = 3 if bias else 2

    def nodes(self):
        return [_Node(self._dtype)] * self._count

    def out_dtype(self):
        return self._out

    def mnk_symbolic(self):
        return self._mnk


class _Graph:
    """What not_prefetched reads of Inductor's GraphLowering: the FX module and the node being lowered."""

    def __init__(self):
        self.graph = torch.fx.Graph()
        self.module = torch.fx.GraphModule(torch.nn.Module(), self.graph)
        self.current_node = None

    def input(self, name, *shape, dtype=torch.bfloat16):
        node = self.graph.placeholder(name)
        node.meta["val"] = torch.empty(*shape, dtype=dtype, device="meta")
        return node

    def t(self, node):
        out = self.graph.call_function(torch.ops.aten.permute.default, (node, [1, 0]))
        out.meta["val"] = node.meta["val"].permute(1, 0)
        return out

    def mm(self, a, b):
        node = self.graph.call_function(torch.ops.aten.mm.default, (a, b))
        node.meta["val"] = torch.empty(a.meta["val"].shape[0], b.meta["val"].shape[1],
                                       dtype=a.meta["val"].dtype, device="meta")
        return node

    def addmm(self, bias, a, b):
        node = self.graph.call_function(torch.ops.aten.addmm.default, (bias, a, b))
        node.meta["val"] = torch.empty(a.meta["val"].shape[0], b.meta["val"].shape[1],
                                       dtype=a.meta["val"].dtype, device="meta")
        return node


def _lone(m, k, n, dtype=torch.bfloat16):
    """A graph of one mm, x[m, k] @ w[k, n], and that mm."""
    g = _Graph()
    return g, g.mm(g.input("x", m, k, dtype=dtype), g.input("w", k, n, dtype=dtype))


def _offered(g, node, **kwargs):
    """The templates mm_resident offers `node` (an mm or addmm) of `g`, as its KernelInputs would read it."""
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.virtualized import V
    op = "addmm" if node.target is torch.ops.aten.addmm.default else "mm"
    a, b = (x.meta["val"] for x in node.args[-2:])
    g.current_node = node
    with V.set_graph_handler(g):
        return mm_resident.templates_for(_Inputs(a.shape[0], a.shape[1], b.shape[1], bias=op == "addmm",
                                                 **kwargs), ["aten", mm_template], op)


def test_a_held_mm_is_offered_the_resident_template_alone():
    assert _offered(*_lone(512, 768, 768)) == [mm_resident.mm_resident_template]


def test_declined_mms_keep_the_mm_template():
    """Breaks if a vector product, a dtype the dot does not take, an out_dtype, too large a pair or another
    device reaches the resident mm."""
    from torch._inductor.kernel.mm import mm_template
    for args, kwargs in (((1, 4096, 4096), {}), ((64, 64, 64), {"dtype": torch.float64}),
                         ((512, 768, 768), {"out": torch.float32}), ((8192, 8192, 8192), {}),
                         ((512, 768, 768), {"device": "cpu"})):
        g, node = _lone(*args, dtype=kwargs.get("dtype", torch.bfloat16))
        assert _offered(g, node, **kwargs) == ["aten", mm_template], (args, kwargs)


def test_the_largest_graph_input_held_is_the_one_prefetched():
    """Breaks if two mms holding graph inputs both run resident, if the smaller one does, or if an input
    reached through a view is not seen as one."""
    from torch._inductor.kernel.mm import mm_template
    g = _Graph()
    x, w1, w2 = g.input("x", 1024, 4096), g.input("w1", 4096, 1024), g.input("w2", 4096, 4096)
    small, big = g.mm(x, w1), g.mm(x, g.t(w2))
    assert _offered(g, big) == [mm_resident.mm_resident_template]
    assert _offered(g, small) == ["aten", mm_template]


def test_an_addmm_holding_the_largest_input_is_the_one_prefetched():
    """Breaks if an addmm is left out of the election (its weight the largest input, as a layer's
    down-projection with the residual added) or is not offered the resident mm."""
    from torch._inductor.kernel.mm import mm_template
    g = _Graph()
    x, w1, h, w2, r = (g.input("x", 1024, 4096), g.input("w1", 4096, 4096), g.input("h", 1024, 12288),
                       g.input("w2", 12288, 4096), g.input("r", 1024, 4096))
    proj, down = g.mm(x, w1), g.addmm(r, h, w2)
    assert _offered(g, down) == [mm_resident.mm_resident_template]
    assert _offered(g, proj) == ["aten", mm_template]


def test_an_mm_holding_a_computed_operand_keeps_the_mm_template():
    """Breaks if the resident mm is offered where XLA could not prefetch: the held operand is made in the
    graph, not an input of it."""
    from torch._inductor.kernel.mm import mm_template
    g = _Graph()
    x, w1, w2 = g.input("x", 1024, 4096), g.input("w1", 4096, 4096), g.input("w2", 4096, 24576)
    first = g.mm(x, w1)
    second = g.mm(first, w2)
    assert _offered(g, second) == ["aten", mm_template]
    assert _offered(g, first) == [mm_resident.mm_resident_template]


def test_its_config_is_the_plan():
    """Breaks if the config holds another operand than the graph's prefetch (x, the larger input here)."""
    from torch._inductor.virtualized import V
    g, node = _lone(8192, 4096, 4096)
    g.current_node = node
    with V.set_graph_handler(g):
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


def test_a_compiled_graph_runs_one_mm_resident():
    """Breaks if lowering a graph of two mms holding graph inputs offers the resident mm to both, or to the
    one holding the smaller input, or the values are off."""
    from torch._inductor import config as inductor_config
    offered, real = [], mm_resident.templates_for

    def spy(kernel_inputs, templates, op_name):
        from torch._inductor.kernel.mm import mm_template
        got = real(kernel_inputs, templates, op_name)
        if mm_template in templates:
            offered.append((mm_resident._label(kernel_inputs), got == [mm_resident.mm_resident_template]))
        return got

    a, b = _ab(m=256, k=1024, n=1024)
    c = torch.randn(1024, 2048, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    mm_resident.templates_for = spy
    try:
        with inductor_config.patch(fx_graph_cache=False):
            out, _ = _launched(lambda x, w1, w2: (x @ w1) @ w2, a, b, c)
    finally:
        mm_resident.templates_for = real
    want = (a.float() @ b.float()).to(torch.bfloat16).float() @ c.float()
    assert (out.cpu().float() - want).abs().max() <= 1e-2 * want.abs().max(), (out, want)
    assert sorted(offered) == [("256x1024x1024", False), ("256x1024x2048", True)], offered


def test_a_compiled_addmm_runs_resident_with_its_bias():
    """Breaks if an addmm is lowered without the resident mm, or its bias is not added in the epilogue."""
    from torch._inductor import config as inductor_config
    offered, real = [], mm_resident.templates_for

    def spy(kernel_inputs, templates, op_name):
        from torch._inductor.kernel.mm import mm_template
        got = real(kernel_inputs, templates, op_name)
        if mm_template in templates:
            offered.append((op_name, got == [mm_resident.mm_resident_template]))
        return got

    a, b = _ab(m=256, k=512, n=768)
    bias = torch.randn(256, 768, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    mm_resident.templates_for = spy
    try:
        with inductor_config.patch(fx_graph_cache=False):
            out, _ = _launched(torch.addmm, bias, a, b)
    finally:
        mm_resident.templates_for = real
    _close(out, bias.float() + a.float() @ b.float())
    assert offered == [("addmm", True)], offered


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


def test_a_timed_resident_candidate_holds_its_operand():
    """Breaks if a resident mm candidate is timed without the operand it holds: built as a kernel that holds
    nothing it gets half the bank, and a held operand over that is refused (timing mode only)."""
    if not config.pytorchsim_timing_mode:
        print("SKIP test_a_timed_resident_candidate_holds_its_operand: TORCHSIM_TIMING_MODE is off")
        return
    a, b = _ab(m=256, k=512, n=768)
    bias = torch.randn(256, 768, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    _launched(torch.addmm, bias, a, b)
    select = os.path.join(config.get_dump_path(), "select")
    metas = [json.load(open(p)) for p in (os.path.join(select, d, "meta.json") for d in os.listdir(select))
             if os.path.isfile(p)]
    resident = [m for m in metas if m["kernel_name"].startswith("triton_mm_resident")]
    assert resident and all(m["graph_prefetch"] == ["arg_B"] for m in resident), \
        [(m["kernel_name"], m["graph_prefetch"]) for m in resident]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
