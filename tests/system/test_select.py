"""A template's config is Inductor's selection: in timing mode each offered config of a bmm is timed on TOGSim
and the fewest cycles wins, an epilogue fused when one is there; the pick is recorded and built again untimed.
Each compile runs in a child process. A bmm, as an mm copy_node.plan holds is planned, not timed."""
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)

import torch  # noqa: E402
from torch._inductor.select_algorithm import TritonTemplateCaller  # noqa: E402

from pytorchsim.backend import template_record, template_timing  # noqa: E402

SHAPE = (128, 256, 256)

#: One compile of OP, template_timing's record kept in SELECT_RECORD_DIR and what was timed written to
#: SELECT_TIMED as [config, cycles] (unfused choices) and [meta, cycles] (fusion benchmarks).
CHILD = r"""
import json, os, sys
import torch
from pytorchsim.frontend.inductor.codegen.scheduling import TritonNPUScheduling
from pytorchsim.backend import template_timing
from pytorchsim.backend import template_record

template_record.RECORD_DIR = os.environ["SELECT_RECORD_DIR"]
timed = {"choices": [], "modules": []}

def time_choices(choices, _orig=template_timing.time_choices):
    out = _orig(choices)
    for c, t in out.items():
        ktc = c.annotations.get("ktc")
        if ktc is not None and t != float("inf"):
            timed["choices"].append([ktc.params.to_kwargs(), t])
    return out
template_timing.time_choices = time_choices

def bench(self, mod, *a, _orig=TritonNPUScheduling.benchmark_codegened_module, **k):
    t, path = _orig(self, mod, *a, **k)
    timed["modules"].append([self._benchmarked[mod.__file__][1], t])
    return t, path
TritonNPUScheduling.benchmark_codegened_module = bench

op = sys.argv[1]
m, k, n = map(int, sys.argv[2:5])
a = torch.randn(2, m, k).to(torch.bfloat16).to("npu:0")
b = torch.randn(2, k, n).to(torch.bfloat16).to("npu:0")
f = {"bmm": torch.bmm, "bmm_relu": lambda x, y: torch.relu(torch.bmm(x, y)),
     "relu": lambda x, y: torch.relu(x)}[op]
torch.compile(dynamic=False)(f)(a, b)
with open(os.environ["SELECT_TIMED"], "w") as fh:
    json.dump(timed, fh)
"""


def _compile(root, name, op, timing):
    """(dump path, what was timed) of one compile in a child process."""
    dump = os.path.join(root, name)
    os.makedirs(dump, exist_ok=True)
    env = dict(os.environ, TORCHSIM_TIMING_MODE=str(int(timing)), TORCHSIM_FUNCTIONAL_MODE="0",
               TORCHSIM_DUMP_PATH=dump, SELECT_RECORD_DIR=os.path.join(root, "record"),
               SELECT_TIMED=os.path.join(dump, "timed.json"))
    proc = subprocess.run([sys.executable, "-c", CHILD, op, *map(str, SHAPE)], env=env,
                          cwd=TORCHSIM_DIR, capture_output=True, text=True)
    assert proc.returncode == 0, f"{name}: {proc.stderr[-4000:]}"
    with open(env["SELECT_TIMED"]) as f:
        return dump, json.load(f)


def _kernels(dump):
    """{workdir name: meta} of every kernel the compile built."""
    out = {}
    for name in os.listdir(dump):
        meta = os.path.join(dump, name, "meta.json")
        if name.startswith("triton_") and os.path.isfile(meta):
            with open(meta) as f:
                out[name] = json.load(f)
    return out


def _templates(dump):
    return {k: m for k, m in _kernels(dump).items() if "tile_source" in m}


def _picks(dump):
    with open(os.path.join(dump, template_record.RECORD_LOG)) as f:
        return [json.loads(line) for line in f]


def _merge(root, dump):
    """scripts/selection_merge.py over a timed run's picks, into the record under `root`."""
    script = os.path.join(TORCHSIM_DIR, "scripts", "selection_merge.py")
    code = (f"import runpy, sys, torch; from pytorchsim.backend import template_record; "
            f"template_record.RECORD_DIR = {os.path.join(root, 'record')!r}; "
            f"sys.argv = [{script!r}, {os.path.join(dump, template_record.RECORD_LOG)!r}]; "
            f"runpy.run_path({script!r}, run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], cwd=TORCHSIM_DIR, capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]


def _same_kernel(meta):
    """What decides a kernel's cycles, without the names it is called by."""
    return {"signature": meta["signature"], "constants": meta["constants"],
            "args": [(a["role"], a["dtype"], a["numel"]) for a in meta["args"]],
            "numels": meta["numels"], "fixed_config": meta["fixed_config"],
            "template_grid": meta["template_grid"], "inside_reduction": meta["inside_reduction"]}


def test_select_record_and_rebuild():
    """Timed alone: the fewest cycles is picked, recorded and built as timed. Timed with an epilogue: the top configs
    are timed fused, the epilogue alone at its built blocks. Recorded: built again untimed. A record timed with
    another compiler is not used."""
    with tempfile.TemporaryDirectory() as root:
        mm, mm_timed = _compile(root, "mm_timed", "bmm", timing=True)
        cycles = [t for _, t in mm_timed["choices"]]
        best = mm_timed["choices"][cycles.index(min(cycles))][0]
        (pick,) = _picks(mm)
        assert {k: v for k, v in pick["config"].items()
                if k not in ("template_id", "template_hash")} == best
        (built,) = _templates(mm).values()
        assert built["tile_source"] == template_record.TIMED
        bench = [json.load(open(os.path.join(mm, "select", d, "meta.json")))
                 for d in os.listdir(os.path.join(mm, "select"))]
        assert _same_kernel(built) in [_same_kernel(b) for b in bench]

        relu, relu_timed = _compile(root, "mm_relu_timed", "bmm_relu", timing=True)
        fused = [m for m, _ in relu_timed["modules"] if m["template_grid"] is not None]
        assert 1 <= len(fused) <= len(relu_timed["choices"])
        (built_relu,) = _templates(relu).values()
        assert built_relu["tile_source"] == template_record.TIMED
        alone = [m for m, _ in relu_timed["modules"] if m["template_grid"] is None]
        plain, _ = _compile(root, "relu", "relu", timing=False)
        (relu_kernel,) = _kernels(plain).values()
        blocks = lambda m: (m["numels"], m["fixed_config"], m["inside_reduction"])
        assert alone and all(blocks(m) == blocks(relu_kernel) for m in alone), (alone, relu_kernel)

        _merge(root, mm)
        for timing in (False, True):
            again, timed = _compile(root, f"mm_recorded_{int(timing)}", "bmm", timing=timing)
            assert timed == {"choices": [], "modules": []}
            assert {k: m["tile_source"] for k, m in _templates(again).items()} == \
                {k: template_record.RECORDED for k in _templates(mm)}
        fused_again, _ = _compile(root, "mm_relu_recorded", "bmm_relu", timing=False)
        assert [m["tile_source"] for m in _templates(fused_again).values()] == \
            [template_record.RECORDED_UNFUSED]

        path = os.path.join(root, "record", os.listdir(os.path.join(root, "record"))[0])
        rec = json.load(open(path))
        rec["identity"]["compiler"] = "another"
        json.dump(rec, open(path, "w"))
        stale, _ = _compile(root, "mm_stale", "bmm", timing=False)
        assert [m["tile_source"] for m in _templates(stale).values()] == [template_record.HEURISTIC]


def _choice(name, uid="triton::mm"):
    c = object.__new__(TritonTemplateCaller)
    c.name, c.description = name, name
    c.annotations = {"ktc": mock.Mock(template=mock.Mock(uid=uid), annotations={})}
    c.bmreq = mock.Mock(kernel_name="triton_mm", module_cache_key=name)
    return c


def _rendered(timed=None):
    """Patches for time_choices off Inductor: each choice's source is its name, no prefetch, nothing timed yet."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(mock.patch.object(template_timing.kernel_metadata, "benchmark_request_meta",
                                          side_effect=lambda req: (req.module_cache_key, {"args": []})))
    stack.enter_context(mock.patch.object(template_timing.graph_prefetch, "choice_statement",
                                          return_value=[]))
    stack.enter_context(mock.patch.dict(template_timing._timed, timed or {}, clear=True))
    return stack


def test_a_refused_config_is_inf_and_none_left_is_an_error():
    a, b = _choice("a"), _choice("b")
    with _rendered(), \
         mock.patch.object(template_timing, "_time_tile",
                           side_effect=lambda src, meta, name, wd, cut: 7.0 if wd.endswith("b") else float("inf")):
        assert template_timing.time_choices([a, b]) == {a: float("inf"), b: 7.0}
    with _rendered(), \
         mock.patch.object(template_timing, "_time_tile", return_value=float("inf")):
        try:
            template_timing.time_choices([a, b])
        except RuntimeError as e:
            assert "no template config" in str(e)
        else:
            raise AssertionError("every config refused, and time_choices returned")


def test_a_config_running_past_the_cut_is_inf():
    """The configs run at once; one still simulating CUT_FACTOR times after the first finished is cut."""
    import time

    def tile(src, meta, name, wd, cut):
        if wd.endswith("fast"):
            time.sleep(0.2)
            return 5.0
        while not cut():
            time.sleep(0.05)
        return float("inf")
    fast, slow = _choice("fast"), _choice("slow")
    with _rendered(), mock.patch.object(template_timing, "_time_tile", side_effect=tile):
        t0 = time.monotonic()
        assert template_timing.time_choices([slow, fast]) == {slow: float("inf"), fast: 5.0}
        assert time.monotonic() - t0 < 0.2 * template_timing.CUT_FACTOR + 1


def test_a_config_timed_once_is_not_timed_again():
    """A config rendered the same at another node takes the cycles this process timed it at."""
    tile = mock.Mock(side_effect=lambda src, meta, name, wd, cut: 3.0 if wd.endswith("a") else 4.0)
    with _rendered(), mock.patch.object(template_timing, "_time_tile", tile):
        a, b = _choice("a"), _choice("b")
        assert template_timing.time_choices([a, b]) == {a: 3.0, b: 4.0}
        a2, b2 = _choice("a"), _choice("b")
        assert template_timing.time_choices([a2, b2]) == {a2: 3.0, b2: 4.0}
    assert tile.call_count == 2


def _machine(root, sub, **change):
    """tpuv6e's machine file with `change` applied, at <root>/<sub>/m.yml: one name, other content."""
    import yaml
    from pytorchsim import config
    with open(os.path.join(config.CONFIG_TOGSIM_CONFIG_DIR, "tpuv6e.yml")) as f:
        raw = yaml.safe_load(f)
    raw.update(change)
    os.makedirs(os.path.join(root, sub))
    path = os.path.join(root, sub, "m.yml")
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return path


def test_graph_cache_key_names_the_machine():
    """Inductor's FX cache key differs between machines with no recorded picks, so one machine's
    cached graph is not handed to another: the core count and the lanes' scratchpad shape it."""
    from pytorchsim.frontend.inductor.registration import npu_choices_class
    with tempfile.TemporaryDirectory() as root:
        paths = [_machine(root, "base"), _machine(root, "cores", num_cores=2),
                 _machine(root, "spad", vpu_spad_size_kb_per_lane=512)]
        keys = []
        for path in paths:
            with mock.patch.dict(os.environ, TORCHSIM_TARGET=path), \
                 mock.patch.object(template_record, "RECORD_DIR", os.path.join(root, "record")):
                assert template_record.record_digest() == "none"
                keys.append(npu_choices_class()().uuid())
                assert keys[-1] == npu_choices_class()().uuid()
        assert len(set(keys)) == len(keys), keys


def test_a_convolution_config_is_timed():
    """Breaks if a convolution's tile is taken in the order it was offered: with its template out of
    TIMED_TEMPLATES, times() is False and pick_config ranks by index, so tile 0 always wins."""
    from torch._inductor.kernel.conv import conv2d_template, conv3d_template
    for uid in (conv2d_template.uid, conv3d_template.uid):
        assert template_record.is_timed(_choice("conv", uid=uid)), uid
    assert template_timing.times([_choice("conv", uid=conv2d_template.uid)])


def test_a_convolution_is_offered_every_tile_a_timed_run_ranks():
    """Breaks if the conv heuristic offers one tile while the strategy autotunes: a lone candidate is
    taken whatever its cycles, so timing it decides nothing (rule 6)."""
    from pytorchsim import config
    from pytorchsim.frontend.inductor.autotune import template_kernel_configs
    from torch._inductor.heuristics.template.triton import BaseConfigHeuristic
    from torch._inductor.virtualized import V

    graph = mock.Mock(sizevars=mock.Mock(optimization_hint=int))
    with V.set_graph_handler(graph), \
         mock.patch.object(BaseConfigHeuristic, "preprocess_mm_configs",
                           lambda self, m, n, k, configs, **kw: configs):
        got = template_kernel_configs.conv_config_heuristic().get_conv_configs()(
            25600, 256, 512, dtype_size=2)
    blocks = {(c.block_m, c.block_n, c.block_k) for c in got}
    assert 1 < len(blocks) <= config.codegen_autotune_template_topk, blocks


if __name__ == "__main__":
    test_a_convolution_config_is_timed()
    test_a_convolution_is_offered_every_tile_a_timed_run_ranks()
    test_graph_cache_key_names_the_machine()
    test_a_config_running_past_the_cut_is_inf()
    test_a_refused_config_is_inf_and_none_left_is_an_error()
    test_a_config_timed_once_is_not_timed_again()
    test_select_record_and_rebuild()
    print("PASS")
