"""A template's config is Inductor's selection. In timing mode each offered config is timed on
TOGSim and the fewest cycles wins, with an epilogue fused when one is there; the pick is recorded,
and a recorded pick is built again, timed or not, without timing anything. Each compile runs in a
child process, since the timing mode is fixed for a process."""
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
from torch._inductor.select_algorithm import TritonTemplateCaller  # noqa: E402

from pytorchsim.driver import select  # noqa: E402

SHAPE = (128, 256, 256)

# One compile of OP, with select's record kept in SELECT_RECORD_DIR, and what was timed written to
# SELECT_TIMED as [config, cycles] (unfused choices) and [meta, cycles] (fusion benchmarks).
CHILD = r"""
import json, os, sys
import torch
from pytorchsim.codegen.inductor_hooks import TritonNPUScheduling
from pytorchsim.driver import select

select.RECORD_DIR = os.environ["SELECT_RECORD_DIR"]
timed = {"choices": [], "modules": []}

def time_choices(choices, _orig=select.time_choices):
    out = _orig(choices)
    for c, t in out.items():
        ktc = c.annotations.get("ktc")
        if ktc is not None and t != float("inf"):
            timed["choices"].append([ktc.params.to_kwargs(), t])
    return out
select.time_choices = time_choices

def bench(self, mod, *a, _orig=TritonNPUScheduling.benchmark_codegened_module, **k):
    t, path = _orig(self, mod, *a, **k)
    timed["modules"].append([self._benchmarked[mod.__file__][1], t])
    return t, path
TritonNPUScheduling.benchmark_codegened_module = bench

op = sys.argv[1]
m, k, n = map(int, sys.argv[2:5])
a = torch.randn(m, k).to(torch.bfloat16).to("npu:0")
b = torch.randn(k, n).to(torch.bfloat16).to("npu:0")
f = {"mm": lambda x, y: x @ y, "mm_relu": lambda x, y: torch.relu(x @ y),
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
    with open(os.path.join(dump, select.RECORD_LOG)) as f:
        return [json.loads(line) for line in f]


def _merge(root, dump):
    """scripts/selection_merge.py over a timed run's picks, into the record under `root`."""
    script = os.path.join(TORCHSIM_DIR, "scripts", "selection_merge.py")
    code = (f"import runpy, sys, torch; from pytorchsim.driver import select; "
            f"select.RECORD_DIR = {os.path.join(root, 'record')!r}; "
            f"sys.argv = [{script!r}, {os.path.join(dump, select.RECORD_LOG)!r}]; "
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
    with tempfile.TemporaryDirectory() as root:
        # timed, no epilogue: the pick is the fewest cycles, recorded, and built as it was timed
        mm, mm_timed = _compile(root, "mm_timed", "mm", timing=True)
        cycles = [t for _, t in mm_timed["choices"]]
        best = mm_timed["choices"][cycles.index(min(cycles))][0]
        (pick,) = _picks(mm)
        assert {k: v for k, v in pick["config"].items()
                if k not in ("template_id", "template_hash")} == best
        (built,) = _templates(mm).values()
        assert built["tile_source"] == select.TIMED
        bench = [json.load(open(os.path.join(mm, "select", d, "meta.json")))
                 for d in os.listdir(os.path.join(mm, "select"))]
        assert _same_kernel(built) in [_same_kernel(b) for b in bench]

        # timed, with an epilogue: Inductor times the top configs fused, and builds one kernel
        relu, relu_timed = _compile(root, "mm_relu_timed", "mm_relu", timing=True)
        fused = [m for m, _ in relu_timed["modules"] if m["template_grid"] is not None]
        assert 1 <= len(fused) <= len(relu_timed["choices"])
        (built_relu,) = _templates(relu).values()
        assert built_relu["tile_source"] == select.TIMED
        # the epilogue alone is timed at the blocks it would be built at (its arguments are the
        # fused graph's: there relu runs in place on the mm's output)
        alone = [m for m, _ in relu_timed["modules"] if m["template_grid"] is None]
        plain, _ = _compile(root, "relu", "relu", timing=False)
        (relu_kernel,) = _kernels(plain).values()
        blocks = lambda m: (m["numels"], m["fixed_config"], m["inside_reduction"])
        assert alone and all(blocks(m) == blocks(relu_kernel) for m in alone), (alone, relu_kernel)

        # recorded: built again, timed or not, as the timed run built it, and nothing is timed
        _merge(root, mm)
        for timing in (False, True):
            again, timed = _compile(root, f"mm_recorded_{int(timing)}", "mm", timing=timing)
            assert timed == {"choices": [], "modules": []}
            assert {k: m["tile_source"] for k, m in _templates(again).items()} == \
                {k: select.RECORDED for k in _templates(mm)}
        fused_again, _ = _compile(root, "mm_relu_recorded", "mm_relu", timing=False)
        assert [m["tile_source"] for m in _templates(fused_again).values()] == \
            [select.RECORDED_UNFUSED]

        # a record timed with another compiler is not used
        path = os.path.join(root, "record", os.listdir(os.path.join(root, "record"))[0])
        rec = json.load(open(path))
        rec["identity"]["compiler"] = "another"
        json.dump(rec, open(path, "w"))
        stale, _ = _compile(root, "mm_stale", "mm", timing=False)
        assert [m["tile_source"] for m in _templates(stale).values()] == [select.HEURISTIC]


def _choice(name):
    c = object.__new__(TritonTemplateCaller)
    c.name, c.description = name, name
    c.annotations = {"ktc": mock.Mock(template=mock.Mock(uid="triton::mm"), annotations={})}
    c.bmreq = mock.Mock(kernel_name="triton_mm", module_cache_key=name)
    return c


def test_a_refused_config_is_inf_and_none_left_is_an_error():
    a, b = _choice("a"), _choice("b")
    with mock.patch.object(select.kernel_metadata, "benchmark_request_meta",
                           return_value=("", {})), \
         mock.patch.object(select, "_time_tile",
                           side_effect=lambda src, meta, name, wd, cut: 7.0 if wd.endswith("b") else float("inf")):
        assert select.time_choices([a, b]) == {a: float("inf"), b: 7.0}
    with mock.patch.object(select.kernel_metadata, "benchmark_request_meta",
                           return_value=("", {})), \
         mock.patch.object(select, "_time_tile", return_value=float("inf")):
        try:
            select.time_choices([a, b])
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
    with mock.patch.object(select.kernel_metadata, "benchmark_request_meta", return_value=("", {})), \
         mock.patch.object(select, "_time_tile", side_effect=tile):
        t0 = time.monotonic()
        assert select.time_choices([slow, fast]) == {slow: float("inf"), fast: 5.0}
        assert time.monotonic() - t0 < 0.2 * select.CUT_FACTOR + 1


if __name__ == "__main__":
    test_a_config_running_past_the_cut_is_inf()
    test_a_refused_config_is_inf_and_none_left_is_an_error()
    test_select_record_and_rebuild()
    print("PASS")
