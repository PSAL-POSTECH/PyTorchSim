"""Write the kernels PyTorchSim builds for the compiler's validation workloads:
    python scripts/ci/bundle.py OUT [SUITE...]      (every suite when none is named)

The workloads are the rows of the compiler's TPU measurements (validation/<suite>/tpu_v6e_*.csv), so
the bundle covers what the compiler compares against and nothing else. Each one is compiled and run
once, and every kernel it launches is copied as the compiler takes it. Run the gemm suite in timing
mode (TORCHSIM_TIMING_MODE=1, which a process keeps from its start), so a template's config is the
one Inductor picks by TOGSim cycles, or the one the selection record holds:

  OUT/bundle.json                     the PyTorchSim commit, torch, the compiler pin, the machine
  OUT/<workload>/manifest.json        suite, shape, dtype and the launches in order
  OUT/<workload>/<NN>_<kernel>/       <kernel>_spec.py, <kernel>_triton.py, meta.json

A template kernel neither timed nor recorded (tile_source "heuristic") fails the bundle: it is not the
kernel a timed run picks. One recorded without its fused epilogue ("recorded_unfused") is kept and
named in a warning. The picks the GEMMs were timed to are appended to selection.jsonl in the dump
path, for scripts/selection_merge.py.
"""

import csv
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402  (first: its device autoload installs pytorchsim)

from pytorchsim import config  # noqa: E402
from pytorchsim.driver import codecache, compiler_bridge, select  # noqa: E402
from pytorchsim.simulator import timing  # noqa: E402

#: The head dimension the compiler's attention measurements were taken at.
HEAD_DIM = 128

DTYPE = "bf16"


def _attention(kind, Hq, Hkv, S):
    """SDPA as a model calls it: prefill is causal over S queries, decode is one query over S keys."""
    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=kind == "prefill", enable_gqa=True)
    L = S if kind == "prefill" else 1
    return sdpa, [(1, Hq, L, HEAD_DIM), (1, Hkv, S, HEAD_DIM), (1, Hkv, S, HEAD_DIM)]


#: The vector programs, as the compiler's measurements name them.
PROGRAMS = {
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "max": lambda x: torch.max(x).float(),
    "sum": lambda x: x.float().sum(),
    "add1": lambda x: x + 1,
    "slice": lambda x: x[:1, :1].float() + 1,
}

#: (suite, the csv under validation/<suite>/, the columns that name a workload, the program a row
#: without a `program` column runs, the function and its input shapes). The suite is the compiler's.
SUITES = [
    ("gemm", "tpu_v6e_torchxla.csv", ("M", "K", "N"), None,
     lambda d: (torch.matmul, [(d["M"], d["K"]), (d["K"], d["N"])])),
    ("vector", "tpu_v6e_softmax_torchxla.csv", ("program", "R", "C"), "softmax",
     lambda d: (PROGRAMS[d["program"]], [(d["R"], d["C"])])),
    ("vector", "tpu_v6e_load_torchxla.csv", ("program", "R", "C"), None,
     lambda d: (PROGRAMS[d["program"]], [(d["R"], d["C"])])),
    ("attention", "tpu_v6e_attention_torchxla.csv", ("kind", "Hq", "Hkv", "S"), None,
     lambda d: _attention(**d)),
]

#: Columns that name a program or a kind rather than a size.
TEXT_COLUMNS = {"program", "kind"}


def workloads(suites=None):
    """(name, suite, dims, fn, input shapes) per measured row of `suites` (default all)."""
    for suite, csv_name, keys, program, call in SUITES:
        if suites and suite not in suites:
            continue
        with open(os.path.join(compiler_bridge.compiler_dir(), "validation", suite, csv_name)) as f:
            for row in csv.DictReader(f):
                if row["dtype_in"] != DTYPE:
                    continue
                row = dict(row, program=program) if program else row
                dims = {k: row[k] if k in TEXT_COLUMNS else int(row[k]) for k in keys}
                fn, shapes = call(dims)
                name = "_".join([suite, *(dims[k] for k in keys if k in TEXT_COLUMNS),
                                 "x".join(str(dims[k]) for k in keys if k not in TEXT_COLUMNS), DTYPE])
                yield name, suite, dims, fn, shapes


LAUNCHED = []
codecache.observe_every_launch(LAUNCHED.append)


def kernel_files(launcher):
    name = launcher.kernel_name
    return [f"{name}_spec.py", f"{name}_triton.py", timing.META_JSON]


def capture(name, fn, shapes, out):
    """Run `fn` once and copy each kernel it launches into out/<name>; its launches in order."""
    LAUNCHED.clear()
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(*s, generator=g).to(torch.bfloat16).to("npu:0") for s in shapes]
    torch.compile(fn, dynamic=False)(*xs)
    dirs, launches = {}, []
    for launcher in LAUNCHED:
        if launcher.workdir not in dirs:
            d = f"{len(dirs):02d}_{launcher.kernel_name}"
            os.makedirs(os.path.join(out, name, d))
            for f in kernel_files(launcher):
                shutil.copy(os.path.join(launcher.workdir, f), os.path.join(out, name, d, f))
            dirs[launcher.workdir] = d
        launches.append({"kernel": launcher.kernel_name, "dir": dirs[launcher.workdir],
                         "tile_source": launcher.meta.get("tile_source")})
    return launches


def pts_commit():
    return os.environ.get("PTS_COMMIT") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=config.CONFIG_TORCHSIM_DIR).stdout.strip()


def main(out, suites=None):
    """Write `suites`' workloads (default all) into `out`, replacing each one's directory there."""
    os.makedirs(out, exist_ok=True)
    unpicked, unfused = [], []
    for name, suite, dims, fn, shapes in workloads(suites):
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)
        launches = capture(name, fn, shapes, out)
        with open(os.path.join(out, name, "manifest.json"), "w") as f:
            json.dump({"workload": name, "suite": suite, **dims, "dtype": DTYPE, "launches": launches},
                      f, indent=1)
        unpicked += [(name, la["kernel"]) for la in launches if la["tile_source"] == select.HEURISTIC]
        unfused += [(name, la["kernel"]) for la in launches
                    if la["tile_source"] == select.RECORDED_UNFUSED]
        print(f"[bundle] {name}: {[la['kernel'] for la in launches]}", flush=True)
    with open(os.path.join(out, "bundle.json"), "w") as f:
        json.dump({"pytorchsim": pts_commit(), "torch": torch.__version__, **select.identity()}, f,
                  indent=1, sort_keys=True)
    for name, kernel in unfused:
        print(f"[bundle] warning: {name} {kernel} is the recorded pick without its fused epilogue")
    if unpicked:
        sys.exit(f"[bundle] neither timed nor recorded: {unpicked}")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), set(sys.argv[2:]))
