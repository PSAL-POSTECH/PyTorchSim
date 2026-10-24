"""Write the kernels PyTorchSim builds for the compiler's validation workloads:
    python scripts/ci/bundle.py OUT [SUITE...] [--jobs N]     (every suite when none is named)

The workloads are the rows of the compiler's TPU measurements (validation/<suite>/tpu_v6e_*.csv), so
the bundle covers what the compiler compares against and nothing else. Each one is compiled and run
once, and every kernel it launches is copied as the compiler takes it. Run the gemm and layer suites
in timing mode (TORCHSIM_TIMING_MODE=1, which a process keeps from its start), so a template's config
is the one Inductor picks by TOGSim cycles, or the one the selection record holds:

  OUT/bundle.json                     the PyTorchSim commit, torch, the compiler pin, the machine
  OUT/<workload>/manifest.json        suite, shape, dtype, the launches in order, and the externs:
                                      each call Inductor left to aten (not simulated), with `at`, the
                                      launches before it
  OUT/<workload>/<NN>_<kernel>/       <kernel>_spec.py, <kernel>_triton.py, and <kernel>_inputs.pt
                                      when its manifest reads runtime attributes from args: their
                                      launch values, which its make_inputs takes instead of drawing

The layer suite holds only the rows under the compiler's CI cap (validation/layer/simulate_shapes.py
under_cap). --jobs N builds the workloads in N processes, each taking the next one, largest first by
the compiler's rank for the suite (token_pairs for the layer, its pool's cost, M*K*N, for the
GEMMs); a process keeps its timing mode.

Only what the compiler builds from: how a config was chosen (timed now, or recorded by an earlier run)
is not in the bundle, so the same kernels are the same bundle.

A template kernel neither timed nor recorded (tile_source "heuristic") fails the bundle: it is not the
kernel a timed run picks. One recorded without its fused epilogue ("recorded_unfused") is kept and
named in a warning. The picks the GEMMs were timed to are appended to selection.jsonl in the dump
path, for scripts/selection_merge.py.

--reuse PREV copies PREV/out/<workload> when PREV/keys/<workload>.json holds the key this run writes to
KEYS (--keys, default OUT/../keys, not in OUT): a hash of env PTS_INPUTS and BASE_IMAGE, torch, the
selection identity, the run settings and the row. Without PTS_INPUTS nothing is keyed or reused.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402  (first: its device autoload installs pytorchsim)

from pytorchsim import config, contract  # noqa: E402
from pytorchsim.backend import compiler_bridge, compiler_input, launch_hooks, template_record  # noqa: E402
from pytorchsim.simulator import functional  # noqa: E402

#: The head dimension the compiler's attention measurements were taken at.
HEAD_DIM = 128

DTYPE = "bf16"


def randn(*shapes, scale=1.0):
    """The maker of a workload's inputs: from a generator, one bf16 normal tensor per shape, in order."""
    return lambda g: [(torch.randn(*s, generator=g) * scale).to(torch.bfloat16) for s in shapes]


def _attention(kind, Hq, Hkv, S):
    """SDPA as a model calls it: prefill is causal over S queries, decode is one query over S keys."""
    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=kind == "prefill", enable_gqa=True)
    L = S if kind == "prefill" else 1
    return sdpa, randn((1, Hq, L, HEAD_DIM), (1, Hkv, S, HEAD_DIM), (1, Hkv, S, HEAD_DIM))


#: The vector programs, as the compiler's measurements name them.
PROGRAMS = {
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "max": lambda x: torch.max(x).float(),
    "sum": lambda x: x.float().sum(),
    "add1": lambda x: x + 1,
    "slice": lambda x: x[:1, :1].float() + 1,
}

#: Qwen3-30B-A3B's decoder layer as validation/layer measured it (tpu-perf-archive
#: harness/qwen_moe_layer_v1.py): its config, and each piece written as that harness writes it.
HID, HQ, HKV, E, TOPK, FF = 2048, 32, 4, 128, 8, 768
EPS, THETA = 1e-6, 1e6


def rmsnorm(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


def rope(x, cos, sin):
    """x [T, H, 128], cos / sin [T, 128] (each half repeated): x * cos + rotate_half(x) * sin."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return (x.float() * cos[:, None, :] + rot.float() * sin[:, None, :]).to(x.dtype)


def rope_tables(t, start=0):
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    ang = torch.outer(torch.arange(start, start + t, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], dim=-1)
    return [ang.cos(), ang.sin()]


def router_post(logits):
    p = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(p, TOPK, dim=-1)
    return (w / w.sum(-1, keepdim=True)).to(logits.dtype), ids


def uniform_ids(t):
    """[T, 8] expert ids: token i takes experts (8i .. 8i+7) mod 128, so every expert gets T*8/128 rows."""
    i = torch.arange(t, dtype=torch.int64)[:, None] * TOPK + torch.arange(TOPK, dtype=torch.int64)[None, :]
    return i % E


def group_sizes(t):
    return torch.bincount(uniform_ids(t).reshape(-1), minlength=E).to(torch.int32)


def dispatch(x, ids):
    """(x's rows in expert order [T*8, 2048], the order, rows per expert)."""
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sizes = (flat[:, None] == torch.arange(E, device=flat.device)[None, :]).sum(0).to(torch.int32)
    return x.index_select(0, order // TOPK), order, sizes


def inverse(ids):
    """Where each (token, expert) pair sits in expert order, counted as the harness's `positions` unsort
    (the csv's unsort column): its expert's offset plus the pairs of that expert before it."""
    flat = ids.reshape(-1)
    oh = (flat[:, None] == torch.arange(E, device=flat.device)[None, :]).to(torch.int32)
    within = (torch.cumsum(oh, 0) * oh).sum(1) - 1
    sizes = oh.sum(0)
    start = torch.cumsum(sizes, 0) - sizes
    return start.index_select(0, flat) + within


def combine(y, ids, w):
    """y [T*8, 2048] in expert order back to token order, weighted, summed over the 8."""
    t = w.shape[0]
    return (y.index_select(0, inverse(ids)).view(t, TOPK, -1).float() * w.float()[..., None]).sum(1).to(y.dtype)


def silu_mul(gu):
    f = gu.shape[-1] // 2
    g, u = gu[:, :f], gu[:, f:]
    return (torch.nn.functional.silu(g.float()) * u.float()).to(gu.dtype)


def gmm(x, w, sizes):
    """torch_xla's gmm (rows of x grouped by `sizes`, group e times w[e]) as PyTorch states it."""
    return torch.nn.functional.grouped_mm(x, w, offs=torch.cumsum(sizes, 0, dtype=torch.int32))


#: The layer's weights in the order the harness draws them, each (shape, scale, offset).
WEIGHTS = {"ln1": ((HID,), 0.1, 1), "ln2": ((HID,), 0.1, 1), "qn": ((HEAD_DIM,), 0.1, 1),
           "kn": ((HEAD_DIM,), 0.1, 1), "wq": ((HID, HQ * HEAD_DIM), HID ** -0.5, 0),
           "wk": ((HID, HKV * HEAD_DIM), HID ** -0.5, 0), "wv": ((HID, HKV * HEAD_DIM), HID ** -0.5, 0),
           "wo": ((HQ * HEAD_DIM, HID), (HQ * HEAD_DIM) ** -0.5, 0), "wr": ((HID, E), HID ** -0.5, 0),
           "wgu": ((E, HID, 2 * FF), HID ** -0.5, 0), "wd": ((E, FF, HID), FF ** -0.5, 0)}


def _weights(g):
    return [randn(s, scale=k)(g)[0] + b for s, k, b in WEIGHTS.values()]


def _layer(name, T, S):
    """The decoder layer, inputs (x, cos, sin, uniform ids, weights..., decode's K and V cache). Decode is one
    token over a cache of S, its own K/V not written; the cache is [1, 4, S, 128] each, the layout the
    decode kernel reads as ragged paged attention reads its pages."""
    def run(x, cos, sin, ids, *rest):
        W = dict(zip(WEIGHTS, rest))
        t = x.shape[0]
        h = rmsnorm(x, W["ln1"])
        q = (h @ W["wq"]).view(t, HQ, HEAD_DIM)
        k = (h @ W["wk"]).view(t, HKV, HEAD_DIM)
        v = (h @ W["wv"]).view(t, HKV, HEAD_DIM)
        q = rope(rmsnorm(q, W["qn"]), cos, sin)
        k = rope(rmsnorm(k, W["kn"]), cos, sin)
        if name == "prefill":
            k, v = k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        else:
            k, v = rest[len(WEIGHTS):]
        a = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k, v, is_causal=name == "prefill", enable_gqa=True)
        a = a[0].transpose(0, 1).reshape(t, HQ * HEAD_DIM)
        x2 = x + a @ W["wo"]
        h2 = rmsnorm(x2, W["ln2"])
        w, _ = router_post(h2 @ W["wr"])
        xs, order, sizes = dispatch(h2, ids)
        y = gmm(silu_mul(gmm(xs, W["wgu"], sizes)), W["wd"], sizes)
        return x2 + combine(y, ids, w)

    def make(g):
        W = _weights(g)
        cos_sin = rope_tables(T, start=S - T)
        cache = randn((1, HKV, S, HEAD_DIM), (1, HKV, S, HEAD_DIM))(g) if name == "decode" else []
        return randn((T, HID))(g) + cos_sin + [uniform_ids(T)] + W + cache
    return run, make


def _op(name, T):
    """One piece of the layer alone at the layer's shapes for T tokens, with the harness's operands."""
    m = T * TOPK
    norm = lambda g: randn((HEAD_DIM,), scale=0.1)(g)[0] + 1
    return {
        "rmsnorm": (rmsnorm, lambda g: randn((T, HID))(g) + [randn((HID,), scale=0.1)(g)[0] + 1]),
        "q_norm": (rmsnorm, lambda g: randn((T, HQ, HEAD_DIM))(g) + [norm(g)]),
        "k_norm": (rmsnorm, lambda g: randn((T, HKV, HEAD_DIM))(g) + [norm(g)]),
        "rope_q": (rope, lambda g: randn((T, HQ, HEAD_DIM))(g) + rope_tables(T)),
        "rope_k": (rope, lambda g: randn((T, HKV, HEAD_DIM))(g) + rope_tables(T)),
        "residual": (lambda a, b: a + b, randn((T, HID), (T, HID))),
        "router_post": (lambda lg: router_post(lg)[0], randn((T, E))),
        "dispatch": (lambda x, ids: dispatch(x, ids)[0], lambda g: randn((T, HID))(g) + [uniform_ids(T)]),
        "gmm_gate_up": (gmm, lambda g: randn((m, HID))(g) + randn((E, HID, 2 * FF), scale=HID ** -0.5)(g)
                        + [group_sizes(T)]),
        "silu_mul": (silu_mul, randn((m, 2 * FF))),
        "gmm_down": (gmm, lambda g: randn((m, FF))(g) + randn((E, FF, HID), scale=FF ** -0.5)(g)
                     + [group_sizes(T)]),
        "combine": (combine, lambda g: randn((m, HID))(g) + [uniform_ids(T), randn((T, TOPK), scale=0.1)(g)[0].abs()]),
    }[name]


def _qwen_layer(section, name, T, S):
    return _layer(name, T, S) if section == "layer" else _op(name, T)


#: Qwen3-8B's decoder layer as validation/qwen3_8b measured it (tpu-perf-archive harness/qwen3_8b_layer_v1.py):
#: its config, and each piece written as that harness writes it; rmsnorm, rope and silu_mul are the 30B's.
D8_HID, D8_HQ, D8_HKV, D8_FF = 4096, 32, 8, 12288
D8_GEMMS = {"q_proj": (D8_HID, D8_HQ * HEAD_DIM), "k_proj": (D8_HID, D8_HKV * HEAD_DIM),
            "v_proj": (D8_HID, D8_HKV * HEAD_DIM), "o_proj": (D8_HQ * HEAD_DIM, D8_HID),
            "gate_up": (D8_HID, 2 * D8_FF), "down": (D8_FF, D8_HID)}
D8_WEIGHTS = {"ln1": ((D8_HID,), 0.1, 1), "ln2": ((D8_HID,), 0.1, 1), "qn": ((HEAD_DIM,), 0.1, 1),
              "kn": ((HEAD_DIM,), 0.1, 1), "wq": ((D8_HID, D8_HQ * HEAD_DIM), D8_HID ** -0.5, 0),
              "wk": ((D8_HID, D8_HKV * HEAD_DIM), D8_HID ** -0.5, 0),
              "wv": ((D8_HID, D8_HKV * HEAD_DIM), D8_HID ** -0.5, 0),
              "wo": ((D8_HQ * HEAD_DIM, D8_HID), (D8_HQ * HEAD_DIM) ** -0.5, 0),
              "wgu": ((D8_HID, 2 * D8_FF), D8_HID ** -0.5, 0), "wd": ((D8_FF, D8_HID), D8_FF ** -0.5, 0)}


def _dense_layer(name, T, S):
    """Qwen3-8B's decoder layer, inputs (x, cos, sin, weights..., decode's K and V cache): the 30B's attention
    half, cache layout and decode as _layer has them, then one dense MLP."""
    def run(x, cos, sin, *rest):
        W = dict(zip(D8_WEIGHTS, rest))
        t = x.shape[0]
        h = rmsnorm(x, W["ln1"])
        q = (h @ W["wq"]).view(t, D8_HQ, HEAD_DIM)
        k = (h @ W["wk"]).view(t, D8_HKV, HEAD_DIM)
        v = (h @ W["wv"]).view(t, D8_HKV, HEAD_DIM)
        q = rope(rmsnorm(q, W["qn"]), cos, sin)
        k = rope(rmsnorm(k, W["kn"]), cos, sin)
        if name == "prefill":
            k, v = k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        else:
            k, v = rest[len(D8_WEIGHTS):]
        a = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k, v, is_causal=name == "prefill", enable_gqa=True)
        a = a[0].transpose(0, 1).reshape(t, D8_HQ * HEAD_DIM)
        x2 = x + a @ W["wo"]
        return x2 + silu_mul(rmsnorm(x2, W["ln2"]) @ W["wgu"]) @ W["wd"]

    def make(g):
        W = [randn(s, scale=k)(g)[0] + b for s, k, b in D8_WEIGHTS.values()]
        cache = randn((1, D8_HKV, S, HEAD_DIM), (1, D8_HKV, S, HEAD_DIM))(g) if name == "decode" else []
        return randn((T, D8_HID))(g) + rope_tables(T, start=S - T) + W + cache
    return run, make


def _dense_op(name, T, S):
    """One piece of Qwen3-8B's layer alone for T tokens (an attention op over S), with the harness's operands."""
    if name in D8_GEMMS:
        k, n = D8_GEMMS[name]
        return torch.matmul, lambda g: randn((T, k))(g) + randn((k, n), scale=k ** -0.5)(g)
    if name in ("attn_prefill", "attn_decode"):
        return _attention(name[len("attn_"):], D8_HQ, D8_HKV, S)
    norm = lambda g: randn((HEAD_DIM,), scale=0.1)(g)[0] + 1
    return {
        "rmsnorm": (rmsnorm, lambda g: randn((T, D8_HID))(g) + [randn((D8_HID,), scale=0.1)(g)[0] + 1]),
        "q_norm": (rmsnorm, lambda g: randn((T, D8_HQ, HEAD_DIM))(g) + [norm(g)]),
        "k_norm": (rmsnorm, lambda g: randn((T, D8_HKV, HEAD_DIM))(g) + [norm(g)]),
        "rope_q": (rope, lambda g: randn((T, D8_HQ, HEAD_DIM))(g) + rope_tables(T)),
        "rope_k": (rope, lambda g: randn((T, D8_HKV, HEAD_DIM))(g) + rope_tables(T)),
        "residual": (lambda a, b: a + b, randn((T, D8_HID), (T, D8_HID))),
        "silu_mul": (silu_mul, randn((T, 2 * D8_FF))),
    }[name]


def _dense_qwen_layer(section, name, T, S):
    return _dense_layer(name, T, S) if section == "layer" else _dense_op(name, T, S)


#: (suite, the csv under validation/<suite>/, the columns that name a workload, the program a row
#: without a `program` column runs, the function and the maker of its inputs from a generator). The
#: suite is the compiler's.
SUITES = [
    ("gemm", "tpu_v6e_torchxla.csv", ("M", "K", "N"), None,
     lambda d: (torch.matmul, randn((d["M"], d["K"]), (d["K"], d["N"])))),
    ("gemm", "tpu_v6e_qwen3moe_torchxla.csv", ("M", "K", "N"), None,
     lambda d: (torch.matmul, randn((d["M"], d["K"]), (d["K"], d["N"])))),
    ("vector", "tpu_v6e_softmax_torchxla.csv", ("program", "R", "C"), "softmax",
     lambda d: (PROGRAMS[d["program"]], randn((d["R"], d["C"])))),
    ("vector", "tpu_v6e_load_torchxla.csv", ("program", "R", "C"), None,
     lambda d: (PROGRAMS[d["program"]], randn((d["R"], d["C"])))),
    ("attention", "tpu_v6e_attention_torchxla.csv", ("kind", "Hq", "Hkv", "S"), None,
     lambda d: _attention(**d)),
    ("layer", "tpu_v6e_qwen3moe_layer_torchxla.csv", ("section", "name", "T", "S"), None,
     lambda d: _qwen_layer(**d)),
    ("qwen3_8b", "tpu_v6e_qwen3_8b_layer_torchxla.csv", ("section", "name", "T", "S"), None,
     lambda d: _dense_qwen_layer(**d)),
]

#: Columns that name a program, a kind or a piece of a model rather than a size.
TEXT_COLUMNS = {"program", "kind", "section", "name"}


def compiler_suite(suite):
    """The compiler's validation/<suite>/simulate_shapes.py, which defines its CI cap and rank."""
    path = os.path.join(compiler_bridge.compiler_dir(), "validation", suite, "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location(f"{suite}_simulate_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The suites the bundle holds only under the compiler's CI cap, by the compiler's own definition.
CAPPED = ("layer", "qwen3_8b")


def workloads(suites=None):
    """(name, suite, dims, fn, inputs maker, csv) per measured row of `suites` (default all); a CAPPED
    suite's rows over the compiler's CI cap are left out."""
    for suite, csv_name, keys, program, call in SUITES:
        if suites and suite not in suites:
            continue
        under_cap = compiler_suite(suite).under_cap if suite in CAPPED else None
        with open(os.path.join(compiler_bridge.compiler_dir(), "validation", suite, csv_name)) as f:
            for row in csv.DictReader(f):
                if row["dtype_in"] != DTYPE:
                    continue
                row = dict(row, program=program) if program else row
                dims = {k: row[k] if k in TEXT_COLUMNS else int(row[k]) for k in keys}
                if under_cap and not under_cap(dims):
                    continue
                fn, make = call(dims)
                name = "_".join([suite, *(dims[k] for k in keys if k in TEXT_COLUMNS),
                                 "x".join(str(dims[k]) for k in keys if k not in TEXT_COLUMNS), DTYPE])
                yield name, suite, dims, fn, make, csv_name


LAUNCHED = []
launch_hooks.observe_every_launch(LAUNCHED.append)
EXTERNS = []
launch_hooks.observe_every_extern(lambda call: EXTERNS.append({"call": call, "at": len(LAUNCHED)}))


def attribute_inputs(launcher, args):
    """The launch values, as the kernel sees them, of the args `launcher`'s manifest says its runtime
    attributes are read from; a kernel with no manifest stops the bundle by name."""
    manifest = contract.kernel_object(launcher.workdir)
    if manifest is None:
        raise RuntimeError(f"[bundle] {launcher.kernel_name}: {launcher.workdir} holds no "
                           f"{contract.KERNEL_MANIFEST}, so the args its runtime attributes are read from "
                           f"are unknown")
    names = contract.attribute_sources(manifest)
    return {m["name"]: functional.flat_input(m, t) for m, t in functional.tensor_args(launcher.meta, args)
            if m["name"] in names}


RECORDED = []
launch_hooks.observe_every_launch_inputs(lambda launcher, args: RECORDED.append(attribute_inputs(launcher, args)))


def kernel_files(launcher):
    name = launcher.kernel_name
    return [f"{name}_spec.py", f"{name}_triton.py"]


def copy_kernel(launcher, recorded, dest):
    """`launcher`'s kernel files into `dest`; with `recorded` ({arg: launch value}) its kernel file is written
    again to take those args from the values saved beside it."""
    if not recorded:
        for f in kernel_files(launcher):
            shutil.copy(os.path.join(launcher.workdir, f), os.path.join(dest, f))
        return
    spec, triton = kernel_files(launcher)
    with open(os.path.join(launcher.workdir, triton)) as f:
        src = f.read()
    compiler_input.write(src, launcher.meta, os.path.join(dest, spec), recorded=recorded)


def same_values(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def capture(name, fn, make, out):
    """Run `fn` once on make's inputs and copy each kernel it launches into out/<name>; its launches in
    order, their tile sources and the extern calls. A kernel launched again on other values of the args its
    runtime attributes are read from stops the bundle: its directory holds one launch's. This process's launch
    directories in the kernels' workdirs go after, their .raw tensors being only the launch's."""
    LAUNCHED.clear()
    EXTERNS.clear()
    RECORDED.clear()
    torch._dynamo.reset()
    xs = [x.to("npu:0") for x in make(torch.Generator().manual_seed(0))]
    torch.compile(fn, dynamic=False)(*xs)
    dirs, kept, launches = {}, {}, []
    for launcher, recorded in zip(LAUNCHED, RECORDED, strict=True):
        if launcher.workdir not in dirs:
            d = f"{len(dirs):02d}_{launcher.kernel_name}"
            os.makedirs(os.path.join(out, name, d))
            copy_kernel(launcher, recorded, os.path.join(out, name, d))
            dirs[launcher.workdir], kept[launcher.workdir] = d, recorded
        elif not same_values(kept[launcher.workdir], recorded):
            raise RuntimeError(f"[bundle] {name}: {launcher.kernel_name} is launched again on other values of "
                               f"{sorted(recorded)}, which its runtime attributes are read from")
        launches.append({"kernel": launcher.kernel_name, "dir": dirs[launcher.workdir]})
    for workdir in dirs:
        shutil.rmtree(contract.dir_for(workdir))
    return launches, [launcher.meta.get("tile_source") for launcher in LAUNCHED], list(EXTERNS)


def pts_commit():
    return os.environ.get("PTS_COMMIT") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=config.CONFIG_TORCHSIM_DIR).stdout.strip()


KEYED_ENV = ("TORCHSIM_", "TORCHINDUCTOR_", "TRITON_")
PER_RUN_ENV = {"TORCHSIM_DUMP_PATH", "TORCHSIM_LOG_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"}


def keyed_env():
    """The environment PyTorchSim, Inductor and Triton read, less the cache and output paths each run
    picks for itself (Inductor sets TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR while compiling)."""
    return {k: v for k, v in os.environ.items() if k.startswith(KEYED_ENV) and k not in PER_RUN_ENV}


def key_inputs(suite, name, dims, csv_name):
    """What decides workload `name`'s kernels, as ids and values; None without env PTS_INPUTS."""
    pts = os.environ.get("PTS_INPUTS", "").split()
    if not pts:
        return None
    return {"pts": pts, "base_image": os.environ.get("BASE_IMAGE", ""), "torch": torch.__version__,
            "env": keyed_env(),
            "identity": template_record.identity(), "target": config.CONFIG_TORCHSIM_TARGET,
            "run": {s: getattr(config, s) for s in config.RUN_SETTINGS},
            "workload": {"suite": suite, "csv": csv_name, "name": name, "dims": dims, "dtype": DTYPE}}


def workload_key(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def write_key(keys, name, inputs, sources, externs):
    os.makedirs(keys, exist_ok=True)
    with open(os.path.join(keys, f"{name}.json"), "w") as f:
        json.dump({"key": workload_key(inputs), "inputs": inputs, "tile_sources": sources,
                   "externs": externs}, f, indent=1, sort_keys=True)


def build(out, suite, name, keys=None):
    """Build `suite`'s workload `name` into out/<name>, and its key into keys/<name>.json unless a launch
    is unpicked: (name, kernels launched, unpicked, unfused, externs)."""
    dims, fn, make, csv_name = next(w[2:] for w in workloads({suite}) if w[0] == name)
    shutil.rmtree(os.path.join(out, name), ignore_errors=True)
    if keys:
        _remove(os.path.join(keys, f"{name}.json"))
    launches, sources, externs = capture(name, fn, make, out)
    with open(os.path.join(out, name, "manifest.json"), "w") as f:
        json.dump({"workload": name, "suite": suite, **dims, "dtype": DTYPE, "launches": launches,
                   "externs": externs}, f, indent=1)
    unpicked = [la["kernel"] for la, s in zip(launches, sources) if s == template_record.HEURISTIC]
    unfused = [la["kernel"] for la, s in zip(launches, sources) if s == template_record.RECORDED_UNFUSED]
    calls = [e["call"] for e in externs]
    inputs = key_inputs(suite, name, dims, csv_name)
    if keys and inputs is not None and not unpicked:
        write_key(keys, name, inputs, sources, calls)
    return name, [la["kernel"] for la in launches], unpicked, unfused, calls


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def reuse(prev, out, keys, suite, name, dims, csv_name):
    """Copy prev/out/<name> into out/<name> when prev/keys/<name>.json holds today's key and no unpicked
    launch: (name, kernels, [], unfused, externs), else None."""
    inputs = key_inputs(suite, name, dims, csv_name)
    try:
        with open(os.path.join(prev, "keys", f"{name}.json")) as f:
            old = json.load(f)
        with open(os.path.join(prev, "out", name, "manifest.json")) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    if inputs is None or old.get("key") != workload_key(inputs) \
            or template_record.HEURISTIC in old.get("tile_sources", [template_record.HEURISTIC]):
        return None
    shutil.rmtree(os.path.join(out, name), ignore_errors=True)
    shutil.copytree(os.path.join(prev, "out", name), os.path.join(out, name))
    write_key(keys, name, inputs, old["tile_sources"], old["externs"])
    kernels = [la["kernel"] for la in manifest["launches"]]
    unfused = [k for k, s in zip(kernels, old["tile_sources"]) if s == template_record.RECORDED_UNFUSED]
    return name, kernels, [], unfused, old["externs"]


def largest_first(suites):
    """(suite, workload name) of `suites`, each suite's ordered by the compiler's rank where it has one."""
    names = []
    for suite in dict.fromkeys(s[0] for s in SUITES if not suites or s[0] in suites):
        mine = [(w[0], w[2]) for w in workloads({suite})]
        rank = _rank(suite)
        names += [(suite, n) for n, d in (sorted(mine, key=lambda w: -rank(w[1])) if rank else mine)]
    return names


def _rank(suite):
    """The compiler's cost of a workload of `suite`: token_pairs for a capped suite, else its pool's `cost`."""
    mod = compiler_suite(suite)
    if suite in CAPPED:
        return getattr(mod, "token_pairs", None)
    keys = next(s[2] for s in SUITES if s[0] == suite)
    return (lambda d: mod.cost(d, keys)) if hasattr(mod, "cost") else None


def main(out, suites=None, jobs=1, keys=None, prev=None):
    """Write `suites`' workloads (default all) into `out`, replacing each one's directory there, in
    `jobs` processes; each one whose key `prev` holds is copied from it instead of built."""
    os.makedirs(out, exist_ok=True)
    keys = keys or os.path.join(os.path.dirname(out), "keys")
    unpicked, unfused = [], []
    names, reused = largest_first(suites), []
    if prev:
        rows = {(w[1], w[0]): (w[2], w[5]) for w in workloads(suites)}
        reused = [r for r in (reuse(prev, out, keys, s, n, *rows[s, n]) for s, n in names) if r]
        names = [n for n in names if n[1] not in {r[0] for r in reused}]
    if jobs > 1:
        pool = multiprocessing.get_context("spawn").Pool(jobs)
        built = pool.imap_unordered(_build, [(out, *n, keys) for n in names])
    else:
        pool, built = None, (build(out, *n, keys) for n in names)
    for name, kernels, up, uf, externs in reused:
        unfused += [(name, k) for k in uf]
        print(f"[bundle] {name}: reused {kernels}" + (f" extern {externs}" if externs else ""), flush=True)
    for name, kernels, up, uf, externs in built:
        unpicked += [(name, k) for k in up]
        unfused += [(name, k) for k in uf]
        print(f"[bundle] {name}: {kernels}" + (f" extern {externs}" if externs else ""), flush=True)
    print(f"[bundle] {len(reused)} reused, {len(names)} built", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    with open(os.path.join(out, "bundle.json"), "w") as f:
        json.dump({"pytorchsim": pts_commit(), "torch": torch.__version__, **template_record.identity()}, f,
                  indent=1, sort_keys=True)
    for name, kernel in unfused:
        print(f"[bundle] warning: {name} {kernel} is the recorded pick without its fused epilogue")
    if unpicked:
        sys.exit(f"[bundle] neither timed nor recorded: {unpicked}")


def _build(args):
    return build(*args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("suites", nargs="*", help="suites to build (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="processes building workloads at once")
    ap.add_argument("--keys", help="where the workloads' keys are written (default: OUT/../keys)")
    ap.add_argument("--reuse", metavar="PREV", help="a previous bundle (PREV/out, PREV/keys) to copy matches from")
    a = ap.parse_args()
    main(os.path.abspath(a.out), set(a.suites), a.jobs, a.keys and os.path.abspath(a.keys),
         a.reuse and os.path.abspath(a.reuse))
