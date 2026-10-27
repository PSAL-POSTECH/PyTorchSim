"""TORCHSIM_NPUWATTCH=1: one npuwattch_<run-id>.json beside the TOGSim results, in the
pytorchsim.run_stats 1.1 form NPUWattch reads. Each standalone launch is one kernel entry: its TOGSim
results and logs, and its kernel object's gem5 samples, MLIR and metadata, totalled by the TOG's rows."""

import datetime
import json
import math
import os
import re
import subprocess
import threading
import uuid

import yaml

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge

logger = config.setup_logger()

ENV = "TORCHSIM_NPUWATTCH"
PREFIX = "npuwattch"
SCHEMA = "pytorchsim.run_stats"
SCHEMA_VERSION = "1.1"

MM_OPS = {"mm", "matmul", "addmm", "linear", "mv"}
BMM_OPS = {"bmm", "baddbmm"}
CONV_OPS = {"convolution", "conv2d", "conv1d", "conv3d", "_convolution"}
SFU_CLASSES = {"vexp": "CustomVexp", "sfu_push": "CustomVexp2", "verf": "CustomVerf",
               "vtanh": "CustomVtanh", "vsin": "CustomVsin", "vcos": "CustomVcos"}
CLASSES = ("CustomMatMul", "CustomMatMulwVpush", "CustomMatMuliVpush", "CustomMatMulvpop",
           "CustomVexp", "CustomVexp2", "CustomVerf", "CustomVtanh", "CustomVsin", "CustomVcos")
ROWS_IN_WORD = {0: 1, 1: 2, 3: 2}
POP_BYTES = 4
DTYPE_BITS = {"f32": 32, "f16": 16, "bf16": 16, "f64": 64, "i8": 8, "i16": 16, "i32": 32,
              "float32": 32, "float16": 16, "bfloat16": 16, "int8": 8, "int16": 16, "int32": 32}
DTYPE_NAME = {"f32": "fp32", "f16": "fp16", "bf16": "bf16", "f64": "fp64", "float32": "fp32",
              "float16": "fp16", "bfloat16": "bf16", "float64": "fp64"}

INST = re.compile(r"Core \[(\d+)\] : (MOVIN|MOVOUT|COMP|MEMORY_BAR)\s+inst_count: (\d+)")
NUMA = re.compile(r"Core \[(\d+)\] : NUMA local memory:\s*(\d+) requests?, remote memory:\s*(\d+)")
DRAM_ECHO = re.compile(r"\[Config/DRAM\] Total bandwidth [\d.]+ GB/s, ([\d.]+) MHz, (\d+) channels, "
                       r"(\d+) bytes per request")
RAMULATOR_ECHO = "[Config/DRAM] Ramulator2 configuration :"
DRAM_STATS = "=== DRAM statistics ==="
CHANNEL = re.compile(r"^---\s*channel\s+(\d+)\s*---")
STAT_LINE = re.compile(r"^\s*(\w+):\s*(-?\d+)\s*$")
POLICY = re.compile(r"^\s*row_policy:\s*$")
IMPL = re.compile(r"^\s*impl:\s*(\w+)\s*$")
CORE_READ = re.compile(r"^read_row_(?:hits|misses|conflicts)_core_(\d+)$")
BOOKSIM_KEY = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(\S+)\s*$")
PACKET_LENGTH = re.compile(r"^Injected packet length average\s*=\s*([\d.eE+-]+)\s*$", re.M)
NODE_DEF = re.compile(r"togsim::Node (\w+) = \{\.id = (\d+), \.count = (\d+)\}")
COMPUTE = re.compile(r"togsim::compute\(\w+, \{\.node = (\w+)(?:, \.variant = (\d+))?")
WORK_ITEM = re.compile(r"^static void tog_work_item\(.*?^}", re.M | re.S)
VCIX_IV = re.compile(r'"vcix\.iv"\(%\w+,\s*(%\w+)\)\s*<\{imm = (-?\d+) : i64, opcode = (\d+) : i64[^}]*\}>'
                     r"\s*\{pytorchsim\.tile_id = (\d+) : i64\}")
VL_CONST = re.compile(r"^%c(\d+)_i64$")
MAP_DEF = re.compile(r"^(#\w+)\s*=\s*affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>", re.M)
DOT = re.compile(r'"pytorchsim\.dot"\([^)]*\)\s*<\{indexing_maps\s*=\s*\[([^\]]*)\]\}>\s*:\s*\('
                 r"(memref<[^>]*>)\s*,\s*(memref<[^>]*>)\s*,\s*(memref<[^>]*>)\)")
MATH_VECTOR = re.compile(r"math\.(?:exp2?|erf|tanh|sin|cos|log|atan)\b[^\n:]*:\s*vector<(\d+)x")


def enabled():
    """Whether this run writes npuwattch.json."""
    return os.environ.get(ENV, "0") not in ("0", "", "no", "false", "False")


class Run:
    """One process's kernel entries and the file they are written to after each launch."""

    def __init__(self):
        self.kernels = []
        self.config = None
        self.producers = None
        self.lock = threading.Lock()
        self.run_id = (datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])

    def path(self):
        """Where this run's npuwattch.json goes: beside the TOGSim results."""
        return os.path.join(config.CONFIG_TORCHSIM_LOG_PATH, f"{PREFIX}_{self.run_id}.json")


RUN = Run()


def record(workdir, launch_dir, result_path, copy_result_paths=()):
    """Add the standalone launch whose TOGSim result is `result_path` (its copy nodes' in
    `copy_result_paths`) to this run's npuwattch.json; a failure is logged, never raised."""
    if not enabled():
        return
    try:
        with RUN.lock:
            machine = _machine()
            if RUN.config is None:
                RUN.config = run_config(machine, result_path)
                RUN.producers = producers(result_path)
            entry = kernel_entry(workdir, launch_dir, machine, RUN.config, result_path, copy_result_paths,
                                 len(RUN.kernels))
            RUN.kernels.append(entry)
            _write()
    except Exception as e:
        logger.warning("[npuwattch] %s: no kernel entry (%s: %s)", os.path.basename(workdir),
                       type(e).__name__, e)


def _write():
    """Rewrite this run's file with every entry so far."""
    os.makedirs(os.path.dirname(RUN.path()), exist_ok=True)
    doc = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "pytorchsim_version": _git_describe(),
           "producers": RUN.producers, "config": RUN.config, "kernels": RUN.kernels}
    with open(RUN.path(), "w") as f:
        json.dump(doc, f, indent=2)


def _machine():
    """The machine file TOGSim timed on, parsed, and the directory its relative paths start from."""
    path = compiler_bridge.target().file
    with open(path) as f:
        return {"path": path, "dir": os.path.dirname(path), "yaml": yaml.safe_load(f)}


def _git_describe():
    """`git describe` of the PyTorchSim checkout, or None."""
    try:
        out = subprocess.run(["git", "-C", config.CONFIG_TORCHSIM_DIR, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def producers(result_path):
    """The versions of the tools behind the counters: TOGSim's result version, gem5's banner, and the
    compiler pin, which fixes the BookSim2 and Ramulator2 TOGSim carries."""
    with open(result_path) as f:
        togsim = json.load(f).get("version")
    pin_path = os.path.join(config.CONFIG_TORCHSIM_DIR, "third_party", "pytorchsim-triton-backend.json")
    try:
        with open(pin_path) as f:
            pin = json.load(f)["pytorchsim_triton_backend"]["ref"]
    except (OSError, ValueError, KeyError):
        pin = None
    gem5 = None
    try:
        out = subprocess.run([compiler_bridge.toolchain().GEM5], capture_output=True, text=True, timeout=30)
        found = re.search(r"gem5 version (\S+)", out.stdout + out.stderr)
        gem5 = found.group(1) if found else None
    except (OSError, subprocess.SubprocessError, AttributeError, compiler_bridge.CompilerError):
        pass
    return {"togsim": togsim, "gem5": gem5, "booksim2": f"compiler pin {pin}" if pin else None,
            "ramulator2": f"compiler pin {pin}" if pin else None}


def _log_of(result_path):
    """The log TOGSim wrote beside its result."""
    return os.path.splitext(result_path)[0] + ".log"


def _ramulator(text):
    """The Ramulator2 configuration the log echoes, or None."""
    i = text.find(RAMULATOR_ECHO)
    if i < 0:
        return None
    lines = []
    for line in text[i + len(RAMULATOR_ECHO):].splitlines()[1:]:
        if line.startswith("["):
            break
        lines.append(line)
    try:
        return json.loads("\n".join(lines))
    except ValueError:
        return None


def _refresh(ram):
    """(refresh manager, banks one all-bank refresh covers, DRAM standard) of the first controller."""
    try:
        ctrl = ram["memory_system"]["controllers"][0]
        count = ctrl["dram"]["org"]["count"]
        banks = int(count[2]) * int(count[3]) if len(count) == 6 else None
        return ctrl.get("refresh_manager", {}).get("impl"), banks, ctrl["dram"].get("impl")
    except (KeyError, IndexError, TypeError):
        return None, None, None


def _booksim(text):
    """The BookSim2 [config] block of the log as key -> value, or None."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "[config]")
    except StopIteration:
        return None
    out = {}
    for line in lines[start + 1:]:
        s = line.strip()
        if s.startswith("["):
            break
        m = BOOKSIM_KEY.match(s)
        if m:
            v = m.group(2)
            out[m.group(1)] = int(v) if v.lstrip("-").isdigit() else v
    return out


def run_config(machine, result_path):
    """The config section, from the machine file and the first launch's log."""
    y = machine["yaml"]
    text = open(_log_of(result_path), errors="ignore").read()
    lanes = int(y["vpu_num_lanes"])
    vcix = y.get("vcix") or {}
    dim = int(vcix.get("tpu_mxu_registers", 1)) * lanes
    zvl = next((int(e[3:-1]) for e in y.get("isa", []) if str(e).startswith("zvl") and str(e).endswith("b")),
               None)
    echo = DRAM_ECHO.search(text)
    _, _, standard = _refresh(_ramulator(text))
    types = y.get("core_type")
    core_type = (sorted(set(types))[0] if isinstance(types, list) and len(set(types)) == 1
                 else types if types else "ws_mesh")
    noc = {"icnt_type": y.get("icnt_type")}
    if y.get("icnt_type") == "booksim2":
        bs = _booksim(text) or {}
        path = y.get("booksim_config_path")
        noc.update({
            "artifacts_dir": os.path.normpath(os.path.join(machine["dir"], path)) if path else None,
            "topology": bs.get("topology"), "n": bs.get("n"), "k": bs.get("k"),
            "flit_size_byte": bs.get("flit_size"), "input_buffer_size_flits": bs.get("input_buffer_size"),
            "freq_mhz": y.get("icnt_freq_mhz"),
            "dies": [{"id": 0, "routers": [0], "cores": list(range(int(y.get("num_cores", 1)))),
                      "dram_channels": list(range(int(y["dram_channels"])))}],
            "routers": [],
        })
    return {
        "num_cores": int(y.get("num_cores", 1)),
        "core_type": core_type,
        "vpu_num_lanes": lanes,
        "vpu_vector_length_bits": zvl,
        "vpu_spad_size_kb_per_lane": y.get("vpu_spad_size_kb_per_lane"),
        "systolic_array": {"width": dim, "height": dim,
                           "per_core": int(y.get("num_systolic_array_per_core", 1))},
        "core_freq_mhz": y.get("core_freq_mhz"),
        "dram": {"type": standard.lower() if standard else y.get("dram_type"),
                 "channels": int(y["dram_channels"]),
                 "req_size_byte": int(echo.group(3)) if echo else y.get("dram_req_size_byte"),
                 "freq_mhz": float(echo.group(1)) if echo else y.get("dram_freq_mhz")},
        "noc": noc,
    }


def _launch_counters(result_path):
    """The counters of one TOGSim launch: its result file, and from its log the instruction counts,
    NUMA split, Ramulator2 channel statistics and BookSim2 packet length."""
    with open(result_path) as f:
        res = json.load(f)
    text = open(_log_of(result_path), errors="ignore").read()
    inst = {}
    for m in INST.finditer(text):
        inst.setdefault(int(m.group(1)), {})[m.group(2)] = int(m.group(3))
    numa = {int(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in NUMA.finditer(text)}
    channels, core_reads, policy = {}, {}, None
    i = text.rfind(DRAM_STATS)
    if i >= 0:
        cur, want_impl = None, False
        for line in text[i + len(DRAM_STATS):].splitlines():
            m = CHANNEL.match(line.strip())
            if m:
                cur = channels.setdefault(int(m.group(1)), {})
                continue
            if cur is None:
                continue
            if POLICY.match(line):
                want_impl = True
                continue
            if want_impl:
                m = IMPL.match(line)
                if m:
                    policy, want_impl = m.group(1).lower(), False
                continue
            m = STAT_LINE.match(line)
            if m:
                cur[m.group(1)] = int(m.group(2))
                c = CORE_READ.match(m.group(1))
                if c:
                    core_reads[int(c.group(1))] = core_reads.get(int(c.group(1)), 0) + int(m.group(2))
    lengths = PACKET_LENGTH.findall(text)
    ram = _ramulator(text)
    return {"result": res, "inst": inst, "numa": numa, "channels": channels, "core_reads": core_reads,
            "policy": policy, "packet_length": float(lengths[-1]) if lengths else None,
            "refresh": _refresh(ram)}


def _rows_per_inst(path):
    """{node: {"vmatmul"|"vmatpush": rows}} from the vcix MLIR: an instruction's vl in words times the
    rows a word holds in its format; a node whose instructions of a kind differ gets no entry."""
    found = {}
    if not os.path.isfile(path):
        return {}
    text = open(path, errors="ignore").read()
    for m in VCIX_IV.finditer(text):
        kind = {0: "vmatmul", 1: "vmatpush"}.get(int(m.group(3)))
        per_word = ROWS_IN_WORD.get(int(m.group(2)) & 3)
        vl = VL_CONST.match(m.group(1))
        if kind is None:
            continue
        rows = int(vl.group(1)) * per_word if vl and per_word else None
        found.setdefault(int(m.group(4)), {}).setdefault(kind, set()).add(rows)
    return {n: {k: next(iter(v)) for k, v in kinds.items() if len(v) == 1 and None not in v}
            for n, kinds in found.items()}


def _memref(text):
    """(shape, element type) of a memref<...> type."""
    body = text[len("memref<"):-1].split(",", 1)[0].split("x")
    return tuple(int(p) for p in body[:-1] if p.isdigit()), body[-1].strip()


def _dots(path):
    """[(in_free, w_free, k, operand type, accumulator type)] of the pytorchsim.dot ops in the MLIR."""
    if not os.path.isfile(path):
        return []
    text = open(path, errors="ignore").read()
    maps = {n: [d.strip() for d in r.split(",")] for n, _, r in MAP_DEF.findall(text)}
    out = []
    for m in DOT.finditer(text):
        names = [n.strip() for n in m.group(1).split(",")]
        ops = [_memref(m.group(i)) for i in (2, 3, 4)]
        if len(names) != 3 or any(n not in maps for n in names):
            continue
        size = {}
        for dims, (shape, _) in zip((maps[n] for n in names), ops):
            for d, s in zip(dims, shape):
                size[d] = s
        out_dims = set(maps[names[2]])
        free0 = math.prod(size[d] for d in maps[names[0]] if d in out_dims)
        free1 = math.prod(size[d] for d in maps[names[1]] if d in out_dims)
        k = math.prod(size[d] for d in set(maps[names[0]]) | set(maps[names[1]]) if d not in out_dims)
        out.append((free0, free1, k, ops[0][1], ops[2][1]))
    return out


def _sfu_vector_elems(workdir):
    """The widest vector a math op of the kernel MLIR works on, per lane, or None."""
    widths = []
    for name in ("08-transformed.mlir", "09-custom.mlir"):
        path = os.path.join(workdir, name)
        if os.path.isfile(path):
            widths += [int(w) for w in MATH_VECTOR.findall(open(path, errors="ignore").read())]
    return max(widths) if widths else None


def _fused_name(kernel_name):
    """What Inductor named the kernel's ops, `pytorchsim_triton_compiler_fused__softmax_<hash>` ->
    `softmax`, or None."""
    m = re.match(r"pytorchsim_triton_compiler_(?:fused_)?(.+)_[a-z0-9]+$", kernel_name or "")
    return m.group(1).strip("_") or None if m else None


def _base_op(target):
    """`aten.mm.default` -> `mm`."""
    parts = str(target).split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "aten" else parts[0]


def useful_macs(op, args):
    """The MACs of the operator `op` from its arguments' shapes (role, sizes), or None."""
    ins = [tuple(a["sizes"]) for a in args if a.get("role") in ("in", "inout") and a.get("sizes")]
    outs = [tuple(a["sizes"]) for a in args if a.get("role") in ("out", "inout") and a.get("sizes")]
    if op in MM_OPS:
        mats = [s for s in ins if len(s) == 2]
        for a in mats:
            for b in mats:
                if a is not b and a[1] == b[0]:
                    return a[0] * a[1] * b[1]
    if op in BMM_OPS:
        mats = [s for s in ins if len(s) == 3]
        for a in mats:
            for b in mats:
                if a is not b and a[0] == b[0] and a[2] == b[1]:
                    return a[0] * a[1] * a[2] * b[2]
    if op in CONV_OPS:
        out = next((s for s in outs if len(s) >= 3), None)
        w = next((s for s in ins if out and len(s) == len(out) and s[0] == out[1]), None)
        if out and w:
            return out[0] * out[1] * math.prod(out[2:]) * math.prod(w[1:])
    return None


def _row_runs(workdir, comp_total):
    """({row: runs}, nodes, note): each compute row of the TOG's work item times the work items, which
    is COMP over the computes a work item makes; note says why it could not be counted."""
    tog = os.path.join(workdir, "tog.cpp")
    text = open(tog, errors="ignore").read()
    nodes = {name: (int(i), int(c)) for name, i, c in NODE_DEF.findall(text)}
    item = WORK_ITEM.search(text)
    if not item or not nodes:
        return {}, len(nodes), "tog.cpp has no work item"
    if re.search(r"\b(for|while)\s*\(", item.group(0)):
        return {}, len(nodes), "the work item loops"
    per_item = {}
    for name, variant in COMPUTE.findall(item.group(0)):
        node_id, count = nodes[name]
        row = node_id + count * int(variant or 0)
        per_item[row] = per_item.get(row, 0) + 1
    calls = sum(per_item.values())
    if not calls or comp_total % calls:
        return {}, len(nodes), f"COMP {comp_total} is not a multiple of the {calls} computes of a work item"
    items = comp_total // calls
    return {row: n * items for row, n in per_item.items()}, len(nodes), None


def _check_units(dumps, runs, result, lanes, arrays, mxu_unit):
    """The units whose active cycles the rows predict otherwise than TOGSim counted, as text."""
    predicted = {}
    for row, n in runs.items():
        for unit, u in dumps[row]["units"].items():
            cap = max(1, int(u.get("capacity", 1)))
            admitted = u.get("admitted", 0)
            div = cap * (min(arrays, max(1, math.ceil(admitted / lanes))) if unit == mxu_unit else 1)
            predicted[unit] = predicted.get(unit, 0.0) + n * admitted / div
    logged = {}
    for by_unit in result.get("units", {}).values():
        for unit, u in by_unit.items():
            logged[unit] = logged.get(unit, 0) + u.get("active_cycles", 0)
    tol = 0.5 * max(1, len(result.get("units", {}))) + 1e-6
    return [f"{u} {predicted.get(u, 0):.1f} vs {logged.get(u, 0)}"
            for u in sorted(set(predicted) | set(logged)) if abs(predicted.get(u, 0) - logged.get(u, 0)) > tol]


def kernel_entry(workdir, launch_dir, machine, cfg, result_path, copy_result_paths, index):
    """One kernels[] entry: the launch's TOG and its copy nodes, totalled."""
    meta = json.load(open(os.path.join(workdir, "meta.json")))
    manifest = contract.kernel_object(workdir) or {}
    y = machine["yaml"]
    lanes, dim = cfg["vpu_num_lanes"], cfg["systolic_array"]["width"]
    arrays = cfg["systolic_array"]["per_core"]
    mxu_unit = f"{y['systolic_array_unit']}.rows" if y.get("systolic_array_unit") else None
    main = _launch_counters(result_path)
    copies = [_launch_counters(p) for p in copy_result_paths]
    launches = [main] + copies
    notes = []

    comp_total = sum(i.get("COMP", 0) for i in main["inst"].values())
    runs, n_nodes, why = _row_runs(workdir, comp_total)
    if why:
        notes.append(f"gem5 counts not totalled: {why}")
    dumps = []
    if runs:
        m5out = os.path.join(os.path.dirname(contract.artifact(workdir, "cycle_elf") or
                                             os.path.join(workdir, "x")), "m5out")
        dumps = compiler_bridge.compiler_contract("gem5_samples").read(m5out, n_nodes)
        off = _check_units(dumps, runs, main["result"], lanes, arrays, mxu_unit)
        if off:
            notes.append("rows' runs disagree with TOGSim's unit totals: " + "; ".join(off))

    committed, admitted_rows, sample_cycles = {}, 0.0, 0
    for row, n in runs.items():
        d = dumps[row]
        sample_cycles += int(d.get("cycles", 0))
        for enc, v in d.get("committed", {}).items():
            committed[enc] = committed.get(enc, 0) + v * n
        admitted_rows += d["units"].get(mxu_unit, {}).get("admitted", 0) * n
    per_inst = _rows_per_inst(os.path.join(workdir, "09-custom.mlir"))
    rows = weight_rows = 0.0
    per_array = [0.0] * arrays
    ratio = round(admitted_rows / committed["vmatmul"]) if committed.get("vmatmul") else 0
    for row, n in runs.items():
        d = dumps[row]
        node = d["tile"]
        r_mm = d.get("committed", {}).get("vmatmul", 0) * per_inst.get(node, {}).get("vmatmul", ratio)
        r_wp = d.get("committed", {}).get("vmatpush", 0) * per_inst.get(node, {}).get("vmatpush", ratio)
        rows += r_mm * n
        weight_rows += r_wp * n
        n_eff = min(arrays, max(1, math.ceil(r_mm / lanes))) if r_mm else 1
        for a in range(n_eff):
            per_array[a] += r_mm * n / n_eff
    if not committed.get("vmatmul") and admitted_rows:
        rows = admitted_rows

    dots = _dots(os.path.join(workdir, "08-transformed.mlir"))
    args = meta.get("args", [])
    if dots:
        op_type, acc_type = dots[0][3], dots[0][4]
        dtype = {"operand": DTYPE_NAME.get(op_type, op_type), "accumulator": DTYPE_NAME.get(acc_type, acc_type),
                 "source": "mlir"}
    else:
        first = next((a["dtype"] for a in args if a.get("dtype")), "float32")
        op_type = first
        dtype = {"operand": DTYPE_NAME.get(first, first), "accumulator": DTYPE_NAME.get(first, first),
                 "source": "meta"}
    bits = DTYPE_BITS.get(op_type, 32)
    vec_elems = max(1, (cfg["vpu_vector_length_bits"] or 256) // bits)
    unit = lanes * vec_elems
    classes = {c: 0 for c in CLASSES}
    classes["CustomMatMulwVpush"] = weight_rows * dim / unit
    classes["CustomMatMuliVpush"] = rows * dim / unit
    classes["CustomMatMul"] = classes["CustomMatMuliVpush"]
    classes["CustomMatMulvpop"] = rows * dim * POP_BYTES / (unit * bits / 8)
    for enc, cls in SFU_CLASSES.items():
        classes[cls] += committed.get(enc, 0)

    ops = [_base_op(o) for o in meta.get("aten_ops", [])]
    operation = next((o for o in ops if o in MM_OPS | BMM_OPS | CONV_OPS),
                     _fused_name(meta.get("kernel_name")) or (ops[0] if ops else None))
    macs = useful_macs(operation, args)
    tensors = {"inputs": [], "outputs": []}
    for a in args:
        t = {"name": a.get("buffer") or a.get("name"), "dtype": DTYPE_NAME.get(a.get("dtype"), a.get("dtype")),
             "shape": a.get("sizes")}
        tensors["outputs" if a.get("role") == "out" else "inputs"].append(t)

    cores = []
    for c in sorted(set().union(*(set(l["inst"]) | set(l["numa"]) for l in launches),
                                set(int(k) for k in main["result"].get("dma", {})))):
        units = main["result"].get("units", {}).get(f"Core [{c}]", {})
        inst = {k: sum(l["inst"].get(c, {}).get(k, 0) for l in launches) for k in ("COMP", "MOVIN", "MOVOUT")}
        dma = [l["result"].get("dma", {}).get(str(c), {}) for l in launches]
        reads = sum(l["core_reads"].get(c, 0) for l in launches)
        writes = (sum(sum(ch.get("num_write_reqs", 0) for ch in l["channels"].values()) for l in launches)
                  if cfg["num_cores"] == 1 else None)
        local = sum(l["numa"].get(c, (0, 0))[0] for l in launches)
        remote = sum(l["numa"].get(c, (0, 0))[1] for l in launches)
        core = {"core_id": c,
                "systolic": [{"array_id": a, "active_cycles": per_array[a] if c == 0 else 0}
                             for a in range(arrays)],
                "vector": {"active_cycles": units.get("Vpu", {}).get("active_cycles", 0)},
                "inst_count": {"comp": inst["COMP"], "gemm": 0, "vector": inst["COMP"],
                               "movin": inst["MOVIN"], "movout": inst["MOVOUT"]},
                "dma": {k: sum(d.get(k, 0) for d in dma) for k in ("active_cycles", "idle_cycles", "responses")},
                "dram_requests": {"reads": reads, **({"writes": writes} if writes is not None else {})},
                "numa": {"local_requests": local, "remote_requests": remote}}
        cores.append(core)
    if cfg["num_cores"] > 1:
        notes.append("the matrix-unit rows are all charged to core 0's arrays")

    per_channel = []
    impl, banks, _ = main["refresh"]
    for ch in sorted(set().union(*(set(l["channels"]) for l in launches))):
        s = {k: sum(l["channels"].get(ch, {}).get(k, 0) for l in launches)
             for k in ("num_read_reqs", "num_write_reqs", "row_hits", "row_misses", "row_conflicts",
                       "num_maintenance_reqs")}
        refresh = s["num_maintenance_reqs"] / banks if impl == "PerBank" and banks else s["num_maintenance_reqs"]
        per_channel.append({"channel": ch, "reads": s["num_read_reqs"], "writes": s["num_write_reqs"],
                            "row_hits": s["row_hits"], "row_misses": s["row_misses"],
                            "row_conflicts": s["row_conflicts"], "row_policy": main["policy"],
                            "refresh_cmds": refresh})

    entry = {
        "id": index, "order": index, "hash": os.path.basename(os.path.normpath(workdir)),
        "name": meta.get("kernel_name"), "operation": operation,
        "artifacts_dir": os.path.normpath(workdir),
        "has_matmul": bool(dots),
        "dtype": dtype,
        "tensors": tensors,
        "useful_macs": macs,
        "gemm_tile": {"M": dots[0][0], "K": dots[0][2], "N": dots[0][1]} if dots else None,
        "sfu_vector_elems": _sfu_vector_elems(workdir),
        "total_cycles": sum(int(l["result"].get("total_cycles", 0)) for l in launches),
        "cores": cores,
        "dram": {"per_channel": per_channel},
    }
    if cfg["noc"].get("icnt_type") == "booksim2":
        apl = main["packet_length"] or 1.0
        reqs = sum(c["reads"] + c["writes"] for c in per_channel)
        flits = 2 * reqs * apl
        entry["noc"] = {"injected_flits": flits, "avg_packet_length": apl,
                        "per_router_traversals": [{"router": 0, "flits": flits}],
                        "per_d2d_channel_flits": []}
    entry["gem5"] = {"committed_inst_by_class": classes,
                     "tile_graph": {"compute_nodes": n_nodes, "total_executions": comp_total},
                     "sample_program_cycles": sample_cycles}
    if notes:
        entry["notes"] = notes
    for key in ("useful_macs", "gemm_tile", "sfu_vector_elems"):
        if entry[key] is None:
            entry.pop(key)
    return entry
