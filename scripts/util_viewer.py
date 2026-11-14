#!/usr/bin/env python3
"""Turn TOGSim logs into a self-contained NPU profile page: Profile (overview, one trace viewer over all runs,
op profile, units) and Lowering (per-kernel PyTorch origin, graphs, compile time). Inputs and flags: --help.
Run it on togsim_results/; --log_level trace logs add per-instruction bars (trace_timeline.py)."""
import argparse
import fnmatch
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim import contract  # noqa: E402
from pytorchsim.simulator import wallclock  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_timeline  # noqa: E402

_MOJIBAKE = re.compile("\u00c2|\u00c3|\u00e2\u20ac|\u00e2\u2013")
_MOJIBAKE_INPUTS = []
_CMDLINE = re.compile(r"--(?:tog_so|trace_so) (\S+)")
_ENQUEUED = re.compile(r"\[Scheduler (\d+)\] Enqueued kernel_id: (\d+), tog_path: (\S+?),")
_CMDLINE_FULL = re.compile(r"\[TOGSim\] Command line: (.*)$", re.M)
_CONFIG_LOADED = re.compile(r'Loaded configuration file "([^"]+)"')
_CONFIG_TOP = re.compile(r"^([A-Za-z_]\w*): ?(.*)$")
_CONFIG_INT = re.compile(r"^(\w+): (\d+)$")
_CONFIG_SA_UNIT = re.compile(r"^systolic_array_unit: (\S+)$")
_PEAK_BW = re.compile(r"Total bandwidth ([\d.]+) GB/s, (\d+) MHz, (\d+) channels, (\d+) bytes")
_DMA = re.compile(r"Core \[(\d+)\] : DMA active_cycles: (\d+), DMA idle_cycles: (\d+), "
                  r"DRAM BW: ([\d.]+) GB/s \((\d+) responses\)")
_UNIT = re.compile(r"Core \[(\d+)\] : (\S+?) utilization\(%\): ([\d.]+), "
                   r"active_cycles: (\d+), idle_cycles: (\d+)")
_TOTCYC = re.compile(r"Core \[(\d+)\] : Total_cycles: (\d+)")
_DRAM_WCH = re.compile(r"\[DRAM\] channel (\d+) \| ([\d.]+) GB/s avg\., ([\d.]+)% of utilization "
                       r"\| (\d+) reads, (\d+) writes \(interval")
_DRAM_WINDOW = re.compile(r"\[DRAM\] all \d+ channels combined \| ([\d.]+) GB/s "
                          r"aggregate, ([\d.]+)% of utilization[^|]*\| (\d+) reads, (\d+) writes"
                          r"(?: \(interval (\d+) cycles\))?")
_DRAM_TOTAL = re.compile(r"\[DRAM\] channels \d+\.\.\d+ combined \| ([\d.]+) GB/s "
                         r"aggregate, ([\d.]+)% of utilization")
_KERNEL_DONE = re.compile(r"\[Scheduler (\d+)\] Kernel (\d+) .* finished at cycle (\d+)")
_TOTAL_EXEC = re.compile(r"Total execution cycles: (\d+)")
_STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
_TILE_SCHED = re.compile(r"\[(\d+)\]\[Core (\d+)\]\[TILE_SCHEDULED")
_REQ_DRAM = re.compile(r"\[(\d+)\]\[Core \d+\]\[REQ_TO_DRAM")

_FINAL_MARKERS = ("=== DRAM statistics ===", "[DRAM] Per-channel average bandwidth",
                  "===== Instructions count =====")

_PATH_MAP = []
_UNIT_DESC = {"MXU": "matrix unit (systolic arrays)",
              "XLU": "cross-lane unit (transpose, shuffle, cross-lane reduce)",
              "SFU": "special function unit (exp, rsqrt, ...)",
              "MISC": "other instructions", "Issue": "instruction issue slots",
              "Vpu": "vector unit"}

_TT_HEAD = re.compile(r"^\S+ \S+ \d+ \S+\] (\{.*\})\s*$")
_TT_KERNEL = re.compile(r"kernel_name='(\w+)'")
_TT_CALL = re.compile(r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]")
_TT_CALLEE = re.compile(r"^\s*(\w+)(?:\.run)?\(")
_TT_FILE = re.compile(r"^\s*# File: (.*)$")
_TT_NODE = re.compile(r"^\s*(\w+)(?:: [^=]*)? = ")
_TT_GRAPHS = (("dynamo_output_graph", "Dynamo FX graph"), ("aot_inference_graph", "AOTAutograd graph"),
              ("aot_forward_graph", "AOTAutograd forward graph"),
              ("aot_backward_graph", "AOTAutograd backward graph"),
              ("inductor_post_grad_graph", "Inductor post-grad graph"),
              ("inductor_output_code", "Inductor output code"))

_SVG_PROLOG = re.compile(r"^\s*(?:<\?xml[^>]*>\s*)?(?:<!DOCTYPE[^>]*>\s*)?(?:<!--.*?-->\s*)*", re.S)
_DIAGRAMS = (("orig_fx_graph_diagram.svg", "FX graph with fusion groups"),
             ("graph_diagram.svg", "Post-fusion graph"))
_PROVENANCE = "inductor_provenance_tracking_node_mappings.json"

_WANTED_CONFIG = ("num_cores", "core_freq_mhz", "core_stats_print_period_cycles",
                  "vpu_num_lanes", "vpu_vector_length_bits",
                  "vpu_spad_size_kb_per_lane", "dram_channels", "dram_freq_mhz",
                  "core_spad_size_kb", "num_systolic_array_per_core")


def _flag_mojibake(path, text):
    """Record `path` when UTF-8 text already holds classic mojibake (UTF-8 read as Latin-1 upstream)."""
    if path not in _MOJIBAKE_INPUTS and _MOJIBAKE.search(text):
        _MOJIBAKE_INPUTS.append(path)


def _read_text(path):
    """The whole file decoded as UTF-8 (undecodable bytes replaced), flagged if it already has mojibake."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    _flag_mojibake(path, text)
    return text


def kernel_name(tog_so):
    """Name a kernel by its `triton_<hash>` dir and the test dir above it."""
    parts = [p for p in tog_so.split(os.sep) if p]
    kdir = next((p for p in reversed(parts) if p.startswith("triton_")), None)
    if kdir is None:
        kdir = parts[-2] if len(parts) > 1 else tog_so
    i = parts.index(kdir) if kdir in parts else -1
    source = parts[i - 1] if i > 0 else ""
    if source in ("outputs", ".triton_sweep", "scratchpad"):
        source = ""
    return kdir, source



def _host_path(path):
    """`path` as this host sees it: a container path that does not exist is mapped through _PATH_MAP."""
    if os.path.exists(path):
        return path
    for src, dst in _PATH_MAP:
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def parse_log(path):
    """Parse one TOGSim log into a run dict, or None if it has no stats. A run is one simulation from
    cycle 0 and may schedule several kernels back to back."""
    text = _read_text(path)
    if "Total execution cycles" not in text:
        return None

    cut = min((text.index(m) for m in _FINAL_MARKERS if m in text), default=len(text))
    head, tail = text[:cut], text[cut:]
    m = _STAMP.match(text)
    run = {"log": os.path.basename(path), "stamp": m.group(1) if m else "", "config": {},
           "wins": [], "cores": {}, "dram_win": [], "kernels": [], "reqs": [], "tiles": []}

    for line in head.splitlines():
        cm = _CONFIG_INT.match(line.strip())
        if cm and cm.group(1) in _WANTED_CONFIG:
            run["config"].setdefault(cm.group(1), int(cm.group(2)))
        sm = _CONFIG_SA_UNIT.match(line.strip())
        if sm:
            run["config"].setdefault("systolic_array_unit", sm.group(1))
    run["machine"] = _get_machine(head)
    m = _PEAK_BW.search(head)
    if m:
        run["config"]["peak_dram_gbps"] = float(m.group(1))
        run["config"]["dram_mhz"] = int(m.group(2))
        run["config"]["dram_req_bytes"] = int(m.group(4))

    interval = run["config"].get("core_stats_print_period_cycles", 0)
    cur, chans, enq, done = None, [], {}, {}
    for line in head.splitlines():
        if "========= Core stat =========" in line:
            cur = {"units": {}, "dma": {}}
            run["wins"].append(cur)
            continue
        if "[DRAM]" in line:
            m = _DRAM_WCH.search(line)
            if m:
                chans.append(float(m.group(3)))
                continue
            m = _DRAM_WINDOW.search(line)
            if m:
                run["dram_win"].append({"g": float(m.group(1)), "u": float(m.group(2)),
                                        "r": int(m.group(3)), "w": int(m.group(4)),
                                        "ch": chans or None, "span": int(m.group(5) or 0)})
                chans = []
            continue
        if "Scheduler" in line:
            m = _ENQUEUED.search(line)
            if m:
                enq[int(m.group(2))] = m.group(3)
            m = _KERNEL_DONE.search(line)
            if m:
                done[int(m.group(2))] = int(m.group(3))
            continue
        if "[TILE_SCHEDULED" in line:
            m = _TILE_SCHED.search(line)
            if m:
                run["tiles"].append((int(m.group(2)), int(m.group(1))))
            continue
        if "[REQ_TO_DRAM" in line:
            m = _REQ_DRAM.search(line)
            if m:
                run["reqs"].append(int(m.group(1)))
            continue
        if cur is None:
            continue
        m = _UNIT.search(line)
        if m:
            cur["units"].setdefault(m.group(2), {})[int(m.group(1))] = int(m.group(4))
            continue
        m = _DMA.search(line)
        if m:
            cur["dma"][int(m.group(1))] = int(m.group(2))
    for kid in sorted(enq):
        run["kernels"].append({"id": kid, "tog": _host_path(enq[kid]), "end": done.get(kid)})
    if not run["kernels"]:
        m = _CMDLINE.search(head)
        if m:
            run["kernels"].append({"id": 0, "tog": _host_path(m.group(1)), "end": None})

    for m in _UNIT.finditer(tail):
        _core(run["cores"], m.group(1))["units"][m.group(2)] = int(m.group(4))
    for m in _DMA.finditer(tail):
        _core(run["cores"], m.group(1))["dma"] = int(m.group(2))
    for m in _TOTCYC.finditer(tail):
        _core(run["cores"], m.group(1))["cycles"] = int(m.group(2))
    m = _DRAM_TOTAL.search(tail)
    run["dram_gbps"], run["dram_util"] = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    m = _TOTAL_EXEC.search(tail)
    run["cycles"] = int(m.group(1)) if m else 0
    run["interval"] = interval

    if "[INST_ID=" in text:
        num_sa = run["config"].get("num_systolic_array_per_core", 1)
        trace = trace_timeline.to_chrome(trace_timeline.parse(text.splitlines()), num_sa)
        run["events"] = trace["traceEvents"]
    else:
        run["events"] = None
    return run


def _get_machine(head):
    """The config file TOGSim loaded, its command line, and the top-level keys of the `PyTorchSim config:`
    block it printed (nested blocks kept as their one-line text)."""
    cm, lm = _CMDLINE_FULL.search(head), _CONFIG_LOADED.search(head)
    full, on = {}, False
    for line in head.splitlines():
        if "PyTorchSim config:" in line:
            on = True
            continue
        if on and _STAMP.match(line):
            break
        m = _CONFIG_TOP.match(line) if on else None
        if m:
            full[m.group(1)] = m.group(2).strip()
    path = lm.group(1) if lm else ""
    return {"cmd": cm.group(1).strip() if cm else "", "path": path,
            "yml": os.path.basename(path), "full": full}


def _blank_core():
    return {"units": {}, "dma": 0, "cycles": 0}


def _core(cores, cid):
    return cores.setdefault(int(cid), _blank_core())


def _read_sidecar(tog_so, name):
    """Lines of `name` in the directory holding `tog_so` (its launch-* component dropped), or None."""
    parts = [p for p in tog_so.split(os.sep) if not p.startswith(f"{contract.LAUNCH_DIR_PREFIX}-")]
    try:
        return _read_text(os.sep + os.path.join(*parts[:-1], name)).splitlines()
    except OSError:
        return None


def read_cycle_table(tog_so):
    """The compute cycles of tog_tile_cycles.tsv beside `tog_so`, line n for tile id n (TOGSim's
    load_cycle_table), or None if there is none."""
    lines = _read_sidecar(tog_so, "tog_tile_cycles.tsv")
    if lines is None:
        return None
    out = []
    for ln in lines:
        f = ln.split()
        if len(f) >= 2 and f[0].isdigit():
            out.append(int(f[0]))
    return out or None


TILE_TIMELINES = {}


def read_tile_timeline(tog_so, kernel):
    """row -> {cycles, events} from the tog_tile_timeline.json beside `tog_so` (or the --tile-timeline
    file for `kernel`), or None; a file of another schema is refused."""
    path = TILE_TIMELINES.get(kernel)
    try:
        doc = json.loads(_read_text(path)) if path else json.loads("\n".join(
            _read_sidecar(tog_so, "tog_tile_timeline.json") or ["null"]))
    except (OSError, ValueError):
        return None
    if doc is None:
        return None
    if doc.get("schema") != "tog-tile-timeline/1":
        raise ValueError(f"tile timeline of {kernel}: unknown schema {doc.get('schema')!r}")
    return {r["row"]: {"cycles": r["cycles"], "events": r["events"]} for r in doc["rows"]}


def read_unit_table(tog_so):
    """tile -> [(unit, unit_of_work, capacity_per_cycle, admitted)] from the tog_unit_table.tsv beside
    `tog_so`, or None if there is none."""
    lines = _read_sidecar(tog_so, "tog_unit_table.tsv")
    if lines is None:
        return None
    table = {}
    for ln in lines:
        f = ln.split("\t")
        if len(f) >= 5:
            try:
                table.setdefault(int(f[0]), []).append((f[1], f[2], int(f[3]), int(f[4])))
            except ValueError:
                continue
    return table or None


def label_units(names):
    """raw unit name -> display name: the prefix common to the prefixed names dropped, then only the part
    before the first '.'; names that collide keep their raw spelling."""
    pref = [n for n in names if "_" in n]
    common = os.path.commonprefix(pref) if pref else ""
    common = common[:common.rfind("_") + 1] if "_" in common else ""
    disp = {n: (n[len(common):] if common and n.startswith(common) else n).split(".")[0] for n in names}
    count = {}
    for d in disp.values():
        count[d] = count.get(d, 0) + 1
    return {n: (d if count[d] == 1 else n) for n, d in disp.items()}


def fold_windows(run):
    """Edges of the run's stat windows (a short last window for the cycles after the final print) and each
    unit's and the DMA engine's mean-over-cores active cycles per window."""
    I, cycles, wins = run["interval"], run["cycles"], run["wins"]
    n = len(wins) if I and wins and len(wins) * I <= cycles else 0
    edges = [i * I for i in range(n + 1)] if n else [0]
    if edges[-1] < cycles:
        edges.append(cycles)
    nc = max(1, len(run["cores"]))

    def mean(d):
        return sum(d.values()) / nc

    def fold(per_window, total):
        w = [round(per_window(i), 2) for i in range(n)]
        if len(edges) - 1 > n:
            w.append(round(max(0.0, total - sum(w)), 2))
        return {"t": round(total, 2), "w": w}

    units = {}
    for name in dict.fromkeys(k for c in run["cores"].values() for k in c["units"]):
        total = mean({cid: c["units"].get(name, 0) for cid, c in run["cores"].items()})
        units[name] = fold(lambda i: mean(wins[i]["units"].get(name, {})), total)
    dma = fold(lambda i: mean(wins[i]["dma"]), mean({cid: c["dma"] for cid, c in run["cores"].items()}))
    return edges, units, dma


def fold_dram(run):
    """The DRAM windows with their start-to-start length in core cycles (the stats period in DRAM clocks
    scaled by core/DRAM frequency), each with its channels and the REQ_TO_DRAM count that fell in it."""
    cfg, wins = run["config"], run["dram_win"]
    span = next((w["span"] for w in wins if w["span"]), 0) or 200
    core_mhz, dram_mhz = cfg.get("core_freq_mhz"), cfg.get("dram_mhz")
    wl = span * core_mhz / dram_mhz if core_mhz and dram_mhz else float(run["interval"] or span)
    out = []
    for j, w in enumerate(wins):
        if j * wl >= run["cycles"]:
            break
        out.append({"g": w["g"], "u": w["u"], "r": w["r"], "w": w["w"],
                    "ch": [round(x, 1) for x in w["ch"]] if w["ch"] else None, "q": 0})
    for c in run["reqs"]:
        j = int(c / wl)
        if j < len(out):
            out[j]["q"] += 1
    return {"wl": round(wl, 3), "gbps": run["dram_gbps"], "util": run["dram_util"], "w": out}


def read_timing(path):
    """Load one the compiler timing.json, or None if it is unreadable or mid-write."""
    try:
        rec = json.loads(_read_text(path))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("stages") else None


def find_timing(roots):
    """Map each workdir holding a the compiler timing.json to its record."""
    out = {}
    for root in roots:
        if os.path.isdir(root):
            files = glob.glob(os.path.join(root, "**", "timing.json"), recursive=True)
        elif os.path.basename(root) == "timing.json":
            files = [root]
        else:
            files = glob.glob(root)
        for f in files:
            rec = read_timing(f)
            if rec:
                out[os.path.dirname(os.path.abspath(f))] = rec
    return out


def read_breakdown(path):
    """Load one PyTorchSim breakdown.json, or None if it is unreadable."""
    try:
        rec = json.loads(_read_text(path))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("components") else None


def find_breakdowns(roots):
    """Collect every PyTorchSim breakdown_<run>.json (or older breakdown.json) under the roots."""
    name = f"{wallclock.PREFIX}*.json"
    out = []
    for root in roots:
        if os.path.isdir(root):
            files = glob.glob(os.path.join(root, "**", name), recursive=True)
        elif fnmatch.fnmatch(os.path.basename(root), name):
            files = [root]
        else:
            files = glob.glob(root)
        for f in sorted(files):
            rec = read_breakdown(f)
            if rec:
                out.append(dict(rec, path=os.path.abspath(f)))
    return out


def breakdown_groups(rec):
    """Fold a breakdown's components into wallclock's groups, then any other, plus elsewhere."""
    groups = {}
    for comp, e in (rec.get("components") or {}).items():
        g = comp.split("/")[0]
        cur = groups.setdefault(g, {"calls": 0, "seconds": 0.0, "parts": []})
        cur["calls"] += e["calls"]
        cur["seconds"] += e["seconds"]
        cur["parts"].append({"name": comp, "calls": e["calls"], "seconds": e["seconds"]})
    wall = rec.get("wall") or 0.0
    elsewhere = wall - (rec.get("measured") or 0.0)
    order = [g for g in wallclock.GROUPS if g in groups]
    order += [g for g in sorted(groups) if g not in order]
    out = [dict(groups[g], name=g) for g in order]
    if elsewhere > 0:
        out.append({"name": "elsewhere", "calls": 0, "seconds": elsewhere, "parts": []})
    return out


def timing_from_breakdowns(breakdowns):
    """Pull the torchsim-compile records a breakdown embeds, keyed by kernel label."""
    out = {}
    for bd in breakdowns:
        for e in bd.get(wallclock.TORCHSIM_COMPILE) or []:
            if e.get("kind", "compile") != "compile":
                continue
            rec = e.get("timing")
            if isinstance(rec, dict) and rec.get("stages"):
                out.setdefault(e.get("kernel") or rec.get("label"), rec)
    return out


def merge_timing(records):
    """Sum per-kernel the compiler records into one of the same shape, in stage order."""
    stages, passes, tools, order = {}, {}, {}, []
    for rec in records:
        for s in rec.get("stages", []):
            if s["name"] not in stages:
                order.append(s["name"])
            stages[s["name"]] = stages.get(s["name"], 0.0) + s["seconds"]
        for stage, entries in (rec.get("passes") or {}).items():
            acc = passes.setdefault(stage, {})
            for e in entries:
                acc[e["name"]] = acc.get(e["name"], 0.0) + e["seconds"]
        for stage, entries in (rec.get("tools") or {}).items():
            acc = tools.setdefault(stage, {})
            for t, e in entries.items():
                cur = acc.setdefault(t, {"calls": 0, "seconds": 0.0})
                cur["calls"] += e["calls"]
                cur["seconds"] += e["seconds"]
    return {"label": None, "total": sum(stages.values()),
            "stages": [{"name": n, "seconds": stages[n]} for n in order],
            "passes": {s: [{"name": n, "seconds": dt} for n, dt in v.items()]
                       for s, v in passes.items()},
            "tools": tools, "kernels": len(records)}


def collect(paths):
    """Expand dirs and globs into parsed runs, newest log last."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.log")))
        else:
            files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    runs = []
    for f in sorted(set(files)):
        try:
            r = parse_log(f)
        except Exception as exc:
            print(f"skip {f}: {exc}", file=sys.stderr)
            continue
        if r:
            runs.append(r)
    return runs


def read_torch_trace(paths):
    """Each compile in PyTorch's TORCH_TRACE logs, in log order: its id, the user frames Dynamo started
    from, and the text of each graph in _TT_GRAPHS."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.log"))) if os.path.isdir(p) else sorted(glob.glob(p))
    compiles = []
    for f in files:
        strings, by_id = {}, {}
        for head, payload in _read_torch_trace_events(f):
            if "str" in head:
                strings[head["str"][1]] = head["str"][0]
                continue
            if "frame_id" not in head:
                continue
            cid = f"{head['frame_id']}/{head['frame_compile_id']}"
            name = next((k for k, _ in _TT_GRAPHS if k in head), None)
            name = name or (head.get("artifact") or {}).get("name")
            if "dynamo_start" not in head and name not in dict(_TT_GRAPHS):
                continue
            c = by_id.get(cid)
            if c is None:
                c = by_id[cid] = {"id": cid, "log": os.path.basename(f), "stack": [], "graphs": {}}
                compiles.append(c)
            if "dynamo_start" in head:
                c["stack"] = [f"{strings.get(fr.get('filename'), fr.get('filename'))}:{fr.get('line')} in "
                              f"{fr.get('name')}" + (f": {fr['loc']}" if fr.get("loc") else "")
                              for fr in head["dynamo_start"].get("stack", [])]
            else:
                c["graphs"][dict(_TT_GRAPHS)[name]] = payload
    return compiles


def _read_torch_trace_events(path):
    """(header dict, payload text) for each record of one TORCH_TRACE log; payload lines start with a tab."""
    head, payload = None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            _flag_mojibake(path, line)
            if line.startswith("\t"):
                payload.append(line[1:].rstrip("\n"))
                continue
            if head is not None:
                yield head, "\n".join(payload)
            m = _TT_HEAD.match(line)
            head, payload = (json.loads(m.group(1)) if m else None), []
    if head is not None:
        yield head, "\n".join(payload)


def map_torch_kernels(compiles):
    """Inductor kernel name -> the compile that made it, its source nodes, ATen ops and the model lines of
    those nodes, from the output code's call-site comments and the Dynamo graph's `# File:` lines (the source
    nodes are Dynamo's names; the post-grad graph is the fallback)."""
    out = {}
    for i, c in enumerate(compiles):
        code = c["graphs"].get("Inductor output code", "")
        lines = {**_get_node_lines(c["graphs"].get("Inductor post-grad graph", "")),
                 **_get_node_lines(c["graphs"].get("Dynamo FX graph", ""))}
        for k in _TT_KERNEL.findall(code):
            out.setdefault(k, {"compile": i, "nodes": [], "aten": [], "lines": []})
        for k, found in _get_call_sites(code, lines).items():
            if out[k]["compile"] == i:
                out[k].update(found)
    return out


def _get_call_sites(code, lines):
    """Kernel name -> {nodes, aten, lines} from the `# Topologically Sorted Source Nodes` comment above each
    call of a kernel `code` defines; `lines` maps a node to its model line."""
    names = set(_TT_KERNEL.findall(code))
    out = {k: {"nodes": [], "aten": [], "lines": []} for k in names}
    call = None
    for ln in code.splitlines():
        m = _TT_CALL.search(ln)
        if m:
            call = m
            continue
        if ln.lstrip().startswith("#"):
            continue
        k = _TT_CALLEE.match(ln)
        if call and k and k.group(1) in names:
            rec = out[k.group(1)]
            for n in filter(None, (x.strip() for x in call.group(1).split(","))):
                if n not in rec["nodes"]:
                    rec["nodes"].append(n)
                    if lines.get(n) and lines[n] not in rec["lines"]:
                        rec["lines"].append(lines[n])
            for a in filter(None, (x.strip() for x in call.group(2).split(","))):
                if a not in rec["aten"]:
                    rec["aten"].append(a)
        call = None
    return out


def _get_node_lines(graph):
    """Node name -> the `# File:` comment FX wrote above it (the model line it was traced from)."""
    lines, cur = {}, None
    for ln in graph.splitlines():
        m = _TT_FILE.match(ln)
        if m:
            cur = m.group(1)
            continue
        m = _TT_NODE.match(ln)
        if m and cur:
            lines.setdefault(m.group(1), cur)
        elif not ln.strip():
            cur = None
    return lines


def read_inductor_debug(paths):
    """Each Inductor debug dir (TORCH_COMPILE_DEBUG) under `paths` that drew a diagram: its SVGs, each
    kernel's post-grad nodes (provenance mapping) and its model lines and ATen ops (`origin`)."""
    dirs = []
    for root in paths:
        found = glob.glob(os.path.join(root, "**", "output_code.py"), recursive=True)
        dirs += sorted({os.path.dirname(f) for f in found})
    out = []
    for d in dirs:
        svgs = {}
        for fname, label in _DIAGRAMS:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                svgs[label] = _SVG_PROLOG.sub("", _read_text(path), count=1)
        if not svgs:
            continue
        code = _read_text(os.path.join(d, "output_code.py"))
        kernels = {k: [] for k in _TT_KERNEL.findall(code)}
        lines = {}
        for fname in ("fx_graph_readable.py", "fx_graph_transformed.py"):
            try:
                lines = {**_get_node_lines(_read_text(os.path.join(d, fname))), **lines}
            except OSError:
                continue
        sites = _get_call_sites(code, lines)
        try:
            code_to_post = json.loads(_read_text(os.path.join(d, _PROVENANCE))).get("cppCodeToPost", {})
        except (OSError, ValueError):
            code_to_post = {}
        for key, nodes in code_to_post.items():
            acc = kernels.setdefault(key.rsplit(":", 1)[0], [])
            acc += [n for n in nodes if n not in acc]
        for k, nodes in kernels.items():
            for n in nodes:
                if k in sites and lines.get(n) and lines[n] not in sites[k]["lines"]:
                    sites[k]["lines"].append(lines[n])
        out.append({"dir": os.path.basename(d), "svgs": svgs, "kernels": kernels, "origin": sites})
    return out


def read_kernel_source(workdir):
    """The Inductor kernel name (meta.json) and Triton source (from @triton.jit on) of the nearest dir at or
    above `workdir` holding meta.json, so a launch's copy-node runs share their kernel's."""
    for _ in range(4):
        if os.path.exists(os.path.join(workdir, "meta.json")) or os.path.dirname(workdir) == workdir:
            break
        workdir = os.path.dirname(workdir)
    try:
        name = json.loads(_read_text(os.path.join(workdir, "meta.json"))).get("kernel_name")
    except (OSError, ValueError):
        name = None
    src = None
    for f in sorted(glob.glob(os.path.join(workdir, "*_triton.py"))):
        text = _read_text(f)
        src = text[text.find("@triton.jit"):] if "@triton.jit" in text else text
        break
    return name, src


def build_gantt(run, unit_index, sa_unit, num_sa):
    """The run's instruction timeline from its trace: DMA bars, COMP bars drawn on each unit its tile's
    tog_unit_table.tsv row admitted work to (width = admitted/capacity capped at compute_cycle, share =
    admitted/(capacity*compute_cycle)), and tile boundaries. None for a log without a trace."""
    events = run["events"]
    if not events:
        return None
    ends = [k["end"] if k["end"] is not None else run["cycles"] for k in run["kernels"]]

    def row_of(k, cycles):
        """The cycle-table line (TOGSim tile id) whose cycle the COMP carries; trace `tile=` is only a
        dispatch-group counter. Equal lines: the last (warm call); a split-shortened COMP: next longer."""
        cyc = run["kernels"][k].get("cycles") or []
        table = run["kernels"][k].get("table") or {}
        same = [t for t, c in enumerate(cyc) if c == cycles and t in table]
        if same:
            return same[-1]
        longer = sorted((c, -t) for t, c in enumerate(cyc) if c > cycles and t in table)
        return -longer[0][1] if longer else None

    def kernel_of(ts):
        return next((i for i, e in enumerate(ends) if ts <= e), len(ends) - 1)

    def tile_of(e):
        t = (e.get("args") or {}).get("tile")
        return int(t) if t is not None and str(t).lstrip("-").isdigit() else None

    comps = [e for e in events if e.get("ph") == "X" and e["tid"] not in ("dram-rd", "dram-wr")]
    dma = [[e["pid"], 0 if e["tid"] == "dram-rd" else 1, e["ts"], e["dur"], e["name"], tile_of(e)]
           for e in events if e.get("ph") == "X" and e["tid"] in ("dram-rd", "dram-wr")]
    row = {id(e): row_of(kernel_of(e["ts"]), e["dur"]) for e in comps}
    per_tile = {}
    for e in comps:
        key = (kernel_of(e["ts"]), row[id(e)], tile_of(e))
        per_tile[key] = per_tile.get(key, 0) + 1

    def sub_of(e):
        tid = e["tid"]
        return int(tid[2:]) if tid.startswith("sa") and tid[2:].isdigit() else 0

    def fallback(e):
        if e["tid"] == "vector":
            return next((i for n, i in unit_index.items() if n.split(".")[0].endswith("Vpu")), -1), -1
        if e["tid"].startswith("sa"):
            return unit_index.get(sa_unit, -1), (sub_of(e) if num_sa > 1 else -1)
        return -1, -1

    out = []
    for e in sorted(comps, key=lambda e: e["ts"]):
        k, tile = kernel_of(e["ts"]), tile_of(e)
        table = run["kernels"][k].get("table") if k >= 0 else None
        rows = (table or {}).get(row[id(e)], [])
        dur, n = max(e["dur"], 1), per_tile[(k, row[id(e)], tile)]
        placed = False
        for name, _uow, cap, adm in rows:
            adm_e = adm / n
            if adm_e <= 0 or name not in unit_index:
                continue
            busy = adm_e / cap if cap else dur
            sub = (sub_of(e) if num_sa > 1 else -1) if name == sa_unit else -1
            out.append([e["pid"], unit_index[name], sub, e["ts"], round(max(1.0, min(busy, dur)), 2),
                        dur, round(adm_e, 2), round(min(1.0, adm_e / (cap * dur)) if cap else 1.0, 3),
                        tile, k, e["name"], row[id(e)]])
            placed = True
        if not placed:
            u, sub = fallback(e)
            out.append([e["pid"], u, sub, e["ts"], dur, dur, None, 1.0 if not rows else 0.15, tile, k,
                        e["name"], row[id(e)]])
    return {"comp": out, "dma": dma, "tiles": [[c, t] for c, t in run["tiles"]]}


def get_origin(torch_kernel, by_torch, diagrams):
    """The model lines and ATen ops of an Inductor kernel: TORCH_TRACE's when it has any, else Inductor's
    debug dir's. Both tabs read this one lookup."""
    t = by_torch.get(torch_kernel)
    if t and (t["aten"] or t["lines"]):
        return {"aten": t["aten"], "lines": t["lines"], "nodes": t["nodes"], "from": "TORCH_TRACE"}
    for dg in diagrams:
        o = dg.get("origin", {}).get(torch_kernel)
        if o and (o["aten"] or o["lines"]):
            return dict(o, **{"from": "Inductor debug " + dg["dir"]})
    return None


def get_short_name(torch_kernel, fallback):
    """Display name of a kernel: the Inductor name without the backend prefix and the hash suffix."""
    if not torch_kernel:
        return fallback
    name = re.sub(r"^pytorchsim_triton_compiler_", "", torch_kernel)
    return re.sub(r"_[a-z0-9]{8}$", "", name) or fallback


def _main_kernel(kernels):
    """Index of the kernel a run is named after: the last non-copy kernel with PyTorch data, else the last
    non-copy kernel, else the first."""
    real = [i for i, k in enumerate(kernels) if not k["copy"]]
    return next((i for i in reversed(real) if kernels[i]["torch"]), real[-1] if real else 0)


def build_payload(runs, timing=None, breakdowns=None, compiles=None, diagrams=None):
    """Shape parsed runs into the JSON the page renders from: runs in log-timestamp order laid end to end
    on one cycle axis (each at its `offset`), the unit table, and each kernel's compile record."""
    timing = timing or {}
    runs = sorted(runs, key=lambda r: (r["stamp"], r["log"]))
    empty = [{"log": r["log"], "cycles": r["cycles"]} for r in runs if not r["kernels"] or r["cycles"] <= 1]
    runs = [r for r in runs if r["kernels"] and r["cycles"] > 1]
    by_torch = map_torch_kernels(compiles or [])
    names = list(dict.fromkeys(k for r in runs for k in fold_windows(r)[1]))
    labels = label_units(names)
    meta = {n: {"raw": n, "name": labels[n], "desc": _UNIT_DESC.get(labels[n].split(".")[0], ""),
                "uow": "", "cap": None} for n in names}
    unit_index = {n: i for i, n in enumerate(names)}

    items, offset = [], 0
    for r in runs:
        edges, units, dma = fold_windows(r)
        for k in r["kernels"]:
            k["table"] = read_unit_table(k["tog"])
            k["cycles"] = read_cycle_table(k["tog"])
            k["timeline"] = read_tile_timeline(k["tog"], kernel_name(k["tog"])[0])
            for rows in (k["table"] or {}).values():
                for name, uow, cap, _adm in rows:
                    if name in meta and meta[name]["cap"] is None:
                        meta[name]["uow"], meta[name]["cap"] = uow, cap
        cfg = r["config"]
        sa_unit = cfg.get("systolic_array_unit", "")
        gantt = build_gantt(r, unit_index, sa_unit, cfg.get("num_systolic_array_per_core", 1))
        kernels, prev = [], 0
        for k in r["kernels"]:
            name, source = kernel_name(k["tog"])
            end = k["end"] if k["end"] is not None else r["cycles"]
            copy = "copy_node" in k["tog"]
            wd = contract.workdir_of(k["tog"])
            tk, src = (None, None) if copy else read_kernel_source(wd)
            kernels.append({"id": k["id"], "name": "copy" if copy else name, "dir": name, "source": source,
                            "copy": copy, "start": prev, "end": end, "workdir": wd,
                            "torch_kernel": tk, "triton": src, "table": bool(k["table"]),
                            "timeline": k["timeline"],
                            "torch": by_torch.get(tk) if tk else None,
                            "origin": get_origin(tk, by_torch, diagrams or []) if tk else None,
                            "short": get_short_name(tk, name),
                            "diagram": next((i for i, dg in enumerate(diagrams or [])
                                             if tk in dg["kernels"]), None) if tk else None})
            prev = end
        main = _main_kernel(kernels) if kernels else None
        mk = kernels[main] if kernels else None
        items.append({"log": r["log"], "label": mk["dir"] if mk else r["log"],
                      "source": mk["source"] if mk else "", "offset": offset, "cycles": r["cycles"],
                      "config": cfg, "machine": r["machine"], "edges": edges, "units": units, "dma": dma,
                      "dram": fold_dram(r),
                      "kernels": kernels, "main": main, "gantt": gantt,
                      "cores": max(1, len(r["cores"]))})
        offset += r["cycles"]

    by_base = {}
    for wd in timing:
        by_base.setdefault(os.path.basename(wd), wd)
    records, index, used = [], {}, set()
    for item in items:
        for k in item["kernels"]:
            wd = k["workdir"]
            if wd not in timing:
                wd = by_base.get(os.path.basename(wd), wd)
            rec = timing.get(wd) if not k["copy"] else None
            if rec is None:
                k["timing"] = None
                continue
            if wd not in index:
                index[wd] = len(records)
                records.append(dict(rec, workdir=wd, kernel=k["dir"]))
            k["timing"] = index[wd]
            used.add(wd)
    orphans = []
    for wd, rec in sorted(timing.items()):
        if wd not in used:
            orphans.append(len(records))
            records.append(dict(rec, workdir=wd, kernel=rec.get("label") or os.path.basename(wd)))
    merged = merge_timing(records) if records else None

    bds = [{"path": bd.get("path", ""), "wall": bd.get("wall") or 0.0,
            "measured": bd.get("measured") or 0.0, "kernels": len(bd.get("kernels") or {}),
            "groups": breakdown_groups(bd)} for bd in (breakdowns or [])]
    return {"runs": items, "empty": empty, "logs": len(runs) + len(empty), "total": offset,
            "units": [meta[n] for n in names],
            "timing": {"records": records, "merged": merged, "orphans": orphans},
            "breakdowns": bds, "compiles": compiles or [], "diagrams": diagrams or []}


_PAGE_HEAD = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>TOGSim Profile</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #8a5cd0; --s6: #d0457a; --s7: #3aa7b8; --s8: #7f8c2a;
  --track: #ececE6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #9b72e0; --s6: #e05a8d; --s7: #3bb5c6; --s8: #98a636;
    --track: #2c2c2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --plane: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #9b72e0; --s6: #e05a8d; --s7: #3bb5c6; --s8: #98a636;
  --track: #2c2c2a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 32px 24px 64px;
}
.wrap { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--ink-2); margin: 0 0 28px; font-size: 13px; }
.card {
  background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px;
  padding: 18px 20px; margin-bottom: 20px;
}
.card > h2 {
  font-size: 13px; font-weight: 600; margin: 0 0 2px; letter-spacing: 0.01em;
}
.card > .note { color: var(--muted); font-size: 12px; margin: 0 0 16px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.kpi { background: var(--surface-1); padding: 14px 16px; border: 1px solid var(--ring); border-radius: 8px; }
.kpi { min-width: 0; }
.kpi .lab { color: var(--muted); font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.04em; margin-bottom: 6px; overflow-wrap: anywhere; }
.kpi .val { font-size: 26px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; }
.kpi .val small { font-size: 13px; font-weight: 500; color: var(--ink-2); margin-left: 2px; }
.kpi .sub2 { color: var(--ink-2); font-size: 12px; margin-top: 4px;
             font-variant-numeric: tabular-nums; }
.meter { height: 6px; border-radius: 3px; background: var(--track); margin-top: 8px;
         overflow: hidden; }
.meter > i { display: block; height: 100%; border-radius: 3px; background: var(--s1); }
.chart-scroll { overflow-x: auto; }
svg { display: block; }
svg text:not([fill]) { fill: var(--muted); }
svg text { font-size: 11px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.04em; cursor: pointer; user-select: none; }
th:first-child, td:first-child { text-align: left; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: color-mix(in srgb, var(--s1) 7%, transparent); }
tbody tr.on { background: color-mix(in srgb, var(--s1) 12%, transparent); }
td .k { font-family: ui-monospace, monospace; font-size: 12px; }
td .src { color: var(--muted); font-size: 11px; }
.bar { display: inline-block; width: 46px; height: 6px; border-radius: 3px;
       background: var(--track); vertical-align: 1px; margin-right: 7px; overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--s1); border-radius: 3px; }
.tbl-scroll { overflow-x: auto; max-height: 460px; overflow-y: auto; }
#tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
       background: var(--surface-1); border: 1px solid var(--ring); border-radius: 8px;
       padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.14);
       z-index: 9; font-variant-numeric: tabular-nums; }
#tip .th { font-weight: 600; margin-bottom: 5px; font-size: 11px; color: var(--ink-2); }
#tip .r { display: flex; gap: 10px; justify-content: space-between; white-space: nowrap; }
#tip .r b { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
            margin-right: 5px; vertical-align: 0; }
.ct-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 24px; }
.card h3 { font-size: 11px; font-weight: 600; color: var(--muted); margin: 0 0 10px;
           text-transform: uppercase; letter-spacing: 0.04em; }
.card h3.sub-h { margin-top: 26px; }
pre.code { max-height: 420px; overflow: auto; margin: 6px 0 12px; padding: 10px 12px;
  background: var(--plane); border: 1px solid var(--grid); border-radius: 6px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre; }
pre.code mark { background: rgba(237, 161, 0, 0.30); color: inherit; border-radius: 2px; }
details.graph > summary { cursor: pointer; font-size: 13px; padding: 4px 0; color: var(--ink-2); }
ul.origin { margin: 0 0 12px; padding-left: 18px; font-size: 13px; }
ul.origin code { font-size: 12px; }
.diagram { position: relative; overflow: hidden; height: 460px; margin: 6px 0 12px; background: #ffffff;
  border: 1px solid var(--grid); border-radius: 6px; cursor: grab; touch-action: none; }
.diagram.grab { cursor: grabbing; }
.diagram .vp-in { position: absolute; left: 0; top: 0; transform-origin: 0 0; }
.diagram svg { max-width: none; }
.vp-bar { position: absolute; top: 6px; right: 6px; display: flex; gap: 4px; z-index: 2; }
.vp-bar button, .cp { appearance: none; border: 1px solid var(--ring); border-radius: 6px; background: var(--surface-1);
  color: var(--ink-2); font: inherit; font-size: 11px; padding: 2px 8px; cursor: pointer; }
.vp-bar button:hover, .cp:hover { border-color: var(--s1); color: var(--s1); }
.oh { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; margin: 0 0 12px; }
.oh .nm { font-size: 15px; }
.srcd { font-size: 11px; color: var(--muted); }
.srcd summary { cursor: pointer; }
.srcd div { padding: 2px 0; }
.mc { border: 1px solid var(--grid); border-radius: 6px; background: var(--plane); margin: 4px 0 14px; overflow: auto; }
.mc-h { display: flex; align-items: center; gap: 8px; padding: 5px 10px; font-size: 12px; color: var(--ink-2);
  border-top: 1px solid var(--grid); }
.mc-h:first-child { border-top: 0; }
.mc-r { display: flex; gap: 12px; padding: 2px 10px; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.mc-r .ln { color: var(--muted); min-width: 28px; text-align: right; flex: none; }
.mc-r code { white-space: pre; }
.tag { color: var(--muted); font-size: 11px; border: 1px solid var(--grid); border-radius: 4px; padding: 0 6px; }
.chips { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 4px 0 14px; }
.chips .arr { color: var(--muted); }
.chip { font-family: ui-monospace, monospace; }
#ov-minor { margin: 8px 0 0; }
.verdict.warn { color: var(--ink); }
.verdict.warn b { color: var(--s2); }
.zoombar button[aria-pressed="true"] { border-color: var(--s1); color: var(--s1); }
details.empty-runs { margin-top: 12px; font-size: 12px; color: var(--ink-2); }
details.empty-runs summary { cursor: pointer; }
#op-tbl .bar { width: 34px; margin-right: 5px; }
#op-tbl th, #op-tbl td { padding: 6px 8px; font-size: 12px; }
#op-tbl td.l { white-space: normal; min-width: 220px; max-width: 420px; }
.diagram svg text { fill: #000000; }
.diagram g.hit > path, .diagram g.hit > polygon, .diagram g.hit > ellipse {
  fill: #ffd27a !important; stroke: #d97a00 !important; stroke-width: 3px; }
.diagram g.cluster.hit > polygon { fill: rgba(255, 210, 122, 0.25) !important; }
.seg { display: inline-flex; border: 1px solid var(--ring); border-radius: 7px;
       overflow: hidden; margin-bottom: 18px; }
.seg button { appearance: none; border: 0; background: var(--surface-1); color: var(--ink-2);
              font: inherit; font-size: 12px; padding: 5px 13px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--ring); }
.seg button[aria-pressed="true"] { background: var(--s1); color: #fff; }
.seg button:focus-visible { outline: 2px solid var(--s1); outline-offset: -2px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px 2px 7px;
        border-radius: 999px; border: 1px solid var(--ring); font-size: 11px;
        color: var(--ink-2); white-space: nowrap; }
.chip i { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.empty { color: var(--muted); font-size: 13px; padding: 8px 0; }
.machine { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: -14px 0 18px; }
.machine .mlab { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 4px; }
.mchips { display: contents; }
.chip.warn { border-color: var(--s4); color: var(--ink); background: color-mix(in srgb, var(--s4) 14%, transparent); }
.mbtn { appearance: none; background: var(--surface-1); font: inherit; font-size: 11px; cursor: pointer; }
.mbtn:hover { border-color: var(--s1); color: var(--s1); }
.mach-det { font-size: 12px; max-height: 60vh; overflow: auto; }
.mach-det h3 { margin: 10px 0 6px; }
.mach-det h3:first-child { margin-top: 0; }
.mono { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; margin: 0 0 4px; }
table.mt { width: auto; }
table.mt td { white-space: normal; padding: 3px 12px 3px 0; border-bottom: 1px solid var(--grid); vertical-align: top; }
.zoombar { display: flex; align-items: center; gap: 10px; margin: 0 0 10px;
           font-size: 12px; color: var(--ink-2); }
.zoombar button { appearance: none; border: 1px solid var(--ring); border-radius: 6px;
                  background: var(--surface-1); color: var(--ink-2); font: inherit;
                  font-size: 12px; padding: 3px 10px; cursor: pointer; }
.zoombar button:hover:not(:disabled) { border-color: var(--s1); color: var(--s1); }
.zoombar button:disabled { opacity: 0.45; cursor: default; }
.zoombar button:focus-visible { outline: 2px solid var(--s1); outline-offset: 1px; }
.zoombar .win { font-variant-numeric: tabular-nums; }
.zoombar .hint { color: var(--muted); margin-left: auto; }
.verdict { margin: 14px 0 0; font-size: 13px; color: var(--ink-2); }
.verdict b { color: var(--ink); }
.top-row { display: grid; grid-template-columns: minmax(80px, 260px) 1fr minmax(0, 270px); gap: 10px;
           align-items: center; padding: 4px 0; cursor: pointer; font-size: 13px; }
.top-row:hover { background: color-mix(in srgb, var(--s1) 7%, transparent); }
.top-row .bar { width: 100%; margin: 0; display: block; height: 8px; }
.top-row .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tv { position: relative; width: 100%; }
.tv canvas { display: block; width: 100%; touch-action: pan-y; outline: none; border-radius: 6px; }
.tv canvas:focus-visible { box-shadow: 0 0 0 2px var(--s1); }
.zoombar { flex-wrap: wrap; }
.zoombar .hint { margin-left: 0; flex-basis: 100%; }
td.g { font-weight: 600; }
td.ind { padding-left: 26px; }
tbody tr.grp { background: color-mix(in srgb, var(--ink) 4%, transparent); }
tbody tr.out { opacity: 0.45; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 7px; }
.uh { cursor: help; border-bottom: 1px dotted var(--muted); }
details.card > summary { cursor: pointer; font-size: 13px; font-weight: 600; }
details.card[open] > summary { margin-bottom: 12px; }
th.l, td.l { text-align: left; }
@media (max-width: 640px) {
  body { padding: 20px 16px 48px; }
  .card { padding: 14px 14px; }
  .top-row { grid-template-columns: 1fr; }
  .top-row .bar { grid-column: 1 / -1; }
}
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--ring); margin: 0 0 20px; }
.tabs button { appearance: none; border: 0; background: none; color: var(--ink-2); font: inherit;
               font-size: 13px; font-weight: 500; padding: 8px 14px; cursor: pointer;
               border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabs button:hover { color: var(--ink); }
.tabs button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--s1); }
.tabs button:focus-visible { outline: 2px solid var(--s1); outline-offset: -2px; }
[hidden] { display: none !important; }
.lw-grid { display: grid; grid-template-columns: minmax(220px, 300px) minmax(0, 1fr); gap: 20px; align-items: start; }
.lw-grid > * { min-width: 0; }
.lw-list { position: sticky; top: 12px; max-height: 80vh; overflow: auto; }
.lw-item { display: block; width: 100%; text-align: left; appearance: none; background: none; color: var(--ink);
           border: 0; border-left: 3px solid transparent; padding: 7px 10px; font: inherit; cursor: pointer; }
.lw-item:hover { background: color-mix(in srgb, var(--s1) 7%, transparent); }
.lw-item[aria-pressed="true"] { border-left-color: var(--s1); background: color-mix(in srgb, var(--s1) 12%, transparent); }
.lw-item .src { display: block; overflow-wrap: anywhere; }
.lw-item .k { display: block; overflow-wrap: anywhere; }
a.lk, #lw-show { color: var(--s1); text-decoration: none; }
a.lk:hover, #lw-show:hover { text-decoration: underline; }
@media (max-width: 760px) {
  .lw-grid { grid-template-columns: minmax(0, 1fr); }
  .lw-list { position: static; max-height: 280px; }
}
</style>
</head><body>"""


def render(payload):
    """Emit the self-contained HTML page for a parsed payload."""
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return _PAGE_HEAD + "\n" + _PAGE_BODY.replace("__DATA__", data)


_PAGE_BODY = r"""
<div class="wrap">
  <h1>TOGSim Profile</h1>
  <p class="sub" id="subtitle"></p>
  <div class="machine" id="machine">
    <span class="mlab">Machine</span><span class="mchips" id="mach-chips"></span>
    <button type="button" class="chip mbtn" id="mach-btn" aria-expanded="false" aria-controls="mach-det">Details</button>
  </div>
  <div class="card mach-det" id="mach-det" hidden></div>
  <nav class="tabs" role="tablist">
    <button type="button" role="tab" id="tab-b-profile" data-tab="profile" aria-selected="true">Profile</button>
    <button type="button" role="tab" id="tab-b-lowering" data-tab="lowering" aria-selected="false">Lowering</button>
  </nav>
  <div id="tab-profile" role="tabpanel">

  <div class="card" id="ov-card">
    <h2>Overview</h2>
    <p class="note" id="ov-note"></p>
    <div class="kpis" id="ov-kpis"></div>
    <p class="verdict" id="ov-verdict"></p>
    <p class="note minor" id="ov-minor"></p>
    <p class="verdict chan" id="ov-chan" hidden></p>
    <h3 class="sub-h">Heaviest kernels</h3>
    <div id="ov-top"></div>
  </div>

  <div class="card" id="tv-card">
    <h2>Trace viewer</h2>
    <p class="note" id="tv-note"></p>
    <div class="zoombar" id="zoombar">
      <button id="z-out" type="button" title="Zoom out (S)">-</button>
      <button id="z-in" type="button" title="Zoom in (W)">+</button>
      <button id="z-sel" type="button" title="Zoom to the selected range (Z)">Zoom to selection</button>
      <button id="z-reset" type="button" title="Show the whole workload (0)">Reset</button>
      <button id="z-heat" type="button" title="Show the per-channel DRAM heatmap" aria-pressed="false">DRAM channels</button>
      <span class="win" id="z-win"></span>
      <span class="hint">wheel or W/S zoom &middot; A/D or shift-drag pan &middot; drag to measure &middot; click a kernel to zoom to it &middot; click a group header to fold it &middot; drag the overview strip</span>
    </div>
    <div class="tv" id="tv"><canvas id="tv-cv" tabindex="0"></canvas></div>
    <p class="note" id="tv-foot" style="margin:10px 0 0"></p>
    <p class="verdict" id="tv-focus" style="margin:8px 0 0"></p>
  </div>

  <details class="card" id="td-card">
    <summary>Tile detail</summary>
    <p class="note" id="td-body">Click a compute bar in the trace viewer.</p>
  </details>

  <div class="card" id="op-card">
    <h2>Op profile</h2>
    <p class="note" id="op-note"></p>
    <div class="tbl-scroll"><table id="op-tbl"><thead></thead><tbody></tbody></table></div>
    <details class="empty-runs" id="op-empty" hidden></details>
  </div>

  <div class="card" id="un-card">
    <h2>Units</h2>
    <p class="note" id="un-note"></p>
    <div class="tbl-scroll"><table id="un-tbl"><thead><tr>
      <th class="l">Unit</th><th class="l">Unit of work</th><th>Capacity / cycle</th>
      <th>Utilization</th><th>Active cycles</th>
    </tr></thead><tbody></tbody></table></div>
  </div>

  </div>

  <div id="tab-lowering" role="tabpanel" hidden>
    <div class="lw-grid">
      <div class="card lw-list"><h2>Kernels</h2><p class="note" id="lw-note"></p><div id="lw-list"></div></div>
      <div id="lw-main">
        <div class="card" id="lw-head"></div>
        <div class="card" id="pt-card"><h2>PyTorch origin</h2><div id="pt-body"></div></div>
        <div class="card" id="ct-card">
          <h2>Compile time</h2>
          <p class="note" id="ct-note"></p>
          <div class="seg" id="ct-seg"></div>
          <div class="ct-grid">
            <div><h3 id="ct-h-left">Pipeline stages</h3>
              <div class="chart-scroll"><svg id="ct-stage" width="500" height="10"></svg></div></div>
            <div><h3 id="ct-h-right">Passes and tools</h3>
              <div class="chart-scroll"><svg id="ct-pass" width="500" height="10"></svg></div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<div id="tip"></div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const RUNS = DATA.runs, UNITS = DATA.units, TOTAL = DATA.total;
const TIMING = DATA.timing || {records: [], merged: null, orphans: []};
const BREAKDOWNS = DATA.breakdowns || [];
const NS = 'http://www.w3.org/2000/svg';
const tip = document.getElementById('tip');
const fmt = n => Math.round(n).toLocaleString('en-US');
const f1 = n => n.toFixed(1);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const FREQ = (RUNS[0] && RUNS[0].config.core_freq_mhz) || 0;
const NAMED_HUES = {MXU: '--s1', Vpu: '--s2', XLU: '--s7', SFU: '--s6', MISC: '--s8', Issue: '--s5'};
const FALLBACK_HUES = ['--s5', '--s6', '--s7', '--s8', '--s1'];
const unitHue = i => NAMED_HUES[UNITS[i].name] || FALLBACK_HUES[i % FALLBACK_HUES.length];
const RUN_HUES = ['--s1', '--s3', '--s5', '--s7', '--s2', '--s6', '--s8', '--s4'];
let view = null;

function esc(t) {
  return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function el(p, tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  p.appendChild(n); return n;
}

function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }

function textWidth(strs, weight, size, spacing) {
  const g = document.createElement('canvas').getContext('2d');
  g.font = `${weight} ${size}px ${getComputedStyle(document.body).fontFamily}`;
  return Math.ceil(Math.max(0, ...strs.map(t => g.measureText(t).width + spacing * t.length)));
}

function freshSvg(id) {
  const old = document.getElementById(id);
  const n = old.cloneNode(false);
  old.replaceWith(n);
  return n;
}

function bar(v, hue) {
  return `<span class="bar"><i style="width:${Math.min(100, v).toFixed(1)}%;` +
         `background:var(${hue})"></i></span>${f1(v)}%`;
}

function secs(v) { return v >= 10 ? v.toFixed(1) + 's' : v.toFixed(2) + 's'; }

function kpi(lab, val, unit, sub, pct, hue) {
  const d = document.createElement('div'); d.className = 'kpi';
  d.innerHTML = `<div class="lab">${lab}</div><div class="val">${val}` +
                (unit ? `<small>${unit}</small>` : '') + `</div>` +
                (sub ? `<div class="sub2">${sub}</div>` : '') +
                (pct == null ? '' :
                  `<div class="meter"><i style="width:${Math.min(100, pct).toFixed(1)}%;` +
                  `background:var(${hue})"></i></div>`);
  return d;
}

function unitTitle(u) {
  return esc(u.raw + (u.desc ? ' - ' + u.desc : ''));
}

function overlap(a0, a1, b0, b1) { return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0)); }

function activeIn(run, s, lo, hi) {
  if (!s) return 0;
  const l0 = lo - run.offset, l1 = hi - run.offset;
  if (l0 <= 0 && l1 >= run.cycles) return s.t;
  let sum = 0;
  for (let i = 0; i < s.w.length; i++) {
    const a = run.edges[i], b = run.edges[i + 1], ov = overlap(a, b, l0, l1);
    if (ov > 0) sum += s.w[i] * ov / (b - a);
  }
  return sum;
}

function rangeStats(lo, hi) {
  const st = {cycles: 0, units: UNITS.map(() => 0), dma: 0, gbps: 0, runs: []};
  RUNS.forEach((r, i) => {
    const ov = overlap(r.offset, r.offset + r.cycles, lo, hi);
    if (ov <= 0) return;
    st.runs.push(i);
    st.cycles += ov;
    UNITS.forEach((u, j) => { st.units[j] += activeIn(r, r.units[u.raw], lo, hi); });
    st.dma += activeIn(r, r.dma, lo, hi);
    st.gbps += r.dram.gbps * ov;
  });
  return st;
}

const KG = [];
(function buildKernelGroups() {
  const by = {}, score = x => (x.torch ? 2 : 0) + (x.origin ? 1 : 0);
  RUNS.forEach((r, ri) => {
    const real = r.kernels.map((k, i) => (k.copy ? -1 : i));
    r.kernels.forEach((k, ki) => {
      let o = ki;
      if (k.copy) {
        o = real.findIndex((v, i) => i > ki && v >= 0);
        for (let i = ki - 1; i >= 0 && o < 0; i--) if (real[i] >= 0) o = i;
        if (o < 0) return;
      }
      const ok = r.kernels[o];
      const e = by[ok.dir] = by[ok.dir] || {dir: ok.dir, kernel: ok, spans: [], copies: [], excl: 0, cp: 0, runs: []};
      if (!KG.includes(e)) KG.push(e);
      const sp = {run: ri, ki, a: r.offset + k.start, b: r.offset + k.end};
      if (k.copy) { e.copies.push(sp); e.cp += sp.b - sp.a; return; }
      e.spans.push(sp); e.excl += sp.b - sp.a;
      if (!e.runs.includes(ri)) e.runs.push(ri);
      if (score(k) > score(e.kernel)) e.kernel = k;
    });
  });
  KG.forEach(e => { e.incl = e.excl + e.cp; e.short = e.kernel.short; });
  KG.sort((a, b) => b.excl - a.excl);
})();

const ST_ALL = rangeStats(0, TOTAL);
const UV = UNITS.map((u, i) => (ST_ALL.cycles ? 100 * ST_ALL.units[i] / ST_ALL.cycles : 0));
const HOT = (() => {
  const idx = UNITS.map((u, i) => i), hot = idx.filter(i => UV[i] >= 0.1);
  return hot.length >= 2 ? hot : idx.sort((a, b) => UV[b] - UV[a]).slice(0, 2).sort((a, b) => a - b);
})();
const CH = (() => {
  let sums = null;
  RUNS.forEach(r => r.dram.w.forEach(w => {
    if (!w.ch) return;
    sums = sums || w.ch.map(() => 0);
    w.ch.forEach((u, c) => { sums[c] += u; });
  }));
  if (!sums) return null;
  const mean = sums.reduce((a, b) => a + b, 0) / sums.length, max = Math.max(...sums);
  return mean > 0 ? {ratio: max / mean, ch: sums.indexOf(max)} : null;
})();
let heatOpen = !!(CH && CH.ratio > 1.5);

function kernelStats(e) {
  const out = {units: UNITS.map(() => 0), dma: 0};
  e.spans.forEach(sp => {
    const r = RUNS[sp.run];
    UNITS.forEach((u, j) => { out.units[j] += activeIn(r, r.units[u.raw], sp.a, sp.b); });
    out.dma += activeIn(r, r.dma, sp.a, sp.b);
  });
  return out;
}

function cyclesText(e) {
  return e.cp ? `${fmt(e.incl)} incl. copy / ${fmt(e.excl)} excl.` : `${fmt(e.excl)} cycles`;
}

function drawOverview() {
  const st = ST_ALL, box = document.getElementById('ov-kpis');
  const launches = KG.reduce((a, e) => a + e.spans.length, 0);
  const peak = (RUNS[0] && RUNS[0].config.peak_dram_gbps) || 0;
  const avg = st.cycles ? st.gbps / st.cycles : 0;
  document.getElementById('ov-note').textContent =
    'Utilization is a share of all cycles over the whole workload: the sum of separately simulated runs, weighted by cycles.';
  clear(box);
  box.appendChild(kpi('Total cycles', fmt(TOTAL), '', FREQ ? `${f1(TOTAL / FREQ)} us @ ${FREQ} MHz` : '', null));
  box.appendChild(kpi('Kernels', fmt(KG.length), '', `${fmt(launches)} launches in ${fmt(RUNS.length)} runs`, null));
  const cands = [];
  UNITS.forEach((u, i) => {
    cands.push({name: u.name, v: UV[i]});
    if (HOT.includes(i)) box.appendChild(kpi(esc(u.name), f1(UV[i]), '%', '', UV[i], unitHue(i)));
  });
  const minor = UNITS.map((u, i) => i).filter(i => !HOT.includes(i));
  document.getElementById('ov-minor').textContent = minor.length
    ? minor.map(i => `${UNITS[i].name} ${f1(UV[i])}`).join(' · ') + ' %' : '';
  const dma = st.cycles ? 100 * st.dma / st.cycles : 0;
  cands.push({name: 'DMA', v: dma});
  box.appendChild(kpi('DMA busy', f1(dma), '%', '', dma, '--s3'));
  box.appendChild(kpi('DRAM', f1(avg), 'GB/s', peak ? `${f1(100 * avg / peak)}% of ${f1(peak)} GB/s peak` : 'average', peak ? 100 * avg / peak : null, '--s4'));
  if (peak) cands.push({name: 'DRAM bandwidth', v: 100 * avg / peak});
  const labW = textWidth([...box.querySelectorAll('.lab')].map(n => n.textContent.toUpperCase()), 400, 11, 0.44) + 34;
  box.style.gridTemplateColumns = `repeat(auto-fit, minmax(min(100%, ${Math.max(150, labW)}px), 1fr))`;
  const top = cands.reduce((a, b) => (b.v > a.v ? b : a), {name: '', v: -1});
  document.getElementById('ov-verdict').innerHTML = !cands.length ? '' : top.v < 20
    ? `<b>No dominant resource.</b> The busiest is ${esc(top.name)} at ${f1(top.v)}%, so the workload is latency- or dependency-bound, not throughput-bound.`
    : `<b>${esc(top.name)}-bound.</b> ${esc(top.name)} is the busiest resource at ${f1(top.v)}% of all cycles.`;
  const chan = document.getElementById('ov-chan');
  chan.hidden = !CH;
  if (CH) {
    chan.classList.toggle('warn', CH.ratio > 1.5);
    chan.innerHTML = `DRAM channel balance: busiest channel / mean = <b>${CH.ratio.toFixed(2)}</b> (channel ${CH.ch})` +
      (CH.ratio > 1.5 ? ' &nbsp;<b>imbalanced</b>' : '') + ` &nbsp;<a href="#profile" class="lk" id="ov-heat">${heatOpen ? 'hide' : 'show'} the channel heatmap</a>`;
    document.getElementById('ov-heat').addEventListener('click', ev => { ev.preventDefault(); toggleHeat(); });
  }
  const order = KG.slice(0, 5);
  document.getElementById('ov-top').innerHTML = order.map((e, i) => {
    const p = TOTAL ? 100 * e.incl / TOTAL : 0;
    return `<div class="top-row" data-dir="${esc(e.dir)}"><span class="nm" title="${esc(e.kernel.torch_kernel || e.dir)}"><b>${esc(e.short)}</b>` +
      (e.short !== e.dir ? ` <span class="src">${esc(e.dir)}</span>` : '') + `</span>` +
      `<span style="min-width:60px"><span class="bar" style="width:100%"><i style="width:${p.toFixed(1)}%;background:var(${RUN_HUES[KG.indexOf(e) % 8]})"></i></span></span>` +
      `<span>${cyclesText(e)} &middot; ${f1(p)}%</span></div>`;
  }).join('');
  document.querySelectorAll('#ov-top .top-row').forEach(n =>
    n.addEventListener('click', () => zoomKernel(KG.find(e => e.dir === n.dataset.dir))));
}

function unitCell(u, i, st, k) {
  return st.cycles ? 100 * st.units[i] / st.cycles : 0;
}

function drawUnits(st) {
  const lo = view ? view.lo : 0, hi = view ? view.hi : TOTAL;
  document.getElementById('un-note').innerHTML =
    `Utilization over ${view ? 'the zoomed range, cycle ' + fmt(lo) + ' to ' + fmt(hi) : 'the whole workload'} ` +
    `(${fmt(st.cycles)} cycles). It is admitted work / capacity per cycle / cycles: throughput-equivalent busy ` +
    `cycles from TOGSim's unit table, not measured occupancy.`;
  document.querySelector('#un-tbl tbody').innerHTML = UNITS.map((u, i) => {
    const v = unitCell(u, i, st);
    return `<tr><td class="l"><span class="dot" style="background:var(${unitHue(i)})"></span>` +
      `<span class="uh" title="${unitTitle(u)}">${esc(u.name)}</span></td>` +
      `<td class="l">${esc(u.uow)}</td><td>${u.cap == null ? '' : fmt(u.cap)}</td>` +
      `<td>${bar(v, unitHue(i))}</td><td>${fmt(st.units[i])}</td></tr>`;
  }).join('');
  document.getElementById('un-card').style.display = UNITS.length ? '' : 'none';
}

function originKey(o) {
  return o ? o.aten.join(', ') + '|' + (o.lines[0] || '') : '';
}

function parseLine(s) {
  const m = /^(.*?):(\d+) in ([^,]+), code: ([\s\S]*)$/.exec(s);
  return m ? {path: m[1], file: m[1].split('/').pop(), line: m[2], fn: m[3].trim(), code: m[4]} : {raw: s};
}

function drawOpProfile() {
  const groups = new Map();
  KG.forEach(e => {
    const o = e.kernel.origin, key = originKey(o);
    if (!groups.has(key)) {
      const p = o && o.lines.length ? parseLine(o.lines[0]) : null;
      groups.set(key, {kernels: [], excl: 0, incl: 0, label: o && o.aten.length ? o.aten.join(' → ') : (o ? 'model code' : 'No PyTorch origin'),
        sub: o ? (p && p.file ? `${p.file}:${p.line} · ${p.code}` : (o.lines[0] || '')) : 'no TORCH_TRACE or Inductor debug dir names this kernel', title: o ? (o.lines[0] || '') : ''});
    }
    const g = groups.get(key);
    g.kernels.push(e); g.excl += e.excl; g.incl += e.incl;
  });
  const list = [...groups.values()].sort((a, b) => b.excl - a.excl);
  document.getElementById('op-note').textContent =
    'Kernels grouped by the PyTorch op or model line they were lowered from, heaviest first. Cycles are the kernel itself; copy kernels that serve it are added in the incl. column. Click a row to zoom the trace to it.';
  document.querySelector('#op-tbl thead').innerHTML = '<tr><th class="l">Origin / kernel</th><th>Cycles</th><th>Incl. copy</th><th>Share</th>' +
    HOT.map(j => `<th title="${unitTitle(UNITS[j])}">${esc(UNITS[j].name)}</th>`).join('') + '<th>DMA</th></tr>';
  const stats = new Map(KG.map(e => [e, kernelStats(e)]));
  const cells = es => {
    const cyc = es.reduce((a, e) => a + e.excl, 0), pct = v => (cyc ? 100 * v / cyc : 0);
    return HOT.map(j => `<td>${bar(pct(es.reduce((a, e) => a + stats.get(e).units[j], 0)), unitHue(j))}</td>`).join('') +
      `<td>${bar(pct(es.reduce((a, e) => a + stats.get(e).dma, 0)), '--s3')}</td>`;
  };
  const share = c => `<td>${bar(TOTAL ? 100 * c / TOTAL : 0, '--s1')}</td>`;
  const rows = [];
  list.forEach(g => {
    const firsts = g.kernels.map(e => e.spans[0]);
    rows.push(`<tr class="grp" data-lo="${Math.min(...firsts.map(x => x.a))}" data-hi="${Math.max(...firsts.map(x => x.b))}"><td class="l g"><span title="${esc(g.title)}">${esc(g.label)}</span>` +
      (g.sub ? `<br><span class="src">${esc(g.sub)}</span>` : '') + `</td><td>${fmt(g.excl)}</td><td>${fmt(g.incl)}</td>${share(g.excl)}${cells(g.kernels)}</tr>`);
    g.kernels.forEach(e => {
      const sp = e.spans[0], r = RUNS[sp.run];
      rows.push(`<tr data-lo="${sp.a}" data-hi="${sp.b}"><td class="l ind"><b class="k" title="${esc(e.kernel.torch_kernel || e.dir)}">${esc(e.short)}</b>` +
        (e.short !== e.dir ? ` <span class="src">${esc(e.dir)}</span>` : '') +
        `<br><span class="src">${esc(r.log)}${e.spans.length > 1 ? ` +${e.spans.length - 1} launch(es)` : ''}</span> &nbsp;<a href="#lowering/${encodeURIComponent(e.dir)}" class="lk">Lowering &rarr;</a></td>` +
        `<td>${fmt(e.excl)}</td><td>${fmt(e.incl)}</td>${share(e.excl)}${cells([e])}</tr>`);
    });
  });
  const tb = document.querySelector('#op-tbl tbody');
  tb.innerHTML = rows.join('');
  tb.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', ev => { if (!ev.target.closest('a')) setView(+tr.dataset.lo, +tr.dataset.hi); }));
  const em = document.getElementById('op-empty');
  em.hidden = !DATA.empty.length;
  em.innerHTML = `<summary>Empty runs (${DATA.empty.length})</summary><p class="note">Runs that scheduled no kernel or lasted at most one cycle; they are left off the axis and the tables.</p>` +
    DATA.empty.map(x => `<div class="src">${esc(x.log)} &middot; ${fmt(x.cycles)} cycle(s)</div>`).join('');
  markOpRows();
}

function markOpRows() {
  document.querySelectorAll('#op-tbl tbody tr').forEach(tr => {
    tr.classList.toggle('on', !!view && overlap(+tr.dataset.lo, +tr.dataset.hi, view.lo, view.hi) > 0);
  });
}

function codeBlock(text, hit) {
  return '<pre class="code">' + text.split('\n').map(l => {
    const h = esc(l);
    return hit && hit(l) ? '<mark>' + h + '</mark>' : h;
  }).join('\n') + '</pre>';
}

function graphHit(label, k, t) {
  if (label === 'Inductor output code') return l => l.includes(k.torch_kernel);
  if (label === 'Dynamo FX graph') {
    const names = new Set(t.nodes);
    return l => { const m = /^\s*(\w+)(?:: [^=]*)? = /.exec(l); return !!m && names.has(m[1]); };
  }
  return l => t.lines.some(x => l.includes('# File: ' + x));
}

function modelCode(lines) {
  let html = '', cur = null;
  lines.map(parseLine).forEach(p => {
    if (p.raw != null) { html += `<div class="mc-r"><code>${esc(p.raw)}</code></div>`; cur = null; return; }
    const key = p.path + '|' + p.fn;
    if (key !== cur) {
      html += `<div class="mc-h" title="${esc(p.path)}"><span class="mc-f">${esc(p.file)}</span><span class="tag">${esc(p.fn)}</span></div>`;
      cur = key;
    }
    html += `<div class="mc-r" title="${esc(p.path + ':' + p.line)}"><span class="ln">${esc(p.line)}</span><code>${esc(p.code)}</code></div>`;
  });
  return `<div class="mc">${html}</div>`;
}

function lowering(k) {
  const t = k.torch, o = k.origin, c = t ? DATA.compiles[t.compile] : null, parts = [];
  const full = k.torch_kernel || k.dir, src = [];
  if (k.torch_kernel) src.push(`Inductor kernel <code>${esc(full)}</code>`);
  if (c) src.push(`TORCH_TRACE compile ${esc(c.id)} in ${esc(c.log)}`);
  if (o) src.push(`model lines and ATen ops from ${esc(o.from)}`);
  let head = `<div class="oh"><b class="nm" title="${esc(full)}">${esc(k.short)}</b>` +
    (k.torch_kernel ? ` <button type="button" class="cp" data-copy="${esc(full)}" title="Copy the full Inductor name">Copy name</button>` : '') +
    (src.length ? ` <details class="srcd"><summary>source</summary>${src.map(x => `<div>${x}</div>`).join('')}</details>` : '') + '</div>';
  if (!o && !c) {
    head += '<p class="note">' + (DATA.compiles.length || DATA.diagrams.length ? 'No TORCH_TRACE compile or Inductor debug dir names this kernel.'
      : 'Run with TORCH_TRACE=&lt;dir&gt; and pass --torch-trace &lt;dir&gt; to see its graphs.') + '</p>';
  }
  if (o && o.lines.length) parts.push('<h3>Model code</h3>' + modelCode(o.lines));
  if (o && o.aten.length) parts.push('<h3>ATen ops</h3><div class="chips">' +
    o.aten.map(a => `<span class="chip">${esc(a)}</span>`).join('<span class="arr">→</span>') + '</div>');
  if (c) {
    if (c.stack.length) parts.push('<details class="graph"><summary>Compiled from</summary><ul class="origin">' + c.stack.map(x => `<li><code>${esc(x)}</code></li>`).join('') + '</ul></details>');
    for (const [label, text] of Object.entries(c.graphs)) {
      parts.push(`<details class="graph"><summary>${esc(label)}</summary>${codeBlock(text, graphHit(label, k, t))}</details>`);
    }
  }
  const dg = k.diagram == null ? null : DATA.diagrams[k.diagram];
  if (dg) {
    const diag = Object.entries(dg.svgs).map(([label, svg]) =>
      `<details class="graph" open><summary>${esc(label)} &nbsp;·&nbsp; ${esc(dg.dir)}</summary><div class="diagram">${svg}</div></details>`).join('');
    const at = (o && o.lines.length ? 2 : 0) + (o && o.aten.length ? 0 : 0);
    parts.splice(Math.min(at, parts.length), 0, diag);
  }
  if (k.triton) parts.push('<details class="graph" open><summary>Triton kernel</summary>' + codeBlock(k.triton, null) + '</details>');
  return head + parts.join('');
}

function setupViewport(box) {
  const svg = box.querySelector('svg');
  if (!svg) return;
  const vb = svg.viewBox && svg.viewBox.baseVal, w = (vb && vb.width) || svg.getBBox().width, h = (vb && vb.height) || svg.getBBox().height;
  svg.setAttribute('width', w); svg.setAttribute('height', h);
  const inner = document.createElement('div'); inner.className = 'vp-in';
  svg.replaceWith(inner); inner.appendChild(svg);
  const bar = document.createElement('div'); bar.className = 'vp-bar';
  bar.innerHTML = '<button type="button" data-a="in" title="Zoom in">+</button><button type="button" data-a="out" title="Zoom out">-</button><button type="button" data-a="fit" title="Fit to width">Fit</button>';
  box.appendChild(bar);
  const st = {k: 1, x: 0, y: 0, touched: false};
  const apply = () => { inner.style.transform = `translate(${st.x}px,${st.y}px) scale(${st.k})`; };
  const fit = () => {
    const bw = box.clientWidth;
    if (!bw) return;
    st.k = Math.min(1, (bw - 2) / w); st.x = 0; st.y = 0; st.touched = false; apply();
  };
  const zoom = (f, px, py) => {
    const k = Math.max(0.03, Math.min(6, st.k * f)), r = k / st.k;
    st.x = px - (px - st.x) * r; st.y = py - (py - st.y) * r; st.k = k; st.touched = true; apply();
  };
  box.addEventListener('wheel', ev => {
    ev.preventDefault();
    const b = box.getBoundingClientRect();
    zoom(Math.exp(-ev.deltaY * 0.0015), ev.clientX - b.left, ev.clientY - b.top);
  }, {passive: false});
  let drag = null;
  box.addEventListener('pointerdown', ev => {
    if (ev.target.closest('.vp-bar')) return;
    drag = {x: ev.clientX - st.x, y: ev.clientY - st.y}; box.setPointerCapture(ev.pointerId); box.classList.add('grab');
  });
  box.addEventListener('pointermove', ev => {
    if (!drag) return;
    st.x = ev.clientX - drag.x; st.y = ev.clientY - drag.y; st.touched = true; apply();
  });
  const end = () => { drag = null; box.classList.remove('grab'); };
  box.addEventListener('pointerup', end); box.addEventListener('pointercancel', end);
  bar.addEventListener('click', ev => {
    const a = ev.target.dataset.a;
    if (a === 'fit') fit();
    else if (a) zoom(a === 'in' ? 1.3 : 1 / 1.3, box.clientWidth / 2, box.clientHeight / 2);
  });
  fit();
  if (window.ResizeObserver) new ResizeObserver(() => { if (!st.touched) fit(); }).observe(box);
}

function markDiagram(body, nodes) {
  const fused = new Set();
  body.querySelectorAll('.diagram g.node').forEach(g => {
    const title = g.querySelector('title');
    if (!title || !nodes.has(title.textContent)) return;
    g.classList.add('hit');
    g.querySelectorAll('text').forEach(x => {
      const m = /^buf=(\S+)$/.exec(x.textContent);
      if (m) fused.add(m[1]);
    });
  });
  body.querySelectorAll('.diagram g.cluster, .diagram g.node').forEach(g => {
    const title = g.querySelector('title');
    if (title && (fused.has(title.textContent) || fused.has(title.textContent.replace(/^cluster_/, ''))))
      g.classList.add('hit');
  });
}

let ctScope = 'kernel';
let lwSel = null, lwDone = false;

function drawLowering() {
  lwDone = true;
  if (!KG.find(k => k.dir === lwSel)) lwSel = KG.length ? KG[0].dir : null;
  document.getElementById('lw-note').textContent = `${KG.length} kernels, heaviest first.`;
  document.getElementById('lw-list').innerHTML = KG.map(e => {
    const o = e.kernel.origin;
    return `<button type="button" class="lw-item" data-dir="${esc(e.dir)}" aria-pressed="${e.dir === lwSel}" title="${esc(e.kernel.torch_kernel || e.dir)}">` +
      `<b class="k">${esc(e.short)}</b><span class="src">${e.short !== e.dir ? esc(e.dir) + ' &middot; ' : ''}${o && o.aten.length ? esc(o.aten.join(', ')) : 'no PyTorch origin'}` +
      ` &middot; ${cyclesText(e)}</span></button>`;
  }).join('');
  document.querySelectorAll('#lw-list .lw-item').forEach(b => b.addEventListener('click', () => go('lowering', b.dataset.dir)));
  const main = document.getElementById('lw-main');
  const e = KG.find(x => x.dir === lwSel);
  main.style.display = e ? '' : 'none';
  if (!e) return;
  const k = e.kernel, r = RUNS[e.spans[0].run];
  document.getElementById('lw-head').innerHTML = `<h2>${esc(e.short)}${e.short !== e.dir ? ` <span class="src">${esc(e.dir)}</span>` : ''}</h2><p class="note">${esc(r.log)} &nbsp;·&nbsp; ` +
    `${cyclesText(e)} in ${e.spans.length} launch(es) &nbsp;·&nbsp; <a href="#" id="lw-show">Show in trace &rarr;</a></p>`;
  document.getElementById('lw-show').addEventListener('click', ev => { ev.preventDefault(); showInTrace(e); });
  const body = document.getElementById('pt-body');
  body.innerHTML = lowering(k);
  body.querySelectorAll('[data-copy]').forEach(b => b.addEventListener('click', () => {
    try { navigator.clipboard.writeText(b.dataset.copy); b.textContent = 'Copied'; } catch (x) { b.textContent = 'Copy failed'; }
  }));
  const dg = k.diagram != null ? DATA.diagrams[k.diagram] : null;
  if (dg) markDiagram(body, new Set(dg.kernels[k.torch_kernel] || []));
  body.querySelectorAll('.diagram').forEach(setupViewport);
  drawCompile(e);
}

function mergeRecords(recs) {
  const stages = {}, order = [], passes = {}, tools = {};
  recs.forEach(rec => {
    rec.stages.forEach(s => { if (!(s.name in stages)) order.push(s.name); stages[s.name] = (stages[s.name] || 0) + s.seconds; });
    for (const st in (rec.passes || {})) rec.passes[st].forEach(e => {
      const a = passes[st] = passes[st] || {}; a[e.name] = (a[e.name] || 0) + e.seconds; });
    for (const st in (rec.tools || {})) for (const t in rec.tools[st]) {
      const a = tools[st] = tools[st] || {}, c = a[t] = a[t] || {calls: 0, seconds: 0};
      c.calls += rec.tools[st][t].calls; c.seconds += rec.tools[st][t].seconds; }
  });
  return {total: order.reduce((a, n) => a + stages[n], 0), stages: order.map(n => ({name: n, seconds: stages[n]})),
          passes: Object.fromEntries(Object.entries(passes).map(([s, v]) => [s, Object.entries(v).map(([name, seconds]) => ({name, seconds}))])), tools};
}

function hbars(svg, entries, total, limit) {
  const RH = 22, R = 74, avail = svg.parentNode.clientWidth || 520;
  let rows = entries.slice().sort((a, b) => b[1] - a[1]);
  if (limit && rows.length > limit) {
    const rest = rows.slice(limit);
    rows = rows.slice(0, limit);
    rows.push([`(${rest.length} smaller)`, rest.reduce((a, e) => a + e[1], 0), true]);
  }
  const P = {l: textWidth(rows.map(r => r[0]), 400, 11, 0) + 22, t: 4, b: 4};
  const plotW = Math.max(110, Math.min(260, avail - P.l - R)), W = P.l + plotW + R;
  svg.setAttribute('width', W);
  svg.setAttribute('height', Math.max(10, P.t + rows.length * RH + P.b));
  const max = Math.max(...rows.map(r => r[1]), 1e-9);
  rows.forEach((row, i) => {
    const y = P.t + i * RH, w = Math.max(1, plotW * row[1] / max);
    el(svg, 'rect', {x: P.l, y: y + 5, width: plotW, height: RH - 12, rx: 3, fill: css('--track')});
    el(svg, 'rect', {x: P.l, y: y + 5, width: w, height: RH - 12, rx: 3, fill: row[2] ? css('--muted') : css('--s1')});
    const lab = el(svg, 'text', {x: P.l - 10, y: y + RH / 2 + 3, 'text-anchor': 'end', fill: css('--ink-2')}, row[0]);
    if (row[3]) el(lab, 'title', {}, row[3]);
    el(svg, 'text', {x: P.l + plotW + 8, y: y + RH / 2 + 3}, `${secs(row[1])}  ${total ? (100 * row[1] / total).toFixed(1) + '%' : ''}`);
  });
}

function drawCompile(entry) {
  const card = document.getElementById('ct-card');
  if (!TIMING.records.length && !BREAKDOWNS.length) { card.style.display = 'none'; return; }
  card.style.display = '';
  const seg = document.getElementById('ct-seg'), note = document.getElementById('ct-note');
  const own = entry && entry.kernel.timing != null ? [entry.kernel.timing] : [];
  const ids = ctScope === 'all' ? TIMING.records.map((x, i) => i) : own;
  if (ctScope === 'run' && !BREAKDOWNS.length) ctScope = 'kernel';
  seg.innerHTML = `<button data-s="kernel" aria-pressed="${ctScope === 'kernel'}">This kernel</button>` +
    `<button data-s="all" aria-pressed="${ctScope === 'all'}">All ${TIMING.records.length} compiles</button>` +
    (BREAKDOWNS.length ? `<button data-s="run" aria-pressed="${ctScope === 'run'}">Whole run</button>` : '');
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => { ctScope = b.dataset.s; drawCompile(entry); }));
  const stageSvg = freshSvg('ct-stage'), passSvg = freshSvg('ct-pass'), hLeft = document.getElementById('ct-h-left');
  if (ctScope === 'run') {
    hLeft.textContent = 'Where the wall clock went';
    const wall = BREAKDOWNS.reduce((a, b) => a + b.wall, 0), kern = BREAKDOWNS.reduce((a, b) => a + b.kernels, 0);
    note.innerHTML = `Wall clock of ${BREAKDOWNS.length} instrumented run(s): ${secs(wall)} over ${kern} kernel(s). Components are ` +
      `self time; <b>elsewhere</b> is torch, Inductor and Python. Written by TORCHSIM_BREAKDOWN=1.`;
    const acc = {}, parts = {};
    BREAKDOWNS.forEach(b => b.groups.forEach(g => {
      acc[g.name] = (acc[g.name] || 0) + g.seconds;
      g.parts.forEach(pt => { parts[pt.name] = (parts[pt.name] || 0) + pt.seconds; });
    }));
    hbars(stageSvg, Object.entries(acc), wall);
    hbars(passSvg, Object.entries(parts), wall, 14);
    return;
  }
  hLeft.textContent = 'Pipeline stages';
  if (!ids.length) { note.textContent = 'No compiler timing.json for this kernel.'; return; }
  const rec = mergeRecords(ids.map(i => TIMING.records[i]));
  note.innerHTML = `Wall clock of ${ids.length} compile(s): ${secs(rec.total)}. This is compile time, not simulated cycles.`;
  hbars(stageSvg, rec.stages.map(s => [s.name, s.seconds]), rec.total);
  const items = [];
  for (const st in rec.passes) for (const e of rec.passes[st]) items.push([`${st}/${e.name}`, e.seconds]);
  for (const st in rec.tools) for (const t in rec.tools[st]) items.push([`${st}/${t.split('/').pop()} x${rec.tools[st][t].calls}`, rec.tools[st][t].seconds, false, `${st}/${t}`]);
  hbars(passSvg, items, rec.total, 14);
}

function machineValue(r, key) {
  if (key === 'yml') return r.machine.yml;
  if (key === 'dram_mhz' || key === 'peak_dram_gbps') return r.config[key];
  const v = r.machine.full[key];
  return v != null && v !== '' ? v : r.config[key];
}

const MACHINE_CHIPS = [
  ['config', ['yml'], r => machineValue(r, 'yml')],
  ['cores', ['num_cores'], r => `${machineValue(r, 'num_cores')} core(s)`],
  ['core clock', ['core_freq_mhz'], r => `${machineValue(r, 'core_freq_mhz')} MHz`],
  ['vector', ['vpu_num_lanes', 'vpu_vector_length_bits'], r => `${machineValue(r, 'vpu_num_lanes')} lanes` + (machineValue(r, 'vpu_vector_length_bits') ? ` × ${machineValue(r, 'vpu_vector_length_bits')} b` : '')],
  ['systolic arrays', ['num_systolic_array_per_core'], r => `${machineValue(r, 'num_systolic_array_per_core')} array(s)/core`],
  ['scratchpad', ['vpu_spad_size_kb_per_lane', 'core_spad_size_kb'], r => machineValue(r, 'core_spad_size_kb') ? `${machineValue(r, 'core_spad_size_kb')} KB spad` : `${machineValue(r, 'vpu_spad_size_kb_per_lane')} KB/lane spad`],
  ['DRAM', ['dram_channels', 'dram_mhz'], r => `${machineValue(r, 'dram_channels')} ch` + (machineValue(r, 'dram_mhz') ? ` × ${machineValue(r, 'dram_mhz')} MHz` : '')],
  ['DRAM peak', ['peak_dram_gbps'], r => `${machineValue(r, 'peak_dram_gbps')} GB/s peak`]
];

function drawMachine() {
  const row = document.getElementById('machine');
  if (!RUNS.length) { row.hidden = true; return; }
  const chips = [], differs = new Set();
  MACHINE_CHIPS.forEach(([label, keys, f]) => {
    if (keys.every(k => RUNS.every(r => machineValue(r, k) == null))) return;
    const vals = RUNS.map(r => f(r)), distinct = [...new Set(vals)];
    keys.forEach(k => { if (new Set(RUNS.map(r => machineValue(r, k))).size > 1) differs.add(k); });
    const warn = distinct.length > 1;
    chips.push(`<span class="chip${warn ? ' warn' : ''}" title="${esc(label + (warn ? ': runs disagree, ' + distinct.join(' | ') : ''))}">${warn ? '⚠ ' : ''}${esc(warn ? label + ' ' + distinct.join(' | ') : vals[0])}</span>`);
  });
  document.getElementById('mach-chips').innerHTML = chips.join('');
  const keys = [...new Set(RUNS.flatMap(r => Object.keys(r.machine.full)))], diffs = [];
  ['yml', ...keys, 'dram_mhz', 'peak_dram_gbps'].forEach(k => {
    const groups = new Map();
    RUNS.forEach(r => { const v = String(machineValue(r, k)); if (!groups.has(v)) groups.set(v, []); groups.get(v).push(r.log); });
    if (groups.size > 1) diffs.push(`<tr><td class="l">${esc(k)}</td><td class="l">` +
      [...groups].map(([v, logs]) => `<b>${esc(v)}</b> <span class="src">${logs.map(esc).join(', ')}</span>`).join('<br>') + '</td></tr>');
  });
  const r0 = RUNS[0], cmds = [...new Set(RUNS.map(r => r.machine.cmd))], paths = [...new Set(RUNS.map(r => r.machine.path))];
  document.getElementById('mach-det').innerHTML =
    (diffs.length ? `<h3>Runs disagree (${diffs.length} value(s)); the summed axis mixes machines</h3><table class="mt">${diffs.join('')}</table>` : '') +
    `<h3>Config file</h3><p class="mono">${paths.map(esc).join('<br>') || 'not logged'}</p>` +
    `<h3>TOGSim command line${cmds.length > 1 ? 's' : ''}</h3>` + cmds.map(c => `<p class="mono">${esc(c) || 'not logged'}</p>`).join('') +
    `<h3>Parsed config${RUNS.length > 1 ? ' (' + esc(r0.log) + ')' : ''}</h3><table class="mt">` +
    Object.entries(r0.machine.full).map(([k, v]) => `<tr><td class="l">${esc(k)}</td><td class="l mono">${esc(v)}</td></tr>`).join('') + '</table>';
}

document.getElementById('mach-btn').addEventListener('click', () => {
  const d = document.getElementById('mach-det'), b = document.getElementById('mach-btn');
  d.hidden = !d.hidden; b.setAttribute('aria-expanded', !d.hidden);
});

const cv = document.getElementById('tv-cv'), cx = cv.getContext('2d'), host = document.getElementById('tv');
const RULER = 30, GH = 22, MINI = 50, PADR = 10, MIN_SPAN = 40;
const NUM_SA = Math.max(1, ...RUNS.map(r => r.config.num_systolic_array_per_core || 1));
const SA_RAW = (RUNS.find(r => r.config.systolic_array_unit) || {config: {}}).config.systolic_array_unit;
const SA_IDX = UNITS.findIndex(u => u.raw === SA_RAW);
const CORES = Array.from({length: Math.max(1, ...RUNS.map(r => r.config.num_cores || r.cores))}, (x, i) => i);
const compBars = {}, dmaBars = {}, tileMarks = {};
let sel = null, collapsed = {}, hover = null, geo = {rows: [], L: 100, H: 100, plotW: 100}, drag = null;
let hasOther = false, painting = false, focusRun = null;

RUNS.forEach((r, ri) => {
  const g = r.gantt;
  if (!g) return;
  g.comp.forEach(e => {
    const [core, u, sub, s, w, d, adm, share, tile, k, name, row] = e;
    if (u < 0) hasOther = true;
    (compBars[`${core}|${u}|${sub}`] = compBars[`${core}|${u}|${sub}`] || []).push(
      {s: r.offset + s, w, d, adm, share, tile, k, name, row, run: ri, local: s});
  });
  g.dma.forEach(e => {
    const [core, dir, s, d, name, tile] = e;
    (dmaBars[`${core}|${dir}`] = dmaBars[`${core}|${dir}`] || []).push({s: r.offset + s, d, name, tile, run: ri});
  });
  g.tiles.forEach(([core, c]) => { (tileMarks[core] = tileMarks[core] || []).push({c: r.offset + c, run: ri}); });
});

function groupsForView() {
  const gs = [{id: 'kern', label: 'Kernels', lanes: [{k: 'kern', label: 'Kernels', h: 24}]}];
  CORES.forEach(c => {
    const lanes = [];
    UNITS.forEach((u, i) => {
      if (i === SA_IDX && NUM_SA > 1) for (let s = 0; s < NUM_SA; s++) lanes.push({k: 'comp', core: c, u: i, sub: s, label: `${u.name} sa${s}`, h: 22, hue: unitHue(i)});
      else lanes.push({k: 'comp', core: c, u: i, sub: -1, label: u.name, h: 22, hue: unitHue(i)});
    });
    if (hasOther) lanes.push({k: 'comp', core: c, u: -1, sub: -1, label: 'other', h: 22, hue: '--muted'});
    gs.push({id: 'cu' + c, label: `Core ${c} · Compute units`, lanes});
    gs.push({id: 'dm' + c, label: `Core ${c} · DMA`, lanes: [
      {k: 'dmab', core: c, dir: 0, label: 'dram-rd', h: 22, hue: '--s3'},
      {k: 'dmab', core: c, dir: 1, label: 'dram-wr', h: 22, hue: '--s4'},
      {k: 'tiles', core: c, label: 'tiles', h: 16, hue: '--ink-2'}]});
  });
  const dram = [{k: 'dmaw', label: 'DMA engine', h: 30, hue: '--s3'}, {k: 'bw', label: 'DRAM GB/s', h: 44, hue: '--s4'}];
  if (RUNS.some(r => r.dram.w.some(w => w.q))) dram.push({k: 'req', label: 'DRAM requests', h: 30, hue: '--s8'});
  const nch = Math.max(0, ...RUNS.flatMap(r => r.dram.w.map(w => (w.ch || []).length)));
  if (nch && heatOpen) dram.push({k: 'heat', label: `DRAM channels`, h: Math.min(160, Math.max(64, nch * 4)), hue: '--s4', nch});
  gs.push({id: 'dram', label: 'DRAM', lanes: dram});
  return gs;
}

function vspan() { return view ? [view.lo, view.hi] : [0, TOTAL]; }
function fmtTick(v, step) {
  if (v >= 1e6) return (v / 1e6).toFixed(step >= 1e6 ? 0 : (step >= 1e5 ? 1 : 2)) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(step >= 1e3 ? 0 : (step >= 100 ? 1 : 2)) + 'k';
  return String(Math.round(v));
}
function niceStep(span, px) {
  const target = span / Math.max(1, px / 90), p = Math.pow(10, Math.floor(Math.log10(target)));
  return [1, 2, 5, 10].map(m => m * p).find(v => v >= target);
}

function rr(x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  cx.beginPath();
  cx.moveTo(x + r, y); cx.arcTo(x + w, y, x + w, y + h, r); cx.arcTo(x + w, y + h, x, y + h, r);
  cx.arcTo(x, y + h, x, y, r); cx.arcTo(x, y, x + w, y, r); cx.closePath();
}

function setFont(w, s) { cx.font = `${w} ${s}px ${getComputedStyle(document.body).fontFamily}`; }

function paint() {
  if (painting) return;
  painting = true;
  requestAnimationFrame(() => { painting = false; paintNow(); });
}

function paintNow() {
  const W = host.clientWidth;
  if (!W || host.offsetParent === null) return;
  const gs = groupsForView(), [lo, hi] = vspan(), span = hi - lo;
  setFont(500, 11);
  const labelW = textWidth(gs.flatMap(g => g.lanes.map(l => l.label)), 500, 11, 0) + 22;
  const L = Math.round(Math.min(labelW, W * 0.42)), plotW = Math.max(40, W - L - PADR);
  const rows = [];
  let y = RULER;
  gs.forEach(g => {
    rows.push({t: 'group', g, y, h: GH}); y += GH;
    if (!collapsed[g.id]) g.lanes.forEach((l, i) => { rows.push({t: 'lane', g, l, y, h: l.h, i}); y += l.h; });
  });
  const miniY = y + 8, H = miniY + MINI + 6;
  geo = {rows, L, plotW, H, miniY, W};
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + 'px';
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const col = {ink: css('--ink'), ink2: css('--ink-2'), muted: css('--muted'), grid: css('--grid'),
    axis: css('--axis'), surface: css('--surface-1'), plane: css('--plane')};
  cx.fillStyle = col.surface; cx.fillRect(0, 0, W, H);
  const X = c => L + (c - lo) / span * plotW;
  geo.X = X; geo.lo = lo; geo.hi = hi;

  rows.forEach(r => {
    if (r.t === 'group') {
      cx.fillStyle = col.plane; cx.fillRect(0, r.y, W, r.h);
      cx.fillStyle = col.ink2; setFont(600, 11);
      cx.textBaseline = 'middle'; cx.textAlign = 'left';
      cx.fillText((collapsed[r.g.id] ? '▸ ' : '▾ ') + r.g.label, 8, r.y + r.h / 2);
      return;
    }
    if (r.i % 2 === 0) { cx.fillStyle = col.ink; cx.globalAlpha = 0.035; cx.fillRect(0, r.y, W, r.h); cx.globalAlpha = 1; }
    setFont(500, 11);
    cx.fillStyle = col.ink2; cx.textAlign = 'left'; cx.textBaseline = 'middle';
    cx.save(); cx.beginPath(); cx.rect(0, r.y, L - 6, r.h); cx.clip();
    cx.fillText(r.l.label, 10, r.y + r.h / 2); cx.restore();
    cx.save(); cx.beginPath(); cx.rect(L, r.y, plotW, r.h); cx.clip();
    drawLane(r, X, col, lo, hi);
    cx.restore();
  });

  cx.strokeStyle = col.axis; cx.lineWidth = 1;
  RUNS.forEach(r => {
    if (r.offset <= lo || r.offset >= hi) return;
    cx.setLineDash([3, 4]); cx.beginPath(); cx.moveTo(X(r.offset) + 0.5, RULER); cx.lineTo(X(r.offset) + 0.5, miniY - 8); cx.stroke();
  });
  cx.setLineDash([]);
  cx.beginPath(); cx.moveTo(L, RULER - 0.5); cx.lineTo(W - PADR, RULER - 0.5); cx.stroke();
  cx.beginPath(); cx.moveTo(L + 0.5, RULER); cx.lineTo(L + 0.5, miniY - 8); cx.stroke();

  const step = niceStep(span, plotW);
  setFont(400, 11); cx.fillStyle = col.muted; cx.textBaseline = 'middle';
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const x = X(v);
    cx.strokeStyle = col.grid; cx.beginPath(); cx.moveTo(x + 0.5, RULER - 7); cx.lineTo(x + 0.5, RULER); cx.stroke();
    cx.textAlign = 'center';
    const t = fmtTick(v, step), tw = cx.measureText(t).width;
    cx.fillText(t, Math.max(L + tw / 2, Math.min(W - PADR - tw / 2, x)), 10);
  }
  cx.textAlign = 'left'; cx.fillStyle = col.muted;
  cx.fillText('cycle', 8, 10);
  if (sel) {
    const a = Math.max(L, X(sel.a)), b = Math.min(W - PADR, X(sel.b));
    if (b > a) {
      cx.fillStyle = css('--s1'); cx.globalAlpha = 0.16; cx.fillRect(a, RULER, b - a, miniY - 8 - RULER); cx.globalAlpha = 1;
      cx.strokeStyle = css('--s1'); cx.beginPath();
      cx.moveTo(a + 0.5, RULER); cx.lineTo(a + 0.5, miniY - 8); cx.moveTo(b - 0.5, RULER); cx.lineTo(b - 0.5, miniY - 8); cx.stroke();
      const d = sel.b - sel.a, label = `Δ ${fmt(d)} cycles` + (FREQ ? ` (${f1(d / FREQ)} us)` : '');
      setFont(600, 11);
      const tw = cx.measureText(label).width + 12, tx = Math.max(L, Math.min(W - PADR - tw, (a + b) / 2 - tw / 2));
      cx.fillStyle = css('--s1'); rr(tx, 14, tw, 15, 4); cx.fill();
      cx.fillStyle = '#fff'; cx.textAlign = 'left'; cx.fillText(label, tx + 6, 22);
    }
  }
  drawMini(col, L, plotW, miniY, W);
  if (hover && hover.x >= L && hover.x <= W - PADR && hover.y >= RULER && hover.y < miniY - 8) {
    cx.strokeStyle = col.ink2; cx.globalAlpha = 0.6; cx.beginPath();
    cx.moveTo(hover.x + 0.5, RULER); cx.lineTo(hover.x + 0.5, miniY - 8); cx.stroke(); cx.globalAlpha = 1;
    const c = lo + (hover.x - L) / plotW * span, label = fmt(c);
    setFont(600, 11);
    const tw = cx.measureText(label).width + 10, tx = Math.max(L, Math.min(W - PADR - tw, hover.x - tw / 2));
    cx.fillStyle = col.ink; rr(tx, 14, tw, 15, 4); cx.fill();
    cx.fillStyle = col.surface; cx.textAlign = 'left'; cx.fillText(label, tx + 5, 22);
  }
}

function drawMini(col, L, plotW, y0, W) {
  const mx = c => L + c / Math.max(1, TOTAL) * plotW;
  cx.fillStyle = col.plane; rr(L, y0, plotW, MINI, 4); cx.fill();
  RUNS.forEach((r, i) => {
    cx.fillStyle = css(RUN_HUES[i % 8]); cx.globalAlpha = 0.28;
    cx.fillRect(mx(r.offset), y0, Math.max(1, mx(r.offset + r.cycles) - mx(r.offset) - 1), MINI);
    cx.globalAlpha = 0.85;
    for (let w = 0; w < r.edges.length - 1; w++) {
      const len = r.edges[w + 1] - r.edges[w];
      let peak = r.dma.w[w] / len;
      UNITS.forEach(u => { const s = r.units[u.raw]; if (s) peak = Math.max(peak, s.w[w] / len); });
      const h = Math.min(1, peak || 0) * (MINI - 6);
      if (h > 0.5) cx.fillRect(mx(r.offset + r.edges[w]), y0 + MINI - h, Math.max(1, mx(r.offset + r.edges[w + 1]) - mx(r.offset + r.edges[w])), h);
    }
    cx.globalAlpha = 1;
  });
  if (view) {
    const a = mx(view.lo), b = mx(view.hi);
    cx.fillStyle = col.surface; cx.globalAlpha = 0.6;
    cx.fillRect(L, y0, a - L, MINI); cx.fillRect(b, y0, L + plotW - b, MINI); cx.globalAlpha = 1;
    cx.strokeStyle = css('--s1'); cx.lineWidth = 2; cx.strokeRect(a + 1, y0 + 1, Math.max(4, b - a - 2), MINI - 2); cx.lineWidth = 1;
  }
  setFont(400, 10); cx.fillStyle = col.muted; cx.textAlign = 'right'; cx.textBaseline = 'middle';
  cx.fillText('overview', L - 8, y0 + MINI / 2);
}

function bandText(x0, x1, y, h, text, fg) {
  setFont(600, 11);
  if (x1 - x0 < 24) return;
  cx.save(); cx.beginPath(); cx.rect(x0 + 3, y, x1 - x0 - 6, h); cx.clip();
  cx.fillStyle = fg; cx.textAlign = 'left'; cx.textBaseline = 'middle';
  cx.fillText(text, Math.max(x0, geo.L) + 5, y + h / 2); cx.restore();
}

function cbar(x0, x1, y, h, fill, alpha, text) {
  const w = Math.max(1, x1 - x0 - (x1 - x0 > 3 ? 1 : 0));
  cx.globalAlpha = alpha; cx.fillStyle = fill; rr(x0, y, w, h, 3); cx.fill(); cx.globalAlpha = 1;
  if (text && w > 28) bandText(x0, x0 + w, y, h, text, '#fff');
}

function drawLane(r, X, col, lo, hi) {
  const l = r.l, y = r.y + 2, h = r.h - 4, color = l.hue ? css(l.hue) : col.ink2;
  if (l.k === 'kern') RUNS.forEach((run, i) => run.kernels.forEach(k => {
    const a = run.offset + k.start, b = run.offset + k.end;
    if (b < lo || a > hi) return;
    cbar(X(a), X(b), y, h, css(RUN_HUES[i % 8]), k.copy ? 0.45 : 0.75, k.copy ? 'copy' : k.short);
  }));
  else if (l.k === 'comp') {
    const bars = compBars[`${l.core}|${l.u}|${l.sub}`] || [];
    bars.forEach(b => {
      if (b.s + b.d < lo || b.s > hi) return;
      cbar(X(b.s), X(b.s + b.d), y, h, color, 0.14);
      cbar(X(b.s), X(b.s + b.w), y, h, color, 0.35 + 0.65 * b.share, b.name);
    });
    RUNS.forEach(run => {
      if (run.gantt || l.u < 0 || run.offset + run.cycles < lo || run.offset > hi) return;
      stepBars(run, run.units[UNITS[l.u].raw], X, y, h, color);
    });
  } else if (l.k === 'dmab') (dmaBars[`${l.core}|${l.dir}`] || []).forEach(b => {
    if (b.s + b.d < lo || b.s > hi) return;
    cbar(X(b.s), X(b.s + b.d), y, h, color, 0.85, b.name);
  });
  else if (l.k === 'tiles') (tileMarks[l.core] || []).forEach(t => {
    if (t.c < lo || t.c > hi) return;
    const x = Math.round(X(t.c));
    cx.fillStyle = col.ink2; cx.fillRect(x, y, 1.5, h);
    cx.beginPath(); cx.moveTo(x - 3, y); cx.lineTo(x + 4.5, y); cx.lineTo(x + 0.75, y + 5); cx.closePath(); cx.fill();
  });
  else if (l.k === 'dmaw') RUNS.forEach(run => stepBars(run, run.dma, X, y, h, color));
  else if (l.k === 'bw' || l.k === 'req') drawDram(l, X, y, h, color, col);
  else if (l.k === 'heat') drawHeat(l, X, y, h, color);
}

function stepBars(run, s, X, y, h, color) {
  if (!s) return;
  for (let i = 0; i < s.w.length; i++) {
    const a = run.edges[i], b = run.edges[i + 1], v = Math.min(1, s.w[i] / (b - a));
    if (v <= 0.002) continue;
    const x0 = X(run.offset + a), x1 = X(run.offset + b);
    if (x1 < geo.L || x0 > geo.W) continue;
    cx.globalAlpha = 0.9; cx.fillStyle = color;
    cx.fillRect(x0, y + h - v * h, Math.max(1, x1 - x0 - (x1 - x0 > 3 ? 1 : 0)), v * h);
    cx.globalAlpha = 1;
  }
}

function drawDram(l, X, y, h, color, col) {
  const key = l.k === 'bw' ? 'g' : 'q';
  const peak = RUNS.reduce((a, r) => Math.max(a, r.config.peak_dram_gbps || 0), 0);
  const mx = Math.max(1e-9, l.k === 'bw' ? Math.max(peak, ...RUNS.flatMap(r => r.dram.w.map(w => w.g))) : Math.max(...RUNS.flatMap(r => r.dram.w.map(w => w.q))));
  RUNS.forEach(run => run.dram.w.forEach((w, j) => {
    const a = run.offset + j * run.dram.wl, b = Math.min(run.offset + run.cycles, a + run.dram.wl);
    const v = w[key] / mx;
    if (v <= 0.002 || X(b) < geo.L || X(a) > geo.W) return;
    cx.globalAlpha = 0.9; cx.fillStyle = color;
    cx.fillRect(X(a), y + h - v * h, Math.max(1, X(b) - X(a) - (X(b) - X(a) > 3 ? 1 : 0)), v * h);
    cx.globalAlpha = 1;
  }));
  if (l.k === 'bw' && peak) {
    cx.strokeStyle = col.muted; cx.setLineDash([4, 4]);
    cx.beginPath(); cx.moveTo(geo.L, y + h - peak / mx * h + 0.5); cx.lineTo(geo.W, y + h - peak / mx * h + 0.5); cx.stroke(); cx.setLineDash([]);
  }
}

function drawHeat(l, X, y, h, color) {
  const ch = h / l.nch;
  RUNS.forEach(run => run.dram.w.forEach((w, j) => {
    if (!w.ch) return;
    const a = run.offset + j * run.dram.wl, b = Math.min(run.offset + run.cycles, a + run.dram.wl);
    if (X(b) < geo.L || X(a) > geo.W) return;
    w.ch.forEach((u, c) => {
      if (u <= 0) return;
      cx.globalAlpha = Math.min(1, 0.06 + u / 100 * 0.94); cx.fillStyle = color;
      cx.fillRect(X(a), y + c * ch, Math.max(1, X(b) - X(a) - 0.5), Math.max(1, ch - 0.5));
    });
    cx.globalAlpha = 1;
  }));
}

function cycleAt(x) { return geo.lo + (x - geo.L) / geo.plotW * (geo.hi - geo.lo); }
function rowAt(y) { return geo.rows.find(r => y >= r.y && y < r.y + r.h); }

function runAt(c) { return RUNS.findIndex(r => c >= r.offset && c < r.offset + r.cycles); }

function tipFor(r, c, y) {
  const ri = runAt(c), run = RUNS[ri], local = run ? c - run.offset : 0;
  const rows = [], row = (a, b) => rows.push(`<div class="r"><span>${a}</span><span>${b}</span></div>`);
  let head = r.l.label;
  const l = r.l;
  if (l.k === 'kern' && run) {
    const k = run.kernels.find(k => local >= k.start && local < k.end);
    if (!k) return '';
    head = k.copy ? 'copy kernel (' + k.dir + ')' : k.short;
    row('cycles', `${fmt(run.offset + k.start)} - ${fmt(run.offset + k.end)}`); row('length', fmt(k.end - k.start));
    const g = KG.find(e => e.dir === k.dir);
    if (g && g.cp) row('kernel', `${fmt(g.incl)} incl. copy / ${fmt(g.excl)} excl.`);
    row('run log', esc(run.log)); row('run total', fmt(run.cycles)); row('', 'click to zoom to this kernel');
  } else if (l.k === 'comp') {
    const px = 3 * (geo.hi - geo.lo) / geo.plotW;
    const b = (compBars[`${l.core}|${l.u}|${l.sub}`] || []).find(b => c >= b.s - px && c <= b.s + Math.max(b.d, b.w) + px);
    if (b) {
      const u = UNITS[l.u], t = b.run != null ? RUNS[b.run] : null;
      head = `${u ? u.name : 'other'} · ${b.name}`;
      row('start', fmt(b.s)); row('compute_cycle', fmt(b.d)); row('tile', b.tile == null ? '-' : b.tile);
      if (b.adm != null && u) {
        row('admitted', `${b.adm} ${esc(u.uow)}`); row('capacity / cycle', u.cap == null ? '-' : u.cap);
        row('busy cycles', f1(b.w)); row('busy share', f1(100 * b.share) + '%');
      } else row('unit table', 'none for this kernel');
      row('', 'derived from per-tile counts; click for tile detail');
    } else {
      const s = run && run.units[UNITS[l.u] ? UNITS[l.u].raw : ''];
      if (!run || run.gantt || !s) return '';
      const i = windowAt(run, local);
      head = UNITS[l.u].name + ' (window average)';
      row('window', `${fmt(run.offset + run.edges[i])} - ${fmt(run.offset + run.edges[i + 1])}`);
      row('utilization', f1(100 * s.w[i] / (run.edges[i + 1] - run.edges[i])) + '%'); row('active cycles', f1(s.w[i]));
    }
  } else if (l.k === 'dmab') {
    const b = (dmaBars[`${l.core}|${l.dir}`] || []).find(b => c >= b.s && c <= b.s + b.d);
    if (!b) return '';
    head = b.name; row('start', fmt(b.s)); row('duration', fmt(b.d)); row('tile', b.tile == null ? '-' : b.tile);
  } else if (l.k === 'tiles') {
    const px = 4 * (geo.hi - geo.lo) / geo.plotW, t = (tileMarks[l.core] || []).find(t => Math.abs(t.c - c) <= px);
    if (!t) return '';
    head = 'tile scheduled'; row('cycle', fmt(t.c));
  } else if (l.k === 'dmaw' && run) {
    const i = windowAt(run, local);
    head = 'DMA engine (window average)';
    row('window', `${fmt(run.offset + run.edges[i])} - ${fmt(run.offset + run.edges[i + 1])}`);
    row('busy', f1(100 * run.dma.w[i] / (run.edges[i + 1] - run.edges[i])) + '%');
  } else if ((l.k === 'bw' || l.k === 'req' || l.k === 'heat') && run) {
    const j = Math.floor(local / run.dram.wl), w = run.dram.w[j];
    if (!w) return '';
    head = l.label; row('window', `${fmt(run.offset + j * run.dram.wl)} +${fmt(run.dram.wl)}`);
    row('bandwidth', f1(w.g) + ' GB/s'); row('utilization', f1(w.u) + '%'); row('reads / writes', `${fmt(w.r)} / ${fmt(w.w)}`);
    if (w.q) row('requests', fmt(w.q));
    if (l.k === 'heat' && w.ch) {
      const n = Math.min(w.ch.length - 1, Math.max(0, Math.floor((y - r.y - 2) / ((r.h - 4) / l.nch))));
      row('channel ' + n, f1(w.ch[n]) + '%');
    }
  } else return '';
  return `<div class="th">${esc(head)}</div>` + rows.join('') + (run ? `<div class="r"><span>run cycle</span><span>${fmt(local)}</span></div>` : '');
}

function dividerTip(p) {
  if (p.y >= RULER || p.x < geo.L) return '';
  const px = 5 * (geo.hi - geo.lo) / geo.plotW, c = cycleAt(p.x);
  const run = RUNS.find(r => Math.abs(r.offset - c) <= px);
  if (!run) return '';
  return `<div class="th">run starts at cycle ${fmt(run.offset)}</div><div class="r"><span>log</span><span>${esc(run.log)}</span></div>` +
    `<div class="r"><span>run total</span><span>${fmt(run.cycles)} cycles</span></div>`;
}

function windowAt(run, local) {
  let i = 0;
  while (i < run.edges.length - 2 && local >= run.edges[i + 1]) i++;
  return i;
}

function showTip(ev, html) {
  if (!html) { tip.style.opacity = 0; return; }
  tip.innerHTML = html; tip.style.opacity = 1;
  tip.style.left = Math.max(8, Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 10)) + 'px';
  tip.style.top = Math.max(8, ev.clientY - tip.offsetHeight - 12) + 'px';
}

function setView(lo, hi) {
  const span = Math.max(Math.min(MIN_SPAN, TOTAL), hi - lo);
  lo = Math.max(0, Math.min(lo, TOTAL - span));
  hi = lo + span;
  view = (lo <= 0 && hi >= TOTAL) ? null : {lo, hi};
  viewChanged();
}

function zoomKernel(e) { const sp = e.spans[0]; setView(sp.a, sp.b); focusKernel(sp.run, sp.ki); }

function zoomBy(f, pivot) {
  const [lo, hi] = vspan(), p = pivot == null ? (lo + hi) / 2 : pivot, fr = (p - lo) / (hi - lo);
  const span = Math.min(TOTAL, Math.max(MIN_SPAN, (hi - lo) * f));
  setView(p - fr * span, p - fr * span + span);
}

function panBy(frac) {
  const [lo, hi] = vspan();
  setView(lo + (hi - lo) * frac, hi + (hi - lo) * frac);
}

function focusKernel(ri, ki) {
  const run = RUNS[ri], k = run.kernels[ki];
  const box = document.getElementById('tv-focus');
  if (!k || k.copy) { box.innerHTML = k ? `Copy kernel (${esc(k.dir)}) in ${esc(run.log)}.` : ''; return; }
  box.innerHTML = `<b>${esc(k.short)}</b> &nbsp;·&nbsp; ${esc(run.log)} &nbsp;·&nbsp; ${fmt(k.end - k.start)} cycles &nbsp;·&nbsp; ` +
    `<a href="#lowering/${encodeURIComponent(k.dir)}" class="lk">Lowering &rarr;</a>`;
}

let tileShown = null;

function onCompBarClick(run, tile, computeCycle, start, bar) {
  const body = document.getElementById('td-body');
  const card = document.getElementById('td-card');
  card.open = true;
  const k = bar && run.kernels[bar.k], tl = k && k.timeline && bar.row != null ? k.timeline[bar.row] : null;
  tileShown = tl ? {k, row: bar.row, tl} : null;
  if (!tl) {
    body.className = 'note';
    body.innerHTML = 'gem5 tile timeline not available for this tile. Compile with <code>TORCHSIM_GEM5_TRACE=1</code> to record it.';
    return;
  }
  body.className = '';
  drawTileDetail(body);
}

function tileLanes(tl) {
  const insn = [], units = {};
  tl.events.forEach(e => {
    if (e.ph === 'X') { if (!insn.includes(e.name)) insn.push(e.name); }
    else if (e.ph === 'C') (units[e.name] = units[e.name] || []).push(e);
  });
  const primary = a => ['rows_in_flight', 'elements_in_flight'].find(n => n in a) ||
    Object.keys(a).find(n => n.includes('in_flight')) || Object.keys(a)[0];
  const comps = Object.keys(units).map(n => {
    const ev = units[n], key = primary(ev[0].args);
    return {name: n, ev, key, max: Math.max(1, ...ev.map(e => e.args[key] || 0))};
  });
  return {insn, comps};
}

function drawTileDetail(body) {
  const {k, row, tl} = tileShown, {insn, comps} = tileLanes(tl);
  body.innerHTML = `<p class="verdict" style="margin:0 0 8px"><b>${esc(k.short)}</b> &nbsp;·&nbsp; tile row ${row} &nbsp;·&nbsp; ` +
    `${fmt(tl.cycles)} cycles &nbsp;·&nbsp; warm call</p><div class="tv"><canvas id="td-cv"></canvas></div>` +
    `<p class="note" style="margin:8px 0 0">gem5 coprocessor view of one tile. DMA is outside this window (modelled by TOGSim).</p>`;
  const c = document.getElementById('td-cv'), g = c.getContext('2d');
  const W = Math.max(280, body.clientWidth), L = Math.min(150, W * 0.28), R = 10, RUL = 22, LH = 16, CH = 36;
  const H = RUL + insn.length * LH + 6 + comps.length * (CH + 4) + 6, pw = W - L - R, dpr = window.devicePixelRatio || 1;
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); c.style.height = H + 'px';
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const col = {ink: css('--ink'), ink2: css('--ink-2'), muted: css('--muted'), grid: css('--grid'), axis: css('--axis'), surface: css('--surface-1')};
  const font = `${getComputedStyle(document.body).fontFamily}`;
  const X = t => L + t / tl.cycles * pw;
  g.fillStyle = col.surface; g.fillRect(0, 0, W, H);
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find(v => tl.cycles / v <= Math.max(3, pw / 70)) || 1000;
  g.font = `400 10px ${font}`; g.textBaseline = 'middle'; g.textAlign = 'center';
  for (let t = 0; t <= tl.cycles; t += step) {
    g.strokeStyle = col.grid; g.beginPath(); g.moveTo(X(t) + .5, RUL - 4); g.lineTo(X(t) + .5, H); g.stroke();
    g.fillStyle = col.muted; g.fillText(fmt(t), Math.min(W - 14, Math.max(L, X(t))), 9);
  }
  const hit = [];
  insn.forEach((n, i) => {
    const y = RUL + i * LH, hue = css(RUN_HUES[i % 8]);
    g.fillStyle = col.ink2; g.textAlign = 'left'; g.font = `400 10px ${font}`; g.fillText(n, 6, y + LH / 2);
    tl.events.filter(e => e.ph === 'X' && e.name === n).forEach(e => {
      const x = X(e.ts), w = Math.max(2, X(e.ts + e.dur) - x);
      g.globalAlpha = e.args && e.args.squashed ? 0.35 : 0.85; g.fillStyle = hue; g.fillRect(x, y + 2, w, LH - 4); g.globalAlpha = 1;
      hit.push({x, w, y: y + 2, h: LH - 4, html: `<b>${esc(n)}</b><div class="r"><span>cycle</span><span>${fmt(e.ts)} + ${fmt(e.dur)}</span></div>` +
        Object.entries(e.args || {}).map(([a, v]) => `<div class="r"><span>${esc(a)}</span><span>${esc(v)}</span></div>`).join('')});
    });
  });
  comps.forEach((u, i) => {
    const y = RUL + insn.length * LH + 6 + i * (CH + 4), hue = css(RUN_HUES[(i + insn.length) % 8]);
    g.fillStyle = col.ink2; g.textAlign = 'left'; g.font = `600 10px ${font}`; g.fillText(u.name, 6, y + 9);
    g.font = `400 10px ${font}`; g.fillStyle = col.muted; g.fillText(`${u.key} (max ${fmt(u.max)})`, 6, y + 22);
    const base = y + CH - 2;
    g.fillStyle = hue; g.globalAlpha = 0.55; g.beginPath(); g.moveTo(X(0), base);
    let last = 0;
    u.ev.forEach(e => { g.lineTo(X(e.ts), base - last / u.max * (CH - 6)); last = e.args[u.key] || 0; g.lineTo(X(e.ts), base - last / u.max * (CH - 6)); });
    g.lineTo(X(tl.cycles), base - last / u.max * (CH - 6)); g.lineTo(X(tl.cycles), base); g.closePath(); g.fill(); g.globalAlpha = 1;
    g.strokeStyle = col.axis; g.beginPath(); g.moveTo(L, base + .5); g.lineTo(L + pw, base + .5); g.stroke();
    hit.push({x: L, w: pw, y, h: CH, unit: u});
  });
  c.onpointermove = ev => {
    const b = c.getBoundingClientRect(), px = ev.clientX - b.left, py = ev.clientY - b.top;
    const h = hit.find(h => h.unit ? py >= h.y && py < h.y + h.h && px >= L && px <= L + pw : px >= h.x && px <= h.x + h.w && py >= h.y && py <= h.y + h.h);
    if (!h) { showTip(ev, ''); return; }
    if (!h.unit) { showTip(ev, h.html); return; }
    const t = Math.floor((px - L) / pw * tl.cycles), cur = h.unit.ev.filter(e => e.ts <= t).pop();
    showTip(ev, `<b>${esc(h.unit.name)}</b><div class="r"><span>cycle</span><span>${fmt(t)}</span></div>` +
      Object.entries(cur ? cur.args : {}).map(([a, v]) => `<div class="r"><span>${esc(a)}</span><span>${esc(v)}</span></div>`).join(''));
  };
  c.onpointerleave = () => showTip(null, '');
}

addEventListener('resize', () => { if (tileShown) drawTileDetail(document.getElementById('td-body')); });
{
  const m = /[?&]tile=(\d+)/.exec(location.search);
  const run = m && RUNS.find(r => r.kernels.some(k => k.timeline));
  if (run) onCompBarClick(run, 0, 0, 0, {k: run.kernels.findIndex(k => k.timeline), row: +m[1]});
}

function clickAt(ev, x, y) {
  const r = rowAt(y);
  if (!r) return;
  if (r.t === 'group') { collapsed[r.g.id] = !collapsed[r.g.id]; paint(); return; }
  if (x < geo.L) return;
  const c = cycleAt(x), ri = runAt(c), run = RUNS[ri], l = r.l;
  if (l.k === 'kern' && run) {
    const local = c - run.offset, ki = run.kernels.findIndex(k => local >= k.start && local < k.end);
    if (ki >= 0) { setView(run.offset + run.kernels[ki].start, run.offset + run.kernels[ki].end); focusKernel(ri, ki); }
  } else if (l.k === 'comp') {
    const px = 3 * (geo.hi - geo.lo) / geo.plotW;
    const b = (compBars[`${l.core}|${l.u}|${l.sub}`] || []).find(b => c >= b.s - px && c <= b.s + Math.max(b.d, b.w) + px);
    if (b) onCompBarClick(RUNS[b.run], b.tile, b.d, b.local, b);
  } else if (sel) { sel = null; paint(); drawZoomBar(); }
}

function miniCycle(x) { return Math.max(0, Math.min(TOTAL, (x - geo.L) / geo.plotW * TOTAL)); }

function pointer(ev) {
  const b = cv.getBoundingClientRect();
  return {x: ev.clientX - b.left, y: ev.clientY - b.top};
}

cv.addEventListener('pointerdown', ev => {
  if (ev.button !== 0) return;
  const p = pointer(ev);
  cv.focus({preventScroll: true});
  if (p.y >= geo.miniY) {
    const c = miniCycle(p.x), [lo, hi] = vspan();
    const near = x => Math.abs(p.x - (geo.L + x / TOTAL * geo.plotW)) <= 6;
    drag = view && near(lo) ? {k: 'ml', fixed: hi} : view && near(hi) ? {k: 'mr', fixed: lo}
      : view && c >= lo && c <= hi ? {k: 'mm', off: c - lo, span: hi - lo} : {k: 'mm', off: (hi - lo) / 2, span: hi - lo};
    if (drag.k === 'mm') setView(c - drag.off, c - drag.off + drag.span);
  } else if (p.x >= geo.L || p.y >= RULER) {
    drag = {k: ev.shiftKey ? 'pan' : 'sel', x0: p.x, y0: p.y, moved: false, c0: cycleAt(p.x), lo0: geo.lo, hi0: geo.hi};
    if (p.x < geo.L) drag.k = 'click';
  }
  if (drag) cv.setPointerCapture(ev.pointerId);
});

cv.addEventListener('pointermove', ev => {
  const p = pointer(ev);
  hover = p;
  if (drag) {
    if (Math.abs(p.x - drag.x0) > 4 || drag.k.startsWith('m')) drag.moved = true;
    if (drag.k === 'mm') { const c = miniCycle(p.x); setView(c - drag.off, c - drag.off + drag.span); }
    else if (drag.k === 'ml') setView(Math.min(miniCycle(p.x), drag.fixed - MIN_SPAN), drag.fixed);
    else if (drag.k === 'mr') setView(drag.fixed, Math.max(miniCycle(p.x), drag.fixed + MIN_SPAN));
    else if (drag.k === 'pan' && drag.moved) { const d = (drag.x0 - p.x) / geo.plotW * (drag.hi0 - drag.lo0); setView(drag.lo0 + d, drag.hi0 + d); }
    else if (drag.k === 'sel' && drag.moved) {
      const a = drag.c0, b = Math.max(0, Math.min(TOTAL, cycleAt(p.x)));
      sel = {a: Math.min(a, b), b: Math.max(a, b)}; drawZoomBar();
    }
    tip.style.opacity = 0; paint(); return;
  }
  const r = rowAt(p.y);
  showTip(ev, r && r.t === 'lane' && p.x >= geo.L ? tipFor(r, cycleAt(p.x), p.y) : dividerTip(p));
  cv.style.cursor = r && (r.t === 'group' || (r.t === 'lane' && ['kern', 'comp'].includes(r.l.k))) ? 'pointer' : 'crosshair';
  paint();
});

cv.addEventListener('pointerup', ev => {
  if (!drag) return;
  const d = drag, p = pointer(ev);
  drag = null;
  if ((d.k === 'sel' || d.k === 'click' || d.k === 'pan') && !d.moved) clickAt(ev, p.x, p.y);
  paint();
});

cv.addEventListener('pointerleave', () => { hover = null; tip.style.opacity = 0; paint(); });

cv.addEventListener('wheel', ev => {
  const p = pointer(ev);
  if (p.x < geo.L || p.y < RULER || p.y >= geo.miniY) return;
  ev.preventDefault();
  if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) panBy((ev.deltaX || ev.deltaY) / 800);
  else zoomBy(Math.exp(ev.deltaY * 0.0015), cycleAt(p.x));
}, {passive: false});

document.addEventListener('keydown', ev => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey || document.getElementById('tab-profile').hidden) return;
  if (!(hover || document.activeElement === cv)) return;
  const pivot = hover && hover.x >= geo.L ? cycleAt(hover.x) : null;
  const k = ev.key.toLowerCase();
  if (k === 'w' || k === '+' || k === '=' || k === 'arrowup') zoomBy(0.8, pivot);
  else if (k === 's' || k === '-' || k === 'arrowdown') zoomBy(1.25, pivot);
  else if (k === 'a' || k === 'arrowleft') panBy(-0.15);
  else if (k === 'd' || k === 'arrowright') panBy(0.15);
  else if (k === 'z') zoomToSelection();
  else if (k === '0' || k === 'escape') { sel = null; setView(0, TOTAL); }
  else return;
  ev.preventDefault();
});

function toggleHeat() {
  heatOpen = !heatOpen;
  document.getElementById('z-heat').setAttribute('aria-pressed', heatOpen);
  document.getElementById('z-heat').hidden = !CH && !RUNS.some(r => r.dram.w.some(w => w.ch));
  if (CH) drawOverview();
  paint();
}

function zoomToSelection() { if (sel && sel.b > sel.a) { const s = sel; sel = null; setView(s.a, s.b); } }

function drawZoomBar() {
  const [lo, hi] = vspan();
  document.getElementById('z-win').textContent = view
    ? `cycle ${fmt(lo)} - ${fmt(hi)} (${fmt(hi - lo)} of ${fmt(TOTAL)} cycles, ${(TOTAL / (hi - lo)).toFixed(1)}x)`
    : `whole workload, ${fmt(TOTAL)} cycles`;
  document.getElementById('z-reset').disabled = !view;
  document.getElementById('z-out').disabled = !view;
  document.getElementById('z-in').disabled = hi - lo <= MIN_SPAN;
  document.getElementById('z-sel').disabled = !(sel && sel.b > sel.a);
}

let unitsTimer = 0;
function viewChanged() {
  drawZoomBar(); paint();
  clearTimeout(unitsTimer);
  unitsTimer = setTimeout(() => { const [lo, hi] = vspan(); drawUnits(rangeStats(lo, hi)); markOpRows(); }, 40);
}

function showInTrace(e) { go('profile'); zoomKernel(e); }

document.getElementById('z-in').addEventListener('click', () => zoomBy(0.6));
document.getElementById('z-out').addEventListener('click', () => zoomBy(1.6));
document.getElementById('z-reset').addEventListener('click', () => { sel = null; setView(0, TOTAL); });
document.getElementById('z-sel').addEventListener('click', zoomToSelection);
document.getElementById('z-heat').setAttribute('aria-pressed', heatOpen);
document.getElementById('z-heat').hidden = !RUNS.some(r => r.dram.w.some(w => w.ch));
document.getElementById('z-heat').addEventListener('click', toggleHeat);
if (window.ResizeObserver) new ResizeObserver(() => paint()).observe(host);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { paint(); if (lwDone) drawLowering(); });

let profileDone = false, machineDone = false;
function go(tab, dir) {
  location.hash = tab === 'lowering' ? '#lowering' + (dir ? '/' + encodeURIComponent(dir) : '') : '#profile';
}

function route() {
  if (!machineDone) { machineDone = true; drawMachine(); }
  const m = /^#(profile|lowering)(?:\/(.*))?$/.exec(location.hash), tab = m ? m[1] : 'profile';
  if (tab === 'lowering') { const d = m[2] ? decodeURIComponent(m[2]) : null; if (d) lwSel = d; }
  ['profile', 'lowering'].forEach(t => {
    document.getElementById('tab-' + t).hidden = t !== tab;
    document.getElementById('tab-b-' + t).setAttribute('aria-selected', t === tab);
  });
  if (tab === 'profile') {
    if (!profileDone) { profileDone = true; drawOverview(); drawOpProfile(); drawUnits(rangeStats(0, TOTAL)); }
    drawZoomBar(); paintNow();
  } else drawLowering();
}

document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => go(b.dataset.tab)));
window.addEventListener('hashchange', route);

document.getElementById('subtitle').textContent =
  `${KG.length} kernels in ${RUNS.length} runs, ${fmt(TOTAL)} cycles from ${DATA.logs} TOGSim logs` +
  (DATA.empty.length ? ` (${DATA.empty.length} empty left out).` : '.');
const traced = RUNS.filter(r => r.gantt).length;
document.getElementById('tv-note').textContent =
  'One cycle axis made of the runs laid end to end in log order; dashed dividers mark where a run starts (hover the ruler for its log). ' +
  'Each run was simulated separately from cycle 0, so the axis is a sum of simulated cycles, not wall-clock time.';
document.getElementById('tv-foot').textContent =
  `Compute-unit bars are derived from per-tile counts in tog_unit_table.tsv, not per-cycle traces: width is ` +
  `admitted work / capacity per cycle (capped at the instruction's compute_cycle), fill strength is its share of capacity. ` +
  (traced < RUNS.length ? `${RUNS.length - traced} of ${RUNS.length} runs have no per-instruction trace (TOGSim was not run with --log_level trace); ` +
    `their unit lanes show window-averaged utilization from the Core stat windows as filled step bars.` : '');
route();
</script>
</body></html>
"""


def join_workdir_timing(runs, timing):
    """Add to `timing` the timing.json of each kernel's workdir that no root gave."""
    for r in runs:
        for k in r["kernels"]:
            wd = contract.workdir_of(k["tog"])
            if wd not in timing:
                rec = read_timing(os.path.join(wd, "timing.json"))
                if rec:
                    timing[wd] = rec


def main(argv=None):
    """Parse the given logs and write the viewer page."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=["togsim_results"],
                    help="log files, globs, or directories (default: togsim_results)")
    ap.add_argument("-o", "--out", default="util.html", help="output HTML path")
    ap.add_argument("--timing", nargs="*", default=None, metavar="ROOT",
                    help="extra roots to scan for the compiler timing.json; the workdir each log's "
                         "tog_so is in is always checked")
    ap.add_argument("--torch-trace", nargs="*", default=None, metavar="DIR",
                    help="TORCH_TRACE log dirs or files; adds each kernel's PyTorch lowering (Dynamo, "
                         "AOTAutograd and Inductor graphs, model lines and ATen ops)")
    ap.add_argument("--inductor-debug", nargs="*", default=None, metavar="DIR",
                    help="TORCH_COMPILE_DEBUG dirs; embeds Inductor's graph diagrams "
                         "(INDUCTOR_ORIG_FX_SVG=1, INDUCTOR_POST_FUSION_SVG=1) with the selected kernel's "
                         "nodes marked")
    ap.add_argument("--breakdown", nargs="*", default=None, metavar="ROOT",
                    help="roots to scan for PyTorchSim breakdown_<run>.json (TORCHSIM_BREAKDOWN=1), "
                         "which adds Spike, gem5 and TOGSim wall clock")
    ap.add_argument("--map-path", action="append", default=[], metavar="CONTAINER=HOST",
                    help="read a logged path that does not exist here (a container's) from this prefix")
    ap.add_argument("--tile-timeline", action="append", default=[], metavar="KERNEL=PATH",
                    help="use PATH as KERNEL's tog_tile_timeline.json instead of the one in its dump dir")
    args = ap.parse_args(argv)
    TILE_TIMELINES.update(a.split("=", 1) for a in args.tile_timeline)
    _PATH_MAP[:] = [tuple(a.split("=", 1)) for a in args.map_path]

    runs = collect(args.paths or ["togsim_results"])
    compiles = read_torch_trace(args.torch_trace or [])
    diagrams = read_inductor_debug(args.inductor_debug or [])
    breakdowns = find_breakdowns(args.breakdown or [])
    timing = find_timing(args.timing or [])
    embedded = timing_from_breakdowns(breakdowns)
    join_workdir_timing(runs, timing)
    seen_labels = {rec.get("label") for rec in timing.values()}
    for label, rec in embedded.items():
        if label not in seen_labels:
            timing["label:" + label] = rec
    if not runs and not timing and not breakdowns:
        print("no TOGSim logs, no timing.json and no breakdown.json found", file=sys.stderr)
        return 1
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(build_payload(runs, timing, breakdowns, compiles, diagrams)))
    with open(args.out, encoding="utf-8") as fh:
        bad = _MOJIBAKE.findall(fh.read())
    if bad:
        src = ", ".join(_MOJIBAKE_INPUTS) or "none of the inputs (the generator itself)"
        print(f"error: {args.out} contains mojibake {sorted(set(bad))}; inputs with it: {src}",
              file=sys.stderr)
        return 1
    print(f"{len(runs)} run(s), {len(timing)} compile record(s), {len(breakdowns)} breakdown(s), "
          f"{len(compiles)} torch compile(s), {len(diagrams)} inductor diagram dir(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
