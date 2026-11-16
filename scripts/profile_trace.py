"""TOGSim logs to a Chrome/Perfetto trace on the simulated time axis, and gem5 tile timelines to one each.
Runs are laid end to end in log order; ts and dur are microseconds (cycles / core MHz) and each slice's
args carry the raw cycles. Instruction slices need a --log_level trace log (trace_timeline.py)."""
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorchsim import contract  # noqa: E402
from pytorchsim.simulator import perfetto_trace  # noqa: E402
import profile_sol  # noqa: E402
import trace_timeline  # noqa: E402

_CMDLINE = re.compile(r"--(?:tog_so|trace_so) (\S+)")
_ENQUEUED = re.compile(r"\[Scheduler (\d+)\] Enqueued kernel_id: (\d+), tog_path: (\S+?),")
_CONFIG_LOADED = re.compile(r'Loaded configuration file "([^"]+)"')
_CONFIG_INT = re.compile(r"^(\w+): (\d+)$")
_PEAK_BW = re.compile(r"Total bandwidth ([\d.]+) GB/s, (\d+) MHz, (\d+) channels, (\d+) bytes")
_UNIT = re.compile(r"Core \[(\d+)\] : (\S+?) utilization\(%\): ([\d.]+), "
                   r"active_cycles: (\d+), idle_cycles: (\d+)")
_DMA = re.compile(r"Core \[(\d+)\] : DMA active_cycles: (\d+), DMA idle_cycles: (\d+)")
_DRAM_FINAL_CH = re.compile(r"\[DRAM\] channel \d+ \| ([\d.]+) GB/s avg\., [\d.]+% of utilization "
                            r"\| \d+ reads, \d+ writes$")
_DRAM_WCH = re.compile(r"\[DRAM\] channel (\d+) \| ([\d.]+) GB/s avg\., ([\d.]+)% of utilization "
                       r"\| (\d+) reads, (\d+) writes \(interval")
_DRAM_WINDOW = re.compile(r"\[DRAM\] all \d+ channels combined \| ([\d.]+) GB/s "
                          r"aggregate, ([\d.]+)% of utilization[^|]*\| (\d+) reads, (\d+) writes"
                          r"(?: \(interval (\d+) cycles\))?")
_KERNEL_DONE = re.compile(r"\[Scheduler (\d+)\] Kernel (\d+) .* finished at cycle (\d+)")
_TOTAL_EXEC = re.compile(r"Total execution cycles: (\d+)")
_STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
_TILE_SCHED = re.compile(r"\[(\d+)\]\[Core (\d+)\]\[TILE_SCHEDULED")
_FINAL_MARKERS = ("=== DRAM statistics ===", "[DRAM] Per-channel average bandwidth",
                  "===== Instructions count =====")

_TT_HEAD = re.compile(r"^\S+ \S+ \d+ \S+\] (\{.*\})\s*$")
_TT_KERNEL = re.compile(r"kernel_name='(\w+)'")
_TT_CALL = re.compile(r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]")
_TT_CALLEE = re.compile(r"^\s*(\w+)(?:\.run)?\(")
_TT_FILE = re.compile(r"^\s*# File: (.*)$")
_TT_NODE = re.compile(r"^\s*(\w+)(?:: [^=]*)? = ")
_TT_GRAPHS = (("dynamo_output_graph", "Dynamo FX graph"),
              ("inductor_post_grad_graph", "Inductor post-grad graph"),
              ("inductor_output_code", "Inductor output code"))

PID_DRAM = 1000
TID_KERNELS, TID_COPY, TID_DMA_READ, TID_DMA_WRITE, TID_TILES, TID_COMPUTE = 1, 2, 3, 4, 5, 10


def _read_text(path):
    """The whole file decoded as UTF-8, undecodable bytes replaced."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def get_host_path(path, path_map):
    """`path` as this host sees it: a path that does not exist is mapped through the (from, to) prefixes."""
    if os.path.exists(path):
        return path
    for src, dst in path_map:
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def parse_log(path, path_map=()):
    """One TOGSim log as a run dict, or None when it has no final stats. A run is one simulation from
    cycle 0 and may schedule several kernels back to back."""
    text = _read_text(path)
    if "Total execution cycles" not in text:
        return None
    cut = min((text.index(m) for m in _FINAL_MARKERS if m in text), default=len(text))
    head, tail = text[:cut], text[cut:]
    stamp = _STAMP.match(text)
    loaded = _CONFIG_LOADED.search(head)
    run = {"log": os.path.basename(path), "stamp": stamp.group(1) if stamp else "", "config": {},
           "machine": os.path.basename(loaded.group(1)) if loaded else "", "wins": [],
           "dram_win": [], "dma_wins": [], "kernels": [], "tiles": [], "events": None}
    for line in head.splitlines():
        cm = _CONFIG_INT.match(line.strip())
        if cm and cm.group(1) in ("core_freq_mhz", "core_stats_print_period_cycles", "num_cores",
                                  "num_systolic_array_per_core"):
            run["config"].setdefault(cm.group(1), int(cm.group(2)))
    bw = _PEAK_BW.search(head)
    if bw:
        run["config"]["dram_mhz"] = int(bw.group(2))
        run["config"]["peak_gbps"] = float(bw.group(1))
    cur, chans, enq, done = None, [], {}, {}
    for line in head.splitlines():
        if "========= Core stat =========" in line:
            cur = {}
            run["wins"].append(cur)
            run["dma_wins"].append({})
        elif "[DRAM]" in line:
            m = _DRAM_WCH.search(line)
            if m:
                chans.append(float(m.group(2)))
                continue
            m = _DRAM_WINDOW.search(line)
            if m:
                run["dram_win"].append({"gbps": float(m.group(1)), "util": float(m.group(2)),
                                        "reqs": int(m.group(3)) + int(m.group(4)),
                                        "span": int(m.group(5) or 0), "ch": chans})
                chans = []
        elif "Scheduler" in line:
            m = _ENQUEUED.search(line)
            if m:
                enq[int(m.group(2))] = m.group(3)
            m = _KERNEL_DONE.search(line)
            if m:
                done[int(m.group(2))] = int(m.group(3))
        elif "[TILE_SCHEDULED" in line:
            m = _TILE_SCHED.search(line)
            if m:
                run["tiles"].append((int(m.group(2)), int(m.group(1))))
        elif cur is not None:
            m = _UNIT.search(line)
            if m:
                cur.setdefault(m.group(2), {})[int(m.group(1))] = float(m.group(3))
                continue
            m = _DMA.search(line)
            if m:
                run["dma_wins"][-1][int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    for kid in sorted(enq):
        run["kernels"].append({"id": kid, "tog": get_host_path(enq[kid], path_map), "end": done.get(kid)})
    if not run["kernels"]:
        m = _CMDLINE.search(head)
        if m:
            run["kernels"].append({"id": 0, "tog": get_host_path(m.group(1), path_map), "end": None})
    total = _TOTAL_EXEC.search(tail)
    run["cycles"] = int(total.group(1)) if total else 0
    run["cores"] = max([c for w in run["wins"] for u in w.values() for c in u] or [0]) + 1
    run["total"] = _parse_final_stats(tail)
    run["dram_window_cycles"] = get_dram_window_cycles(run)
    if "[INST_ID=" in text:
        num_sa = run["config"].get("num_systolic_array_per_core", 1)
        insts = trace_timeline.parse(text.splitlines())
        run["events"] = trace_timeline.to_chrome(insts, num_sa)["traceEvents"]
    return run


def _parse_final_stats(tail):
    """The whole-run Core stat table and DRAM channel averages printed after the last window: units, DMA
    (active, idle) per core and the summed DRAM GB/s (None without them)."""
    units, dma, gbps = {}, {}, []
    lines = tail.splitlines()
    start = max((i for i, ln in enumerate(lines) if "========= Core stat =========" in ln), default=None)
    for ln in lines[start:] if start is not None else []:
        m = _UNIT.search(ln)
        if m:
            units.setdefault(m.group(2), {})[int(m.group(1))] = float(m.group(3))
            continue
        m = _DMA.search(ln)
        if m:
            dma[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    for ln in lines:
        m = _DRAM_FINAL_CH.search(ln)
        if m:
            gbps.append(float(m.group(1)))
    return {"units": units, "dma": dma, "dram_gbps": sum(gbps) if gbps else None}


def collect_runs(paths, path_map=()):
    """Parsed runs of every log in `paths` (files, globs or directories), in timestamp order; a log that
    cannot be parsed is reported and skipped."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.log"))) if os.path.isdir(p) else sorted(glob.glob(p))
    runs = []
    for f in sorted(set(files)):
        try:
            run = parse_log(f, path_map)
        except (OSError, ValueError) as exc:
            print(f"skip {f}: {exc}", file=sys.stderr)
            continue
        if run:
            runs.append(run)
    return sorted(runs, key=lambda r: (r["stamp"], r["log"]))


def split_empty(runs):
    """(runs that scheduled a kernel and ran, the rest): a run with no kernel or one cycle is empty."""
    live = [r for r in runs if r["kernels"] and r["cycles"] > 1]
    return live, [r for r in runs if r not in live]


def get_kernel_name(tog_so):
    """The `triton_<hash>` directory name of a tog.so path, else the name of its directory."""
    parts = [p for p in tog_so.split(os.sep) if p]
    return next((p for p in reversed(parts) if p.startswith("triton_")),
                parts[-2] if len(parts) > 1 else tog_so)


def get_short_name(torch_kernel, fallback):
    """Inductor's kernel name without the backend prefix and the hash suffix."""
    if not torch_kernel:
        return fallback
    name = re.sub(r"^pytorchsim_triton_compiler_", "", torch_kernel)
    return re.sub(r"_[a-z0-9]{8}$", "", name) or fallback


def get_sidecar(tog_so, name):
    """Path of `name` in the directory holding `tog_so` (its launch-* component dropped), if it exists."""
    parts = [p for p in tog_so.split(os.sep) if not p.startswith(f"{contract.LAUNCH_DIR_PREFIX}-")]
    path = os.sep + os.path.join(*parts[:-1], name)
    return path if os.path.exists(path) else None


def get_kernel_meta(workdir):
    """(Inductor kernel name from meta.json, path of its Triton source) of the nearest dir at or above
    `workdir` that holds meta.json."""
    for _ in range(4):
        if os.path.exists(os.path.join(workdir, "meta.json")) or os.path.dirname(workdir) == workdir:
            break
        workdir = os.path.dirname(workdir)
    try:
        name = json.loads(_read_text(os.path.join(workdir, "meta.json"))).get("kernel_name")
    except (OSError, ValueError):
        name = None
    return name, next(iter(sorted(glob.glob(os.path.join(workdir, "*_triton.py")))), None)


def read_torch_trace(paths):
    """Each compile in PyTorch's TORCH_TRACE logs: the text of the graphs in _TT_GRAPHS."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.log"))) if os.path.isdir(p) else sorted(glob.glob(p))
    compiles = []
    for f in files:
        by_id = {}
        for head, payload in _read_torch_trace_events(f):
            if "frame_id" not in head:
                continue
            name = next((k for k, _ in _TT_GRAPHS if k in head), None)
            name = name or (head.get("artifact") or {}).get("name")
            if name not in dict(_TT_GRAPHS):
                continue
            cid = f"{head['frame_id']}/{head['frame_compile_id']}"
            c = by_id.get(cid)
            if c is None:
                c = by_id[cid] = {"graphs": {}}
                compiles.append(c)
            c["graphs"][dict(_TT_GRAPHS)[name]] = payload
    return compiles


def _read_torch_trace_events(path):
    """(header dict, payload text) for each record of one TORCH_TRACE log; payload lines start with a tab."""
    head, payload = None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("\t"):
                payload.append(line[1:].rstrip("\n"))
                continue
            if head is not None:
                yield head, "\n".join(payload)
            m = _TT_HEAD.match(line)
            head, payload = (json.loads(m.group(1)) if m else None), []
    if head is not None:
        yield head, "\n".join(payload)


def get_node_lines(graph):
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


def get_call_sites(code, lines):
    """Kernel name -> {nodes, aten, lines} from the `# Topologically Sorted Source Nodes` comment above
    each call of a kernel `code` defines; `lines` maps a node to its model line."""
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


def map_torch_origins(compiles):
    """Inductor kernel name -> {aten, lines, nodes, from} out of TORCH_TRACE's output code and graphs."""
    out = {}
    for c in compiles:
        code = c["graphs"].get("Inductor output code", "")
        lines = {**get_node_lines(c["graphs"].get("Inductor post-grad graph", "")),
                 **get_node_lines(c["graphs"].get("Dynamo FX graph", ""))}
        for k, found in get_call_sites(code, lines).items():
            out.setdefault(k, dict(found, **{"from": "TORCH_TRACE"}))
    return out


def read_inductor_origins(paths):
    """Inductor kernel name -> {aten, lines, nodes, from} from each TORCH_COMPILE_DEBUG dir under `paths`."""
    dirs = []
    for root in paths:
        dirs += sorted({os.path.dirname(f) for f in
                        glob.glob(os.path.join(root, "**", "output_code.py"), recursive=True)})
    out = {}
    for d in dirs:
        code = _read_text(os.path.join(d, "output_code.py"))
        lines = {}
        for fname in ("fx_graph_readable.py", "fx_graph_transformed.py"):
            try:
                lines = {**get_node_lines(_read_text(os.path.join(d, fname))), **lines}
            except OSError:
                continue
        for k, found in get_call_sites(code, lines).items():
            out.setdefault(k, dict(found, **{"from": "Inductor debug " + os.path.basename(d)}))
    return out


def build_kernels(run, origins):
    """The run's kernels laid end to end from cycle 0, each with its cycles, short name, workdir and origin;
    copy_node kernels are counted into the next real kernel's cycles_incl_copy, and its `span` is the
    cycle range from the first of them to its own end (a trailing copy extends the last real kernel)."""
    out, prev, pending = [], 0, []
    for k in run["kernels"]:
        end = max(k["end"] if k["end"] is not None else run["cycles"], prev)
        copy = "copy_node" in k["tog"]
        workdir = contract.workdir_of(k["tog"])
        name = get_kernel_name(k["tog"])
        torch_kernel, source = (None, None) if copy else get_kernel_meta(workdir)
        rec = {"id": k["id"], "tog": k["tog"], "dir": name, "copy": copy, "start": prev, "end": end,
               "workdir": workdir, "torch_kernel": torch_kernel, "triton_source": source,
               "short": "copy" if copy else get_short_name(torch_kernel, name),
               "origin": origins.get(torch_kernel) if torch_kernel else None, "copy_cycles": 0,
               "copies": []}
        if copy:
            pending.append(rec)
        else:
            rec["copies"], pending = pending, []
        out.append(rec)
        prev = end
    real = [r for r in out if not r["copy"]]
    if pending and real:
        real[-1]["copies"] += pending
    for rec in out:
        group = [rec] + rec["copies"]
        rec["copy_cycles"] = sum(c["end"] - c["start"] for c in rec["copies"])
        rec["span"] = (min(c["start"] for c in group), max(c["end"] for c in group))
    return out


def label_units(names):
    """raw unit name -> display name: the common prefix of the prefixed names dropped, then the part before
    the first '.'; names that collide keep their raw spelling."""
    pref = [n for n in names if "_" in n]
    common = os.path.commonprefix(pref) if pref else ""
    common = common[:common.rfind("_") + 1] if "_" in common else ""
    disp = {n: (n[len(common):] if common and n.startswith(common) else n).split(".")[0] for n in names}
    count = {}
    for d in disp.values():
        count[d] = count.get(d, 0) + 1
    return {n: (d if count[d] == 1 else n) for n, d in disp.items()}


def get_dram_window_cycles(run):
    """Length in core cycles of one DRAM stats window: its period in DRAM clocks scaled by the clock ratio."""
    cfg, wins = run["config"], run["dram_win"]
    span = next((w["span"] for w in wins if w["span"]), 0) or 200
    core_mhz, dram_mhz = cfg.get("core_freq_mhz"), cfg.get("dram_mhz")
    if core_mhz and dram_mhz:
        return span * core_mhz / dram_mhz
    return float(cfg.get("core_stats_print_period_cycles") or span)


class _Events:
    """Trace events with their process and thread names, so Perfetto shows clean names in a fixed order."""

    def __init__(self):
        self.events, self.procs, self.threads = [], {}, {}

    def proc(self, pid, name, sort):
        self.procs[pid] = (name, sort)

    def thread(self, pid, tid, name, sort):
        self.threads[(pid, tid)] = (name, sort)

    def add(self, **kw):
        self.events.append(kw)

    def finish(self):
        meta = []
        for pid, (name, sort) in sorted(self.procs.items()):
            meta.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": name}})
            meta.append({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
                         "args": {"sort_index": sort}})
        for (pid, tid), (name, sort) in sorted(self.threads.items()):
            meta.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}})
            meta.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid,
                         "args": {"sort_index": sort}})
        return meta + self.events


def get_rel_path(path, out):
    """`path` relative to the directory `out` with '/' separators, or None when it does not exist."""
    if not path or not out or not os.path.exists(path):
        return None
    return os.path.relpath(path, out).replace(os.sep, "/")


def _get_kernel_links(k, out, tiles):
    """tile_trace, source and dump_dir of a kernel as paths relative to `out` (those that exist)."""
    links = {"tile_trace": (tiles or {}).get(k["dir"]), "source": get_rel_path(k["triton_source"], out),
             "dump_dir": get_rel_path(k["workdir"], out)}
    return {key: v for key, v in links.items() if v}


def build_trace(runs, origins=None, dram_channels=False, out=None, tiles=None):
    """The Chrome trace of `runs` (parsed, in order), kernels via build_kernels; returns (trace dict,
    per-run kernel records) so the index page reads the same numbers. With the dump directory `out`, kernel
    slices carry their source and dump_dir paths relative to it, and tile_trace from `tiles` (kernel ->
    path)."""
    live, empty = split_empty(runs)
    origins = origins or {}
    ev, kernel_runs, offset_us = _Events(), [], 0.0
    names = (u for r in live for u in [*(u for w in r["wins"] for u in w), *r["total"]["units"]])
    labels = label_units(list(dict.fromkeys(names)))
    for r in live:
        mhz = r["config"].get("core_freq_mhz") or 1000
        kernels = build_kernels(r, origins)
        kernel_runs.append(kernels)
        for k in kernels:
            k["sol"] = profile_sol.get_sol(r, k["span"][0], k["span"][1], labels)

        def us(cycles):
            return offset_us + cycles / mhz

        ev.add(ph="i", s="g", name=f"run {r['log']}", pid=0, tid=0, ts=us(0),
               args={"log": r["log"], "cycles": r["cycles"], "start_cycle": 0, "core_mhz": mhz})
        for c in range(r["cores"]):
            pid = c + 1
            ev.proc(pid, f"NPU core {c}", c)
            ev.thread(pid, TID_KERNELS, "Kernels", 0)
            ev.thread(pid, TID_COPY, "copy", 1)
            for k in kernels:
                dur = k["end"] - k["start"]
                span = k["copy_cycles"]
                args = {"triton_kernel": k["dir"], "inductor_kernel": k["torch_kernel"],
                        "start_cycle": k["start"], "cycle_end": k["end"], "cycles": dur,
                        "cycles_excl_copy": dur, "cycles_incl_copy": dur + span, "workdir": k["workdir"],
                        "log": r["log"], **profile_sol.get_trace_args(k["sol"]),
                        **_get_kernel_links(k, out, tiles)}
                if k["origin"]:
                    args["origin"] = k["origin"]
                tid = TID_COPY if k["copy"] else TID_KERNELS
                ev.add(ph="X", name=k["short"], cat="kernel", pid=pid, tid=tid, ts=us(k["start"]),
                       dur=dur / mhz, args=args)
        _add_unit_counters(ev, r, labels, us)
        _add_dram_counters(ev, r, us, dram_channels)
        _add_instruction_slices(ev, r, us)
        offset_us += r["cycles"] / mhz
    ev.proc(PID_DRAM, "DRAM", 1000)
    mhz = next((r["config"]["core_freq_mhz"] for r in live if r["config"].get("core_freq_mhz")), 1000)
    machine = next((r["machine"] for r in live if r["machine"]), "unknown")
    trace = {"traceEvents": ev.finish(), "displayTimeUnit": "ns",
             "otherData": {"time_unit": "us = cycles / core_freq_mhz", "machine": machine, "core_mhz": mhz,
                           "time_axis": f"simulated runs concatenated in log order; 1 us = {mhz} cycles",
                           "verdict_note": profile_sol.get_note(),
                           "runs": [r["log"] for r in live], "empty_runs": [r["log"] for r in empty]}}
    return trace, kernel_runs


def _add_unit_counters(ev, run, labels, us):
    """One counter track per core and unit: the utilization % each Core stat window printed."""
    interval = run["config"].get("core_stats_print_period_cycles", 0)
    for i, win in enumerate(run["wins"]):
        if not interval or i * interval >= run["cycles"]:
            break
        for name, per_core in win.items():
            for core, util in per_core.items():
                ev.add(ph="C", name=f"{labels[name]} util %", pid=core + 1, ts=us(i * interval),
                       args={"util %": util})


def _add_dram_counters(ev, run, us, dram_channels):
    """DRAM counters per stats window: total GB/s, utilization %, requests, and with `dram_channels`
    each channel's GB/s."""
    wl = get_dram_window_cycles(run)
    for j, w in enumerate(run["dram_win"]):
        if j * wl >= run["cycles"]:
            break
        ts = us(j * wl)
        ev.add(ph="C", name="DRAM GB/s", pid=PID_DRAM, ts=ts, args={"GB/s": w["gbps"]})
        ev.add(ph="C", name="DRAM utilization %", pid=PID_DRAM, ts=ts, args={"util %": w["util"]})
        ev.add(ph="C", name="DRAM requests", pid=PID_DRAM, ts=ts, args={"requests": w["reqs"]})
        if dram_channels:
            for ch, gbps in enumerate(w["ch"]):
                ev.add(ph="C", name=f"DRAM channel {ch} GB/s", pid=PID_DRAM, ts=ts, args={"GB/s": gbps})


def _add_instruction_slices(ev, run, us):
    """DMA read/write slices, compute-lane slices and tile markers from a trace-level log."""
    if not run["events"]:
        return
    mhz = run["config"].get("core_freq_mhz") or 1000
    lanes = {}
    for e in run["events"]:
        if e.get("ph") != "X":
            continue
        pid, lane = e["pid"] + 1, e["tid"]
        if lane == "dram-rd":
            tid, tname = TID_DMA_READ, "DMA read"
        elif lane == "dram-wr":
            tid, tname = TID_DMA_WRITE, "DMA write"
        else:
            tid = lanes.setdefault((pid, lane), TID_COMPUTE + len(lanes))
            tname = f"compute {lane}"
        ev.thread(pid, tid, tname, tid)
        args = dict(e.get("args") or {}, start_cycle=e["ts"], cycles=e["dur"])
        ev.add(ph="X", name=e["name"], cat="insn", pid=pid, tid=tid, ts=us(e["ts"]), dur=e["dur"] / mhz,
               args=args, **({"cname": e["cname"]} if "cname" in e else {}))
    for core, cycle in run["tiles"]:
        ev.thread(core + 1, TID_TILES, "tiles", 9)
        ev.add(ph="i", s="t", name="tile scheduled", pid=core + 1, tid=TID_TILES, ts=us(cycle),
               args={"start_cycle": cycle, "cycles": 0})


def check_trace(trace):
    """Raise ValueError unless every X has ts >= 0 and dur >= 0, every counter has numeric args and no
    number is NaN or infinite."""
    for e in trace["traceEvents"]:
        for key in ("ts", "dur"):
            if key in e and (not isinstance(e[key], (int, float)) or not math.isfinite(e[key]) or e[key] < 0):
                raise ValueError(f"bad {key} in {e}")
        if e["ph"] == "X" and ("ts" not in e or "dur" not in e):
            raise ValueError(f"slice without ts/dur: {e}")
        if e["ph"] == "C":
            vals = e.get("args") or {}
            if not vals or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals.values()):
                raise ValueError(f"counter without numeric args: {e}")
        _check_args(e.get("args"), e)
    check_nesting(trace)


def check_nesting(trace):
    """Raise ValueError when two X slices of one thread overlap without one containing the other, or a
    slice's pid or tid is not an int: the binary format can only nest."""
    by_thread = {}
    for e in trace["traceEvents"]:
        if e["ph"] == "X":
            if not isinstance(e["pid"], int) or not isinstance(e["tid"], int):
                raise ValueError(f"pid and tid must be ints: {e}")
            by_thread.setdefault((e["pid"], e["tid"]), []).append(e)
    for key, slices in by_thread.items():
        stack = []
        for e in sorted(slices, key=lambda e: (e["ts"], -e["dur"])):
            while stack and stack[-1] <= e["ts"] + 1e-9:
                stack.pop()
            end = e["ts"] + e["dur"]
            if stack and end > stack[-1] + 1e-9:
                raise ValueError(f"slice {e['name']!r} overlaps its neighbour on thread {key}")
            stack.append(end)


def _check_args(value, event):
    """Raise ValueError for a NaN or infinite number anywhere in an event's args."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {event}")
    for v in value.values() if isinstance(value, dict) else value if isinstance(value, list) else ():
        _check_args(v, event)


def read_tile_timeline(tog_so, override=None):
    """The event dict of the tog_tile_timeline.pftrace beside `tog_so` (or the file `override` names), or
    None; a trace of another schema is refused."""
    path = override or get_sidecar(tog_so, "tog_tile_timeline.pftrace")
    if not path:
        return None
    trace = perfetto_trace.read_trace(path)
    schema = trace["otherData"].get("schema")
    if schema != perfetto_trace.TILE_TIMELINE_SCHEMA:
        raise ValueError(f"tile timeline {path}: unknown schema {schema!r}")
    return trace


def collect_tile_docs(live, tile_files=None):
    """Kernel dir name -> its tile timeline trace for every non-copy kernel of the runs that has one;
    `tile_files` maps a kernel name to a file to use instead of the one beside its tog.so."""
    docs = {}
    for r in live:
        for k in r["kernels"]:
            name = get_kernel_name(k["tog"])
            if name in docs or "copy_node" in k["tog"]:
                continue
            doc = read_tile_timeline(k["tog"], (tile_files or {}).get(name))
            if doc:
                docs[name] = doc
    return docs


def build_tile_trace(timeline, kernel):
    """`timeline` (a tile timeline's event dict) with the kernel's name in its metadata."""
    return {**timeline, "otherData": {**timeline["otherData"], "kernel": kernel}}
