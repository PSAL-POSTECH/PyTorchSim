"""Where a run's wall clock went: torchsim-compile, Spike, gem5, TOGSim.

Off unless TORCHSIM_BREAKDOWN=1; one table at process exit and a
breakdown_<run-id>.json beside it, stamped so parallel runs do not share a file.
Nested spans are charged self time, so gem5 inside a trace build counts once.
"""

import atexit
import datetime
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()

ENV = "TORCHSIM_BREAKDOWN"
PREFIX = "breakdown"

TORCHSIM_COMPILE = "torchsim-compile"
SPIKE = "spike"
GEM5_SAMPLE = "gem5/sample"
GEM5_BUILD = "gem5/build"
GEM5_RUN = "gem5/run"
TOGSIM_TRACE = "togsim/trace"
TOGSIM_RUN = "togsim/run"

ORDER = [TORCHSIM_COMPILE, SPIKE, GEM5_SAMPLE, GEM5_BUILD, GEM5_RUN, TOGSIM_TRACE, TOGSIM_RUN]


def enabled():
    """Whether this run collects a breakdown."""
    return os.environ.get(ENV, "0") not in ("0", "", "no", "false", "False")


def new_run_id():
    """A per-run stamp, `<YYYYMMDD_HHMMSS>_<8 hex>`, as togsim_results names its logs.

    Parallel experiments sharing one dump path would otherwise write one file.
    """
    return (datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_" + uuid.uuid4().hex[:8])


class Record:
    """One process's spans, per component and per kernel, plus torchsim-compile's own JSON."""

    def __init__(self):
        self.totals = {}
        self.kernels = {}
        self.compiles = []
        self.started = time.perf_counter()
        self.run_id = new_run_id()
        self._local = threading.local()
        self._lock = threading.Lock()

    def stack(self):
        """This thread's open spans, innermost last."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def add(self, component, kernel, seconds):
        """Charge `seconds` of self time to one component and one kernel."""
        with self._lock:
            e = self.totals.setdefault(component, [0, 0.0])
            e[0] += 1
            e[1] += seconds
            k = self.kernels.setdefault(kernel or "?", {})
            k[component] = k.get(component, 0.0) + seconds

    def add_compile(self, kernel, kind, rec):
        """Keep one torchsim-compile record -- a compile or a launch -- for the sub-tables."""
        with self._lock:
            self.compiles.append((kernel, kind, rec))

    @property
    def measured(self):
        return sum(dt for _, dt in self.totals.values())

    def as_dict(self):
        return {
            "run": self.run_id,
            "pid": os.getpid(),
            "wall": time.perf_counter() - self.started,
            "measured": self.measured,
            "components": {c: {"calls": n, "seconds": dt}
                           for c, (n, dt) in self.totals.items()},
            "kernels": self.kernels,
            "torchsim-compile": [{"kernel": k, "kind": kind, "timing": r}
                     for k, kind, r in self.compiles],
        }


REC = Record()
_INSTALLED = [False]


@contextmanager
def span(component, kernel=None):
    """Time one component of one kernel's launch or compile.

    A no-op when the breakdown is off: no clock is read and nothing is stored.
    """
    if not enabled():
        yield
        return
    _install()
    stack = REC.stack()
    frame = [0.0]
    stack.append(frame)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        stack.pop()
        if stack:
            stack[-1][0] += wall
        REC.add(component, kernel, wall - frame[0])


def ingest_compile(workdir, kernel, kind="compile", name="timing.json"):
    """Read the per-stage/per-pass record torchsim-compile left in `workdir`, if any."""
    if not enabled():
        return
    path = os.path.join(workdir, name)
    try:
        with open(path) as f:
            REC.add_compile(kernel, kind, json.load(f))
    except (OSError, ValueError):
        logger.debug("[breakdown] no readable %s", path)


def _install():
    """Register the exit hook once, on the first span that is actually timed."""
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    atexit.register(_at_exit)


def _at_exit():
    if not REC.totals:
        return
    text = render(REC.as_dict())
    print("\n" + text, flush=True)
    try:
        path = os.path.join(extension_config.get_dump_path(),
                            f"{PREFIX}_{REC.run_id}.json")
        with open(path, "w") as f:
            json.dump(REC.as_dict(), f, indent=2)
        print(f"  {os.path.basename(path)}: {path}", flush=True)
    except OSError:
        pass


def _sum_compiles(records, key):
    """Sum every kernel's torchsim-compile `stages` or a stage's `passes` into one dict."""
    out = {}
    order = []
    for rec in records:
        entries = rec.get(key) or []
        for e in entries:
            if e["name"] not in out:
                order.append(e["name"])
            out[e["name"]] = out.get(e["name"], 0.0) + e["seconds"]
    return [(n, out[n]) for n in order]


def _compile_passes(records, stage):
    """Sum one torchsim-compile stage's per-pass times across every compiled kernel."""
    out = {}
    for rec in records:
        for e in (rec.get("passes") or {}).get(stage, []):
            out[e["name"]] = out.get(e["name"], 0.0) + e["seconds"]
    return sorted(out.items(), key=lambda kv: -kv[1])


def _compile_tools(records):
    """Sum every record's external-tool time, keyed `stage/binary`."""
    out = {}
    for rec in records:
        for stage, entries in (rec.get("tools") or {}).items():
            for tool, e in entries.items():
                cur = out.setdefault(f"{stage}/{tool}", [0, 0.0])
                cur[0] += e["calls"]
                cur[1] += e["seconds"]
    return sorted(out.items(), key=lambda kv: -kv[1][1])


def _bar(frac, width=20):
    return "#" * int(round(frac * width)) + "." * (width - int(round(frac * width)))


def _line(name, dt, total, indent=2, width=40, extra=""):
    pct = 100.0 * dt / total if total else 0.0
    return (f"{' ' * indent}{name:<{width - indent}} {dt:9.2f}s {pct:5.1f}%"
            f"{extra}")


def render(rec):
    """The table a human reads, from the same dict breakdown.json holds."""
    wall = rec.get("wall") or 0.0
    comps = rec.get("components") or {}
    kernels = rec.get("kernels") or {}
    entries = rec.get("torchsim-compile") or []
    compiles = [e["timing"] for e in entries if e.get("kind", "compile") == "compile"]
    launch = [e["timing"] for e in entries
              if e.get("kind") in ("spike", "cycle")]

    out = [f"BREAKDOWN  {rec.get('run') or '?'}  pid {rec.get('pid')}",
           f"           {wall:.2f}s since the npu backend loaded, "
           f"{len(kernels)} kernel(s)", ""]
    groups = {}
    for c, e in comps.items():
        groups.setdefault(c.split("/")[0], [0, 0.0])
        groups[c.split("/")[0]][0] += e["calls"]
        groups[c.split("/")[0]][1] += e["seconds"]
    for g in [x for x in ("torchsim-compile", "spike", "gem5", "togsim") if x in groups]:
        calls, dt = groups[g]
        pct = 100.0 * dt / wall if wall else 0.0
        out.append(f"  {g:<14} {dt:9.2f}s {pct:5.1f}%  x{calls:<5} {_bar(pct / 100)}")
        for c in ORDER:
            if c in comps and c.startswith(g + "/"):
                out.append(_line(c.split("/", 1)[1], comps[c]["seconds"], wall,
                                 indent=6, extra=f"  x{comps[c]['calls']}"))
    other = wall - (rec.get("measured") or 0.0)
    pct = 100.0 * other / wall if wall else 0.0
    out.append(f"  {'elsewhere':<14} {other:9.2f}s {pct:5.1f}%  "
               f"{'':<6} {_bar(pct / 100)}   torch, inductor, python")

    if compiles:
        stages = _sum_compiles(compiles, "stages")
        if stages:
            out += ["", "  torchsim-compile, by stage"]
            for name, dt in stages:
                out.append(_line(name, dt, wall, indent=4))
    for records, stage in ((compiles, "transform"), (compiles, "lower"),
                           (launch, "spike")):
        passes = _compile_passes(records, stage)
        if not passes:
            continue
        out += ["", f"  inside torchsim-compile's {stage}"]
        shown = [p for p in passes if p[1] >= 0.05][:15]
        for name, dt in shown:
            out.append(_line(name, dt, wall, indent=4))
        rest = passes[len(shown):]
        if rest:
            out.append(_line(f"({len(rest)} smaller)",
                             sum(d for _, d in rest), wall, indent=4))

    tools = _compile_tools(compiles + launch)
    if tools:
        out += ["", "  external tools, summed over calls -- Spike splits one "
                    "grid over several processes,", "  so its sum can exceed "
                    "the wall clock it ran inside"]
        for name, (calls, dt) in tools:
            out.append(f"    {name:<36} {dt:9.2f}s  x{calls}")

    if kernels:
        out += ["", "  per kernel (self time, seconds)"]
        cols = [g for g in ("torchsim-compile", "spike", "gem5", "togsim") if g in groups]
        head = "".join(f"{c:>10}" for c in cols)
        out.append(f"    {'kernel':<46}{head}{'total':>10}")
        rows = []
        for k, per in kernels.items():
            by_group = {c: 0.0 for c in cols}
            for comp, dt in per.items():
                g = comp.split("/")[0]
                if g in by_group:
                    by_group[g] += dt
            rows.append((k, by_group, sum(per.values())))
        for k, by_group, tot in sorted(rows, key=lambda r: -r[2]):
            cells = "".join(f"{by_group[c]:10.2f}" for c in cols)
            out.append(f"    {k[:46]:<46}{cells}{tot:10.2f}")
    return "\n".join(out)
