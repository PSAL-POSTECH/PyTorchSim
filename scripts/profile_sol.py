"""Utilization-based speed of light of a kernel's cycle span: each unit's, the DMA's and the DRAM's busy share
and a verdict. Not a FLOP roofline (no FLOP counts); numbers are cycle-weighted over the Core stat windows
that overlap the span, or the whole-run table when no window does."""

BOUND = 60.0
LOW = 20.0
VERDICT_COMPUTE = "compute-bound"
VERDICT_MEMORY = "memory-bound (DRAM)"
VERDICT_DMA = "DMA-latency-bound"
VERDICT_LATENCY = "latency/dependency-bound"
VERDICT_UNKNOWN = "unknown"
VERDICT_CLASSES = {VERDICT_MEMORY: "memory", VERDICT_DMA: "dma", VERDICT_LATENCY: "latency",
                   VERDICT_UNKNOWN: "unknown"}


def get_note():
    """One sentence stating how the verdict is decided, with the thresholds."""
    return (f"Utilization-based, not a FLOP roofline: compute-bound when a unit is at least {BOUND:.0f}% "
            f"busy, "
            f"memory-bound when DRAM bandwidth is at least {BOUND:.0f}% of peak, DMA-latency-bound when the "
            f"DMA is at least {BOUND:.0f}% busy while DRAM is under {LOW:.0f}%, otherwise "
            f"latency/dependency-bound.")


def get_verdict_class(verdict):
    """The CSS class suffix of a verdict string."""
    return "compute" if verdict.startswith(VERDICT_COMPUTE) else VERDICT_CLASSES.get(verdict, "unknown")


def get_verdict(compute_pct, compute_unit, dram_pct, dma_pct):
    """Verdict string from the compute peak, DRAM % of peak and DMA busy %; the larger of compute and DRAM
    wins when both reach BOUND. None inputs count as 0; all None is unknown."""
    if compute_pct is None and dram_pct is None and dma_pct is None:
        return VERDICT_UNKNOWN
    comp, dram, dma = compute_pct or 0.0, dram_pct or 0.0, dma_pct or 0.0
    if comp >= BOUND and comp >= dram:
        return f"{VERDICT_COMPUTE} ({compute_unit})"
    if dram >= BOUND:
        return VERDICT_MEMORY
    if dma >= BOUND and dram < LOW:
        return VERDICT_DMA
    return VERDICT_LATENCY


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _mean(values):
    return sum(values) / len(values) if values else None


def _add(acc, key, weight, value):
    cell = acc.setdefault(key, [0.0, 0.0])
    cell[0] += weight
    cell[1] += weight * value


def _finish(acc):
    return {k: v[1] / v[0] for k, v in acc.items() if v[0] > 0}


def _get_window_stats(run, start, end):
    """(unit -> %, DMA busy %, GB/s) averaged over the stat windows overlapping [start, end); a missing
    kind is None or empty."""
    cfg, units, dma, dram = run["config"], {}, {}, {}
    interval = cfg.get("core_stats_print_period_cycles", 0)
    for i, win in enumerate(run["wins"] if interval else []):
        w = _overlap(i * interval, min((i + 1) * interval, run["cycles"]), start, end)
        if not w:
            continue
        for name, per_core in win.items():
            _add(units, name, w, _mean(list(per_core.values())))
        busy = [a / (a + b) * 100.0 for a, b in run["dma_wins"][i].values() if a + b] \
            if i < len(run["dma_wins"]) else []
        if busy:
            _add(dma, "dma", w, _mean(busy))
    wl = run["dram_window_cycles"]
    for j, win in enumerate(run["dram_win"]):
        w = _overlap(j * wl, min((j + 1) * wl, run["cycles"]), start, end)
        if w:
            _add(dram, "dram", w, win["gbps"])
    return _finish(units), _finish(dma).get("dma"), _finish(dram).get("dram")


def _get_total_stats(run):
    """(unit -> %, DMA busy %, GB/s) of the whole-run table printed at the end of the log."""
    tot = run["total"]
    units = {n: _mean(list(c.values())) for n, c in tot["units"].items()}
    busy = [a / (a + b) * 100.0 for a, b in tot["dma"].values() if a + b]
    return units, _mean(busy), tot["dram_gbps"]


def get_sol(run, start, end, labels):
    """Speed of light of cycles [start, end) of a parsed run: per-unit util under display names, the compute
    peak and its unit, DRAM GB/s and % of peak, DMA busy %, source ('windows' or 'run-total') and verdict."""
    units, dma, gbps = _get_window_stats(run, start, end)
    source = "windows"
    if not units and run["total"]["units"]:
        units, source = _get_total_stats(run)[0], "run-total"
    if dma is None:
        dma = _get_total_stats(run)[1]
    if gbps is None:
        gbps = run["total"]["dram_gbps"]
    util = {labels[n]: v for n, v in units.items()}
    peak = max(util.items(), key=lambda kv: kv[1]) if util else (None, None)
    peak_bw = run["config"].get("peak_gbps")
    dram_pct = gbps / peak_bw * 100.0 if gbps is not None and peak_bw else None
    return {"span": max(end - start, 0), "source": source if util else None, "util": util,
            "compute_unit": peak[0], "compute_pct": peak[1], "dram_gbps": gbps, "dram_pct": dram_pct,
            "dma_pct": dma, "verdict": get_verdict(peak[1], peak[0], dram_pct, dma)}


def merge_sols(weighted):
    """One sol from [(weight, sol)] (weights are cycle spans): each number is the weighted mean over those
    that have it, and the verdict is decided again from the means."""
    acc, util = {}, {}
    for w, s in weighted:
        if not w:
            continue
        for key in ("dram_gbps", "dram_pct", "dma_pct"):
            if s[key] is not None:
                _add(acc, key, w, s[key])
        for unit, v in s["util"].items():
            _add(util, unit, w, v)
    mean, util = _finish(acc), _finish(util)
    peak = max(util.items(), key=lambda kv: kv[1]) if util else (None, None)
    return {"span": sum(w for w, _ in weighted), "source": None, "util": util, "compute_unit": peak[0],
            "compute_pct": peak[1], "dram_gbps": mean.get("dram_gbps"), "dram_pct": mean.get("dram_pct"),
            "dma_pct": mean.get("dma_pct"),
            "verdict": get_verdict(peak[1], peak[0], mean.get("dram_pct"), mean.get("dma_pct"))}


def get_trace_args(sol):
    """The slice args of a sol: sol_compute_pct, sol_compute_unit, sol_dram_pct, sol_dram_gbps, dma_busy_pct,
    verdict, sol_source, sol_span_cycles and util_<unit>; numbers rounded to two places, None left out."""
    out = {"sol_span_cycles": sol["span"], "sol_source": sol["source"], "verdict": sol["verdict"],
           "sol_compute_unit": sol["compute_unit"]}
    for key, name in (("compute_pct", "sol_compute_pct"), ("dram_pct", "sol_dram_pct"),
                      ("dram_gbps", "sol_dram_gbps"), ("dma_pct", "dma_busy_pct")):
        out[name] = None if sol[key] is None else round(sol[key], 2)
    out.update({f"util_{u}": round(v, 2) for u, v in sol["util"].items()})
    return {k: v for k, v in out.items() if v is not None}
