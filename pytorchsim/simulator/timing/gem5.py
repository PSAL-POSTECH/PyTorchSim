"""gem5 over the compiler's cycle binary, one sampled tile per dump, and the tables TOGSim reads
beside the TOG from those dumps: tog_tile_cycles.tsv (cycle, overlapping) and tog_unit_table.tsv; with
TORCHSIM_GEM5_TRACE=1 also tog_tile_timeline.json, the coprocessor trace split by cycle-table row."""

import json
import os
import re
import subprocess

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge
from pytorchsim.simulator import wallclock

logger = config.setup_logger()

TRACE_FILE = "coprocessor_trace.json"
TRACE_SCHEMA = "coprocessor-trace/1"
TIMELINE_SCHEMA = "tog-tile-timeline/1"


def run_cycle_binary(workdir, meta):
    """The m5out of the compiler's cycle ELF run under gem5, or None when gem5 could not run it."""

    kernel_name = meta["kernel_name"]
    elf = contract.artifact(workdir, "cycle_elf")
    if elf is None:
        logger.warning("[Gem5] %s has no cycle_elf; it was compiled without --tog", workdir)
        return None
    try:
        with wallclock.span(wallclock.GEM5_RUN, kernel_name):
            return _simulate(
                elf, workdir, contract.kernel_object(workdir), silent_mode=True)
    except Exception as e:
        logger.warning("[Gem5] sampling failed: %s", e)
        return None


def _simulate(target_binary, obj_dir, manifest, silent_mode=False):
    """gem5 over one tile of the kernel object in `obj_dir`, under what it was compiled for."""
    dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
    machine = compiler_bridge.compiled_for(manifest)
    run = compiler_bridge.run_options(obj_dir, manifest)
    gem5_cmd = [compiler_bridge.toolchain().GEM5, "-r", "--stdout-file=sto.log", "-d", dir_path,
                run["gem5_script"], "-c", target_binary,
                "--model", run["model"],
                "--machine-config", run["machine_config"],
                "--vlane", str(machine.vector_lanes), "--vlen", str(machine.vlen_bits)]
    if config.CONFIG_GEM5_TRACE:
        gem5_cmd.append(f"--coprocessor-trace-file={TRACE_FILE}")

    if not silent_mode:
        logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
        logger.info("[Gem5] Gem5 simulation started")

    proc = subprocess.run(gem5_cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(_gem5_failure(proc, os.path.join(dir_path, "sto.log")))

    return dir_path


#: What gem5 says when it dies, as opposed to the libc backtrace that follows it.
_GEM5_SIGNAL = re.compile(
    r"^(?:.*\b(?:panic|fatal):.*|.*Assertion.*|Program aborted at tick.*)$", re.M)


def _gem5_failure(proc, log_path):
    """The message a failed gem5 run raises, from the log it redirects to: -r sends stdout and
    stderr to --stdout-file, so the pipes are empty and the return code alone says nothing."""
    how = (f"signal {-proc.returncode}" if proc.returncode < 0
           else f"exit {proc.returncode}")
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        text = (proc.stderr or "") + (proc.stdout or "")
    hits = [h.strip() for h in _GEM5_SIGNAL.findall(text)]
    if not hits:
        hits = [l.strip() for l in text.strip().splitlines()[-3:]]
    if not hits:
        hits = [f"gem5 left no diagnostic; see {log_path}"]
    return (f"Gem5 Simulation Failed ({how}); full log in {log_path}\n  "
            + "\n  ".join(h[:300] for h in hits[:3]))


def tile_tables(m5out, overlap_offsets):
    """The cycle table and the unit rows of a gem5 run: a row per dump the compiler's gem5_samples
    reads, in the order it returns them; whatever that reader refuses is raised, not sampled around."""
    dumps = compiler_bridge.compiler_contract("gem5_samples").read(m5out, len(overlap_offsets))
    return cycle_table(overlap_offsets, dumps), unit_rows(dumps)


def write_tile_timeline(m5out, table, path):
    """tog_tile_timeline.json from the coprocessor trace gem5 left in `m5out`, one row per cycle-table
    row with its events rebased to the tile's reset; None when tracing was off. Fails on a cycle mismatch."""
    trace = os.path.join(m5out, TRACE_FILE)
    if not os.path.isfile(trace):
        return None
    with open(trace) as fh:
        rows = split_trace_by_tile(json.load(fh), [c for c, _ in table])
    with open(path, "w") as fh:
        json.dump({"schema": TIMELINE_SCHEMA, "rows": rows}, fh)
    return path


def split_trace_by_tile(trace, cycles):
    """[{row, cycles, events}] from a coprocessor-trace/1 document: each dump closes the window opened
    by the latest reset; events are clipped to it and rebased; a window not of `cycles[row]` raises."""
    schema = (trace.get("otherData") or {}).get("schema")
    if schema != TRACE_SCHEMA or (trace["otherData"].get("ts_unit") != "cycle"):
        raise ValueError(f"coprocessor trace schema {schema!r}, expected {TRACE_SCHEMA} in cycles")
    events = trace["traceEvents"]
    windows, start = [], None
    for e in sorted((e for e in events if e.get("cat") == "stats"), key=lambda e: e["ts"]):
        if e["name"] == "reset":
            start = e["ts"]
        elif e["name"] == "dump" and start is not None:
            windows.append((start, e["ts"]))
            start = None
    if len(windows) != len(cycles):
        raise ValueError(f"coprocessor trace has {len(windows)} tile windows, "
                         f"the cycle table {len(cycles)} rows")
    rows = []
    for row, ((lo, hi), want) in enumerate(zip(windows, cycles)):
        if hi - lo != int(want):
            raise ValueError(f"coprocessor trace window {row} is {hi - lo} cycles, "
                             f"the cycle table says {want}")
        rows.append({"row": row, "cycles": hi - lo, "events": _clip_events(events, lo, hi)})
    return rows


def _clip_events(events, lo, hi):
    """The insn slices overlapping [lo, hi) clipped to it, and each unit's counter events inside it
    (its value at lo carried in at 0), all with ts - lo."""
    out, held = [], {}
    for e in events:
        if e["ph"] == "X" and e.get("cat") == "insn":
            a, b = max(e["ts"], lo), min(e["ts"] + e["dur"], hi)
            if b > a:
                out.append({**e, "ts": a - lo, "dur": b - a})
        elif e["ph"] == "C" and e.get("cat") == "unit":
            if e["ts"] <= lo:
                held[e["name"]] = e
            elif e["ts"] < hi:
                out.append({**e, "ts": e["ts"] - lo})
    out += [{**e, "ts": 0} for e in held.values()]
    return sorted(out, key=lambda e: (e["ts"], e["ph"] != "C"))


def cycle_table(overlap_offsets, dumps):
    """[(cycle, overlapping_cycle), ...], one per gem5 dump in row order. The compiler states each
    tile's overlap offset (None where nothing of it overlaps), and each dump names its tile."""
    return [(int(d["cycles"]), 0 if (off := overlap_offsets[d["tile"]]) is None
             else max(int(d["cycles"]) - int(off), 0))
            for d in dumps]


def unit_rows(dumps):
    """TOGSim's --unit_table rows, one per table row and unit, from gem5's dumps in row order:
    (row, unit, unit_of_work, capacity per cycle, admitted)."""
    rows = []
    for row, dump in enumerate(dumps):
        for name, u in dump["units"].items():
            if "admitted" not in u or "capacity" not in u:
                raise ValueError(f"gem5 dump of tile {dump['tile']} call {dump['call']}: unit {name} "
                                 f"lacks admitted or capacity")
            rows.append((row, name, u["unit_of_work"], int(u["capacity"]), int(u["admitted"])))
    return rows


def dump_cycle_table_tsv(table, path, origins=None):
    """cycle<TAB>overlapping per line, in row order, for TOGSim's loader.

    origins is appended as a trailing comment line, which the loader stops at.
    """
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
        if origins:
            fh.write("# origins: %s\n" % ", ".join(sorted(str(o) for o in origins)))
    return path


