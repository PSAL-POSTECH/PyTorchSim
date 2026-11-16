#!/usr/bin/env python3
"""Write <out>/trace.pftrace (Perfetto binary), <out>/tiles/<kernel>.pftrace and a thin <out>/index.html of
links from TOGSim logs. Inputs and flags: --help. Instruction slices need --log_level trace logs."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorchsim.simulator import perfetto_trace  # noqa: E402
import profile_index  # noqa: E402
import profile_trace  # noqa: E402


TRACE_FILE = "trace.pftrace"


def write_profile(runs, out, origins=None, dram_channels=False, tile_files=None):
    """trace.pftrace and tiles/<kernel>.pftrace under `out` for parsed `runs`; returns (trace event dict,
    kernels per run, live runs, tile paths relative to `out`)."""
    live, _ = profile_trace.split_empty(runs)
    docs = profile_trace.collect_tile_docs(live, tile_files)
    tiles = {name: os.path.join("tiles", name + ".pftrace").replace(os.sep, "/") for name in docs}
    trace, kernel_runs = profile_trace.build_trace(runs, origins, dram_channels, out, tiles)
    profile_trace.check_trace(trace)
    os.makedirs(os.path.join(out, "tiles") if docs else out, exist_ok=True)
    perfetto_trace.write_trace(os.path.join(out, TRACE_FILE), trace)
    for name, doc in docs.items():
        tile = profile_trace.build_tile_trace(doc, name)
        profile_trace.check_trace(tile)
        perfetto_trace.write_trace(os.path.join(out, tiles[name]), tile)
    return trace, kernel_runs, live, tiles


def dump(paths, out, timing_roots=(), torch_trace=(), inductor_debug=(), path_map=(), tile_files=None,
         dram_channels=False, tlparse=None):
    """Write trace.pftrace, tiles/ and index.html under `out` for the logs in `paths`; None when no log,
    else (trace dict, live runs, tile files)."""
    runs = profile_trace.collect_runs(paths, path_map)
    if not runs:
        return None
    origins = profile_trace.read_inductor_origins(inductor_debug)
    origins.update(profile_trace.map_torch_origins(profile_trace.read_torch_trace(torch_trace)))
    trace, kernel_runs, live, tiles = write_profile(runs, out, origins, dram_channels, tile_files)
    empty = [r for r in runs if r not in live]
    rows = profile_index.summarize_kernels(live, kernel_runs, tiles, profile_index.find_timing(timing_roots))
    links = [(f"TORCH_TRACE: {d}", profile_index.get_link(d, out)) for d in torch_trace]
    if tlparse:
        page = os.path.join(tlparse, "index.html")
        target = page if os.path.exists(page) else tlparse
        links.append((f"tlparse: {tlparse}", profile_index.get_link(target, out)))
    links = [(label, href) for label, href in links if href]
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(profile_index.render_index(out, live, empty, rows, links, TRACE_FILE))
    return trace, live, tiles


def main(argv=None):
    """Parse the given logs and write the dump."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=["togsim_results"], help="log files, globs or directories")
    ap.add_argument("-o", "--out", default="profile", help="output directory")
    ap.add_argument("--timing", nargs="*", default=None, metavar="ROOT",
                    help="roots to scan for the compiler's timing.json")
    ap.add_argument("--torch-trace", nargs="*", default=None, metavar="DIR",
                    help="TORCH_TRACE log dirs or files; adds each kernel's model lines and ATen ops")
    ap.add_argument("--inductor-debug", nargs="*", default=None, metavar="DIR",
                    help="TORCH_COMPILE_DEBUG dirs; the origin when TORCH_TRACE has none")
    ap.add_argument("--map-path", action="append", default=[], metavar="CONTAINER=HOST",
                    help="read a logged path that does not exist here (a container's) from this prefix")
    ap.add_argument("--tile-timeline", action="append", default=[], metavar="KERNEL=PATH",
                    help="use PATH as KERNEL's tog_tile_timeline.pftrace instead of the one in its dump dir")
    ap.add_argument("--tlparse", default=None, metavar="DIR", help="tlparse output dir to link")
    ap.add_argument("--dram-channels", action="store_true", help="also one DRAM counter per channel")
    args = ap.parse_args(argv)
    path_map = [tuple(a.split("=", 1)) for a in args.map_path]
    tile_files = dict(a.split("=", 1) for a in args.tile_timeline)
    result = dump(args.paths, args.out, args.timing or [], args.torch_trace or [], args.inductor_debug or [],
                  path_map, tile_files, args.dram_channels, args.tlparse)
    if result is None:
        print("no TOGSim logs found", file=sys.stderr)
        return 1
    trace, live, tiles = result
    print(f"{len(live)} run(s), {len(trace['traceEvents'])} events, {len(tiles)} tile file(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
