#!/usr/bin/env python3
"""Write <out>/trace.pftrace (Perfetto binary) and <out>/tiles/<kernel>.pftrace from TOGSim logs; open them in
ui.perfetto.dev. Inputs and flags: --help. Instruction slices need --log_level trace logs."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorchsim.simulator import perfetto_trace  # noqa: E402
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
    ap.add_argument("--dram-channels", action="store_true", help="also one DRAM counter per channel")
    args = ap.parse_args(argv)
    path_map = [tuple(a.split("=", 1)) for a in args.map_path]
    runs = profile_trace.collect_runs(args.paths, path_map)
    if not runs:
        print("no TOGSim logs found", file=sys.stderr)
        return 1
    origins = profile_trace.read_inductor_origins(args.inductor_debug or [])
    origins.update(profile_trace.map_torch_origins(profile_trace.read_torch_trace(args.torch_trace or [])))
    tile_files = dict(a.split("=", 1) for a in args.tile_timeline)
    trace, _, live, tiles = write_profile(runs, args.out, origins, args.dram_channels, tile_files)
    print(f"{len(live)} run(s), {len(runs) - len(live)} empty, {len(trace['traceEvents'])} events, "
          f"{len(tiles)} tile file(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
