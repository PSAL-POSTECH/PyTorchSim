from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from ..acs import cycles as acs_cycles
from ..acs import node_mapping
from . import emit, graph, shapes
from .config import read_hardware
from .onnx_ops import resolve

HERE = os.path.dirname(os.path.abspath(__file__))
TOGSIM = os.path.dirname(HERE)
EXAMPLE_DIR = os.path.join(TOGSIM, "example")
INCLUDE_DIR = os.path.join(TOGSIM, "include")
SIMULATOR = os.path.join(TOGSIM, "build", "bin", "Simulator")

CYCLE_RE = re.compile(r"Total execution cycles:\s*(\d+)")


def simulate(so: str, table: str, config: str) -> int:
    cmd = [SIMULATOR, "--config", config, "--trace_so", so, "--cycle_table", table]
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = CYCLE_RE.search(res.stdout + res.stderr)
    if not m:
        tail = (res.stderr or res.stdout)[-400:]
        raise RuntimeError(f"TOGSim reported no cycle count: {tail}")
    return int(m.group(1))


def run_node(node, kernel, shp, hw, config, out_dir):
    values = shapes.constants(node, kernel, shp, hw)
    tag = "_".join(str(values[n]) for n in kernel.size_names if n in values)
    stem = f"{kernel.stem}_{tag}" if tag else kernel.stem

    cpp = os.path.join(out_dir, stem + ".cpp")
    emit.emit(os.path.join(EXAMPLE_DIR, kernel.directory, kernel.stem + ".cpp"),
              values, cpp)
    so = emit.compile_so(cpp, INCLUDE_DIR, os.path.join(out_dir, stem + ".so"))

    desc = node_mapping.lookup(kernel.stem)
    tile = shapes.tile_for_cost(kernel, values)
    table = os.path.join(out_dir, stem + ".tsv")
    acs_cycles.write_table(acs_cycles.build_table(desc, tile, hw), table)

    return simulate(so, table, config)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run an ONNX graph on TOGSim from the kernel library.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "result"))
    ap.add_argument("--sa", type=int, default=None,
                    help="systolic array dimension; inferred from the config name")
    ap.add_argument("--no-optimize", action="store_true",
                    help="skip the ORT fusion pass")
    args = ap.parse_args(argv)

    hw = read_hardware(args.config, args.sa)
    os.makedirs(args.out, exist_ok=True)
    model, shp = graph.load(args.model, optimize_graph=not args.no_optimize)

    print(f"config   : {os.path.basename(args.config)}")
    print(f"hardware : SA={hw.sa} lanes={hw.vlane} vlen={hw.vlen_bits}b "
          f"({hw.throughput} elem/cycle)")
    print()

    total = 0
    ran = no_compute = 0
    unmapped: dict[str, int] = {}
    rows = []

    for node in model.graph.node:
        kernels = resolve(node.op_type)
        if kernels is None:
            unmapped[node.op_type] = unmapped.get(node.op_type, 0) + 1
            continue
        if not kernels:
            no_compute += 1
            continue
        node_cycles = 0
        try:
            for kernel in kernels:
                node_cycles += run_node(node, kernel, shp, hw, args.config,
                                        args.out)
        except (KeyError, ValueError, RuntimeError,
                subprocess.CalledProcessError) as exc:
            reason = f"{node.op_type} ({type(exc).__name__})"
            unmapped[reason] = unmapped.get(reason, 0) + 1
            print(f"  !! {node.op_type:<18} {exc}", file=sys.stderr)
            continue

        total += node_cycles
        ran += 1
        rows.append({"op": node.op_type, "name": node.name or f"#{ran}",
                     "cycles": node_cycles})
        print(f"  {node.op_type:<20} {node_cycles:>12,} cycles")

    print()
    print(f"nodes run      : {ran}")
    print(f"nodes no-op    : {no_compute}")
    print(f"total cycles   : {total:,}")
    if unmapped:
        print("\nUNMAPPED (not counted in the total):")
        for op, n in sorted(unmapped.items(), key=lambda kv: -kv[1]):
            print(f"  {op}: {n}")

    report = {"model": args.model, "config": args.config,
              "hardware": {"sa": hw.sa, "vlane": hw.vlane,
                           "vlen_bits": hw.vlen_bits},
              "total_cycles": total, "nodes_run": ran,
              "nodes_no_compute": no_compute, "unmapped": unmapped,
              "nodes": rows}
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport         : {os.path.join(args.out, 'report.json')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
