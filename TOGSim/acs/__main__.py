from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from TOGSim.acs import cycles as acs_cycles
from TOGSim.acs import node_mapping
from TOGSim.onnx_frontend.config import read_hardware

TILE_NAMES = {
    "gemm": ("TM", "TN"),
    "conv": ("TILE_M", "TILE_N"),
    "softmax": ("TM", "TN"),
    "layernorm": ("TM", "TN"),
    "embed_layernorm": ("TSEQ", "TDIM"),
    "global_avgpool": ("TC", "THW"),
    "bias_act": ("TM", "TN"),
    "bias_gelu": ("TM", "TN"),
    "maxpool": ("TROW", "TCOL"),
    "adaptive_avgpool": ("TROW", "TCOL"),
    "attention": ("Q_LEN", "DHEAD"),
    "kvcache_concat": ("TTOK", "HIDDEN"),
    "concat": ("TROW", "TD"),
    "flatten": (),
}


def read_constant(source: str, name: str) -> int | None:
    m = re.search(rf"static\s+const\s+\w+(?:_t)?\s+{re.escape(name)}\s*=\s*"
                  r"([^;]+);", source)
    if not m:
        return None
    text = m.group(1).strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def tile_from_source(stem: str, source: str) -> dict[str, int]:
    names = TILE_NAMES.get(stem)
    if names is None:
        raise KeyError(
            f"no tile constants recorded for kernel {stem!r}; add it to "
            f"TILE_NAMES in acs/__main__.py")
    if not names:
        return {"rows": 1, "elems": 1}

    values = []
    for n in names:
        v = read_constant(source, n)
        if v is None:
            raise KeyError(
                f"{stem}: constant {n!r} is not declared, or is not a plain "
                f"integer. Pass --rows and --elems instead.")
        values.append(v)
    rows = values[0]
    elems = rows * values[1] if len(values) > 1 else rows
    return {"rows": rows, "elems": elems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Derive a TOGSim cycle table for one kernel, without gem5.")
    ap.add_argument("kernel", help="path to the kernel .cpp")
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--out", default=None,
                    help="output .tsv (default: cycles.tsv beside the kernel)")
    ap.add_argument("--sa", type=int, default=None,
                    help="systolic array dimension; inferred from the config name")
    ap.add_argument("--rows", type=int, default=None,
                    help="override the tile row count")
    ap.add_argument("--elems", type=int, default=None,
                    help="override the tile element count")
    args = ap.parse_args(argv)

    hw = read_hardware(args.config, args.sa)
    stem = os.path.splitext(os.path.basename(args.kernel))[0]
    source = open(args.kernel).read()

    desc = node_mapping.lookup(stem)
    if args.rows is not None or args.elems is not None:
        tile = {"rows": args.rows or 1, "elems": args.elems or 1}
    else:
        tile = tile_from_source(stem, source)

    rows = acs_cycles.build_table(desc, tile, hw)
    out = args.out or os.path.join(os.path.dirname(args.kernel), "cycles.tsv")
    acs_cycles.write_table(rows, out)

    print(f"kernel   : {stem} ({len(rows)} compute nodes)")
    print(f"hardware : SA={hw.sa} lanes={hw.vlane} vlen={hw.vlen_bits}b "
          f"({hw.throughput} elem/cycle)")
    print(f"tile     : rows={tile['rows']} elems={tile['elems']}")
    print(f"table    : {out}")
    for (c, o), node in zip(rows, desc.nodes):
        print(f"  {node.name:<14} {node.compute_type:<8} {c:>6}\t{o:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
