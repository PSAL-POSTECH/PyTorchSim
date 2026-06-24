"""cycle_table (C3): the precomputed tile_id -> (cycle, overlapping_cycle) table
the C++ trace pipeline looks up at runtime (docs/design/togsim_cpp_trace.md sec
6, sec 9.8 task 4).

A `togsim.compute(tile_id=...)` in the trace says *which* tile to compute, not
how long it takes. Because tiles are fixed size, each tile's cost is invariant
(only the trip count varies with shape), so it is sampled once and stored here,
keyed by `tile_id`. Two numbers per tile, mirroring the legacy TOG:

  * `cycle`            -- full compute latency, sampled by gem5 sample-mode
                          (the existing measurement: `_rewrite_loop_steps` +
                          `_insert_compute_markers` in build_tog, run through
                          CycleSimulator -> the per-tile `cycle_list`).
  * `overlapping_cycle` -- the portion that overlaps the previous instruction in
                          the systolic pipeline; the timing core uses it as
                          `finish = prev.finish + cycle - overlapped` (Core.cc).
                          Derived exactly as the legacy path does
                          (tog_generator.generate_tile_graph):
                              type 0 (VectorCompute)  -> 0
                              type 1 (MatmulCompute)  -> max(cycle - x_offset, 0)
                              type 2 (MatmulPreload)  -> max(cycle - w_offset, 0)

This module only *builds/serializes* the table from a cycle_list; obtaining the
cycle_list reuses the existing sample-mode + gem5 path (wired in P3 task 5). The
`tile_id` order matches build_skeleton's `compute_nodes` order, which matches the
legacy TOG, so the same sampling keys both paths.

Requires the MLIR Python bindings (to read the skeleton's togsim.compute ops).
"""

import json

from . import togsim_ops as ts
from ._mlir_util import walk_ops
from .build_tog import (
    ir,
    VECTOR_COMPUTE,
    MATMUL_COMPUTE,   # noqa: F401 (documents the type enum used by the formula)
    MATMUL_PRELOAD,
)


def overlapping_cycle(cycle, compute_type, x_offset, w_offset):
    """Hideable (pipeline-overlapped) portion of `cycle`. Mirrors
    tog_generator.generate_tile_graph."""
    if compute_type <= VECTOR_COMPUTE:           # VectorCompute: no systolic overlap
        return 0
    offset = w_offset if compute_type == MATMUL_PRELOAD else x_offset
    return max(int(cycle) - int(offset), 0)


def _compute_types(skeleton_module):
    """tile_id-ordered list of compute_type ints, from the skeleton's
    togsim.compute ops."""
    items = []
    for op in walk_ops(skeleton_module.body):
        if op.operation.name != ts.COMPUTE:
            continue
        tid = ir.IntegerAttr(op.operation.attributes[ts.ATTR_TILE_ID]).value
        ct = ir.IntegerAttr(op.operation.attributes[ts.ATTR_COMPUTE_TYPE]).value
        items.append((tid, ct))
    items.sort()
    return [t for _, t in items]


def build_cycle_table(skeleton_module, cycle_list, x_offset, w_offset):
    """Return `[(cycle, overlapping_cycle), ...]` indexed by tile_id.

    `cycle_list` is the per-tile gem5 measurement (compute_nodes order ==
    tile_id order). `x_offset`/`w_offset` are the systolic-fill offsets the
    legacy path computes from the vector-lane size / loop size."""
    types = _compute_types(skeleton_module)
    if len(cycle_list) != len(types):
        raise ValueError(
            "cycle_list (%d) does not match #compute tiles (%d)"
            % (len(cycle_list), len(types)))
    return [(int(c), overlapping_cycle(c, t, x_offset, w_offset))
            for c, t in zip(cycle_list, types)]


def dump_cycle_table(table, path, x_offset=None, w_offset=None):
    """Serialize the table as a sidecar JSON next to the trace `.so`. The P3 C6
    loader reads it and sets compute_cycle + overlapping_cycle on each emitted
    Instruction."""
    with open(path, "w") as fh:
        json.dump({"x_offset": x_offset, "w_offset": w_offset,
                   "table": [list(e) for e in table]}, fh)
    return path


def load_cycle_table(path):
    with open(path) as fh:
        return json.load(fh)


def dump_cycle_table_tsv(table, path):
    """Plain `cycle<TAB>overlapping` per line, in tile_id order -- the trivial
    format the C++ `--cycle_table` loader (main.cc, P3 trace pipeline) reads with
    ifstream (no JSON dependency in TOGSim)."""
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
    return path
