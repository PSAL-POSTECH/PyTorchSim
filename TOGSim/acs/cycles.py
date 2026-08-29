from __future__ import annotations

import math

from ..onnx_frontend.tiling import Hardware
from .node_mapping import ELEMENTWISE, REDUCTION, NodeDesc

#: vslidedown at 2,4,6,8...; each vfmax consumes the last -- one chain, so linear.
FOLD_WIDTH = 8


def preload_cycles(hw: Hardware) -> tuple[int, int]:
    return 2 * hw.sa - 1, 0


def matmul_cycles(rows: int, hw: Hardware) -> tuple[int, int]:
    cycles = 2 * hw.sa - 2 + rows
    return cycles, min(hw.sa, rows)


def vector_cycles(node: NodeDesc, tile_elems: int, hw: Hardware) -> tuple[int, int]:
    total = 0
    for op in node.ops:
        if op.kind == ELEMENTWISE:
            total += math.ceil(tile_elems / hw.throughput) * op.n
        elif op.kind == REDUCTION:
            total += (FOLD_WIDTH - 1) * (1 + op.n)
        else:
            raise ValueError(f"unknown op kind {op.kind!r} in node {node.name!r}")
    # nothing in front of a VPU node to hide behind: fully exposed
    return total, total


def default_cost(node: NodeDesc, tile: dict, hw: Hardware) -> tuple[int, int]:
    if node.compute_type == "preload":
        return preload_cycles(hw)
    if node.compute_type == "matmul":
        return matmul_cycles(int(tile.get("rows", hw.sa)), hw)
    if node.compute_type == "vector":
        return vector_cycles(node, int(tile.get("elems", 1)), hw)
    raise ValueError(f"unknown compute type {node.compute_type!r}")

_cost_function = default_cost


def set_cost_function(fn) -> None:
    global _cost_function
    _cost_function = fn


def build_table(kernel_desc, tile: dict, hw: Hardware) -> list[tuple[int, int]]:
    rows = []
    for node in kernel_desc.nodes:
        cycles, interval = _cost_function(node, tile, hw)
        # column 2 is overlapping, not the interval: cycles - overlapping is
        # what the core sustains. Writing the interval there inverts it.
        rows.append((cycles, max(0, cycles - interval)))
    return rows


def write_table(rows: list[tuple[int, int]], path: str, origin: str = "acs") -> None:
    with open(path, "w") as fh:
        for cycles, overlapping in rows:
            fh.write(f"{cycles}\t{overlapping}\n")
        fh.write(f"# origins: {origin}\n")
