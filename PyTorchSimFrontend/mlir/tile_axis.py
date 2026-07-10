"""One axis of a tile, and everything that axis means in the spaces it is embedded in.

Today a tile descriptor is stored column-wise: one parallel array per space (the tile
extents, the DRAM strides, the SRAM strides), a scalar index into them
(vlane_split_axis) and a scalar riding alongside (vlane_stride). Keeping the columns
aligned across an axis reorder, insert or collapse is the caller's job, and that is
where the mistakes are: the reduction GEMM repeats the same `if nr_rdim` branch over
four of them, `apply_divisor` inserts an axis but forgets `tile_constraint`,
`decompose_transfer` re-indexes three arrays by hand and remaps the lane index
separately.

`Axis` stores the same table row-wise: one iteration dimension, carrying what it means
in DRAM and in the enclosing loop nest. What is *not* a property of a single axis stays
on the tile: which axis rides the lanes, and the order the axes sit in SRAM.
"""
from dataclasses import dataclass
from typing import Optional

import sympy

from PyTorchSimFrontend.mlir import mlir_common


@dataclass(frozen=True)
class Axis:
    """One iteration dimension of a tile.

    extent       how many elements of this axis the tile covers
    dram_stride  distance in DRAM between two neighbours along this axis. This is the
                 stride of the *access*, not of the tensor: conv walks a padded logical
                 layout, so it is an int or a sympy expression, not a layout stride.
    loop         the enclosing loop variable that advances this axis, one tile at a
                 time; None when the axis does not move in DRAM
    """
    extent: int
    dram_stride: object = 0
    loop: Optional[str] = None


def sram_strides(axes, sram_order):
    """SRAM strides, in the axes' declared order.

    `sram_order` lists the axis names outermost first, so the last one is contiguous.
    An extent-1 axis is indexed only at 0, so its stride never reaches an address --
    Spike bounds that axis' loop by its extent -- but it still gets the stride it would
    have if it were not degenerate.
    """
    stride, init = {}, 1
    for name in reversed(sram_order):
        stride[name] = init
        init *= axes[name].extent
    return [stride[name] for name in axes]


def dram_index(axes):
    """The tile's DRAM offset, one term per axis, in the axes' declared order."""
    return [sympy.Integer(0) if a.loop is None else sympy.Symbol(a.loop) * a.dram_stride
            for a in axes.values()]


def build_tile(buffer, vector_lane, axes, sram_order, lane, lane_chunk=1, offset=0):
    """Build the tile descriptor and the DRAM index expression for one operand.

    `buffer` is the SRAM buffer's name, as the template text spells it. `axes` is an
    ordered mapping name -> Axis; its order is the memref's dimension order, which is
    what linalg sees. `sram_order` is the order those axes sit in SRAM, outermost first.
    The two differ whenever the tile is declared transposed -- the GEMM reduction variant
    declares (N, M) but still lays M out contiguously.

    The SRAM strides, the lane axis, the lane stride and the DRAM index expression are
    all derived from that.
    """
    assert set(sram_order) == set(axes), f"{buffer}: sram_order does not cover the axes"
    assert lane in axes, f"{buffer}: lane axis {lane!r} is not an axis"

    names = list(axes)
    extents = [axes[n].extent for n in names]
    desc = mlir_common.MLIRMultiDimTile(extents, vector_lane, names.index(lane), lane_chunk)
    desc.set_tile_size_stride(extents, sram_strides(axes, sram_order))
    desc.set_name(buffer)
    desc.offset = offset

    return desc, dram_index(axes)
