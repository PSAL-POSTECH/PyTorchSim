"""A kernel the compiler refuses as too big is retried at smaller blocks.

A kernel's pinned blocks (its fixed_config) halve when the compiler says they do not fit. A
template's own blocks are its config, which Inductor chose (driver/select.py), and are built as
rendered.
"""

import copy
import itertools

from pytorchsim.driver import compiler_bridge


def too_big(exc):
    """Why this failure means the tile is too big, else None: the scratchpad budget it broke or the
    register file (llc spilled, which the compiler's ELF refuses), as the compiler's failure.json
    names it. Triton's own block limit is never the reason: every block tiling pins is under it."""
    got = getattr(exc, "failure", None) or {}
    kinds = compiler_bridge.compiler_contract("failure")
    if got.get("kind") == kinds.SPAD_OVERFLOW:
        return "%d bytes/lane over a budget of %d" % (got["usage"], got["budget"])
    if got.get("kind") == kinds.SPILL_NOT_ALLOWED:
        return "the vector register file spills"
    return None


def _get_tile_candidates(meta):
    """Every tile the search may try, LARGEST FIRST: each block the kernel takes as
    an argument halves down to 2, never 1. Equal sizes go to the larger reduction
    block; the first is the tile fixed_config_for pinned."""
    cfg = meta.get("fixed_config") or {}
    signature = meta.get("signature") or {}
    movable = [k for k, v in cfg.items() if k in signature and v and v >= 2]
    ranges = []
    for k in movable:
        vs, v = [], cfg[k]
        while v >= 2:
            vs.append(v)
            v //= 2
        ranges.append(vs)

    def _product(tile, keep):
        n = 1
        for k, v in tile.items():
            if keep(k):
                n *= v
        return n
    tiles = [dict(zip(movable, vs)) for vs in itertools.product(*ranges)]
    tiles.sort(key=lambda t: (-_product(t, lambda k: True),
                              -_product(t, lambda k: k.startswith("R")),
                              [-t[k] for k in movable]))
    return tiles


def smaller_tiles(src_code, meta):
    """(tile, source, meta) to retry with after the compiler refuses a kernel as too big: each
    smaller fixed_config tile."""
    tiles = iter(_get_tile_candidates(meta))
    next(tiles, None)
    for tile in tiles:
        meta = copy.deepcopy(meta)
        meta["fixed_config"].update(tile)
        yield {k: v for k, v in tile.items() if k.endswith("BLOCK")}, src_code, meta
