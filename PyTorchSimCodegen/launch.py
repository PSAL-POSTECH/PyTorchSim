"""What shape the launch is: which axes, how many programs, how big a block.

the compiler compiles one binary per kernel and the C wrapper walks the grid as a plain
loop, so there is no autotuner to choose any of this later. Every consumer that
pairs an axis with an extent reads it from here, or two of them disagree.
"""

import os

from torch._inductor.virtualized import V

from .errors import SpecIncomplete

_PARALLEL_PREFIXES = ("z", "y", "x")

_REDUCTION_LIVE_TILES = 12

_DTYPE_BITS = {"float64": 64, "float32": 32, "float16": 16, "bfloat16": 16,
               "float8_e5m2": 8, "float8_e4m3fn": 8,
               "int64": 64, "int32": 32, "int16": 16, "int8": 8,
               "uint64": 64, "uint32": 32, "uint16": 16, "uint8": 8,
               "bool": 8}


def _block_name(prefix):
    return f"{prefix.upper()}BLOCK"


def template_grid(kernel):
    """A TEMPLATE kernel's launch grid as [gridX, gridY, gridZ], else None.

    mm and conv tile with their own BLOCK_M/N, so the grid is theirs to state
    and their numels describe the output. Refuses rather than falling back.
    """
    grid_fn = getattr(kernel, "grid_fn", None)
    sizes = getattr(kernel, "call_sizes", None)
    if grid_fn is None or sizes is None:
        return None
    name = getattr(kernel, "kernel_name", kernel)
    try:
        vals = [int(V.graph.sizevars.optimization_hint(s)) for s in sizes]
        grid = grid_fn(*vals, dict(getattr(kernel, "meta", None) or {}))
        extents = [int(g) for g in grid]
    except Exception as e:
        raise SpecIncomplete(
            f"{name}: template grid_fn raised {type(e).__name__}: {e}") from e
    if len(extents) != 3 or any(g < 1 for g in extents):
        raise SpecIncomplete(
            f"{name}: template grid_fn returned {grid!r}; this route needs three "
            f"static positive extents. Refusing rather than falling back to the "
            f"numels, which describe the output tensor and not the grid.")
    return extents


def combo_grid(kernel):
    """A ComboKernel's grid as [gridX, 1, 1], else None.

    A COMBO KERNEL IS N KERNELS IN ONE, AND IT STATES ITS OWN GRID. Inductor
    fuses independent small kernels horizontally -- `aten.cat` of twelve expert
    outputs, `torch._foreach_*` over a parameter list -- into one @foreach
    kernel that splits `program_id` into one range per piece:

        num_xblocks_0 = tl.cdiv(5120, XBLOCK)
        num_xblocks_1 = num_xblocks_0 + tl.cdiv(512, XBLOCK)   # twelve of these

    So the launch is the SUM of the pieces' block counts, and `kernel.numels`
    does not describe it -- there is no single iteration space. That is the same
    situation `template_grid` answers, which is why this answers in the same
    shape and rides the same `template_grid` path from there on.

    `combo_grid_meta` is Inductor's own answer to the same question (it is what
    the emitted `inductor_meta` carries), so this asks it rather than
    reconstructing the ranges.
    """
    meta_fn = getattr(kernel, "combo_grid_meta", None)
    if meta_fn is None:
        return None
    grid_meta = meta_fn([kernel.select_heuristics(k)[1] for k in kernel.sub_kernels])
    xblock = (grid_meta.get("default_config") or {}).get("XBLOCK")
    if not xblock:
        return None
    blocks = 0
    for i in range(int(grid_meta.get("num_kernels", 0))):
        n = grid_meta.get(f"xnumel_{i}")
        if n is None:
            return None
        blocks += -(-int(n) // int(xblock))
    return [blocks, 1, 1] if blocks else None


def parallel_axes(numels):
    """Grid axes this kernel uses, outermost first."""
    return [p for p in _PARALLEL_PREFIXES if f"{p}numel" in numels]


def launch_axes(meta):
    """The axes the LAUNCH spreads programs over, outermost first.

    A template declares ALL THREE even at extent 1: its source reads
    tl.program_id(0..2) regardless, and the trace producer stops at an extra arg.
    """
    if meta.get("template_grid") is None:
        return parallel_axes(meta["numels"])
    return list(_PARALLEL_PREFIXES)


def launch_extents(meta):
    """The extents of `launch_axes`, in the same order.

    Kept beside the axes rather than recomputed per consumer: the pairing is
    the part that has been got wrong, not either half on its own.
    """
    grid = meta.get("template_grid")
    if grid is None:
        return grid_of(meta)
    by_axis = dict(zip(("x", "y", "z"), grid))
    return tuple(by_axis[p] for p in launch_axes(meta))


def reduction_axes(numels):
    """Reduction prefixes, read off the keys rather than assumed.

    THE PREFIX CARRIES A TRAILING UNDERSCORE -- Inductor names the parameter
    `r0_numel`, so the prefix is "r0_". Looking up "r0numel" silently misses.
    """
    return sorted(k[:-len("numel")] for k in numels
                  if k.endswith("numel") and k.startswith("r"))


def grid_of(meta):
    """Launch grid from the numels and the pinned blocks, outermost first.

    Also read by the timing path with the LAUNCH's numels, so the derivation is
    stated once and only the extents differ. A template states its own grid.
    """
    if meta.get("template_grid") is not None:
        return launch_extents(meta)

    numels = meta["numels"]
    cfg = meta.get("fixed_config") or {}
    axes = parallel_axes(numels)
    if not axes:
        raise SpecIncomplete(
            f"{meta['kernel_name']} has no parallel iteration axis to grid over")

    grid = []
    for prefix in axes:
        n, block = numels.get(f"{prefix}numel"), cfg.get(_block_name(prefix))
        if n is None or not block:
            raise SpecIncomplete(
                f"cannot compute the grid for {meta['kernel_name']} axis "
                f"'{prefix}': {prefix}numel={n!r}, {_block_name(prefix)}={block!r}. "
                f"Inductor defers the grid to triton_heuristics at runtime; this "
                f"route needs it statically (see fixed_config_for).")
        grid.append(-(-int(n) // int(block)))
    return tuple(grid)


def grid_xyz(meta):
    """The same grid in the compiler's order: (gridX, gridY, gridZ).

    the compiler's spec.grid is X-FIRST while grid_of is outermost-first, and the wrong
    order writes only the first XBLOCK columns silently. Built by axis NAME.
    """
    extents = dict(zip(launch_axes(meta), launch_extents(meta)))
    return tuple(extents.get(p, 1) for p in ("x", "y", "z"))


def _element_bits(args):
    """Widest tensor element in the kernel, in bits.

    Widest, not narrowest: the block has to be one register's worth for EVERY
    operand. A kernel with no typed tensor argument falls back to 32.
    """
    bits = [_DTYPE_BITS[a["dtype"]] for a in (args or [])
            if a.get("dtype") in _DTYPE_BITS]
    return max(bits) if bits else 32


def reduction_block_for(extent, elem_bytes=4, lane_bytes=None):
    """The R0_BLOCK this backend pins for a reduction of `extent`.

    Cover the extent and no more, then shrink to the budget. Two callers must
    agree: this pins the block, inductor_patches asks if persistence matches.
    """
    if lane_bytes is None:
        lane_bytes = int(os.environ.get("TORCHSIM_COMPILE_SPAD_SIZE", str(64 * 1024)), 0)
    budget = lane_bytes // 2 // _REDUCTION_LIVE_TILES
    block = 1 << (int(extent) - 1).bit_length()
    while block * elem_bytes > budget and block > 1:
        block //= 2
    return block


def _covers(want, n):
    """`want`, clamped to the smallest legal block that covers `n`.

    A block wider than its numel strands the mask's rank-1 index rows ONE_LANE
    against banked data. A numel of 1 is NOT clamped: it leaves the lanes none.
    """
    if not n or n <= 1:
        return want
    cover = 1
    while cover < n:
        cover *= 2
    return min(want, cover)


def fixed_config_for(kernel, numels, args):
    """Block sizes pinned at codegen time, from the machine and the numels.

    `numels` is the <prefix>numel-keyed dict, NOT kernel.numels. THESE ARE SIZES,
    NOT A LAYOUT -- the lane axis is the compiler's select_lane_axis, passes later.
    """
    from . import compiler_bridge
    machine = compiler_bridge.machine()
    lanes = machine["vector_lanes"]
    per_vector = max(1, machine["vlen_bits"] // _element_bits(args))

    axes = parallel_axes(numels)
    cfg = {}
    for i, p in enumerate(axes):
        if i == 0:
            cfg[_block_name(p)] = _covers(lanes, numels.get(f"{p}numel"))
        elif i == len(axes) - 1:
            cfg[_block_name(p)] = _covers(per_vector, numels.get(f"{p}numel"))
        else:
            cfg[_block_name(p)] = 1
    cfg.setdefault("XBLOCK", _covers(lanes, numels.get("xnumel")))

    if getattr(kernel, "inside_reduction", False):
        lane_bytes = machine["spad_size"]
        elem_bytes = max(1, _element_bits(args) // 8)
        for p in reduction_axes(numels) or ["r0_"]:
            n = numels.get(f"{p}numel")
            if not n:
                cfg[_block_name(p)] = None
                continue
            cfg[_block_name(p)] = reduction_block_for(n, elem_bytes, lane_bytes)
    return cfg
