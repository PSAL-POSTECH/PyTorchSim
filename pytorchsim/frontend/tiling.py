"""How a kernel is cut into blocks: the sizes pinned at codegen time and the mm tiles this machine
takes, decided from the machine and the kernel, since nothing tunes the compiled binary later."""

from pytorchsim import config
from pytorchsim.contract import block_name, parallel_axes, reduction_axes

_REDUCTION_LIVE_TILES = 12


def _element_bits(args):
    """Widest tensor element in the kernel, in bits (32 with no typed tensor argument): the block has to
    be one register's worth for every operand."""
    import torch

    bits = [8 * getattr(torch, a["dtype"]).itemsize for a in (args or []) if a.get("dtype")]
    return max(bits) if bits else 32


def triton_max_numel():
    """Elements Triton accepts in one block tensor, read from the Triton this route compiles with."""
    try:
        from triton._utils import TRITON_MAX_TENSOR_NUMEL
    except ImportError:
        from triton.language.core import TRITON_MAX_TENSOR_NUMEL
    return int(TRITON_MAX_TENSOR_NUMEL)


def reduction_block_for(extent, elem_bytes=4, lane_bytes=None):
    """The R0_BLOCK pinned for a reduction of `extent`: the extent covered and no more, shrunk to one lane's
    scratchpad (the machine's, when `lane_bytes` is left out) over the machine's spad_buffers, shared by the
    live tiles; the pinned block and NPUChoices' persistence answer both come from here."""
    if lane_bytes is None:
        lane_bytes = config.CONFIG_MACHINE.spad_size
    budget = lane_bytes // int(config.CONFIG_MACHINE.spad_buffers) // _REDUCTION_LIVE_TILES
    block = 1 << (int(extent) - 1).bit_length()
    while block * elem_bytes > budget and block > 1:
        block //= 2
    return block


def _covering_block(want, n):
    """`want`, clamped to the smallest legal block that covers `n`: a block wider than its numel
    leaves a mask the DMA cannot state, so the lanes go to another axis."""
    if not n:
        return want
    cover = 1
    while cover < n:
        cover *= 2
    return min(want, cover)


def fixed_config_for(kernel, numels, args, contiguous=None):
    """Block sizes (not a layout) pinned at codegen time from the machine and the <prefix>numel-keyed
    `numels`; `contiguous` ('x' or 'r') is the axis the largest read is unit-stride along, which picks
    kept_axis_contiguous_blocks, lane_axis_reduction_block or kept_axis_rows_block for a reduction."""
    machine = config.CONFIG_MACHINE
    lanes = machine.vector_lanes
    per_vector = max(1, machine.vlen_bits // _element_bits(args))

    axes = parallel_axes(numels)
    cfg = {}
    for i, p in enumerate(axes):
        if i == 0:
            cfg[block_name(p)] = _covering_block(lanes, numels.get(f"{p}numel"))
        elif i == len(axes) - 1:
            cfg[block_name(p)] = _covering_block(per_vector, numels.get(f"{p}numel"))
        else:
            cfg[block_name(p)] = 1
    cfg.setdefault("XBLOCK", _covering_block(lanes, numels.get("xnumel")))
    if not getattr(kernel, "inside_reduction", False) and axes == ["x"] and numels.get("xnumel"):
        cfg["XBLOCK"] = pointwise_block(numels["xnumel"], max(1, _element_bits(args) // 8), lanes,
                                        machine.spad_size, triton_max_numel())

    if getattr(kernel, "inside_reduction", False):
        lane_bytes = machine.spad_size
        elem_bytes = max(1, _element_bits(args) // 8)
        room = triton_max_numel()
        for v in cfg.values():
            room //= max(1, v or 1)
        for p in reduction_axes(numels) or ["r0_"]:
            n = numels.get(f"{p}numel")
            if not n:
                cfg[block_name(p)] = None
                continue
            block = reduction_block_for(n, elem_bytes, lane_bytes)
            while block > max(1, room):
                block //= 2
            cfg[block_name(p)] = block
            room //= max(1, block)
        if contiguous == "x" and set(cfg) == {"XBLOCK", "R0_BLOCK"} \
                and numels.get("xnumel") and numels.get("r0_numel"):
            cfg["XBLOCK"], cfg["R0_BLOCK"] = kept_axis_contiguous_blocks(
                numels["xnumel"], numels["r0_numel"], elem_bytes, lanes, lane_bytes,
                triton_max_numel())
        elif set(cfg) == {"XBLOCK", "R0_BLOCK"} and cfg["XBLOCK"] < lanes \
                and (numels.get("r0_numel") or 0) >= lanes:
            cfg["R0_BLOCK"] = lane_axis_reduction_block(
                cfg["XBLOCK"], numels["r0_numel"], elem_bytes, lanes, lane_bytes,
                triton_max_numel())
        elif set(cfg) == {"XBLOCK", "R0_BLOCK"} and cfg["R0_BLOCK"] and numels.get("xnumel"):
            cfg["XBLOCK"] = kept_axis_rows_block(
                numels["xnumel"], cfg["R0_BLOCK"], elem_bytes, lanes, lane_bytes,
                triton_max_numel())
    return cfg


def kept_axis_rows_block(xnumel, rblock, elem_bytes, lanes, lane_bytes, max_numel):
    """XBLOCK of a reduction along its contiguous axis: rows of `rblock` filling the tile budget
    the other blocks here take (one lane's, times the lanes), at least the lane cover and no more
    than covers `xnumel` or Triton takes in one block."""
    per_lane = reduction_block_for(1 << 62, elem_bytes, lane_bytes)
    rows = min(lanes * per_lane, max_numel) // max(1, rblock)
    floor = _covering_block(lanes, xnumel)
    block = floor
    while block * 2 <= rows and block < _covering_block(1 << 62, xnumel):
        block *= 2
    return block


def pointwise_block(xnumel, elem_bytes, lanes, lane_bytes, max_numel):
    """The XBLOCK of a pointwise kernel over one axis: the extent covered, at least the lane cover, up to
    the tile reduction_block_for's one-lane budget fills on every lane, and no more than Triton takes in
    one block."""
    per_lane = reduction_block_for(1 << 62, elem_bytes, lane_bytes)
    return max(_covering_block(lanes, xnumel), min(_covering_block(1 << 62, xnumel), lanes * per_lane, max_numel))


def lane_axis_reduction_block(xblock, rnumel, elem_bytes, lanes, lane_bytes, max_numel):
    """R0_BLOCK of a reduction whose reduced axis rides the lanes (its rows fill fewer lanes than the
    machine has): the extent covered and no more, within reduction_block_for's one-lane budget times the
    lanes and what Triton takes in one block."""
    per_lane = reduction_block_for(1 << 62, elem_bytes, lane_bytes)
    budget = min(lanes * per_lane, max_numel) // max(1, xblock)
    block = _covering_block(1 << 62, rnumel)
    while block > budget and block > 1:
        block //= 2
    return block


def kept_axis_contiguous_blocks(xnumel, rnumel, elem_bytes, lanes, lane_bytes, max_numel):
    """(XBLOCK, R0_BLOCK) for a reduction whose largest read is contiguous along the kept axis: power-of-two
    blocks within reduction_block_for's one-lane budget times the lanes, XBLOCK at least the lane cover,
    ranked by no padding on x, then the longest contiguous run, then the larger R0_BLOCK."""
    per_lane = reduction_block_for(1 << 62, elem_bytes, lane_bytes)
    budget = min(lanes * per_lane, max_numel)
    floor = _covering_block(lanes, xnumel)
    best = None
    x = floor
    while x <= _covering_block(1 << 62, xnumel):
        r = 1
        while x * r <= budget and r <= _covering_block(1 << 62, rnumel):
            pad = -(-xnumel // x) * x - xnumel
            key = (-pad, min(x, xnumel), r)
            if best is None or key > best[0]:
                best = (key, x, r)
            r *= 2
        x *= 2
    if best is None:
        return floor, reduction_block_for(rnumel, elem_bytes, lane_bytes)
    return best[1], best[2]


def gemm_tiles(m, n, k, dtype_size, out_size=None):
    """This machine's mm tiles for [m, k] @ [k, n], best first, as GemmConfigs. C is held twice
    when it is stored at another width than it is summed at: the template sums halves in fp32."""
    from torch._inductor.heuristics.template.triton import GemmConfig

    size = int(dtype_size)
    out = int(out_size or size)
    acc = 4 if size == 2 else size
    c_bytes = acc if out == acc else acc + out
    tiles = gemm_tile_candidates(int(m), int(n), int(k), c_bytes, size)
    return [GemmConfig(tm, tn, tk, 1, 4) for tm, tn, tk in tiles]


def gemm_tile_candidates(M, N, K, c_bytes, precision_bytes):
    """Every (tile_M, tile_N, tile_K) whose A, B and C are blocks Triton accepts and fit the machine's
    spad_budget, largest first. Each side is a lane multiple times a power of two (TTGIR's layouts
    need them), the tail masked."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    max_spad_per_lane = int(config.CONFIG_MACHINE.spad_budget)
    max_spad = max_spad_per_lane * lanes
    max_numel = triton_max_numel()

    def per_lane(rows, cols):
        return max(rows * ((cols + lanes - 1) // lanes), 2)

    def sides(dim):
        pad = lanes if dim > lanes else 8
        top = _pow2_ceil((dim + pad - 1) // pad * pad)
        if dim <= lanes:
            return [top]
        return [lanes << i for i in range((top // lanes).bit_length())]

    tiles = []
    for tk in sides(max(K, 8)):
        for tm in sides(M):
            for tn in sides(N):
                used = (tm * tk + tk * tn) * precision_bytes + tm * tn * c_bytes
                used_per_lane = ((per_lane(tk, tn) + per_lane(tm, tk)) * precision_bytes
                                 + per_lane(tm, tn) * c_bytes)
                if (max(tm * tk, tk * tn, tm * tn) <= max_numel
                        and used < max_spad and used_per_lane < max_spad_per_lane):
                    tiles.append((used, (tm, tn, tk)))
    tiles.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in tiles]


def _pow2_ceil(v):
    """The smallest power of two at or above v."""
    return 1 << (int(v) - 1).bit_length()
