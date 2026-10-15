"""How a kernel is cut into blocks: the sizes pinned at codegen time, a template's grid and its tiles.

The compiler builds one binary per kernel and nothing tunes it later, so block sizes and the mm
tiles this machine can take are decided here, from the machine and the kernel.
"""

import re

from torch._inductor.virtualized import V

from pytorchsim import config
from pytorchsim.contract import SpecIncomplete, block_name, parallel_axes, reduction_axes

_REDUCTION_LIVE_TILES = 12


_DTYPE_BITS = {"float64": 64, "float32": 32, "float16": 16, "bfloat16": 16,
               "float8_e5m2": 8, "float8_e4m3fn": 8,
               "int64": 64, "int32": 32, "int16": 16, "int8": 8,
               "uint64": 64, "uint32": 32, "uint16": 16, "uint8": 8,
               "bool": 8}


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


def _element_bits(args):
    """Widest tensor element in the kernel, in bits.

    Widest, not narrowest: the block has to be one register's worth for EVERY
    operand. A kernel with no typed tensor argument falls back to 32.
    """
    bits = [_DTYPE_BITS[a["dtype"]] for a in (args or [])
            if a.get("dtype") in _DTYPE_BITS]
    return max(bits) if bits else 32


def triton_max_numel():
    """Elements Triton accepts in one block tensor, read from the Triton this route compiles with."""
    try:
        from triton._utils import TRITON_MAX_TENSOR_NUMEL
    except ImportError:
        from triton.language.core import TRITON_MAX_TENSOR_NUMEL
    return int(TRITON_MAX_TENSOR_NUMEL)


def reduction_block_for(extent, elem_bytes=4, lane_bytes=None):
    """The R0_BLOCK this backend pins for a reduction of `extent`.

    Cover the extent and no more, then shrink to the budget: half of one lane's
    scratchpad, shared by the tiles a reduction keeps live. Two callers must
    agree: this pins the block, inductor_patches asks if persistence matches.
    Both see the machine's scratchpad when `lane_bytes` is left to this
    function; a persistent reduction rendered for a bigger one than the target
    has is a kernel the compiler refuses as over budget, whatever XBLOCK shrinks to.
    """
    if lane_bytes is None:
        lane_bytes = config.CONFIG_MACHINE.spad_size
    budget = lane_bytes // 2 // _REDUCTION_LIVE_TILES
    block = 1 << (int(extent) - 1).bit_length()
    while block * elem_bytes > budget and block > 1:
        block //= 2
    return block


def _covers(want, n):
    """`want`, clamped to the smallest legal block that covers `n`.

    A block wider than its numel strands the mask's rank-1 index rows ONE_LANE
    against banked data, and a numel of 1 under a lane-wide block is a tile of
    digits whose mask the DMA cannot state; the lanes go to another axis.
    """
    if not n:
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
    machine = config.CONFIG_MACHINE
    lanes = machine.vector_lanes
    per_vector = max(1, machine.vlen_bits // _element_bits(args))

    axes = parallel_axes(numels)
    cfg = {}
    for i, p in enumerate(axes):
        if i == 0:
            cfg[block_name(p)] = _covers(lanes, numels.get(f"{p}numel"))
        elif i == len(axes) - 1:
            cfg[block_name(p)] = _covers(per_vector, numels.get(f"{p}numel"))
        else:
            cfg[block_name(p)] = 1
    cfg.setdefault("XBLOCK", _covers(lanes, numels.get("xnumel")))

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
        import os
        if os.environ.get("TORCHSIM_BIG_BLOCKS") == "1":
            _big_blocks(cfg, numels, lanes, lane_bytes)
    return cfg


def _pow2_divisor(n, cap):
    """The largest power of two that divides `n` and is at most `cap`."""
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


def _big_blocks(cfg, numels, lanes, lane_bytes):
    """GEMV experiment: blocks as large as the scratchpad holds. The reduction block covers its
    numel (the compiler cuts the padding, torchsim-fit-to-numel, and runs a big tile's reduce in
    chunks that overlap their loads, torchsim-pipeline-reductions). XBLOCK -- the lane axis, which
    the compiler cuts to whole rows a lane -- is the power of two covering xnumel (masked past it),
    capped by half a lane's scratchpad (4 bytes a tile element: two bf16 operands, held twice)."""
    r = [p for p in reduction_axes(numels) or ["r0_"] if numels.get(f"{p}numel")]
    xn = numels.get("xnumel")
    if len(r) != 1 or not xn or len(parallel_axes(numels)) > 1:
        return
    rb = cfg[block_name(r[0])] or 1
    per_lane = lane_bytes // 2 // (4 * 2)               # tile elements a lane may hold
    cap = max(lanes, lanes * per_lane // max(1, rb))
    xb = _covers(1 << (cap.bit_length() - 1), int(xn))  # covering the numel, masked past it
    if xb >= lanes:
        cfg["XBLOCK"] = xb


def gemm_tiles(m, n, k, dtype_size, out_size=None):
    """This machine's mm tiles for [m, k] @ [k, n], best first, as GemmConfigs. C is held twice
    when it is stored at another width than it is summed at: the template sums halves in fp32."""
    from torch._inductor.template_heuristics.triton import GemmConfig

    size = int(dtype_size)
    out = int(out_size or size)
    acc = 4 if size == 2 else size
    c_bytes = acc if out == acc else acc + out
    tiles = gemm_tile_candidates(int(m), int(n), int(k), c_bytes, size)
    return [GemmConfig(tm, tn, tk, 1, 4) for tm, tn, tk in tiles]


def gemm_tile_candidates(M, N, K, c_bytes, precision_bytes):
    """Every (tile_M, tile_N, tile_K) whose A, B and C are blocks Triton accepts and fit half the
    scratchpad, largest first. Each side is a lane multiple times a power of two (TTGIR's layouts
    need them), the tail masked."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    spad_per_lane = config.CONFIG_MACHINE.spad_size
    max_spad = spad_per_lane * lanes // 2
    max_spad_per_lane = spad_per_lane // 2
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


TILE_RE = {k: re.compile(rf"^(\s*{k}\s*:\s*tl\.constexpr\s*=\s*)(\S+)$", re.M)
            for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "EVEN_K")}


DIM_RE = {k: re.compile(rf"^\s*{k} = (\d+)$", re.M) for k in "MNK"}


def mm_template_shape(src_code, meta):
    """(M, N, K, input bytes, output bytes) of an mm or bmm template kernel -- one program per output
    tile (and batch), its blocks stated as constexprs -- else None."""
    if (meta.get("template_grid") is None
            or not all(r.search(src_code) for r in TILE_RE.values())):
        return None
    dims = [DIM_RE[k].search(src_code) for k in "MNK"]
    if not all(dims):
        return None
    import torch
    nbytes = {a["role"]: getattr(torch, a["dtype"]).itemsize for a in meta["args"]}
    return tuple(int(d.group(1)) for d in dims) + (nbytes["in"], nbytes.get("out", nbytes["in"]))


def template_tiles(src_code, meta):
    """Every mapped tile of an mm template kernel this source can take, best first, as (M, N, K)
    blocks; [] for any other kernel. A body rendered with EVEN_K loads K unmasked, so it takes only
    a BLOCK_K that divides K."""
    shape = mm_template_shape(src_code, meta)
    if shape is None:
        return []
    m, n, k, size, out_size = shape
    even = TILE_RE["EVEN_K"].search(src_code).group(2) == "True"
    return [(c.block_m, c.block_n, c.block_k) for c in gemm_tiles(m, n, k, size, out_size)
            if not even or k % c.block_k == 0]
