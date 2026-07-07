"""Python port of the C++ `-dma-fine-grained` MLIR pass (TestDmaFineGrained.cpp).

Splits the matmul MVIN DMAs (input / weight / optional bias) into subtile loops
(affine.for nests carrying per-subtile DRAM/SRAM offset affine.apply maps) and
fuses the input and weight loop nests, mirroring the C++ pass structurally. Runs
AFTER -test-loop-padding (it reads the padded tile shapes / loop bounds) and
BEFORE -test-pytorchsim-to-vcix, so extension_codecache splits the single mlir-opt
invocation around this pass.

The C++ pass fuses the two subtile loop nests by cloning their bodies with an
IRMapping; the MLIR Python bindings expose no IRMapping, so this port builds the
fused nest directly and emits each DMA inside it using the fused induction vars
(equivalence target: same loop structure / counts, same offset maps, same
togsim.transfer operands+attrs -- validated against mlir-opt -dma-fine-grained and
the end-to-end gemm/conv/model tests, not byte-exact SSA text).

Operates on the togsim.transfer convention (see mlir_codegen_backend.emit_transfer
and lower_transfer_to_gemmini): operands = dram, dram_idx, sram, sram_idx, tag,
tag_idx, dma_type, vst(=vlane_stride)[, offset]; attrs = dma_kind, vlane_split_axis
(i64), dram_stride[], tile_stride[], padding, [subtile_size, async]. Direction is
derived from dma_kind / dma_type: MVIN => src=dram, dst=sram; MVOUT => src=sram,
dst=dram. tile shape = the sram memref shape for BOTH directions. MVIN dma_type in
{2,1,14}.

Pipeline entry point: run_fine_grained(in_path, out_path, vectorlane).
"""
import itertools
import os
import sys

_DEFAULT_BINDINGS = "/riscv-llvm/python_packages/mlir_core"
if os.path.isdir(_DEFAULT_BINDINGS) and _DEFAULT_BINDINGS not in sys.path:
    sys.path.insert(0, _DEFAULT_BINDINGS)

import mlir.ir as ir  # noqa: E402

from ._mlir_util import walk_ops, attr_i64_array

MARKERS = ("subtile_size",)   # only subtile DMAs are split

MVIN, MVIN2, MVIN3, MVOUT = 2, 1, 14, 3

# Per-rank subtile loop order and the fused-loop layout (mirror the C++ loopGroups).
# in_to_fused[d] / w_to_fused[d] give the fused-loop index that the input/weight
# DMA's dim d iterates; n_fused is the number of fused affine.for loops.
_FUSE = {
    2: dict(n_fused=3, in_to_fused=[0, 1],        w_to_fused=[1, 2]),
    3: dict(n_fused=4, in_to_fused=[0, 1, 2],     w_to_fused=[0, 2, 3]),
    4: dict(n_fused=7, in_to_fused=[0, 1, 4, 5],  w_to_fused=[2, 3, 5, 6]),
}


# ---------------------------------------------------------------------------
# Small readers (mirror CustomDMAAttribute.h)
# ---------------------------------------------------------------------------
def _const_int(value, default=-1):
    try:
        return ir.IntegerAttr(value.owner.attributes["value"]).value
    except Exception:
        return default


def _is_block_arg(v):
    return isinstance(v, ir.BlockArgument)


class _Dma:
    """Positional view of a togsim.transfer op.

    operands: dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vst[, offset]
    Direction from dma_kind / dma_type: MVIN => src=dram, dst=sram (MVOUT swaps).
    self.src_idx is a single-element list holding the base DRAM idx for MVIN (the
    SRAM idx for MVOUT); tile_shape is always the sram memref shape.
    """

    def __init__(self, op):
        self.op = op
        operands = list(op.operands)
        # dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vst[, offset]
        self.dram = operands[0]
        self.dram_idx = operands[1]
        self.sram = operands[2]
        self.sram_idx = operands[3]
        self.tag = operands[4]
        self.tag_idx = operands[5]
        self.num_elements = operands[6]           # = dma_type const operand
        self.num_elements_per_stride = operands[7]  # = vlane_stride (vst)
        # trailing operands: [offset (indirect)] then (low, high) per masked axis.
        _extra = operands[8:]
        self.offset = None
        if "indirect" in op.attributes:
            self.offset = _extra[0]
            _extra = _extra[1:]
        self.masked_axes = attr_i64_array(op, "masked_axes", default=[])
        self.masked_ops = _extra   # low0, high0, low1, high1, ...

        self.sram_rank = len(ir.MemRefType(self.sram.type).shape)
        # Direction: MVIN reads dram -> sram; MVOUT writes sram -> dram.
        if self.is_mvin:
            self.src, self.dst = self.dram, self.sram
            self.src_idx = [self.dram_idx]
        else:
            self.src, self.dst = self.sram, self.dram
            self.src_idx = [self.sram_idx]
        self.src_rank = len(ir.MemRefType(self.src.type).shape)
        self.dst_rank = len(ir.MemRefType(self.dst.type).shape)

    @property
    def dma_type(self):
        return _const_int(self.num_elements)

    @property
    def is_mvin(self):
        return self.dma_type in (MVIN, MVIN2, MVIN3)

    @property
    def vlane_split_axis(self):
        return ir.IntegerAttr(self.op.attributes["vlane_split_axis"]).value

    @property
    def vlane_stride(self):
        return _const_int(self.num_elements_per_stride) & 0x7FFF

    def tile_shape(self):
        return list(ir.MemRefType(self.sram.type).shape)

    def subtile_size(self):
        return attr_i64_array(self.op, "subtile_size", default=[])

    def sram_stride(self):
        # togsim.transfer names the spad stride "tile_stride".
        return attr_i64_array(self.op, "tile_stride", default=[])

    def dram_stride(self):
        return attr_i64_array(self.op, "dram_stride", default=[])

    def is_async(self):
        a = self.op.attributes
        if "async" not in a:
            return False
        try:
            return bool(ir.IntegerAttr(a["async"]).value)
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Affine map builders (mirror buildDramAffineMap / buildSramAffineMap)
# ---------------------------------------------------------------------------
def _ceil_div(a, b):
    return (a + b - 1) // b


def _build_dram_map(dma):
    dram = dma.dram_stride()
    sub = dma.subtile_size()
    rank = len(dram)
    expr = ir.AffineConstantExpr.get(0)
    for i in range(rank):
        expr = expr + ir.AffineDimExpr.get(i) * (dram[i] * sub[i])
    return ir.AffineMap.get(rank, 0, [expr])


def _build_sram_map(dma, vectorlane):
    tile_shape = dma.tile_shape()
    tile_stride = dma.sram_stride()
    sub = dma.subtile_size()
    split = dma.vlane_split_axis
    vstride = dma.vlane_stride

    target_stride = tile_stride[split]
    old_size = tile_shape[split]
    nr_outerloop = _ceil_div(old_size, vectorlane * vstride)
    new_size = nr_outerloop * vstride

    expr = None
    for i in range(len(tile_stride)):
        subtilesize = sub[i]
        stride = tile_stride[i]
        if stride > target_stride:
            stride = stride // old_size * new_size
        d = ir.AffineDimExpr.get(i)
        if i != split:
            term = d * (subtilesize * stride)
        else:
            term = ir.AffineExpr.get_floor_div(d * subtilesize, vectorlane) * stride
        expr = term if expr is None else expr + term
    return ir.AffineMap.get(len(tile_stride), 0, [expr])


def _build_tag_map(dma, loop_order):
    """Mirror the tag stride map built inside buildSubtileLoop."""
    tile_sizes = dma.tile_shape()
    sub = dma.subtile_size()
    rank = len(tile_sizes)
    strides = [1] * rank
    for i in range(rank - 2, -1, -1):
        cur, nxt = loop_order[i], loop_order[i + 1]
        strides[cur] = strides[nxt] * _ceil_div(tile_sizes[nxt], sub[nxt])
    expr = ir.AffineConstantExpr.get(0)
    for i in range(rank):
        expr = expr + ir.AffineDimExpr.get(i) * strides[i]
    return ir.AffineMap.get(rank, 0, [expr])


def _loop_counts(dma, loop_order):
    tile_sizes = dma.tile_shape()
    sub = dma.subtile_size()
    return [_ceil_div(tile_sizes[d], sub[d]) for d in range(len(tile_sizes))]


# ---------------------------------------------------------------------------
# DMA emission inside a body
# ---------------------------------------------------------------------------
def _sum_map():
    d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
    return ir.AffineMap.get(2, 0, [d0 + d1])


def _apply(map_, operands, ip):
    from mlir.dialects import affine
    return affine.AffineApplyOp(map_, list(operands), ip=ip).result


def _dma_attrs(dma):
    """Build the emitted togsim.transfer's attrs: copy dma_kind, vlane_split_axis,
    dram_stride, tile_stride (the spad stride), subtile_size and padding straight
    from the source op; set async (BoolAttr) + fine_grained (BoolAttr true)."""
    attrs = {}
    op = dma.op
    for k in ("dma_kind", "vlane_split_axis", "dram_stride", "tile_stride",
              "subtile_size", "padding", "masked_axes", "masked_fill", "indirect",
              "offset_stride", "accumulate", "acc_float"):
        if k in op.attributes:
            attrs[k] = op.attributes[k]
    attrs["async"] = ir.BoolAttr.get(dma.is_async())
    attrs["fine_grained"] = ir.BoolAttr.get(True)
    return attrs


def _remap_bound(bound, iv, sub, is_high, ip):
    """The masked low/high are full-tile-local; shift to this subtile: subtile position
    p maps to full-tile p + iv*sub, so subtile-local high = min(sub, high - iv*sub) and
    low = max(0, low - iv*sub). Out-of-window (neg high / low>sub) -> Spike skips all."""
    from mlir.dialects import affine
    d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
    edge = ir.AffineConstantExpr.get(sub if is_high else 0)
    m = ir.AffineMap.get(2, 0, [edge, d0 - d1 * sub])
    op = affine.AffineMinOp if is_high else affine.AffineMaxOp
    return op(m, [bound, iv], ip=ip).result


def _emit_dma(dma, ivs, vectorlane, ip):
    """Emit one fine-grained togsim.transfer at `ip`, indexed by `ivs` (the fused
    induction vars for this DMA's dims, in dim order)."""
    dram_off = _apply(_build_dram_map(dma), ivs, ip)
    # DRAM base index = the original transfer's dram_idx operand.
    dram_idx = _apply(_sum_map(), [dram_off, dma.dram_idx], ip)

    # SRAM offset is a SINGLE linear sram_idx operand (row-major stride 1).
    sram_off = _apply(_build_sram_map(dma, vectorlane), ivs, ip)
    # Per-subtile tag index (required for async DMA<->barrier pairing downstream).
    tag_idx = _apply(_build_tag_map(dma, list(range(len(dma.tile_shape())))), ivs, ip)

    operands = [dma.dram, dram_idx, dma.sram, sram_off,
                dma.tag, tag_idx, dma.num_elements, dma.num_elements_per_stride]
    if dma.offset is not None:
        operands.append(dma.offset)
    # masked low/high are full-tile-local -> remap each per THIS subtile's offset.
    sub = dma.subtile_size()
    for i, axis in enumerate(dma.masked_axes):
        low, high = dma.masked_ops[2 * i], dma.masked_ops[2 * i + 1]
        operands.append(_remap_bound(low, ivs[axis], sub[axis], False, ip))
        operands.append(_remap_bound(high, ivs[axis], sub[axis], True, ip))
    ir.Operation.create("togsim.transfer", results=[], operands=operands,
                        attributes=_dma_attrs(dma), ip=ip)


def _const_index(v, ip):
    from mlir.dialects import arith
    return arith.ConstantOp(ir.IndexType.get(),
                            ir.IntegerAttr.get(ir.IndexType.get(), v), ip=ip).result


def _fresh_tag(dma):
    """Give this DMA a fresh tag memref.alloc right BEFORE the (pre-split) coarse
    dma_start, and rewire every use of the old tag -- the dma_start re-emitted
    below AND its dma_wait -- to it. The coarse dma sits at the reduction-loop body
    level (it has not been wrapped in a subtile load nest yet), so the alloc there
    dominates both the load nest fine-grained is about to build and the sibling
    wait nest. Each reduction iteration thus allocates its own tag -> successive
    iterations are distinct (multi-tile-K / conv) and the per-iteration tag
    semantics is in the IR, not reconstructed downstream. Old alloc becomes dead."""
    old = dma.tag
    new_tag = ir.Operation.create("memref.alloc", results=[old.type],
                                  operands=[], ip=ir.InsertionPoint(dma.op)).results[0]
    old.replace_all_uses_with(new_tag)
    dma.tag = new_tag
    # the old (func-entry, per-tensor unique) alloc is now dead -- erase it.
    try:
        old.owner.erase()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Loop-nest construction
# ---------------------------------------------------------------------------
def _build_for_nest(bounds, ip):
    """Create a nested affine.for over `bounds` (step 1, marked inner_loop). Returns
    (induction_vars, innermost_body_ip_before_yield)."""
    from mlir.dialects import affine
    ivs = []
    cur_ip = ip
    for b in bounds:
        floop = affine.AffineForOp(0, b, 1, ip=cur_ip)
        floop.operation.attributes["inner_loop"] = ir.BoolAttr.get(True)
        ivs.append(floop.induction_variable)
        with ir.InsertionPoint(floop.body):
            affine.AffineYieldOp([])
        cur_ip = ir.InsertionPoint.at_block_terminator(floop.body)
    return ivs, cur_ip


def _create_subtile_dma(dma, loop_order, ip, vectorlane):
    """Standalone subtile loop for one DMA (used for bias). Mirrors createSubtileDMA."""
    counts = _loop_counts(dma, loop_order)
    bounds = [counts[d] for d in loop_order]
    ivs_in_order, body_ip = _build_for_nest(bounds, ip)
    # map dim -> its induction var (loop_order[k] is the dim of the k-th loop)
    iv_by_dim = [None] * len(counts)
    for k, d in enumerate(loop_order):
        iv_by_dim[d] = ivs_in_order[k]
    _emit_dma(dma, iv_by_dim, vectorlane, body_ip)


# ---------------------------------------------------------------------------
# Operand reachability (mirror traverseOperands)
# ---------------------------------------------------------------------------
def _reaches(value, target):
    if value == target:
        return True
    owner = value.owner
    if isinstance(owner, ir.Block):   # block argument: no defining op to walk
        return False
    for operand in owner.operands:
        if _reaches(operand, target):
            return True
    return False


# ---------------------------------------------------------------------------
# Pass driver
# ---------------------------------------------------------------------------
def _run_func(func, vectorlane):
    from mlir.dialects import linalg
    # First matmul only.
    matmul = None
    dmas = []
    for op in walk_ops(func.regions[0].blocks[0]):
        name = op.operation.name
        if name == "linalg.matmul" and matmul is None:
            matmul = op
        elif name == "togsim.transfer":
            dmas.append(op)
    if matmul is None:
        return

    m_in = matmul.operands[0]
    m_w = matmul.operands[1]
    m_res = list(matmul.operands)[-1]   # output (init) operand

    mvin_input = mvin_weight = mvin_bias = None
    for op in dmas:
        d = _Dma(op)
        if d.dma_type == MVOUT:
            continue
        if _reaches(m_in, d.dst):
            mvin_input = d
        elif _reaches(m_w, d.dst):
            mvin_weight = d
        elif _reaches(m_res, d.dst) and len(d.subtile_size()) > 1:
            mvin_bias = d

    in_async = mvin_input is not None and mvin_input.is_async()
    w_async = mvin_weight is not None and mvin_weight.is_async()
    if not (in_async or w_async):
        return
    if mvin_input is None or mvin_weight is None:
        return

    rank = len(mvin_input.tile_shape())
    if rank not in _FUSE:
        return
    fuse = _FUSE[rank]
    loop_order = list(range(rank))

    # Bias first (standalone), inserted before its own op.
    if mvin_bias is not None:
        brank = len(mvin_bias.tile_shape())
        border = {2: [0, 1], 4: [2, 3, 0, 1]}.get(brank)
        if border is not None:
            _create_subtile_dma(mvin_bias, border,
                                 ir.InsertionPoint(mvin_bias.op), vectorlane)
            mvin_bias.op.erase()

    # Fused input + weight nest. Fused loop bounds: take each fused loop's count
    # from whichever DMA dim maps onto it.
    in_counts = _loop_counts(mvin_input, loop_order)
    w_counts = _loop_counts(mvin_weight, loop_order)
    bounds = [None] * fuse["n_fused"]
    for d, f in enumerate(fuse["in_to_fused"]):
        bounds[f] = in_counts[d]
    for d, f in enumerate(fuse["w_to_fused"]):
        bounds[f] = w_counts[d]

    # Give each load a fresh per-iteration tag alloc just before its coarse dma
    # (rewiring its dma_wait via the old tag's uses), so the tag is distinct per
    # reduction iteration -- positioned to match the per-iteration tag semantics.
    _fresh_tag(mvin_input)
    _fresh_tag(mvin_weight)

    # Insert the fused nest at the weight DMA (the later of the two): both DMAs'
    # original DRAM base indices (src_idx[0], computed in the enclosing loops) must
    # dominate the nest. Codegen emits input before weight, matching the C++ pass
    # which fuses after the weight subtile loop.
    ip = ir.InsertionPoint(mvin_weight.op)
    # Unroll the fused nest, emitting each distinct input/weight subtile ONCE (a load
    # is invariant to the other operand's dims, so the cross-product re-emits it
    # identically). Dedup by the operand's own coords; keep the fused issue order.
    seen_in, seen_w = set(), set()
    for it in itertools.product(*[range(b) for b in bounds]):
        in_key = tuple(it[fuse["in_to_fused"][d]] for d in range(rank))
        if in_key not in seen_in:
            seen_in.add(in_key)
            _emit_dma(mvin_input, [_const_index(c, ip) for c in in_key], vectorlane, ip)
        w_key = tuple(it[fuse["w_to_fused"][d]] for d in range(rank))
        if w_key not in seen_w:
            seen_w.add(w_key)
            _emit_dma(mvin_weight, [_const_index(c, ip) for c in w_key], vectorlane, ip)
    mvin_input.op.erase()
    mvin_weight.op.erase()


def run(module, vectorlane=128, **_):
    """Apply fine-grained DMA subtiling to every func in `module`, in place."""
    from mlir.dialects import func as func_d  # noqa: F401 (ensure dialect loaded)
    for region in module.operation.regions:
        for b in region.blocks:
            for op in list(b.operations):
                if op.operation.name == "func.func" and len(op.operation.regions[0].blocks):
                    _run_func(op, vectorlane)


def run_fine_grained(in_path, out_path, vectorlane=128):
    """Parse `in_path`, run the pass, write `out_path`. Pipeline entry point."""
    with open(in_path) as f:
        text = f.read()
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(text)
        run(module, vectorlane=vectorlane)
        out = str(module)
    with open(out_path, "w") as f:
        f.write(out)


if __name__ == "__main__":
    vl = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    run_fine_grained(sys.argv[1], sys.argv[2], vl)
