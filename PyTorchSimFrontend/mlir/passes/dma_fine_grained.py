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
dma_start operands+attrs -- validated against mlir-opt -dma-fine-grained and the
end-to-end gemm/conv/model tests, not byte-exact SSA text).

Operates on the customized memref.dma_start convention (see lower_dma_to_gemmini):
operands = src, *src_idx, dst, *dst_idx, num_elements(dma_type), tag, *tag_idx,
stride(=vlane_split_axis), num_elements_per_stride(=vlane_stride). MVIN dma_type in
{2,1,14}; tile shape = dst shape for MVIN.

Pipeline entry point: run_fine_grained(in_path, out_path, vectorlane).
"""
import os
import sys

_DEFAULT_BINDINGS = "/riscv-llvm/python_packages/mlir_core"
if os.path.isdir(_DEFAULT_BINDINGS) and _DEFAULT_BINDINGS not in sys.path:
    sys.path.insert(0, _DEFAULT_BINDINGS)

import mlir.ir as ir  # noqa: E402

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


def _int_array_attr(op, key):
    if key not in op.attributes:
        return []
    return [ir.IntegerAttr(a).value for a in ir.ArrayAttr(op.attributes[key])]


def _is_block_arg(v):
    return isinstance(v, ir.BlockArgument)


class _Dma:
    """Positional view of a customized memref.dma_start op."""

    def __init__(self, op):
        self.op = op
        operands = list(op.operands)
        src_rank = len(ir.MemRefType(operands[0].type).shape)
        i = 0
        self.src = operands[i]; i += 1
        self.src_idx = operands[i:i + src_rank]; i += src_rank
        self.dst = operands[i]; i += 1
        dst_rank = len(ir.MemRefType(self.dst.type).shape)
        self.dst_idx = operands[i:i + dst_rank]; i += dst_rank
        self.num_elements = operands[i]; i += 1
        self.tag = operands[i]; i += 1
        tag_rank = len(ir.MemRefType(self.tag.type).shape)
        self.tag_idx = operands[i:i + tag_rank]; i += tag_rank
        self.stride = operands[i]; i += 1          # = vlane_split_axis
        self.num_elements_per_stride = operands[i]  # = vlane_stride
        self.src_rank, self.dst_rank, self.tag_rank = src_rank, dst_rank, tag_rank

    @property
    def dma_type(self):
        return _const_int(self.num_elements)

    @property
    def is_mvin(self):
        return self.dma_type in (MVIN, MVIN2, MVIN3)

    @property
    def vlane_split_axis(self):
        return _const_int(self.stride)

    @property
    def vlane_stride(self):
        return _const_int(self.num_elements_per_stride) & 0x7FFF

    def tile_shape(self):
        mt = ir.MemRefType((self.dst if self.is_mvin else self.src).type)
        return list(mt.shape)

    def subtile_size(self):
        return _int_array_attr(self.op, "subtile_size")

    def sram_stride(self):
        return _int_array_attr(self.op, "sram_stride")

    def dram_stride(self):
        return _int_array_attr(self.op, "dram_stride")

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
    """Mirror getDmaAttrs: keep subtile/sram/dram strides, set async + fine_grained."""
    attrs = {}
    op = dma.op
    for k in ("subtile_size", "sram_stride", "dram_stride"):
        if k in op.attributes:
            attrs[k] = op.attributes[k]
    attrs["async"] = ir.BoolAttr.get(dma.is_async())
    attrs["fine_grained"] = ir.BoolAttr.get(True)
    return attrs


def _emit_dma(dma, ivs, vectorlane, ip):
    """Emit one fine-grained memref.dma_start at `ip`, indexed by `ivs` (the fused
    induction vars for this DMA's dims, in dim order)."""
    idx_ty = ir.IndexType.get()
    zero = _const_index(0, ip)

    dram_off = _apply(_build_dram_map(dma), ivs, ip)
    src_idx0 = dma.src_idx[0]
    dram_idx = _apply(_sum_map(), [dram_off, src_idx0], ip)

    sram_off = _apply(_build_sram_map(dma, vectorlane), ivs, ip)
    tag_idx = _apply(_build_tag_map(dma, list(range(len(dma.tile_shape())))), ivs, ip)

    # SRAM indices: zeros except the last = sram offset (mirror sramIndices.back()).
    sram_indices = [zero] * dma.dst_rank
    sram_indices[-1] = sram_off

    operands = [dma.src, dram_idx, dma.dst, *sram_indices,
                dma.num_elements, dma.tag, tag_idx,
                dma.stride, dma.num_elements_per_stride]
    ir.Operation.create("memref.dma_start", results=[], operands=operands,
                        attributes=_dma_attrs(dma), ip=ip)


def _const_index(v, ip):
    from mlir.dialects import arith
    return arith.ConstantOp(ir.IndexType.get(),
                            ir.IntegerAttr.get(ir.IndexType.get(), v), ip=ip).result


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
def _iter_ops(block):
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _iter_ops(b)


def _run_func(func, vectorlane):
    from mlir.dialects import linalg
    # First matmul only.
    matmul = None
    dmas = []
    for op in _iter_ops(func.regions[0].blocks[0]):
        name = op.operation.name
        if name == "linalg.matmul" and matmul is None:
            matmul = op
        elif name == "memref.dma_start":
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

    # Insert the fused nest at the weight DMA (the later of the two): both DMAs'
    # original DRAM base indices (src_idx[0], computed in the enclosing loops) must
    # dominate the nest. Codegen emits input before weight, matching the C++ pass
    # which fuses after the weight subtile loop.
    ip = ir.InsertionPoint(mvin_weight.op)
    fused_ivs, body_ip = _build_for_nest(bounds, ip)
    in_ivs = [fused_ivs[fuse["in_to_fused"][d]] for d in range(rank)]
    w_ivs = [fused_ivs[fuse["w_to_fused"][d]] for d in range(rank)]
    _emit_dma(mvin_input, in_ivs, vectorlane, body_ip)
    _emit_dma(mvin_weight, w_ivs, vectorlane, body_ip)
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
