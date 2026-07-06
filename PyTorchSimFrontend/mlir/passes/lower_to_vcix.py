"""Python port of the C++ `-test-pytorchsim-to-vcix` conversion pass
(TestPyTorchSimToVCIXConversion.cpp).

Lowers `linalg.matmul` and the transcendental math ops (exp/erf/tanh/sin/cos) to
VCIX dialect ops (RISC-V vector custom instructions). The C++ pass is a
dialect-conversion (`applyPartialConversion`); the MLIR Python bindings expose no
conversion framework, so each matchAndRewrite is reimplemented as imperative IR
rewriting (walk + build replacement + replace uses + erase).

The VCIX dialect is NOT registered in the Python bindings, so vcix ops are created
as unregistered generic ops. This round-trips: mlir-opt / mlir-translate (which do
have vcix registered) re-parse the `{}`-attr generic form fine, and the existing
`run_standard_lowering` already consumes the C++ vcix output via
`allow_unregistered_dialects` -- so emitting generic vcix ops here is consistent
with the current pipeline.

Covers all 6 C++ patterns: linalg.matmul (gemm + conv2d) and exp/erf/tanh/sin/cos.
Wired into extension_codecache (run_to_vcix) after fine-grained, before the standard
lowering; mlir-opt then runs only -test-loop-padding. Validated structurally against
`mlir-opt -test-pytorchsim-to-vcix` (non-constant ops byte-identical incl. dma_wait tag
maps) and numerically end-to-end (gemm/bmm/conv2d/transcendental, Spike+gem5 allclose).
"""
import os
import sys

_DEFAULT_BINDINGS = "/riscv-llvm/python_packages/mlir_core"
if os.path.isdir(_DEFAULT_BINDINGS) and _DEFAULT_BINDINGS not in sys.path:
    sys.path.insert(0, _DEFAULT_BINDINGS)

import mlir.ir as ir  # noqa: E402

MARKERS = ("linalg.matmul", "math.exp", "math.erf", "math.tanh", "math.sin", "math.cos")

# math op name -> (opcode, imm) for the vcix.v.iv lowering (mirror Math*ToVCIX).
_MATH_VIV = {
    "math.exp":  (0b000011, 0),
    "math.erf":  (0b000000, 0),
    "math.tanh": (0b000001, 0),
    "math.sin":  (0b000010, 0),
    "math.cos":  (0b000010, 1),
}


def _sew(elt_ty):
    # Mirror C++ legalizeVectorType: only F32/F64/integer/index get a sew. F16/BF16
    # return 0 so transcendental math ops stay unlowered (-convert-math-to-llvm),
    # matching the validated path -- do NOT emit VCIX for them here.
    if ir.F32Type.isinstance(elt_ty):
        return 32
    if ir.F64Type.isinstance(elt_ty):
        return 64
    if ir.IntegerType.isinstance(elt_ty):
        return ir.IntegerType(elt_ty).width
    if ir.IndexType.isinstance(elt_ty):
        return 64
    return 0


def _log2(x):
    return x.bit_length() - 1


def _legalize_vector_type(vt, vlen):
    """Mirror legalizeVectorType: return (n, legal_vector_type) or (0, None)."""
    if len(vt.shape) != 1:           # C++ guards getRank() != 1
        return 0, None
    elt_ty = vt.element_type
    sew = _sew(elt_ty)
    if sew == 0:
        return 0, None
    elt_count = vt.shape[0]
    lmul = elt_count * sew // 64
    scalable = vt.scalable
    if not scalable:
        n = (_log2(lmul) - 2) if lmul > 32 else 1
        if n == 1:
            return 1, vt
        return n, ir.VectorType.get([vlen // (sew // 8)], elt_ty)
    n = (_log2(lmul) - 2) if lmul > 8 else 1
    return n, ir.VectorType.get([elt_count >> (n - 1)], elt_ty, scalable=[True])


def _i64(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), v)


def _i32(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), v)


def _viv(operand, result_ty, opcode, imm, rvl=None):
    """Create an unregistered vcix.v.iv (vcix::BinaryImmOp) op at the current IP."""
    operands = [operand] if rvl is None else [operand, rvl]
    return ir.Operation.create(
        "vcix.v.iv", results=[result_ty], operands=operands,
        attributes={"opcode": _i64(opcode), "imm": _i32(imm)}).results[0]


def _make_sf_vc_v_iv(vec, op_vt, n, legal_ty, opcode, imm):
    """Mirror make_sf_vc_v_iv: chunk `vec` (type op_vt) into legal-width vcix.v.iv."""
    from mlir.dialects import arith, vector
    total = op_vt.shape[0]
    elt_count = legal_ty.shape[0]
    scalable = legal_ty.scalable
    rvl = None
    if scalable:
        rvl = arith.ConstantOp(ir.IntegerType.get_signless(64), _i64(9)).result
    if n == 1:
        return _viv(vec, legal_ty, opcode, imm, rvl)
    elt_ty = legal_ty.element_type
    zero = ir.DenseElementsAttr.get_splat(op_vt, ir.FloatAttr.get(elt_ty, 0.0))
    res = arith.ConstantOp(op_vt, zero).result
    if scalable:
        for i in range(n):
            ext = vector.ScalableExtractOp(legal_ty, vec, i * elt_count).result
            v = _viv(ext, legal_ty, opcode, imm, rvl)
            res = vector.ScalableInsertOp(v, res, i * elt_count).result
    else:
        for i in range(total // elt_count):
            ext = vector.ExtractStridedSliceOp(
                legal_ty, vec,
                ir.ArrayAttr.get([_i64(i * elt_count)]),
                ir.ArrayAttr.get([_i64(elt_count)]),
                ir.ArrayAttr.get([_i64(1)])).result
            v = _viv(ext, legal_ty, opcode, imm, rvl)
            res = vector.InsertStridedSliceOp(
                v, res, ir.ArrayAttr.get([_i64(i * elt_count)]),
                ir.ArrayAttr.get([_i64(1)])).result
    return res


def _iter_ops(block):
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _iter_ops(b)


# ---------------------------------------------------------------------------
# matmul lowering helpers (mirror MatmulOpLowering)
# ---------------------------------------------------------------------------
def _elt_bits(elt_ty):
    if ir.IntegerType.isinstance(elt_ty):
        return ir.IntegerType(elt_ty).width
    return ir.FloatType(elt_ty).width


def _bool_attr_true(op, key):
    a = op.attributes
    return key in a and ir.BoolAttr(a[key]).value


def _enclosing_loops(op):
    """Walk ancestor ops; return (accumulation, outer, inner) affine.for lists,
    outermost-first (mirror the C++ insert-at-begin)."""
    acc, outer, inner = [], [], []
    parent = op.operation.parent
    while parent is not None:
        if parent.name == "affine.for":
            if _bool_attr_true(parent, "accumulation_loop"):
                acc.insert(0, parent)
            if _bool_attr_true(parent, "outer_loop"):
                outer.insert(0, parent)
            if _bool_attr_true(parent, "inner_loop"):
                inner.insert(0, parent)
        parent = parent.parent
    return acc, outer, inner


def _loop_iv(forop):
    return forop.regions[0].blocks[0].arguments[0]


def _loop_ub(forop):
    # single constant upper bound
    m = ir.AffineMapAttr(forop.attributes["upperBoundMap"]).value
    return ir.AffineConstantExpr(m.results[0]).value


def _block_terminator(forop):
    blk = forop.regions[0].blocks[0]
    ops = list(blk.operations)
    return ops[-1]


def _affine_consts(expr):
    """All AffineConstantExpr values reachable in `expr` (recursive)."""
    out = []
    if ir.AffineConstantExpr.isinstance(expr):
        out.append(ir.AffineConstantExpr(expr).value)
    elif ir.AffineBinaryExpr.isinstance(expr):
        be = ir.AffineBinaryExpr(expr)
        out += _affine_consts(be.lhs)
        out += _affine_consts(be.rhs)
    return out


def _scan_conv_offsets(ow_loop, o_h, k_h, o_w, k_w):
    """Mirror the heuristic offset scan: find affine.apply(o_h,k_h)/(o_w,k_w) in the
    o_w loop and read the constant in its map (default 1)."""
    offset_h = offset_w = 1
    for o in _iter_ops(ow_loop.regions[0].blocks[0]):
        if o.operation.name != "affine.apply":
            continue
        ops = list(o.operation.operands)
        if len(ops) < 2:
            continue
        m = ir.AffineMapAttr(o.operation.attributes["map"]).value
        consts = _affine_consts(m.results[0])
        if ops[0] == o_h and ops[1] == k_h and consts:
            offset_h = consts[-1]
        if ops[0] == o_w and ops[1] == k_w and consts:
            offset_w = consts[-1]
    return offset_h, offset_w


def _mem_space(v):
    mt = ir.MemRefType(v.type)
    ms = mt.memory_space
    return ir.IntegerAttr(ms).value if ms is not None else 0


def _dram_is_write(src, dst):
    """(dram_memref, is_write) by memory space, mirror getDramMemRef."""
    ss, ds = _mem_space(src), _mem_space(dst)
    if ds == 0 and ss == 1:
        return dst, True
    if ds == 1 and ss == 0:
        return src, False
    return None, False


def _idx(v):
    return ir.IntegerAttr.get(ir.IndexType.get(), v)


def _const_index(v):
    from mlir.dialects import arith
    return arith.ConstantOp(ir.IndexType.get(), _idx(v)).result


def _apply(map_, operands):
    from mlir.dialects import affine
    return affine.AffineApplyOp(map_, list(operands)).result


def _spad_maps():
    d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
    s0, s1 = (ir.AffineSymbolExpr.get(i) for i in range(2))
    spad = ir.AffineMap.get(3, 2, [d0 * s0 + d1 * s1 + d2])
    x = ir.AffineMap.get(1, 1, [ir.AffineExpr.get_floor_div(ir.AffineDimExpr.get(0),
                                                            ir.AffineSymbolExpr.get(0))])
    y = ir.AffineMap.get(1, 1, [ir.AffineExpr.get_mod(ir.AffineDimExpr.get(0),
                                                      ir.AffineSymbolExpr.get(0))])
    return spad, x, y


def _transfer_read(vec_ty, source, indices, padding):
    from mlir.dialects import vector
    src_rank = len(ir.MemRefType(source.type).shape)
    vec_rank = len(ir.VectorType(vec_ty).shape)
    perm = ir.AffineMap.get_minor_identity(src_rank, vec_rank)
    return vector.TransferReadOp(vec_ty, source, list(indices), perm, padding,
                                 [False] * vec_rank).result


def _transfer_write(value, dest, indices):
    from mlir.dialects import vector
    dst_rank = len(ir.MemRefType(dest.type).shape)
    vec_rank = len(ir.VectorType(value.type).shape)
    perm = ir.AffineMap.get_minor_identity(dst_rank, vec_rank)
    vector.TransferWriteOp(None, value, dest, list(indices), perm, [False] * vec_rank)


def _dma_wait(tag, idx, num_elements):
    from mlir.dialects import memref
    memref.DmaWaitOp(tag, [idx], num_elements)


def _vcix(name, operands, result_tys, attrs):
    return ir.Operation.create(name, results=result_tys, operands=list(operands),
                               attributes=attrs)


def _reaches(value, target):
    if value == target:
        return True
    owner = value.owner
    if isinstance(owner, ir.Block):
        return False
    for operand in owner.operands:
        if _reaches(operand, target):
            return True
    return False


class _DmaView:
    """Positional view of a customized memref.dma_start (see lower_dma_to_gemmini)."""

    def __init__(self, op):
        self.op = op
        operands = list(op.operands)
        src_rank = len(ir.MemRefType(operands[0].type).shape)
        i = 0
        self.src = operands[i]; i += 1
        i += src_rank
        self.dst = operands[i]; i += 1
        dst_rank = len(ir.MemRefType(self.dst.type).shape)
        i += dst_rank
        i += 1  # num_elements
        self.tag = operands[i]; i += 1
        tag_rank = len(ir.MemRefType(self.tag.type).shape)
        self.tag_idx = operands[i:i + tag_rank]

    def subtile_size(self):
        a = self.op.attributes
        if "subtile_size" not in a:
            return []
        return [ir.IntegerAttr(x).value for x in ir.ArrayAttr(a["subtile_size"])]

    def is_async(self):
        a = self.op.attributes
        if "async" not in a:
            return False
        try:
            return bool(ir.IntegerAttr(a["async"]).value)
        except Exception:
            return True


def _ceil_div(a, b):
    return (a + b - 1) // b


def _lower_matmul(op, SS, vlen):
    """Lower one linalg.matmul (gemm path) to the vcix push/compute/pop sequence.
    Returns True if lowered, False if skipped (conv2d / unexpected nesting -> left
    for the C++ pass / a later port). Mirrors MatmulOpLowering (gemm branch)."""
    from mlir.dialects import arith

    A, B, C = op.operands[0], op.operands[1], op.operands[2]
    mtA, mtB, mtC = ir.MemRefType(A.type), ir.MemRefType(B.type), ir.MemRefType(C.type)
    eltA, eltB, eltC = mtA.element_type, mtB.element_type, mtC.element_type
    M, K, N = mtA.shape[0], mtA.shape[1], mtB.shape[1]

    # Mirror the C++ guard: a dimension > SS must be an exact multiple, else the
    # N//SS / K//SS loop trip counts below silently drop the tail tile.
    for _dim, _name in ((M, "M"), (N, "N"), (K, "K")):
        if _dim > SS and _dim % SS != 0:
            raise NotImplementedError(
                f"matmul {_name}={_dim} must be a multiple of systolic size {SS} when > {SS}")
    
    elenA, elenB, elenC = _elt_bits(eltA), _elt_bits(eltB), _elt_bits(eltC)
    nr_eltA, nr_eltB, nr_eltC = vlen // elenA, vlen // elenB, vlen // elenC
    i64 = ir.IntegerType.get_signless(64)
    def a64(v): return ir.IntegerAttr.get(i64, v)

    acc, outer, inner = _enclosing_loops(op)
    is_conv2d = len(inner) == 4
    if not acc or len(outer) < 2:
        return False
    tile_kw = tile_oh = tile_ow = None
    if is_conv2d:                  # inner = [k_h, k_w, o_h, o_w]
        tile_kw, tile_oh, tile_ow = inner[1], inner[2], inner[3]

    vectorTypeA = ir.VectorType.get([nr_eltA], eltA)
    vectorTypeC = ir.VectorType.get([nr_eltC], eltC)
    
    nr_m = max(min(M, nr_eltA), 2)
    vectorMTypeA = ir.VectorType.get([nr_m], eltA)
    
    nr_m_c = max(min(M, nr_eltC), 2)
    vectorMTypeC = ir.VectorType.get([nr_m_c], eltC)
    spad_map, spadX, spadY = _spad_maps()

    idxMap = [0, 1, 2]
    if "idx_map" in op.attributes:
        idxMap = list(ir.DenseI32ArrayAttr(op.attributes["idx_map"]))

    # Scan the outermost loop for the A/B/Bias load DMAs (tags + subtile).
    ATag = BTag = BiasTag = None
    AAsync = BAsync = BiasAsync = 0
    BiasIdx = None
    subtileM, subtileN, subtileK = M, N, K
    a_subk = b_subk = None
    # Mirror the C++ isAInitialized / isBInitialized flags: an operand is
    # "initialized" either by an MVIN dma_start (tag found below) or by a
    # preceding affine.vector_store into its root memref (the fused case, e.g.
    # SDPA scores.V where B is the softmax output produced in-place, not DMAed).
    isAInit = isBInit = False

    def _root(v):
        owner = v.owner
        if not isinstance(owner, ir.Block):
            nm = owner.name
            if nm in ("memref.reinterpret_cast", "memref.cast"):
                return owner.operands[0]
        return v
    rootA, rootB = _root(A), _root(B)
    for o in _iter_ops(outer[-1].regions[0].blocks[0]):
        if o.operation.name == "affine.vector_store":
            dest = _root(o.operation.operands[1])
            if dest == rootA:
                isAInit = True
            elif dest == rootB:
                isBInit = True
            continue
        if o.operation.name != "memref.dma_start":
            continue
        d = _DmaView(o.operation)
        dram, is_write = _dram_is_write(d.src, d.dst)
        if dram is None or is_write:
            continue
        sram = d.dst                     # MVIN: dst is the spad
        if not any(_reaches(opnd, sram) for opnd in op.operands):
            continue
        if not isinstance(dram.owner, ir.Block):   # must be a block argument
            continue
        argn = ir.BlockArgument(dram).arg_number
        sub = d.subtile_size()
        if argn == idxMap[0]:
            ATag, AAsync = d.tag, d.is_async()
            isAInit = True
            if len(sub) >= 2:
                subtileM, subtileK = sub[-2], sub[-1]
                a_subk = sub[-1]
        elif argn == idxMap[1]:
            BTag, BAsync = d.tag, d.is_async()
            isBInit = True
            if len(sub) >= 2:
                subtileK, subtileN = sub[-2], sub[-1]
                b_subk = sub[-2]
        elif argn == idxMap[2]:
            BiasTag, BiasAsync = d.tag, d.is_async()
            BiasIdx = d.tag_idx
    if not isAInit or not isBInit:
        return False
    # A and B must agree on the K subtile (last-writer-wins would otherwise pick one silently).
    if a_subk is not None and b_subk is not None and a_subk != b_subk:
        raise NotImplementedError(
            f"Mismatched subtile K between A ({a_subk}) and B ({b_subk}) matmul operands")

    KStep = subtileK
    push_length = min(subtileM, SS)
    MStep = min(M, push_length)
    NStep = min(subtileN, SS)
    M_LOOP = min(M, push_length)

    # conv2d builds inside the existing k_w loop; gemm builds at the matmul site.
    ip = (ir.InsertionPoint.at_block_terminator(tile_kw.regions[0].blocks[0])
          if is_conv2d else ir.InsertionPoint(op))
    with ip:
        c0 = _const_index(0)
        rvl = arith.ConstantOp(i64, a64(nr_eltA)).result
        rvl_c = arith.ConstantOp(i64, a64(nr_eltC)).result
        K_val, N_val, M_val = _const_index(K), _const_index(N), _const_index(M)
        push_val = _const_index(push_length)
        num1 = _const_index(1)
        zero_pad = arith.ConstantOp(eltA, ir.FloatAttr.get(eltA, 0.0)).result
        zero_pad_c = arith.ConstantOp(eltC, ir.FloatAttr.get(eltC, 0.0)).result

    # --- inner N / K loops ---
    from mlir.dialects import affine
    body_ip = ip
    n_idx = c0
    k_idx = c0
    nk_inner = None        # innermost n/k loop created (conv2d hosts o_h/o_w here)
    if N > SS:
        with body_ip:
            nl = affine.AffineForOp(0, N // SS, 1)
            nl.operation.attributes["inner_loop"] = ir.BoolAttr.get(True)
        n_idx = nl.induction_variable
        with ir.InsertionPoint(nl.body):
            affine.AffineYieldOp([])
        body_ip = ir.InsertionPoint.at_block_terminator(nl.body)
        nk_inner = nl
    zero_vector = None
    if K > SS:
        with body_ip:
            kl = affine.AffineForOp(0, K // SS, 1)
            kl.operation.attributes["inner_loop"] = ir.BoolAttr.get(True)
        k_idx = kl.induction_variable
        with ir.InsertionPoint(kl.body):
            affine.AffineYieldOp([])
        body_ip = ir.InsertionPoint.at_block_terminator(kl.body)
        nk_inner = kl
    else:
        with body_ip:
            zv = ir.DenseElementsAttr.get_splat(vectorTypeA, ir.FloatAttr.get(eltA, 0.0))
            zero_vector = arith.ConstantOp(vectorTypeA, zv).result

    n_tag = c0 if N == subtileN else n_idx
    k_tag = c0 if K == subtileK else k_idx

    with body_ip:
        # --- B dma_wait ---
        nacc = len(acc)
        acc_ivs = [_loop_iv(l) for l in acc]
        bexpr = ir.AffineDimExpr.get(0) * -1
        for i in range(1, nacc):
            bexpr = bexpr + ir.AffineDimExpr.get(i) * -1
        b_extra = []
        bdo = nacc
        if is_conv2d:
            kW = _loop_ub(tile_kw)
            bdo = nacc + 2
            bexpr = (bexpr
                     + ir.AffineDimExpr.get(bdo - 2) * ((N // subtileN) * (K // subtileK) * kW)
                     + ir.AffineDimExpr.get(bdo - 1) * ((N // subtileN) * (K // subtileK)))
            b_extra = [_loop_iv(inner[0]), _loop_iv(inner[1])]   # k_h, k_w
        bexpr = (bexpr
                 + ir.AffineExpr.get_floor_div(ir.AffineDimExpr.get(bdo), _ceil_div(NStep, SS)) * (K // KStep)
                 + ir.AffineExpr.get_floor_div(ir.AffineDimExpr.get(bdo + 1), _ceil_div(KStep, SS)) * 1)
        bmap = ir.AffineMap.get(bdo + 2, 0, [bexpr])
        btag_idx = _apply(bmap, acc_ivs + b_extra + [n_tag, k_tag])
        if BAsync:
            _dma_wait(BTag, btag_idx, num1)

        # --- weight push loop (K x N) ---
        for i in range(0, SS, nr_eltA):
            if i < K:
                sp = _apply(spad_map, [n_idx, k_idx, _const_index(i), K_val, _const_index(SS)])
                wx = _apply(spadX, [sp, N_val])
                wy = _apply(spadY, [sp, N_val])
                wv = _transfer_read(vectorTypeA, B, [wx, wy], zero_pad)
            else:
                wv = zero_vector
            _vcix("vcix.iv", [wv, rvl], [],
                  {"opcode": a64(1), "imm": a64(0), "rd": a64(0)})

    # conv2d: move the o_h/o_w spatial loops after the weight push and continue the
    # input-push/compute/pop inside the o_w loop (heuristic, mirrors the C++ branch
    # for the no-extra-inner-loop case).
    if is_conv2d:
        # host the o_h/o_w spatial loops inside the innermost n/k loop (so n_idx/k_idx
        # stay in scope) or directly in the k_w loop when no n/k loop was created.
        host = nk_inner if nk_inner is not None else tile_kw
        tile_oh.operation.move_before(_block_terminator(host))
        body_ip = ir.InsertionPoint.at_block_terminator(tile_ow.regions[0].blocks[0])

    # --- M loop ---
    m_idx = c0
    if M > push_length:
        with body_ip:
            ml = affine.AffineForOp(0, M // push_length, 1)
            ml.operation.attributes["inner_loop"] = ir.BoolAttr.get(True)
        m_idx = ml.induction_variable
        with ir.InsertionPoint(ml.body):
            affine.AffineYieldOp([])
        body_ip = ir.InsertionPoint.at_block_terminator(ml.body)
    m_tag = c0 if M == subtileM else m_idx

    with body_ip:
        # --- A dma_wait ---
        aexpr = ir.AffineDimExpr.get(0) * -1
        for i in range(1, nacc):
            aexpr = aexpr + ir.AffineDimExpr.get(i) * -1
        a_extra = []
        ado = nacc
        if is_conv2d:
            k_h, k_w, o_h, o_w = (_loop_iv(inner[j]) for j in range(4))
            kW, oW = _loop_ub(tile_kw), _loop_ub(tile_ow)
            offset_h, offset_w = _scan_conv_offsets(tile_ow, o_h, k_h, o_w, k_w)
            coeff_h = 1 + (oW - 1) * offset_w + (kW - 1)
            ado = nacc + 2
            aexpr = (aexpr
                     + ir.AffineDimExpr.get(ado - 2) * ((K // subtileK) * (M // subtileM) * offset_h * coeff_h)
                     + ir.AffineDimExpr.get(ado - 1) * ((K // subtileK) * (M // subtileM) * offset_w))
            a_extra = [o_h, o_w]
        aexpr = (aexpr
                 + ir.AffineDimExpr.get(ado) * (M // MStep)
                 + ir.AffineExpr.get_floor_div(ir.AffineDimExpr.get(ado + 1), _ceil_div(MStep, SS)))
        amap = ir.AffineMap.get(ado + 2, 0, [aexpr])
        atag_idx = _apply(amap, acc_ivs + a_extra + [k_tag, m_tag])
        if AAsync:
            _dma_wait(ATag, atag_idx, num1)

        # --- Bias dma_wait ---
        if BiasTag is not None:
            bias_is_const = BiasIdx and BiasIdx[0].owner.name == "arith.constant"
            first_i = c0 if bias_is_const else n_tag
            third_i = c0 if bias_is_const else m_tag
            d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
            bias_expr = (ir.AffineExpr.get_floor_div(d0, _ceil_div(NStep, SS)) * (M // MStep)
                         + ir.AffineExpr.get_floor_div(d1, _ceil_div(MStep, SS)))
            bias_map = ir.AffineMap.get(2, 0, [bias_expr])
            bias_tag_idx = _apply(bias_map, [first_i, third_i])
            if BiasAsync:
                _dma_wait(BiasTag, bias_tag_idx, num1)

        # --- input push loop (M x K) ---
        for i in range(0, M_LOOP, nr_eltA):
            sp = _apply(spad_map, [k_idx, m_idx, _const_index(i), M_val, push_val])
            x = _apply(spadX, [sp, K_val])
            y = _apply(spadY, [sp, K_val])
            iv = _transfer_read(vectorMTypeA, A, [x, y], zero_pad)
            _vcix("vcix.iv", [iv, rvl], [],
                  {"opcode": a64(0), "imm": a64(0), "rd": a64(0)})

        # --- compute ---
        _vcix("vcix.i", [rvl], [],
              {"opcode": a64(1), "imm": a64(4), "rd": a64(0), "rs2": a64(0),
               "sew": a64(elenA), "lmul": a64(0)})

        # --- pop loop (M x N) ---
        for i in range(0, M_LOOP, nr_eltC):
            sp = _apply(spad_map, [n_idx, m_idx, _const_index(i), M_val, push_val])
            vpop = _vcix("vcix.v.i", [rvl_c], [vectorMTypeC],
                         {"opcode": a64(2), "imm": a64(0), "rs2": a64(0)}).results[0]
            x = _apply(spadX, [sp, N_val])
            y = _apply(spadY, [sp, N_val])
            prev = _transfer_read(vectorMTypeC, C, [x, y], zero_pad_c)
            if ir.IntegerType.isinstance(eltC):
                out = arith.AddIOp(prev, vpop).result
            else:
                out = arith.AddFOp(prev, vpop).result
            _transfer_write(out, C, [x, y])
    op.erase()
    return True


def run(module, vectorlane=128, vlen=128, **_):
    """Lower linalg.matmul (gemm) + transcendental math ops to VCIX ops, in place."""
    # matmul first (uses surrounding loop structure before any rewrites)
    mms = []
    for region in module.operation.regions:
        for b in region.blocks:
            for o in _iter_ops(b):
                if o.operation.name == "linalg.matmul":
                    mms.append(o.operation)
    for o in mms:
        _lower_matmul(o, vectorlane, vlen)
    targets = []
    for region in module.operation.regions:
        for b in region.blocks:
            for op in _iter_ops(b):
                if op.operation.name in _MATH_VIV:
                    targets.append(op.operation)
    for op in targets:
        opcode, imm = _MATH_VIV[op.name]
        vec = op.operands[0]
        res_ty = op.results[0].type
        vt = ir.VectorType(res_ty)
        n, legal_ty = _legalize_vector_type(vt, vlen)
        if legal_ty is None:
            continue
        with ir.InsertionPoint(op):
            new = _make_sf_vc_v_iv(vec, vt, n, legal_ty, opcode, imm)
        op.results[0].replace_all_uses_with(new)
        op.erase()


def run_to_vcix(in_path, out_path, vectorlane=128, vlen=128):
    with open(in_path) as f:
        text = f.read()
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(text)
        run(module, vectorlane=vectorlane, vlen=vlen)
        out = str(module)
    with open(out_path, "w") as f:
        f.write(out)


if __name__ == "__main__":
    vl = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    vlen_ = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    run_to_vcix(sys.argv[1], sys.argv[2], vl, vlen_)
