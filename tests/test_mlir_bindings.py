"""Exercise the MLIR Python bindings the way a decompose-transfer pass would:
parse a custom op, read its AffineMap attr, build an scf.for loop with
affine.apply + an inner (unregistered) DMA op, erase the original, re-verify.
"""
from mlir.ir import (Context, Module, Location, InsertionPoint, Operation,
                     IndexType, IntegerAttr, AffineMap)
from mlir.dialects import scf, affine, arith, func, memref

ctx = Context()
ctx.allow_unregistered_dialects = True

with ctx, Location.unknown():
    src = '''
    func.func @kernel(%dram: memref<256x256xf16>, %sram: memref<128x128xf16, 1>) {
      "togsim.transfer"(%dram, %sram) {
        dma_kind = "MVIN",
        src_map = affine_map<(d0, d1) -> (d0, d1 floordiv 16, d1 mod 16)>
      } : (memref<256x256xf16>, memref<128x128xf16, 1>) -> ()
      return
    }
    '''
    m = Module.parse(src)
    print("[1] parsed module ok")

    fn = m.body.operations[0]
    blk = fn.regions[0].blocks[0]
    transfer = next(op.operation for op in blk.operations
                    if op.operation.name == "togsim.transfer")
    print("[2] found op:", transfer.name)

    src_map = transfer.attributes["src_map"]
    print("[3] src_map attr:", src_map)

    idx = IndexType.get()
    def cst(v):
        return Operation.create("arith.constant", results=[idx],
                                attributes={"value": IntegerAttr.get(idx, v)}).result

    with InsertionPoint(transfer):
        lb, ub, step = cst(0), cst(2), cst(1)
        loop = scf.ForOp(lb, ub, step)
        with InsertionPoint(loop.body):
            iv = loop.induction_variable
            base = affine.AffineApplyOp(AffineMap.get_identity(1), [iv])
            Operation.create("togsim.dma_descriptor",
                             operands=[base.result], results=[])
            scf.YieldOp([])
    print("[4] built scf.for + affine.apply + inner op")

    transfer.erase()
    print("[5] erased original transfer")

    print("[6] verify:", m.operation.verify())
    print("----- rewritten IR -----")
    print(str(m))
print("ALL GOOD")
