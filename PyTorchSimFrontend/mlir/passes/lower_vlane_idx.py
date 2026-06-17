"""Python out-of-line MLIR pass: lower torchsim.vlane_idx -> per-lane index * offset.

Codegen emits a dedicated `torchsim.vlane_idx` op (generic form, unregistered
dialect) carrying a `vlane_offset` integer attribute. This pass rewrites each
such op to:

    %v = "vcix.v.i"(%K) {opcode = 0, rs2 = 0, imm = 0} : (i64) -> vector<KxT>   // per-lane index
    %n = arith.constant dense<offset> : vector<KxT>
    %r = arith.muli %v, %n : vector<KxT>

and replaces uses of the original op with %r. Replaces the former C++
`-global-idx` pass (which overloaded arith.addi with a vlane_offset attribute).

Pass interface (see passes/__init__.py): MARKERS + run(module). Also runnable
standalone as a CLI:
    python PyTorchSimFrontend/mlir/passes/lower_vlane_idx.py in.mlir [out.mlir]

Requires the MLIR Python bindings on PYTHONPATH
(/riscv-llvm/python_packages/mlir_core). The `vcix` dialect must be registered
in the consuming mlir-opt for the result to round-trip (see
registerVCIXDialectTranslation in mlir-opt.cpp).
"""

OP_NAME = "torchsim.vlane_idx"
MARKERS = (OP_NAME,)


def _iter_ops(block):
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _iter_ops(b)


def run(module, **_):
    """Lower every torchsim.vlane_idx op in `module`, in place.

    Must be called with the module's Context active (the orchestrator provides it).
    """
    from mlir.ir import (InsertionPoint, Operation, IntegerType, IntegerAttr,
                         DenseElementsAttr, VectorType)
    i64 = IntegerType.get_signless(64)
    i32 = IntegerType.get_signless(32)

    targets = []
    for region in module.operation.regions:
        for b in region.blocks:
            for op in _iter_ops(b):
                if op.operation.name == OP_NAME:
                    targets.append(op.operation)

    for op in targets:
        res = op.results[0]
        vt = VectorType(res.type)
        k, et = vt.shape[0], vt.element_type
        offset = IntegerAttr(op.attributes["vlane_offset"]).value
        with InsertionPoint(op):
            rvl = Operation.create("arith.constant", results=[i64],
                                   attributes={"value": IntegerAttr.get(i64, k)}).results[0]
            lane = Operation.create("vcix.v.i", results=[vt], operands=[rvl],
                                    attributes={"opcode": IntegerAttr.get(i64, 0),
                                                "rs2": IntegerAttr.get(i32, 0),
                                                "imm": IntegerAttr.get(i32, 0)}).results[0]
            ovec = Operation.create("arith.constant", results=[vt],
                                    attributes={"value": DenseElementsAttr.get_splat(
                                        vt, IntegerAttr.get(et, offset))}).results[0]
            mul = Operation.create("arith.muli", results=[vt], operands=[lane, ovec]).results[0]
        res.replace_all_uses_with(mul)
        op.erase()


def lower_text(text: str) -> str:
    """Parse `text`, run this pass, return the printed module. CLI/testing helper."""
    if OP_NAME not in text:
        return text
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.parse(text)
        run(m)
        return str(m)


if __name__ == "__main__":
    import sys
    out = lower_text(open(sys.argv[1]).read())
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out)
    else:
        sys.stdout.write(out)
