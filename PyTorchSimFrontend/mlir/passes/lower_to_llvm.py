"""Standard MLIR -> LLVM-dialect lowering via the bindings PassManager.

Runs the upstream *registered* lowering passes (convert-*-to-llvm, lower-affine,
reconcile-unrealized-casts, ...) in-process on the post-custom-pass IR, replacing
the tail of the mlir-opt pipeline. The custom passes (test-loop-padding,
dma-fine-grained, test-pytorchsim-to-vcix, test-tile-operation-graph,
test-memref-to-gemmini) still run in mlir-opt; this picks up right after
memref-to-gemmini. As those custom passes migrate to Python, mlir-opt shrinks and
eventually this becomes the whole back half of an all-in-process flow.

Validated to produce byte-identical LLVM IR to running the same passes inside
mlir-opt. Note: only lower-vector-multi-reduction is func.func-scoped (the
bindings pass-pipeline parser does not auto-nest like the mlir-opt CLI, so it is
wrapped explicitly); order is preserved to match the original pipeline.
"""

STANDARD_PIPELINE = (
    "builtin.module("
    "convert-linalg-to-loops,"
    "convert-vector-to-scf{full-unroll=true},"
    "lower-affine,"
    "expand-strided-metadata,"   # decompose memref.collapse_shape/subview before LLVM
    "finalize-memref-to-llvm,"
    "func.func(lower-vector-multi-reduction),"
    "convert-vector-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-math-to-llvm,"
    "convert-scf-to-cf,"
    "convert-cf-to-llvm,"
    "convert-func-to-llvm,"
    "convert-index-to-llvm,"
    "reconcile-unrealized-casts)"
)


def run_standard_lowering(in_path, out_path=None, timing=False):
    """Lower the post-custom-pass MLIR at `in_path` to the LLVM dialect.

    Runs the imperative Gemmini lowering (memref.dma_start/dma_wait) then the
    registered standard MLIR->LLVM passes. `timing` selects the Gemmini behavior:
    False for the functional/Spike path (emit gemmini asm), True for the gem5
    cycle path (erase dma_start; the TOG already carries DMA timing) -- this
    preserves the old test-memref-to-gemmini `timing=1` semantics.

    Writes the result to `out_path` (defaults to `in_path`, i.e. in place).
    Requires the MLIR Python bindings on PYTHONPATH.
    """
    if out_path is None:
        out_path = in_path
    from mlir.ir import Context, Module, Location
    from mlir.passmanager import PassManager
    from . import lower_dma_to_gemmini
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        # Imperative Python pass: memref.dma_start/dma_wait -> Gemmini asm (replaces
        # the C++ test-memref-to-gemmini), then the registered standard lowering.
        lower_dma_to_gemmini.run(module, timing=timing)
        PassManager.parse(STANDARD_PIPELINE, ctx).run(module.operation)
        with open(out_path, "w") as f:
            f.write(str(module))


if __name__ == "__main__":
    import sys
    run_standard_lowering(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
