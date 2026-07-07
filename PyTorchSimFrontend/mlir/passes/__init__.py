"""Python out-of-line MLIR passes run on each kernel .mlir before mlir-opt.

MLIR's PassManager only schedules *registered C++ passes*, not arbitrary Python
functions, so imperative Python rewrites are orchestrated here instead. The flow
is Module-centric: parse the .mlir once, run each registered pass on the shared
Module, print once. A text marker check skips parsing entirely when no pass's
target op is present (the common case).

To add a pass, create a module exposing MARKERS (tuple of op-name strings) and
run(module) (mutates the Module in place), and append it to PASSES below.
"""
def _ensure_mlir_bindings_on_path():
    """Make `import mlir` work even when PYTHONPATH is not set, by deriving the
    bindings location from TORCHSIM_LLVM_PATH (e.g. /riscv-llvm/bin ->
    /riscv-llvm/python_packages/mlir_core). The container sets PYTHONPATH, but
    plain local runs may not."""
    try:
        import mlir.ir  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import os
    import sys
    from PyTorchSimFrontend import extension_config
    llvm_path = (extension_config.CONFIG_TORCHSIM_LLVM_PATH or "").rstrip("/")
    cand = os.path.join(os.path.dirname(llvm_path), "python_packages", "mlir_core")
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)


_ensure_mlir_bindings_on_path()

from . import lower_vlane_idx
from . import decompose_transfer
from . import dma_fine_grained
from . import lower_to_vcix
from . import peel_transfer
from . import peel_transfer
from .lower_to_llvm import run_standard_lowering  # noqa: F401 (re-exported)
from .build_tog import run_tog  # noqa: F401 (re-exported; replaces C++ test-tile-operation-graph)
from .dma_fine_grained import run_fine_grained  # noqa: F401 (re-exported; standalone/CLI)
from .lower_to_vcix import run_to_vcix  # noqa: F401 (re-exported; standalone/CLI)

# Module rewrite passes around the one remaining mlir-opt pass (-test-loop-padding).
# Each exposes MARKERS + run(module, **opts); run_module_passes parses once per phase.
# togsim.transfer stays through the pipeline (no more memref.dma_start): it lowers
# directly to Gemmini at the end (lower_transfer_to_gemmini); loop-padding runs opaquely.
PRE_OPT_PASSES = [
    lower_vlane_idx,
]
# fine-grained first: splits the matmul DMAs that the vcix lowering then reads.
# peel_transfer last: split any >4D togsim.transfer so build_tog (trace) also sees <=4D.
POST_OPT_PASSES = [
    dma_fine_grained,
    lower_to_vcix,
    peel_transfer,
]


def run_module_passes(in_path, out_path, passes, **opts):
    """Parse `in_path` once, run each marker-matched pass on the shared Module in
    order, print once to `out_path` (in place if equal). `opts` forwarded to each
    run(module, **opts). Returns True if any pass ran."""
    with open(in_path) as f:
        text = f.read()

    active = [p for p in passes if any(mk in text for mk in p.MARKERS)]
    if not active:
        if out_path != in_path:
            import shutil
            shutil.copyfile(in_path, out_path)
        return False

    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        module = Module.parse(text)
        for p in active:
            p.run(module, **opts)
        out = str(module)

    # Atomic write: run_python_passes rewrites the kernel .mlir in place outside
    # load()'s FileLock, so a concurrent compile of the same source must never see a
    # truncated file -- mlir-opt would parse it to an empty module and silently drop
    # the kernel (-> undefined reference to wrapper_kernel at link).
    from torch._inductor.codecache import write_atomic
    write_atomic(out_path, out)
    return True


def run_python_passes(mlir_path, vectorlane=128):
    """Run the pre-mlir-opt Module passes (PRE_OPT_PASSES) on `mlir_path`, in place."""
    return run_module_passes(mlir_path, mlir_path, PRE_OPT_PASSES, vectorlane=vectorlane)
