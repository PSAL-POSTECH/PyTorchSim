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
from .lower_to_llvm import run_standard_lowering  # noqa: F401 (re-exported)

# Ordered passes applied to each kernel .mlir before mlir-opt.
# decompose_transfer first: it lowers togsim.transfer -> memref.dma_start, which
# downstream passes (and the gemmini lowering) expect.
PASSES = [
    decompose_transfer,
    lower_vlane_idx,
]


def run_python_passes(mlir_path):
    """Apply all registered Python MLIR passes to the .mlir at `mlir_path`, in place.

    Returns True if the file was modified, False otherwise.
    """
    with open(mlir_path) as f:
        text = f.read()

    # Fast path: nothing to do if no pass's target op appears in the text.
    active = [p for p in PASSES if any(mk in text for mk in p.MARKERS)]
    if not active:
        return False

    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        module = Module.parse(text)
        for p in active:
            p.run(module)
        out = str(module)

    with open(mlir_path, "w") as f:
        f.write(out)
    return True
