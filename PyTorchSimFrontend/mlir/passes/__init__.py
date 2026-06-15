"""Python out-of-line MLIR passes run on each kernel .mlir before mlir-opt.

MLIR's PassManager only schedules *registered C++ passes*, not arbitrary Python
functions, so imperative Python rewrites are orchestrated here instead. The flow
is Module-centric: parse the .mlir once, run each registered pass on the shared
Module, print once. A text marker check skips parsing entirely when no pass's
target op is present (the common case).

To add a pass, create a module exposing MARKERS (tuple of op-name strings) and
run(module) (mutates the Module in place), and append it to PASSES below.
"""
from . import lower_vlane_idx
from .lower_to_llvm import run_standard_lowering  # noqa: F401 (re-exported)

# Ordered passes applied to each kernel .mlir before mlir-opt.
PASSES = [
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
