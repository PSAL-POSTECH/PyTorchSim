"""The timing half of the Triton route: tnpu IR -> trace.so -> TOGSim.

TOGSim simulates from a compiled trace producer (docs/design/togsim_cpp_trace.md).
PyTorchSim's codegen already emits one; this emits the same from a Triton-shaped
kernel, where the grid must be supplied -- see `lower_to_emitc.WorkItem`.

    emit_trace(workdir, meta)   04-custom.mlir -> trace.so + trace_cycles.tsv
    run_togsim(workdir, ...)    hand them to TOGSim, return its parsed result
"""

import json
import os

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()

#: Name TOGSim derives from the kernel directory (Simulator/simulator.py).
TRACE_SO = "trace.so"
CYCLE_TSV = "trace_cycles.tsv"
META_JSON = "meta.json"

#: Stand-in per-tile cost until gem5 sampling lands. Deliberately not a
#: plausible-looking number: only an obvious non-measurement gets fixed.
PLACEHOLDER_CYCLE = 1


def _runtime_arg_layout(meta):
    """(n_tensor_args, n_scalar_args) of the lowered signature.

    triton-shared lays it out as pointers, user scalars, then its own six
    (gridX,Y,Z / pidX,Y,Z). constexpr params never become arguments.
    """
    sig = meta["signature"]
    tensors = [k for k, v in sig.items() if v.startswith("*")]
    scalars = [k for k, v in sig.items()
               if not v.startswith("*") and v != "constexpr"]
    return len(tensors), len(scalars)


def work_item_for(meta):
    """The WorkItem describing this kernel's program-id args and grid extents."""
    from PyTorchSimFrontend.mlir.passes.lower_to_emitc import WorkItem
    from . import kernel_spec

    n_tensor, n_scalar = _runtime_arg_layout(meta)
    pid_x = n_tensor + n_scalar + 3          # after gridX, gridY, gridZ
    grid = list(kernel_spec.grid_of(meta))
    return WorkItem(parallel_args=list(range(pid_x, pid_x + len(grid))), grid=grid)


def emit_trace(workdir, meta):
    """Build `trace.so` + `trace_cycles.tsv` from tnpu's post-vcix IR.

    Returns the number of compute tiles the cycle table covers.
    """
    from PyTorchSimFrontend.mlir.passes import build_skeleton as bs
    from PyTorchSimFrontend.mlir.passes import cycle_table as ct
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as l2e
    from PyTorchSimFrontend.mlir.passes.build_tog import ir

    postvcix = os.path.join(workdir, "04-custom.mlir")
    if not os.path.isfile(postvcix):
        raise FileNotFoundError(
            f"{postvcix} not found -- tnpu must have run at least to stage 4 "
            f"(the post-vcix IR is what the trace is built from)")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(postvcix).read(), ctx)
        bs.build_skeleton(module)
        n_tiles = len(ct._compute_types(module))
        l2e.skeleton_to_so(module, os.path.join(workdir, TRACE_SO),
                           work_item=work_item_for(meta))

    # Until gem5 sampling lands, every tile costs PLACEHOLDER_CYCLE: TOGSim
    # models the DMA and the dependency structure but NOT compute latency.
    table = [(PLACEHOLDER_CYCLE, 0)] * n_tiles
    ct.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    logger.warning("[Gem5] %s holds PLACEHOLDER cycles (%d per tile x %d tiles); "
                   "compute latency is not modelled yet",
                   CYCLE_TSV, PLACEHOLDER_CYCLE, n_tiles)
    return n_tiles


def run_togsim(workdir, attribute_path=None, timeout_sec=None):
    """Simulate the emitted trace. Returns TOGSimulator's parsed result dict."""
    from Simulator.simulator import TOGSimulator

    so = os.path.join(workdir, TRACE_SO)
    if not os.path.isfile(so):
        raise FileNotFoundError(f"{so} not found -- call emit_trace first")

    # A handle only: TOGSim derives trace.so / trace_cycles.tsv from its
    # DIRECTORY, and reads the file itself only on the STONNE path.
    handle = os.path.join(workdir, "tile_graph.onnx")
    result_path = TOGSimulator.run_standalone(
        handle, attribute_path or os.path.join(workdir, "attribute"),
        timeout_sec=timeout_sec)
    return TOGSimulator.get_result_from_file(result_path)


def store_meta(workdir, meta):
    """Persist codegen metadata beside the artifacts, so the timing step can run
    standalone."""
    with open(os.path.join(workdir, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(workdir):
    with open(os.path.join(workdir, META_JSON)) as f:
        return json.load(f)
