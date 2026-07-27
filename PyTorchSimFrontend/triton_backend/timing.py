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

#: Used only when gem5 sampling fails. Deliberately not a plausible-looking
#: number: only an obvious non-measurement gets fixed.
PLACEHOLDER_CYCLE = 1

SAMPLE_MLIR = "04-sample.mlir"
CYCLE_BIN = "cycle_bin"


def measure_tile_cycles(workdir, meta):
    """Per-compute-node cycle counts for ONE tile, measured under gem5.

    build_tog's sample mode marks each compute node and makes every loop a
    single trip; tnpu lowers that to a binary (in ITS process -- the Gemmini/VCIX
    lowering and its LLVM live there); gem5 runs it. Returns None on any failure,
    and the caller falls back to the placeholder table.
    """
    from PyTorchSimFrontend.mlir.passes.build_tog import run_tog

    kernel_name = meta["kernel_name"]
    spec = os.path.join(workdir, f"{kernel_name}_spec.py")
    if not os.path.isfile(spec):
        logger.warning("[Gem5] %s not found; cannot sample cycles", spec)
        return None

    run_tog(os.path.join(workdir, "04-custom.mlir"),
            os.path.join(workdir, "tog_sample.py"),
            os.path.join(workdir, SAMPLE_MLIR), sample_mode=True)

    import subprocess

    from . import tnpu_bridge
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # keep tnpu on its own MLIR bindings
    proc = subprocess.run(
        [extension_config.CONFIG_TNPU_PYTHON, "-m", "tnpu.cycle", spec, workdir],
        capture_output=True, text=True, cwd=tnpu_bridge.tnpu_dir(), env=env)
    if proc.returncode != 0:
        logger.warning("[Gem5] cycle binary build failed:\n%s",
                       (proc.stdout + proc.stderr)[-2000:])
        return None

    from Simulator.simulator import CycleSimulator
    try:
        return CycleSimulator().compile_and_simulate(
            os.path.join(workdir, CYCLE_BIN), int(extension_config.vpu_num_lanes),
            silent_mode=True)
    except Exception as e:  # noqa: BLE001 - fall back to the placeholder table
        logger.warning("[Gem5] sampling failed: %s", e)
        return None


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


#: triton-shared appends pidX, pidY, pidZ in that order, whatever the tiling is.
_PID_SLOT = {"x": 0, "y": 1, "z": 2}


def work_item_for(meta):
    """The WorkItem describing this kernel's program-id args and grid extents.

    `grid_of` orders axes OUTERMOST first (z, y, x -- x is Inductor's contiguous
    one), while the program-id arguments are always laid out x, y, z. The two
    are zipped downstream, so the argument list is built per axis rather than as
    a range.
    """
    from PyTorchSimFrontend.mlir.passes.lower_to_emitc import WorkItem
    from . import kernel_spec

    n_tensor, n_scalar = _runtime_arg_layout(meta)
    pid_base = n_tensor + n_scalar + 3       # after gridX, gridY, gridZ
    axes = kernel_spec.parallel_axes(meta["numels"])
    grid = list(kernel_spec.grid_of(meta))
    return WorkItem(parallel_args=[pid_base + _PID_SLOT[p] for p in axes],
                    grid=grid)


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

    # Before build_skeleton: both read the post-vcix IR, which it rewrites in place.
    cycles = measure_tile_cycles(workdir, meta)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(postvcix).read(), ctx)
        bs.build_skeleton(module)
        compute_types = ct._compute_types(module)
        n_tiles = len(compute_types)

        if cycles:
            # One numCycles per compute node; pad/truncate as the MLIR route does.
            cl = list(cycles)
            if len(cl) != n_tiles:
                logger.warning("[Gem5] returned %d cycle(s) for %d "
                               "tile(s); padding with the last", len(cl), n_tiles)
                cl = (cl + [cl[-1]] * n_tiles)[:n_tiles]
            # Systolic-array fill; only matmul tiles use it.
            lanes = int(extension_config.vpu_num_lanes)
            table = ct.build_cycle_table(module, cl, x_offset=lanes, w_offset=0)
        else:
            table = [(PLACEHOLDER_CYCLE, 0)] * n_tiles
            logger.warning(
                "[Gem5] %s holds PLACEHOLDER cycles (%d per tile x %d "
                "tiles): gem5 sampling did not produce a measurement, so "
                "compute latency is NOT modelled",
                CYCLE_TSV, PLACEHOLDER_CYCLE, n_tiles)

        l2e.skeleton_to_so(module, os.path.join(workdir, TRACE_SO),
                           work_item=work_item_for(meta))

    ct.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    if cycles:
        logger.info("[Gem5] tile cycles: %s", table)
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
