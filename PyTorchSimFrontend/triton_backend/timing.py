"""The timing half of the Triton route: tnpu IR -> trace.so -> TOGSim.

TOGSim simulates from a compiled trace producer (docs/design/togsim_cpp_trace.md).
PyTorchSim's codegen already emits one; this emits the same from a Triton-shaped
kernel, where the grid must be supplied -- see `lower_to_emitc.WorkItem`.

    emit_trace(workdir, meta)   *-custom.mlir -> trace.so + trace_cycles.tsv
    run_togsim(workdir, ...)    hand them to TOGSim, return its parsed result
"""

import json
import os

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()

#: Name TOGSim derives from the kernel directory (Simulator/simulator.py).
TRACE_SO = "trace.so"
CYCLE_TSV = "trace_cycles.tsv"
SHAPE_TXT = "trace_shape.txt"
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

    from .tnpu_bridge import stage_artifact
    run_tog(stage_artifact(workdir, "custom.mlir"),
            os.path.join(workdir, "tog_sample.py"),
            os.path.join(workdir, SAMPLE_MLIR), sample_mode=True)

    import subprocess

    from . import tnpu_bridge
    # Drops the stale PYTHONPATH (tnpu keeps its own MLIR bindings) and hands
    # over the machine the TOGSim YAML describes.
    env = tnpu_bridge.tnpu_env()
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

    A TEMPLATE KERNEL'S AXIS COUNT IS NOT IN ITS NUMELS, which is the same thing
    kernel_spec.grid_xyz has to say about the extents. mm and conv come from a
    jinja template whose grid is over output tiles; Inductor still hands them a
    numels dict, and it describes the pointwise iteration space rather than that
    grid. Counting axes from it gives 1 for every template kernel.

    That is right often enough to hide: an mm at 128x128 has template grid
    (16, 1, 1), so one axis is the true answer and the numels agree by accident.
    A bmm does not -- measured on tests/ops/fusion/test_prologue_fusion.py,
    triton_npu_fused_add_bmm_mul_4 has template grid (256, 4, 1) while its
    numels hold a single xnumel. One pid argument was replaced and the other was
    left, and _rewrite_signature refuses a kernel argument that still has uses:

        ValueError: kernel arg still used after build_skeleton; cannot drop it

    with %arg6 feeding remsi/divsi (the pid decomposition) and %arg7 feeding
    muli by 262144 (the batch stride). Both allowlist failures on this route,
    test_gqa and test_prologue_fusion, are bmm template kernels dying there.

    So the template's own grid decides the count when it has one, and the numels
    do otherwise. The extents still are not compiled in -- only the COUNT has to
    be, and the launch knows the rest -- so this reads the length and nothing
    more.
    """
    from PyTorchSimFrontend.mlir.passes.lower_to_emitc import WorkItem
    from . import kernel_spec

    n_tensor, n_scalar = _runtime_arg_layout(meta)
    pid_base = n_tensor + n_scalar + 3       # after gridX, gridY, gridZ
    axes = kernel_spec.launch_axes(meta)
    # Extents are left to run time: only the axis COUNT has to be compiled in,
    # and the launch knows the real numels. One trace then serves every shape.
    return WorkItem(parallel_args=[pid_base + _PID_SLOT[p] for p in axes],
                    grid=[None] * len(axes))


def write_shape(workdir, meta, args=()):
    """Write the grid extents the trace producer reads as shape_args.

    `args` is the launch's positional arguments; Inductor appends the numels
    after the tensors, so the trailing values are them, in `meta["numels"]`
    order. Falls back to the compile-time hint when they are absent.

    A TEMPLATE KERNEL'S EXTENTS ARE ITS TEMPLATE GRID, not a numels/BLOCK
    division. `kernel_spec.launch_axes` and `launch_extents` own that pairing --
    both readers ask them, so neither can disagree about which slot is which --
    and this writes what they answer.
    """
    from . import kernel_spec

    numels = dict(meta["numels"])
    # A template kernel's trailing call arguments are its grid, not its numels
    # (FixedGrid passes _grid_0/1/2), and its numels describe the output tensor
    # rather than its iteration space. Reading either into the other silently
    # rescales the launch, so the recorded grid is used verbatim.
    if meta.get("template_grid") is not None:
        grid = list(kernel_spec.launch_extents(meta))
        _write_extents(workdir, grid, "template grid")
        return grid

    # Only the PARALLEL numels ride along on the call -- a reduction axis is
    # looped inside the kernel, so it is not passed and must not consume one of
    # the trailing values. They arrive in kernel order, which is the dict's.
    passed = [k for k in numels if not k.startswith("r")]
    trailing = [a for a in args if isinstance(a, int) and not isinstance(a, bool)]
    if passed and len(trailing) >= len(passed):
        for key, val in zip(passed, trailing[-len(passed):]):
            numels[key] = val

    axes = kernel_spec.parallel_axes(numels)

    cfg = meta.get("fixed_config") or {}
    grid = []
    for p in axes:
        n, block = numels.get(f"{p}numel"), cfg.get(f"{p.upper()}BLOCK")
        if n is None or not block:
            raise ValueError(f"no extent for grid axis '{p}': {n!r} / {block!r}")
        grid.append(-(-int(n) // int(block)))     # ceil-div

    _write_extents(workdir, grid, "grid")
    return grid


#: How many extents the compiled trace reads. Written beside it, checked before
#: every launch -- see `_write_extents`.
AXES_TXT = "trace_axes.txt"


def _write_extents(workdir, ext, what):
    """Write `ext` as the trace's shape_args, refusing a count the trace cannot use.

    THE ABI CARRIES THE COUNT AND NOBODY WAS READING IT. The producer entry is
    `togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n)` and
    `_bind_runtime_bounds` subscripts `shape_args[k]` without ever looking at
    `n`, so a trace compiled for two axes and launched with one extent reads an
    int64 off the end of the list and loops to it.

    THAT IS NOT A WRONG NUMBER, IT IS THE MACHINE. Measured on
    tests/ops/attention/test_gqa.py: triton_npu_fused_bmm_..._1 had
    template_grid [1, 8, 1] and a single xnumel, so the WorkItem compiled two
    pid arguments while this wrote one line, "1". TOGSim allocated against
    whatever was past the end -- tens of gigabytes, killed by hand, and a
    SIGABRT once the address space was capped.

    `kernel_spec.launch_axes` and `launch_extents` are the single source both
    readers use, so they agree by construction. This guards what that cannot: a trace.so is REUSED when it is
    already on disk (see `emit_trace`'s caller), so a stale one can meet a meta
    that counts differently. The count travels with the trace and is compared
    here, where the answer is a diagnostic instead of an allocation.
    """
    path = os.path.join(workdir, AXES_TXT)
    if os.path.isfile(path):
        with open(path) as f:
            want = int(f.read().strip())
        if want != len(ext):
            raise ValueError(
                f"{TRACE_SO} in {workdir} was compiled to read {want} grid "
                f"extent(s) and this launch has {len(ext)} ({ext}); the trace "
                f"would read past the end of shape_args. Delete the workdir to "
                f"rebuild it.")
    with open(os.path.join(workdir, SHAPE_TXT), "w") as f:
        f.write("\n".join(str(int(g)) for g in ext) + "\n")
    logger.info("[TOGSim] %s %s -> %s", what, ext, SHAPE_TXT)


def emit_trace(workdir, meta):
    """Build `trace.so` + `trace_cycles.tsv` from tnpu's post-vcix IR.

    Returns the number of compute tiles the cycle table covers.
    """
    from PyTorchSimFrontend.mlir.passes import build_skeleton as bs
    from PyTorchSimFrontend.mlir.passes import cycle_table as ct
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as l2e
    from PyTorchSimFrontend.mlir.passes.build_tog import ir

    from .tnpu_bridge import stage_artifact
    postvcix = stage_artifact(workdir, "custom.mlir")
    if postvcix is None:
        raise FileNotFoundError(
            f"no *-custom.mlir in {workdir} -- tnpu must run far enough to emit "
            f"the post-vcix IR, which is what the trace is built from")

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

        wi = work_item_for(meta)
        l2e.skeleton_to_so(module, os.path.join(workdir, TRACE_SO), work_item=wi)
    # Beside the trace, so a launch can refuse a shape list of the wrong length
    # instead of reading past the end of it -- see `_write_extents`.
    with open(os.path.join(workdir, AXES_TXT), "w") as f:
        f.write(f"{len(wi.parallel_args)}\n")

    ct.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    if cycles:
        logger.info("[Gem5] tile cycles: %s", table)
    return n_tiles


def run_togsim(workdir, meta=None, args=(), attribute_path=None, timeout_sec=None):
    """Simulate the emitted trace. Returns TOGSimulator's parsed result dict.

    `meta`/`args` supply the grid: the trace producer takes its loop bounds from
    shape_args, so they are written out per launch rather than compiled in.
    """
    from Simulator.simulator import TOGSimulator

    so = os.path.join(workdir, TRACE_SO)
    if not os.path.isfile(so):
        raise FileNotFoundError(f"{so} not found -- call emit_trace first")
    if meta is not None:
        write_shape(workdir, meta, args)

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
