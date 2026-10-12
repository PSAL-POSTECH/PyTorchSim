"""The timing half of the Triton route: the compiler IR -> trace.so -> TOGSim.

    run(workdir, meta, args)    emit if needed, then simulate, under the lock
    emit_trace(workdir, meta)   the compiler's trace.so + gem5 cycles -> trace_cycles.tsv
    run_togsim(workdir, ...)    hand them to TOGSim, return its parsed result
"""

import json
import os
import re
import shutil
import subprocess

from pytorchsim import config

from pytorchsim import contract
from pytorchsim.simulator import wallclock

logger = config.setup_logger()

TRACE_SO = "trace.so"
CYCLE_TSV = "trace_cycles.tsv"
SHAPE_TXT = "trace_shape.txt"
TENSORS_TXT = "trace_tensors.txt"
TENSOR_SHAPES_TXT = "trace_tensor_shapes.txt"
META_JSON = "meta.json"

PLACEHOLDER_CYCLE = 1

AXES_TXT = "trace_axes.txt"

LOCK_NAME = ".timing.lock"
LOCK_TIMEOUT = 1800



def measure_tile_cycles(workdir, meta):
    """Per-compute-node cycle counts for ONE tile: the compiler's cycle ELF, measured under gem5.
    None on any failure."""

    kernel_name = meta["kernel_name"]
    elf = contract.artifact(workdir, "cycle_elf")
    if elf is None:
        logger.warning("[Gem5] %s has no cycle_elf; it was compiled without --tog", workdir)
        return None
    try:
        with wallclock.span(wallclock.GEM5_RUN, kernel_name):
            return CycleSimulator().compile_and_simulate(
                elf, int(config.vpu_num_lanes), silent_mode=True)
    except Exception as e:
        logger.warning("[Gem5] sampling failed: %s", e)
        return None


def write_tensors(workdir):
    """Write each kernel argument's byte size, and its rows cols elem_bytes with every leading
    axis folded into the rows, in argument order, where TOGSim reads them to lay it out in DRAM."""
    import math

    import torch


    manifest = contract.kernel_object(workdir)
    if manifest is None:
        raise FileNotFoundError(f"{workdir} has no kernel.json -- compile the kernel first")
    args = [(a["shape"], getattr(torch, a["dtype"]).itemsize) for a in manifest["args"]]
    with open(os.path.join(contract.dir_for(workdir), TENSORS_TXT), "w") as f:
        f.write("".join(f"{math.prod(shape) * elem}\n" for shape, elem in args))
    with open(os.path.join(contract.dir_for(workdir), TENSOR_SHAPES_TXT), "w") as f:
        f.write("".join(f"{math.prod(shape[:-1])} {shape[-1] if shape else 1} {elem}\n"
                        for shape, elem in args))


def write_shape(workdir, meta, args=()):
    """Write the grid extents the trace producer reads as shape_args.

    The parallel numels are read off `args` by the kernel's parameter names. A
    template's trailing arguments ARE its grid, so its extents are used as is.
    """
    numels = dict(meta["numels"])
    if meta.get("template_grid") is not None:
        grid = list(contract.launch_extents(meta))
        _write_extents(workdir, grid, "template grid")
        return grid

    passed = [k for k in numels if not k.startswith("r")]
    bound = dict(zip(meta.get("call_params", ()), args))
    if bound:
        for key in passed:
            if isinstance(bound.get(key), int) and not isinstance(bound[key], bool):
                numels[key] = bound[key]
    else:
        trailing = [a for a in args if isinstance(a, int) and not isinstance(a, bool)]
        if passed and len(trailing) >= len(passed):
            for key, val in zip(passed, trailing[-len(passed):]):
                numels[key] = val

    grid = list(contract.grid_of({**meta, "numels": numels}))
    _write_extents(workdir, grid, "grid")
    return grid


def _write_extents(workdir, ext, what):
    """Write `ext` as this launch's shape_args, refusing a count it cannot use.

    The count is checked against the shared trace; the file goes in this
    process's launch directory, because the grid belongs to the launch.
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
    with open(os.path.join(contract.dir_for(workdir), SHAPE_TXT), "w") as f:
        f.write("\n".join(str(int(g)) for g in ext) + "\n")
    logger.info("[TOGSim] %s %s -> %s", what, ext, SHAPE_TXT)


def emit_trace(workdir, meta):
    """Build trace.so + trace_cycles.tsv from the compiler's trace producer.

    Returns the number of compute tiles the cycle table covers.
    """
    import json


    kernel = meta["kernel_name"]
    so_path = contract.artifact(workdir, "trace_so")
    types_path = contract.artifact(workdir, "tile_types")
    if so_path is None or types_path is None:
        raise FileNotFoundError(
            f"{workdir} has no trace_so/tile_types -- the compiler must reach "
            f"the tog step (--tog), which is what builds the producer")
    with open(types_path) as fh:
        tiles = json.load(fh)
    offsets = tiles["overlap_offset"]
    axes = tiles["parallel_axes"]
    n_tiles = len(offsets)

    cycles = measure_tile_cycles(workdir, meta)
    if cycles is None:
        raise RuntimeError(
            f"[Gem5] sampling failed for {kernel}, so compute latency would "
            f"not be modelled at all")
    table = cycle_table(offsets, list(cycles))

    if os.path.abspath(so_path) != os.path.abspath(os.path.join(workdir, TRACE_SO)):
        shutil.copyfile(so_path, os.path.join(workdir, TRACE_SO))
    with open(os.path.join(workdir, AXES_TXT), "w") as f:
        f.write(f"{len(axes)}\n")

    dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    logger.info("[Gem5] tile cycles: %s", table)
    return n_tiles


def run_togsim(workdir, meta, args=()):
    """Simulate the emitted trace, or hand it to an open TOGSimulator.

    Standalone returns the parsed result; a queued launch returns its kernel id,
    since the stream's numbers are only whole when the simulator closes.
    """
    import torch

    from pytorchsim.simulator.togsim import TOGSimulator

    for need in (TRACE_SO, CYCLE_TSV):
        if not os.path.isfile(os.path.join(workdir, need)):
            raise FileNotFoundError(f"{os.path.join(workdir, need)} not found -- call emit_trace first")
    mine = contract.link_shared(workdir, (TRACE_SO, CYCLE_TSV))
    write_shape(workdir, meta, args)
    write_tensors(workdir)

    handle = os.path.join(mine, "tile_graph.onnx")
    attribute = os.path.join(mine, "attribute")
    if torch.npu.get_tog_simulator() is not None:
        return torch.npu.launch_kernel(handle, attribute)

    result_path = TOGSimulator.run_standalone(handle, attribute)
    return TOGSimulator.get_result_from_file(result_path)


def run(workdir, meta, args=()):
    """Measure the tiles and write the cycle table if it is missing, then simulate. Returns
    TOGSim's result.

    Only the build is locked, and only against a second build of the same
    trace; simulating is per launch and runs concurrently.
    """
    from filelock import FileLock

    kernel = meta["kernel_name"]
    if not os.path.isfile(os.path.join(workdir, CYCLE_TSV)):
        with FileLock(os.path.join(workdir, LOCK_NAME), timeout=LOCK_TIMEOUT):
            if not os.path.isfile(os.path.join(workdir, CYCLE_TSV)):
                with wallclock.span(wallclock.TOGSIM_TRACE, kernel):
                    emit_trace(workdir, meta)
    with wallclock.span(wallclock.TOGSIM_RUN, kernel):
        return run_togsim(workdir, meta, args)


def store_meta(workdir, meta):
    """Persist codegen metadata beside the artifacts, so the timing step can
    run standalone."""
    with open(os.path.join(workdir, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)


#: What gem5 says when it dies, as opposed to the libc backtrace that follows it.
_GEM5_SIGNAL = re.compile(
    r"^(?:.*\b(?:panic|fatal):.*|.*Assertion.*|Program aborted at tick.*)$", re.M)


def _gem5_failure(proc, log_path):
    """The message a failed gem5 run should raise, from the log it redirects to.

    `-r` sends stdout AND stderr to `--stdout-file`, so the pipes are empty and
    a return code on its own says nothing: gem5 panics and aborts.
    """
    how = (f"signal {-proc.returncode}" if proc.returncode < 0
           else f"exit {proc.returncode}")
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        text = (proc.stderr or "") + (proc.stdout or "")
    hits = [h.strip() for h in _GEM5_SIGNAL.findall(text)]
    if not hits:
        hits = [l.strip() for l in text.strip().splitlines()[-3:]]
    if not hits:
        hits = [f"gem5 left no diagnostic; see {log_path}"]
    return (f"Gem5 Simulation Failed ({how}); full log in {log_path}\n  "
            + "\n  ".join(h[:300] for h in hits[:3]))


class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, vectorlane_size, silent_mode=False):
        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        gem5_cmd = [config.CONFIG_GEM5_PATH, "-r", "--stdout-file=sto.log", "-d", dir_path,
                    config.CONFIG_GEM5_SCRIPT, "-c", target_binary,
                    "--model", config.CONFIG_VCIX_MODEL, "--vlane", str(vectorlane_size),
                    "--vlen", str(config.vpu_vector_length_bits)]

        if not silent_mode:
            logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
            logger.info("[Gem5] Gem5 simulation started")

        proc = subprocess.run(gem5_cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(_gem5_failure(proc, os.path.join(dir_path, "sto.log")))

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        cycle_list = cycle_list[:-1]
        return cycle_list


def cycle_table(overlap_offsets, cycle_list):
    """[(cycle, overlapping_cycle), ...] indexed by tile_id. The compiler states each tile's
    overlap offset (None where nothing of it overlaps); gem5 gives the cycles."""
    if len(cycle_list) != len(overlap_offsets):
        raise ValueError(
            f"gem5 returned {len(cycle_list)} cycle sample(s) for "
            f"{len(overlap_offsets)} compute tile(s): a marker fired a different "
            f"number of times than there are tiles, so the table would be keyed "
            f"by the wrong samples")
    return [(int(c), 0 if off is None else max(int(c) - int(off), 0))
            for c, off in zip(cycle_list, overlap_offsets)]


def dump_cycle_table_tsv(table, path, origins=None):
    """cycle<TAB>overlapping per line, in tile_id order, for TOGSim's loader.

    origins is appended as a trailing comment line, which the loader stops at.
    """
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
        if origins:
            fh.write("# origins: %s\n" % ", ".join(sorted(str(o) for o in origins)))
    return path
