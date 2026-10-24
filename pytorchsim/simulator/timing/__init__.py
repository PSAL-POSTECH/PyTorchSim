"""The timing path: the compiler's tile-operation graph (tog.so) -> TOGSim. run writes the cycle
tables if needed (write_cycle_tables: tog.so + gem5's sampled tiles) and simulates (run_togsim);
write_shape and write_tensors are the per-launch files TOGSim reads."""

import json
import os
import shutil

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge
from pytorchsim.simulator import wallclock
from pytorchsim.simulator.timing import gem5

logger = config.setup_logger()

TOG_SO = "tog.so"
CYCLE_TSV = "tog_cycles.tsv"
UNITS_TSV = "tog_units.tsv"
SHARED = (TOG_SO, CYCLE_TSV, UNITS_TSV)
SHAPE_TXT = "tog_shape.txt"
TENSORS_TXT = "tog_tensors.txt"
TENSOR_SHAPES_TXT = "tog_tensor_shapes.txt"

AXES_TXT = "tog_axes.txt"

LOCK_NAME = ".timing.lock"
LOCK_TIMEOUT = 1800


def run(workdir, meta, args=()):
    """Measure the tiles and write the cycle table if it is missing, then simulate. Returns TOGSim's
    result. Only the build is locked; simulating is per launch and runs concurrently."""
    from filelock import FileLock

    kernel = meta["kernel_name"]

    def emitted():
        return all(os.path.isfile(os.path.join(workdir, name)) for name in (CYCLE_TSV, UNITS_TSV))

    if not emitted():
        with FileLock(os.path.join(workdir, LOCK_NAME), timeout=LOCK_TIMEOUT):
            if not emitted():
                with wallclock.span(wallclock.TOGSIM_TABLES, kernel):
                    write_cycle_tables(workdir, meta)
    with wallclock.span(wallclock.TOGSIM_RUN, kernel):
        return run_togsim(workdir, meta, args)


def run_togsim(workdir, meta, args=()):
    """Simulate the TOG, or hand it to an open TOGSimulator. Standalone returns the result TOGSim wrote,
    read; a queued launch returns its kernel id, as the stream's numbers are whole only when it closes."""
    import torch

    from pytorchsim.simulator.timing.togsim import TOGSimProcess

    for need in SHARED:
        if not os.path.isfile(os.path.join(workdir, need)):
            raise FileNotFoundError(f"{os.path.join(workdir, need)} not found -- call write_cycle_tables first")
    mine = contract.link_shared(workdir, SHARED)
    write_shape(workdir, meta, args)
    write_tensors(workdir)

    handle = os.path.join(mine, "tile_graph.onnx")
    attribute = attribute_dir(workdir)
    if torch.npu.get_tog_simulator() is not None:
        return torch.npu.launch_kernel(handle, _held_for_the_session(mine, attribute))

    result_path = TOGSimProcess.run_standalone(handle, attribute)
    return TOGSimProcess.read_result(result_path)


def attribute_dir(workdir):
    """The runtime attribute records this launch's functional run left (runtime/runtime_attribute/),
    which TOGSim reads; refused when the kernel declares attributes and no functional run left them."""
    manifest = contract.kernel_object(workdir) or {}
    d = os.path.join(contract.runtime_dir(workdir),
                     compiler_bridge.compiler_contract("kernel_object").RUNTIME_ATTRIBUTE_DIR)
    if manifest.get("runtime_attributes") and not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} not found -- {manifest.get('name')} reads runtime attributes, which only the "
            f"functional run records: run with TORCHSIM_FUNCTIONAL_MODE=1")
    return d


def _held_for_the_session(mine, attribute):
    """A copy of `attribute` this launch alone names: an open TOGSimulator reads it when it takes the
    launch, after the next functional run of the kernel has rewritten the original."""
    import tempfile
    held = tempfile.mkdtemp(prefix="runtime_attribute-", dir=mine)
    shutil.copytree(attribute, held, dirs_exist_ok=True)
    return held


def write_cycle_tables(workdir, meta):
    """Place the compiler's tog.so in the workdir and write tog_cycles.tsv and tog_units.tsv beside it.

    Returns the number of compute tiles gem5 measured.
    """


    kernel = meta["kernel_name"]
    so_path = contract.artifact(workdir, "tog_so")
    types_path = contract.artifact(workdir, "tile_types")
    if so_path is None or types_path is None:
        raise FileNotFoundError(
            f"{workdir} has no tog_so/tile_types -- the compiler must reach "
            f"the tog step (--tog), which is what builds the TOG")
    with open(types_path) as fh:
        tiles = json.load(fh)
    offsets = tiles["overlap_offset"]
    axes = tiles["parallel_axes"]
    n_tiles = len(offsets)

    m5out = gem5.run_cycle_binary(workdir, meta)
    if m5out is None:
        raise RuntimeError(
            f"[Gem5] sampling failed for {kernel}, so compute latency would "
            f"not be modelled at all")
    table, rows = gem5.tile_tables(m5out, offsets)

    if os.path.abspath(so_path) != os.path.abspath(os.path.join(workdir, TOG_SO)):
        shutil.copyfile(so_path, os.path.join(workdir, TOG_SO))
    with open(os.path.join(workdir, AXES_TXT), "w") as f:
        f.write(f"{len(axes)}\n")

    with open(os.path.join(workdir, UNITS_TSV), "w") as f:
        f.write("".join("\t".join(str(v) for v in r) + "\n" for r in rows))
    gem5.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    logger.info("[Gem5] tile cycles: %s", table)
    return n_tiles


def write_shape(workdir, meta, args=()):
    """Write the shape_args the TOG reads: the grid extents, the parallel numels read off `args` by
    parameter name (a template's trailing arguments are its grid), then the integer scalars in the
    compiler's order (kernel_object.shape_args)."""
    numels = dict(meta["numels"])
    if meta.get("template_grid") is not None:
        grid = list(contract.launch_extents(meta))
        _write_extents(workdir, grid, "template grid", meta, args)
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
    _write_extents(workdir, grid, "grid", meta, args)
    return grid


def _write_extents(workdir, ext, what, meta, args):
    """Write `ext` and the launch's integer scalars as this launch's shape_args in its launch
    directory, refusing a grid count the shared TOG cannot use."""
    path = os.path.join(workdir, AXES_TXT)
    if os.path.isfile(path):
        with open(path) as f:
            want = int(f.read().strip())
        if want != len(ext):
            raise ValueError(
                f"{TOG_SO} in {workdir} was compiled to read {want} grid "
                f"extent(s) and this launch has {len(ext)} ({ext}); the TOG "
                f"would read past the end of shape_args. Delete the workdir to "
                f"rebuild it.")
    ko = compiler_bridge.compiler_contract("kernel_object")
    shape = ko.shape_args(contract.kernel_object(workdir), dict(zip(meta.get("call_params", ()), args)),
                          grid=ext)
    with open(os.path.join(contract.dir_for(workdir), SHAPE_TXT), "w") as f:
        f.write("\n".join(str(int(g)) for g in shape) + "\n")
    logger.info("[TOGSim] %s %s, scalars %s -> %s", what, ext, shape[len(ext):], SHAPE_TXT)


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
