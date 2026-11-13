"""The timing path: the compiler's tile-operation graph (tog.so) -> TOGSim. run writes the cycle
tables if needed (write_cycle_tables: tog.so + gem5's sampled tiles) and simulates (run_togsim);
write_shape, write_tensors and placement.write_tensor_addrs are the per-launch files TOGSim reads."""

import json
import os
import shutil

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge
from pytorchsim.simulator import placement, wallclock
from pytorchsim.simulator.timing import gem5

logger = config.setup_logger()

TOG_SO = "tog.so"
CYCLE_TSV = "tog_tile_cycles.tsv"
UNITS_TSV = "tog_unit_table.tsv"
SHARED = (TOG_SO, CYCLE_TSV, UNITS_TSV)
SHAPE_TXT = "tog_shape_args.txt"
TENSORS_TXT = "tog_tensor_bytes.txt"
TENSOR_SHAPES_TXT = "tog_tensor_shapes.txt"

TIMELINE_JSON = "tog_tile_timeline.json"

LOCK_NAME = ".timing.lock"
LOCK_TIMEOUT = 1800


def run(workdir, meta, args=(), names=None):
    """Measure the tiles and write the cycle table if it is missing, then simulate. Returns TOGSim's
    result, or None for a launch queued on an open TOGSimulator. Only the build is locked."""
    from filelock import FileLock

    kernel = meta["kernel_name"]

    def emitted():
        return all(os.path.isfile(os.path.join(workdir, name)) for name in (CYCLE_TSV, UNITS_TSV))

    if not emitted():
        with FileLock(os.path.join(workdir, LOCK_NAME), timeout=LOCK_TIMEOUT):
            if not emitted():
                with wallclock.span(wallclock.TOGSIM_TABLES, kernel):
                    write_cycle_tables(workdir, meta)
    else:
        logger.info("[Gem5] tile cycles reused from %s", os.path.join(workdir, CYCLE_TSV))
    with wallclock.span(wallclock.TOGSIM_RUN, kernel):
        return run_togsim(workdir, meta, args, names)


def run_togsim(workdir, meta, args=(), names=None):
    """Simulate the TOG, or hand it to an open TOGSimulator. Standalone returns the result TOGSim wrote,
    read; a queued launch returns None, as the stream's numbers are whole only when it closes."""
    import torch

    for need in SHARED:
        if not os.path.isfile(os.path.join(workdir, need)):
            raise FileNotFoundError(f"{os.path.join(workdir, need)} not found -- call write_cycle_tables first")
    mine = contract.new_launch_dir(workdir, SHARED, build_launch_record(meta, args, names))
    write_shape(workdir, meta, args, mine)
    write_tensors(workdir, mine)
    placement.write_tensor_addrs(workdir, meta, args, launch_dir=mine)

    attribute = attribute_dir(workdir)
    sim = torch.npu.get_tog_simulator()
    if sim is not None:
        for handle in copy_node_handles(workdir, mine):
            torch.npu.launch_kernel(handle, "")
        torch.npu.launch_kernel(os.path.join(mine, "tile_graph.onnx"),
                                _held_for_the_session(sim, mine, attribute) if attribute else "")
        return None
    return simulate_standalone(workdir, mine, attribute or "")


def simulate_standalone(workdir, mine, attribute, **kw):
    """read_result of the launch in `mine`, run alone: the kernel object's copy nodes, then its TOG.
    total_cycles is the launch's sum; kernel_cycles and copy_node_cycles are the parts."""
    from pytorchsim.simulator.timing import npuwattch
    from pytorchsim.simulator.timing.togsim import TOGSimProcess

    copy_paths = [TOGSimProcess.run_standalone(h, "", **kw) for h in copy_node_handles(workdir, mine)]
    copies = [TOGSimProcess.read_result(p)["total_cycles"] for p in copy_paths]
    result_path = TOGSimProcess.run_standalone(os.path.join(mine, "tile_graph.onnx"), attribute, **kw)
    result = TOGSimProcess.read_result(result_path)
    if not kw.get("autotune_mode"):
        npuwattch.record(workdir, mine, str(result_path), [str(p) for p in copy_paths])
    if copies:
        result = dict(result, kernel_cycles=result["total_cycles"], copy_node_cycles=copies,
                      total_cycles=result["total_cycles"] + sum(copies))
    return result


def copy_node_handles(workdir, mine):
    """The handle of each copy node the kernel object lists (the compiler's kernel_object.copy_nodes), in
    the launch directory `mine` beside the tensors it moves: a TOG that only loads, empty tables."""
    ko = compiler_bridge.compiler_contract("kernel_object")
    handles = []
    for node in ko.copy_nodes(contract.kernel_object(workdir) or {}):
        so = os.path.join(workdir, node["tog_so"])
        d = os.path.join(mine, os.path.dirname(node["tog_so"]))
        os.makedirs(d, exist_ok=True)
        if os.path.lexists(os.path.join(d, TOG_SO)):
            os.remove(os.path.join(d, TOG_SO))
        os.symlink(so, os.path.join(d, TOG_SO))
        for name in (CYCLE_TSV, UNITS_TSV):
            open(os.path.join(d, name), "w").close()
        for name in (TENSORS_TXT, TENSOR_SHAPES_TXT, placement.tensor_addrs_file()):
            if os.path.isfile(os.path.join(mine, name)):
                shutil.copyfile(os.path.join(mine, name), os.path.join(d, name))
        handles.append(os.path.join(d, "tile_graph.onnx"))
    return handles


def build_launch_record(meta, args, names=None):
    """Which call a launch was: the kernel, and per argument its parameter name, the buffer the wrapper's call
    names (`names`, codegen's torchsim_arg_names), and a tensor's shape, stride, dtype and address."""
    import torch
    params = list(meta.get("call_params", ()))
    out = []
    for i, a in enumerate(args):
        entry = {"param": params[i] if i < len(params) else None,
                 "name": names[i] if names is not None and i < len(names) else None}
        if torch.is_tensor(a):
            entry.update(shape=list(a.shape), stride=list(a.stride()), dtype=str(a.dtype).split(".")[-1],
                         data_ptr=a.data_ptr())
        elif isinstance(a, (int, float, bool)):
            entry["value"] = a
        out.append(entry)
    return {"kernel": meta.get("kernel_name"), "args": out}


def attribute_dir(workdir):
    """The runtime attribute records this launch's functional run left, which TOGSim reads, or None when
    the kernel declares none; refused when it declares some and no functional run left them."""
    d = compiler_bridge.runtime_attribute_dir(workdir)
    if d is not None and not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} not found -- {os.path.basename(workdir)} reads runtime attributes, which only the "
            f"functional run records: run with TORCHSIM_FUNCTIONAL_MODE=1")
    return d


def _held_for_the_session(sim, mine, attribute):
    """A copy of `attribute` this launch alone names, which `sim` reads when it takes the launch, after
    the next functional run has rewritten the original; `sim` removes it when it shuts down."""
    import tempfile
    held = tempfile.mkdtemp(prefix="runtime_attribute-", dir=mine)
    shutil.copytree(attribute, held, dirs_exist_ok=True)
    sim.held.append(held)
    return held


def write_cycle_tables(workdir, meta):
    """Place the compiler's tog.so in the workdir and write tog_tile_cycles.tsv and tog_unit_table.tsv beside it.

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
    n_tiles = len(offsets)

    m5out = gem5.run_cycle_binary(workdir, meta)
    if m5out is None:
        raise RuntimeError(
            f"[Gem5] sampling failed for {kernel}, so compute latency would "
            f"not be modelled at all")
    table, rows = gem5.tile_tables(m5out, offsets)

    if os.path.abspath(so_path) != os.path.abspath(os.path.join(workdir, TOG_SO)):
        shutil.copyfile(so_path, os.path.join(workdir, TOG_SO))
    with open(os.path.join(workdir, UNITS_TSV), "w") as f:
        f.write("".join("\t".join(str(v) for v in r) + "\n" for r in rows))
    gem5.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    gem5.write_tile_timeline(m5out, table, os.path.join(workdir, TIMELINE_JSON))
    logger.info("[Gem5] tile cycles: %s", table)
    return n_tiles


def write_shape(workdir, meta, args=(), into=None):
    """Write the shape_args the TOG reads: the grid extents, the parallel numels read off `args` by
    parameter name (a template's trailing arguments are its grid), then the integer scalars in the
    compiler's order (kernel_object.shape_args)."""
    numels = dict(meta["numels"])
    if meta.get("template_grid") is not None:
        grid = list(contract.launch_extents(meta))
        _write_extents(workdir, grid, "template grid", meta, args, into)
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
    _write_extents(workdir, grid, "grid", meta, args, into)
    return grid


def _write_extents(workdir, ext, what, meta, args, into=None):
    """Write `ext` and the launch's integer scalars as this launch's shape_args in its launch
    directory (`into`, else this process's). A count the TOG was not compiled for is refused by the TOG itself
    (togsim::shape), before it reads past shape_args."""
    ko = compiler_bridge.compiler_contract("kernel_object")
    shape = ko.shape_args(contract.kernel_object(workdir), dict(zip(meta.get("call_params", ()), args)),
                          grid=ext)
    with open(os.path.join(into or contract.dir_for(workdir), SHAPE_TXT), "w") as f:
        f.write("\n".join(str(int(g)) for g in shape) + "\n")
    logger.info("[TOGSim] %s %s, scalars %s -> %s", what, ext, shape[len(ext):], SHAPE_TXT)


def write_tensors(workdir, into=None):
    """Write each kernel argument's byte size, and its rows cols elem_bytes with every leading
    axis folded into the rows, in argument order, where TOGSim reads them to lay it out in DRAM."""
    import math

    manifest = contract.kernel_object(workdir)
    if manifest is None:
        raise FileNotFoundError(f"{workdir} has no kernel.json -- compile the kernel first")
    into = into or contract.dir_for(workdir)
    with open(os.path.join(into, TENSORS_TXT), "w") as f:
        f.write("".join(f"{_get_arg_bytes(a)}\n" for a in manifest["args"]))
    with open(os.path.join(into, TENSOR_SHAPES_TXT), "w") as f:
        f.write("".join(f"{math.prod(a['shape'][:-1])} {a['shape'][-1] if a['shape'] else 1} {_get_itemsize(a)}\n"
                        for a in manifest["args"]))


def _get_itemsize(arg):
    import torch
    return getattr(torch, arg["dtype"]).itemsize


def _get_arg_bytes(arg):
    import math
    return math.prod(arg["shape"]) * _get_itemsize(arg)
