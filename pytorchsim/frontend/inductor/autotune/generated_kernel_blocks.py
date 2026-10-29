"""The blocks of a kernel Inductor generates from nodes (any but a template) on this machine, from its
numels, widest dtype and contiguous axis through tiling.fixed_config_for: blocks_for is the one place
they are computed; and whether a reduction is persistent or splits over the cores."""

from pytorchsim import config
from pytorchsim.frontend import tiling
from pytorchsim.frontend.inductor import is_lowering_npu

logger = config.setup_logger()


def pinned_blocks(kernel_tiling, features):
    """This machine's blocks for a kernel of `kernel_tiling` as a FixedTritonConfig, or None (logged)
    when its numels are not static."""
    from torch._inductor.codegen.triton import FixedTritonConfig

    if features is None or not kernel_tiling:
        return None
    try:
        numels = {f"{prefix}numel": int(n) for prefix, n in kernel_tiling.items()}
    except TypeError as e:
        logger.info("[torchsim-compile] no pinned blocks for tiling %s: %s", dict(kernel_tiling), e)
        return None
    cfg = {k: int(v) for k, v in blocks_for(numels, features).items() if v and k.endswith("BLOCK")}
    return FixedTritonConfig(cfg) if cfg else None


def blocks_for(numels, features):
    """tiling.fixed_config_for for a scheduled kernel of `numels` (<prefix>numel -> int)."""
    shim = type("K", (), {"inside_reduction": bool(features.is_reduction()), "numels": numels})()
    return tiling.fixed_config_for(shim, numels, _widest_dtype_args(features),
                                   contiguous=_kept_axis_contiguous(features)) or {}


def use_persistent_reduction(features, upstream):
    """Inductor's answer (`upstream`) narrowed: persistent only when the whole reduction fits this
    machine's reduction block."""
    if not upstream or not is_lowering_npu():
        return upstream
    try:
        extent = int(features.reduction_numel)
    except (TypeError, ValueError):
        return upstream
    if extent < 1:
        return upstream
    persistent = 1 << (extent - 1).bit_length()
    try:
        numels = {"xnumel": int(features.numel), "r0_numel": extent}
    except TypeError:
        logger.info("[torchsim-compile] a reduction of %s rows is not static: persistent "
                    "only within one reduction block", features.numel)
        return persistent <= tiling.reduction_block_for(extent)
    return persistent <= int(blocks_for(numels, features)["R0_BLOCK"])


def reduction_split(reduction_numel_hint, numel_hint, upstream):
    """Split a reduction only when a core would otherwise sit idle, and at most cores ways:
    `upstream()` is Inductor's split, asked only on a machine of more than one core."""
    cores = _num_cores()
    if cores <= 1:
        return 1
    want = upstream()
    try:
        if int(numel_hint) >= cores:
            return 1
    except (TypeError, ValueError):
        return 1
    split = min(int(want), cores)
    if split > 1:
        logger.info("[torchsim-compile] splitting a reduction of %s elements %s ways over %s cores "
                    "(upstream wanted %s)", reduction_numel_hint, split, cores, want)
        return split
    return 1


def _num_cores():
    """This machine's core count, `num_cores` of its TOGSim config (every machine file has it)."""
    return max(1, int(config.CONFIG_NUM_CORES))


def _kept_axis_contiguous(features):
    """'x' when the largest read (in bytes) over every index of a reduction runs unit-stride along the
    innermost kept axis -- the one x's index walks first -- from its MemoryDep strides; else None.
    Inductor's OUTER hint is not used: it is DEFAULT for any kernel of num_sm * 64 rows or more."""
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.virtualized import V
    if not features.is_reduction():
        return None
    best = None
    for node in features.reduction_nodes():
        kept, reduced = node._sizes
        for dep in node.read_writes.reads:
            if not isinstance(dep, MemoryDep) or len(dep.var_names) != len(kept) + len(reduced):
                continue
            strides = V.graph.sizevars.stride_hints(dep.index, dep.var_names)
            unit = [i for i, s in enumerate(strides) if s == 1]
            if not unit:
                continue
            size = dep.numbytes_hint()
            if best is None or size > best[0]:
                best = (size, len(kept) - 1 in unit)
    return "x" if best and best[1] else None


_WIDTH_NAME = {64: "float64", 32: "float32", 16: "float16", 8: "int8"}


def _widest_dtype_args(features):
    """The widest tensor a scheduled node makes or reads, as fixed_config_for's args; a node with no dtype
    (no tensor), a read that names no buffer (Inductor's KeyError) and a buffer of fewer elements than the
    machine's lanes, which no block tile holds (a gather's indices, a scalar), are not counted."""
    from torch._inductor.virtualized import V
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    dtypes = []
    for nd in features.scheduler_nodes():
        dtypes.append(nd.node.maybe_get_dtype())
        for dep in nd.read_writes.reads:
            try:
                dtype, numel = V.graph.get_dtype(dep.name), V.graph.get_numel(dep.name)
            except KeyError:
                continue
            try:
                if int(V.graph.sizevars.optimization_hint(numel)) < lanes:
                    continue
            except (TypeError, ValueError):
                pass
            dtypes.append(dtype)
    bits = [8 * d.itemsize for d in dtypes if d is not None]
    return [{"dtype": _WIDTH_NAME[max(bits)]}] if bits else []
