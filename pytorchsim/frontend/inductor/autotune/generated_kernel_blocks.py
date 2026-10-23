"""The blocks of an Inductor kernel on this machine: the kernel's numels, widest dtype and contiguous
axis read off Inductor's scheduler nodes, handed to tiling.fixed_config_for. blocks_for is the one
place they are computed; pinning, persistence and the compile metadata all read it."""

from pytorchsim import config
from pytorchsim.frontend import tiling

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


def _kept_axis_contiguous(features):
    """'x' when the largest read (in bytes) over every index of a reduction runs unit-stride along a
    kept axis, from its MemoryDep strides; else None. Inductor's OUTER hint is not used: it is left
    DEFAULT for any kernel of at least num_sm * 64 rows, whatever its strides."""
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
                best = (size, unit[0] < len(kept))
    return "x" if best and best[1] else None


_WIDTH_NAME = {64: "float64", 32: "float32", 16: "float16", 8: "int8"}


def _widest_dtype_args(features):
    """The widest tensor a scheduled node makes or reads, as fixed_config_for's args; a node or
    read with no dtype (no tensor) is not counted."""
    from torch._inductor.virtualized import V
    dtypes = []
    for nd in features.scheduler_nodes():
        try:
            dtypes.append(nd.node.get_dtype())
        except Exception:  # noqa: BLE001
            pass
        for dep in getattr(getattr(nd, "read_writes", None), "reads", ()) or ():
            try:
                dtypes.append(V.graph.get_dtype(dep.name))
            except Exception:  # noqa: BLE001
                pass
    bits = [8 * d.itemsize for d in dtypes if d is not None]
    return [{"dtype": _WIDTH_NAME[max(bits)]}] if bits else []
