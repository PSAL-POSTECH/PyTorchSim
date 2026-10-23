"""The blocks of an Inductor kernel on this machine: the kernel's numels, widest dtype and contiguous
axis read off Inductor's scheduler nodes, handed to tiling.fixed_config_for."""

import torch

from pytorchsim import config
from pytorchsim.frontend import tiling

logger = config.setup_logger()


def pinned_blocks(kernel_tiling, features):
    """This machine's blocks for a kernel of `kernel_tiling` as a FixedTritonConfig, or None.

    The numels come from the tiling; the dtype width from the widest tensor a scheduled node makes
    or reads; whether the kept axis is the contiguous one from Inductor's hint.
    """
    from torch._inductor.codegen.triton import FixedTritonConfig

    numels = {}
    for prefix, n in (kernel_tiling or {}).items():
        try:
            numels[f"{prefix}numel"] = int(n)
        except Exception:
            pass
    if not numels or features is None:
        return None
    args = _elem_args(features)
    shim = type("K", (), {"inside_reduction": bool(features.is_reduction()),
                          "numels": numels})()
    try:
        cfg = tiling.fixed_config_for(shim, numels, args,
                                      contiguous=_kept_axis_contiguous(features)) or {}
    except Exception as e:  # noqa: BLE001
        logger.info("[torchsim-compile] no fixed_config for this kernel: %s", e)
        return None
    cfg = {k: int(v) for k, v in cfg.items() if v and k.endswith("BLOCK")}
    return FixedTritonConfig(cfg) if cfg else None


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


_DT_BITS = {torch.float64: 64, torch.float32: 32, torch.float16: 16,
            torch.bfloat16: 16, torch.int64: 64, torch.int32: 32,
            torch.int16: 16, torch.int8: 8, torch.uint8: 8, torch.bool: 8}


def _elem_args(features):
    """The widest tensor a scheduled node makes or reads, as fixed_config_for's args."""
    from torch._inductor.virtualized import V
    bits = []
    for nd in features.scheduler_nodes():
        try:
            bits.append(_DT_BITS.get(nd.node.get_dtype(), 32))
        except Exception:
            pass
        for dep in getattr(getattr(nd, "read_writes", None), "reads", ()) or ():
            try:
                bits.append(_DT_BITS.get(V.graph.get_dtype(dep.name), 32))
            except Exception:
                pass
    return ([{"dtype": {64: "float64", 32: "float32", 16: "float16",
                        8: "int8"}[max(bits)]}] if bits else [])
