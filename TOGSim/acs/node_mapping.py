from __future__ import annotations

from dataclasses import dataclass

ELEMENTWISE = "elementwise"
REDUCTION = "reduction"

#: max/min: compare, then select.
MAXMIN = 2


@dataclass(frozen=True)


class Op:
    kind: str
    n: int          # instructions per element (elementwise) or per fold step
    comment: str


@dataclass(frozen=True)


class NodeDesc:
    name: str
    compute_type: str       # "vector" | "matmul" | "preload"
    ops: tuple = ()


@dataclass(frozen=True)


class KernelDesc:
    name: str
    nodes: tuple


def _vec(name, *ops):
    return NodeDesc(name, "vector", tuple(ops))

#: Kernel name -> its compute nodes, in the order the .cpp emits them.
KERNELS: dict[str, KernelDesc] = {

    "gemm": KernelDesc("gemm", (
        _vec("acc_init", Op(ELEMENTWISE, 1, "zero the accumulator")),
        NodeDesc("preload", "preload"),
        NodeDesc("matmul", "matmul"),
    )),

    "conv": KernelDesc("conv", (
        _vec("acc_init", Op(ELEMENTWISE, 1, "zero the accumulator")),
        NodeDesc("preload", "preload"),
        NodeDesc("matmul", "matmul"),
    )),

    "softmax": KernelDesc("softmax", (
        _vec("max_reduce", Op(ELEMENTWISE, MAXMIN, "running max")),
        _vec("max_write", Op(REDUCTION, MAXMIN, "fold the max")),
        _vec("sum_reduce",
             Op(ELEMENTWISE, 1, "x - max"),
             Op(ELEMENTWISE, 1, "exp"),
             Op(ELEMENTWISE, 1, "accumulate")),
        _vec("sum_write", Op(REDUCTION, 1, "fold the sum")),
        _vec("softmax",
             Op(ELEMENTWISE, 1, "x - max"),
             Op(ELEMENTWISE, 1, "exp"),
             Op(ELEMENTWISE, 1, "/ sum")),
    )),

    "layernorm": KernelDesc("layernorm", (
        _vec("stats_reduce",
             Op(ELEMENTWISE, 1, "accumulate x"),
             Op(ELEMENTWISE, 2, "accumulate x*x")),
        _vec("write_mean", Op(REDUCTION, 1, "fold the sum")),
        _vec("write_var", Op(REDUCTION, 1, "fold the sum of squares")),
        _vec("normalize",
             Op(ELEMENTWISE, 1, "x - mean"),
             Op(ELEMENTWISE, 1, "* rstd"),
             Op(ELEMENTWISE, 2, "scale and shift")),
    )),

    "embed_layernorm": KernelDesc("embed_layernorm", (
        _vec("gather",
             Op(ELEMENTWISE, 1, "gather the embedding"),
             Op(ELEMENTWISE, 1, "x - mean"),
             Op(ELEMENTWISE, 1, "* rstd"),
             Op(ELEMENTWISE, 2, "scale and shift")),
    )),

    "global_avgpool": KernelDesc("global_avgpool", (
        _vec("sum_reduce", Op(ELEMENTWISE, 1, "accumulate")),
        _vec("sum_write", Op(REDUCTION, 1, "fold the sum")),
        _vec("scale", Op(ELEMENTWISE, 1, "* 1/HW")),
    )),

    "bias_act": KernelDesc("bias_act", (
        _vec("bias_act",
             Op(ELEMENTWISE, 1, "+ bias"),
             Op(ELEMENTWISE, 1, "max(x, 0)")),
    )),

    "bias_gelu": KernelDesc("bias_gelu", (
        _vec("bias_gelu",
             Op(ELEMENTWISE, 1, "+ bias"),
             Op(ELEMENTWISE, 1, "x / sqrt(2)"),
             Op(ELEMENTWISE, 2, "erf"),
             Op(ELEMENTWISE, 1, "1 + erf"),
             Op(ELEMENTWISE, 1, "* 0.5x")),
    )),

    "maxpool": KernelDesc("maxpool", (
        _vec("maxpool", Op(ELEMENTWISE, MAXMIN, "running max over the window")),
    )),

    "adaptive_avgpool": KernelDesc("adaptive_avgpool", (
        _vec("avgpool",
             Op(ELEMENTWISE, 1, "accumulate the window"),
             Op(ELEMENTWISE, 1, "* 1/n")),
    )),

    # pure DMA: no compute node
    "flatten": KernelDesc("flatten", ()),

    "attention": KernelDesc("attention", (
        NodeDesc("qk_preload", "preload"),
        NodeDesc("qk_matmul", "matmul"),
        _vec("rowmax", Op(REDUCTION, MAXMIN, "running row max")),
        _vec("sub", Op(ELEMENTWISE, 1, "s - max")),
        _vec("exp", Op(ELEMENTWISE, 1, "exp")),
        _vec("rowsum", Op(REDUCTION, 1, "fold the row sum")),
        _vec("mac", Op(ELEMENTWISE, 2, "running l")),
        _vec("rescale", Op(ELEMENTWISE, 2, "rescale the accumulator")),
        NodeDesc("sv_preload", "preload"),
        NodeDesc("sv_matmul", "matmul"),
        _vec("normalize",
             Op(ELEMENTWISE, 1, "/ l"),
             Op(ELEMENTWISE, 1, "scale")),
    )),

    "concat": KernelDesc("concat", ()),

    "kvcache_concat": KernelDesc("kvcache_concat", (
        _vec("split", Op(ELEMENTWISE, 1, "split QKV into query, key, value")),
    )),
}


def lookup(kernel_name: str) -> KernelDesc:
    try:
        return KERNELS[kernel_name]
    except KeyError:
        raise KeyError(
            f"no compute-node table for kernel {kernel_name!r}. Add one to "
            f"acs/node_mapping.py: without it the cost of every node in this "
            f"kernel would be invented.") from None
