from __future__ import annotations

from dataclasses import dataclass

SYSTOLIC = "systolic"
REDUCING = "reducing"
POINTWISE = "pointwise"


@dataclass(frozen=True)


class KernelRef:
    directory: str
    stem: str
    tile_class: str
    size_names: tuple = ()
    tile_names: tuple = ()
    hw_names: tuple = ()

#: op_type -> its kernel.
OPS: dict[str, list[KernelRef]] = {
    "Gemm":   [KernelRef("gemm", "gemm", SYSTOLIC, ("M", "K", "N"),
                         ("TM", "TN", "TK"), ("SA", "VLANE"))],
    "MatMul": [KernelRef("gemm", "gemm", SYSTOLIC, ("M", "K", "N"),
                         ("TM", "TN", "TK"), ("SA", "VLANE"))],

    "Conv": [KernelRef("conv", "conv", SYSTOLIC,
                       ("I_C", "I_H", "I_W", "O_C", "O_H", "O_W", "K_H", "K_W"),
                       ("TILE_M", "TILE_N", "TILE_K"), ("SA", "VLANE"))],

    "Attention": [KernelRef("attention", "attention", SYSTOLIC,
                            ("HEADS", "KV_HEADS", "SEQ", "DHEAD"), (), ("SA",))],
    "MultiHeadAttention": [KernelRef("attention", "attention", SYSTOLIC,
                                     ("HEADS", "KV_HEADS", "SEQ", "DHEAD"), (),
                                     ("SA",))],

    "Softmax": [
        KernelRef("softmax", "softmax", REDUCING, ("M", "N"), ("TM", "TN"))],

    "LayerNormalization": [
        KernelRef("layernorm", "layernorm", REDUCING, ("M", "N"), ("TM", "TN"))],

    "EmbedLayerNormalization": [
        KernelRef("embed_layernorm", "embed_layernorm", REDUCING,
                  ("SEQ", "DIM"), ("TSEQ", "TDIM"))],

    "GlobalAveragePool": [
        KernelRef("global_avgpool", "global_avgpool", REDUCING,
                  ("C", "HW"), ("TC", "THW"))],

    "Relu":  [KernelRef("bias_act", "bias_act", POINTWISE, ("M", "N"), ("TM", "TN"))],
    "Add":   [KernelRef("bias_act", "bias_act", POINTWISE, ("M", "N"), ("TM", "TN"))],
    "Gelu":  [KernelRef("bias_gelu", "bias_gelu", POINTWISE, ("M", "N"), ("TM", "TN"))],

    "MaxPool": [KernelRef("maxpool", "maxpool", POINTWISE,
                          ("ROWS", "COLS"), ("TROW", "TCOL"))],
    "AveragePool": [KernelRef("adaptive_avgpool", "adaptive_avgpool", POINTWISE,
                              ("ROWS", "COLS"), ("TROW", "TCOL"))],

    "Concat":  [KernelRef("concat", "concat", POINTWISE, ("A_ROWS",), ("TROW",))],
    "Flatten": [KernelRef("flatten", "flatten", POINTWISE, (), ())],
}

#: Fused nodes with no kernel, and the operators they decompose to.
DECOMPOSE: dict[str, list[str]] = {
    "FusedMatMul":              ["MatMul"],
    "FusedConv":                ["Conv", "Relu"],
    "BiasGelu":                 ["Add", "Gelu"],
    "FastGelu":                 ["Add", "Gelu"],
    "SkipLayerNormalization":   ["Add", "LayerNormalization"],
}

#: Metadata-only operators: no compute node to charge.
NO_COMPUTE = frozenset({
    "Reshape", "Squeeze", "Unsqueeze", "Identity", "Constant", "Shape",
    "Dropout", "Cast", "Transpose", "Gather", "Slice", "ConstantOfShape",
})


def resolve(op_type: str, _depth: int = 0) -> list[KernelRef] | None:
    if op_type in NO_COMPUTE:
        return []
    if op_type in OPS:
        return list(OPS[op_type])
    if op_type in DECOMPOSE and _depth < 4:
        out: list[KernelRef] = []
        for part in DECOMPOSE[op_type]:
            got = resolve(part, _depth + 1)
            if got is None:
                return None
            out.extend(got)
        return out
    return None
