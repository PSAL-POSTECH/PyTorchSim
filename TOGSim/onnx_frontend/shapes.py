from __future__ import annotations

from .onnx_ops import KernelRef, POINTWISE, REDUCING, SYSTOLIC
from .tiling import (Hardware, check_systolic, pointwise_tile, reducing_tile,
                     systolic_tile)


def rows_cols(shape: list[int]) -> tuple[int, int]:
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows, shape[-1]


def attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else a.i
    return default


def _matmul_dims(node, shapes) -> tuple[int, int, int]:
    a = shapes[node.input[0]]
    b = shapes[node.input[1]]
    m, k = rows_cols(a)
    if len(b) == 1:
        n = b[0]
    else:
        n = b[-1]
        if attr(node, "transB", 0):
            n = b[-2]
    return m, k, n


def _conv_dims(node, shapes) -> dict[str, int]:
    x = shapes[node.input[0]]
    w = shapes[node.input[1]]
    if attr(node, "group", 1) != 1:
        raise ValueError("grouped convolution: no library kernel takes a group count")
    o_c, _, k_h, k_w = w
    strides = attr(node, "strides", [1, 1])
    s = strides[0] if isinstance(strides, list) else strides
    pads = attr(node, "pads", [0, 0, 0, 0])
    p = pads[0] if isinstance(pads, list) else pads
    i_c, i_h, i_w = x[1], x[2], x[3]
    o_h = (i_h + 2 * p - k_h) // s + 1
    o_w = (i_w + 2 * p - k_w) // s + 1
    return dict(I_C=i_c, I_H=i_h, I_W=i_w, O_C=o_c,
                O_H=max(1, o_h), O_W=max(1, o_w), K_H=k_h, K_W=k_w)


def _attention_heads(node, q_shape) -> int:
    n = attr(node, "num_heads", 0)
    return int(n) if n else 1


def _pool_dims(node, shapes) -> tuple[int, int]:
    x = shapes[node.input[0]]
    kernel = attr(node, "kernel_shape", [1, 1])
    strides = attr(node, "strides", [1, 1])
    s = strides[0] if isinstance(strides, list) else strides
    if len(x) == 4:
        return x[1] * max(1, x[2] // max(1, s)), max(1, x[3] // max(1, s))
    return rows_cols(x)


def constants(node, kernel: KernelRef, shapes: dict, hw: Hardware) -> dict[str, int]:
    out: dict[str, int] = {}

    if kernel.directory == "conv":
        dims = _conv_dims(node, shapes)
        out.update(dims)
        m = dims["O_H"] * dims["O_W"]
        n = dims["O_C"]
        k = dims["I_C"] * dims["K_H"] * dims["K_W"]
        check_systolic(n, hw)
        tm, tn, tk = systolic_tile(m, n, k, hw)
        out.update(TILE_M=tm, TILE_N=tn, TILE_K=tk)

    elif kernel.stem == "attention":
        q = shapes[node.input[0]]
        seq = q[1] if len(q) >= 3 else q[0]
        heads = _attention_heads(node, q)
        kv_heads = int(attr(node, "num_kv_heads", 0)) or heads
        dhead = max(1, q[-1] // max(1, heads))
        out.update(HEADS=heads, KV_HEADS=kv_heads, SEQ=seq, DHEAD=dhead)

    elif kernel.tile_class == SYSTOLIC:
        m, k, n = _matmul_dims(node, shapes)
        check_systolic(n, hw)
        tm, tn, tk = systolic_tile(m, n, k, hw)
        out.update(M=m, K=k, N=n, TM=tm, TN=tn, TK=tk)

    elif kernel.stem == "global_avgpool":
        x = shapes[node.input[0]]
        c = x[1] if len(x) >= 2 else x[0]
        hw_elems = (x[2] * x[3]) if len(x) == 4 else 1
        tc, thw = reducing_tile(c, hw_elems, hw)
        out.update(C=c, HW=hw_elems, TC=tc, THW=thw)

    elif not kernel.size_names:
        pass

    else:
        if kernel.directory in ("maxpool", "adaptive_avgpool"):
            rows, cols = _pool_dims(node, shapes)
        else:
            rows, cols = rows_cols(shapes[node.input[0]])
        if kernel.tile_class == REDUCING:
            tm, tn = reducing_tile(rows, cols, hw)
        else:
            tm, tn = pointwise_tile(rows, cols, hw)
        # each kernel names these its own way
        size = kernel.size_names
        tiles = kernel.tile_names
        out[size[0]] = rows
        if len(size) > 1:
            out[size[1]] = cols
        if tiles:
            out[tiles[0]] = tm
        if len(tiles) > 1:
            out[tiles[1]] = tn

    for name in kernel.hw_names:
        out[name] = hw.sa if name == "SA" else hw.vlane
    return out


def tile_for_cost(kernel: KernelRef, values: dict[str, int]) -> dict[str, int]:
    if kernel.directory == "conv":
        return {"rows": values["TILE_M"],
                "elems": values["TILE_M"] * values["TILE_N"]}
    names = kernel.tile_names
    if not names:                      # flatten, attention: no tile constants
        return {"rows": values.get("SEQ", 1),
                "elems": values.get("SEQ", 1) * values.get("DHEAD", 1)}
    rows = values[names[0]]
    elems = rows * values[names[1]] if len(names) > 1 else rows
    return {"rows": rows, "elems": elems}
