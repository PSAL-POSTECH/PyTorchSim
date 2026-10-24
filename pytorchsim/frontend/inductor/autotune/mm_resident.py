"""The resident mm, as v6e XLA runs a GEMM: one program loads whole the operand copy_node.plan holds, then
streams the other two output tiles a trip (two buffers), K whole, each tile stored through store_output so
an epilogue fuses per tile. Offered alone where the rule holds."""

import torch
from torch._inductor.select_algorithm import TritonTemplate

from pytorchsim import config

logger = config.setup_logger()

def _stream(held, streamed):
    """The kernel body once `held` ("A" or "B") is loaded: a loop over the output tiles of the other, TILES of
    them a trip, the second only where it starts inside the output."""
    ext, blk = ("N", "BLOCK_N") if held == "A" else ("M", "BLOCK_M")
    ld = {"A": "a = tl.load(A + rm[:, None] * stride_am + rk[None, :] * stride_ak,\n"
               "{pad}    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)",
          "B": "b = tl.load(B + rk[:, None] * stride_bk + rn[None, :] * stride_bn,\n"
               "{pad}    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)"}
    idx = {"A": "rm", "B": "rn"}
    lines = [f"    {idx[held]} = tl.arange(0, {'BLOCK_M' if held == 'A' else 'BLOCK_N'}).to(INDEX_DTYPE)",
             "    " + ld[held].format(pad="    "),
             f"    for t in range(0, tl.cdiv({ext}, TILES * {blk})):"]
    for h in (0, 1):
        pad = " " * (8 if h == 0 else 12)
        body = [f"{idx[streamed]} = (TILES * t + {h}) * {blk} + tl.arange(0, {blk}).to(INDEX_DTYPE)",
                ld[streamed].format(pad=pad),
                "acc = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)",
                "out_m = rm[:, None]", "out_n = rn[None, :]", "mask = (out_m < M) & (out_n < N)",
                '{{store_output(("out_m", "out_n"), "acc", "mask", indent_width=%d, val_shape=("BLOCK_M", "BLOCK_N"))}}'
                % len(pad)]
        if h == 1:
            lines.append("{% if TILES == 2 %}")
            lines.append(f"        if (TILES * t + 1) * {blk} < {ext}:")
        lines += [pad + line for line in body]
        if h == 1:
            lines.append("{% endif %}")
    return "\n".join(lines) + "\n"


SOURCE = """{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    rk = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)
{% if RESIDENT == 0 %}
""" + _stream("A", "B") + "{% else %}\n" + _stream("B", "A") + "{% endif %}\n"


def grid(m, n, meta, *, cdiv=None):
    """One program: it walks every output tile itself, whatever the launch."""
    return (1, 1, 1)


mm_resident_template = TritonTemplate(name="mm_resident", grid=grid, source=SOURCE)


def plan(m, n, k, in_bytes, out_bytes):
    """The compiler's copy_node.plan of [m, k] @ [k, n] on this machine: which operand is held and the tile.
    """
    from pytorchsim.backend import compiler_bridge
    from pytorchsim.frontend import tiling
    machine = config.CONFIG_MACHINE
    return compiler_bridge.compiler_contract("copy_node").plan(
        m, n, k, in_bytes, 4, out_bytes, int(machine.vector_lanes), machine.spad_size, tiling.triton_max_numel(),
        machine.matmul_width)


def blocks(m, n, k, got):
    """BLOCK_M, BLOCK_N, BLOCK_K of a plan: the held operand's whole extents, the streamed one's tile, K
    whole."""
    from pytorchsim.backend import compiler_bridge
    pad = compiler_bridge.compiler_contract("copy_node").block
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    held_m = got.resident == 0
    return dict(BLOCK_M=pad(m, lanes) if held_m else got.tile, BLOCK_N=got.tile if held_m else pad(n, lanes),
                BLOCK_K=pad(k, lanes))


DOT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def templates_for(kernel_inputs, templates, op_name):
    """`templates` for one npu mm: only the resident mm where copy_node.plan holds an operand, else as
    offered; the decision and why are logged. Other ops, devices and templates are left as they are."""
    from torch._inductor.kernel.mm import mm_template
    if op_name != "mm" or kernel_inputs.device_type != "npu" or mm_template not in templates:
        return templates
    why, got = decline(kernel_inputs)
    if why is not None:
        logger.info("[mm_resident] %s: the mm template, %s", _label(kernel_inputs), why)
        return templates
    logger.info("[mm_resident] %s: %s", _label(kernel_inputs), got.why)
    return [mm_resident_template]


def decline(kernel_inputs):
    """(why the resident mm does not take this mm, or None; the plan)."""
    dtypes = [n.get_dtype() for n in kernel_inputs.nodes()]
    if len(set(dtypes)) != 1 or dtypes[0] not in DOT_DTYPES:
        return f"operands {dtypes}", None
    if kernel_inputs.out_dtype() != dtypes[0]:
        return f"out_dtype {kernel_inputs.out_dtype()}", None
    try:
        m, n, k = _mnk(kernel_inputs)
    except (TypeError, ValueError) as e:
        return f"a size that is not static ({e})", None
    got = plan(m, n, k, dtypes[0].itemsize, kernel_inputs.out_dtype().itemsize)
    return (got.why if got.resident is None else None), got


def _mnk(kernel_inputs):
    """m, n, k as integers: a static size as it is, else the graph's hint for it."""
    import sympy
    from torch._inductor.virtualized import V
    return [int(s) if isinstance(s, (int, sympy.Integer)) else int(V.graph.sizevars.optimization_hint(s))
            for s in kernel_inputs.mnk_symbolic()]


def _label(kernel_inputs):
    try:
        m, n, k = _mnk(kernel_inputs)
        return f"{m}x{k}x{n}"
    except (TypeError, ValueError):
        return str(kernel_inputs.mnk_symbolic())


def configs(kernel_inputs):
    """The resident mm's one config for this mm: the plan's operand, its blocks, fp32 accumulation."""
    why, got = decline(kernel_inputs)
    if why is not None:
        return
    m, n, k = _mnk(kernel_inputs)
    extent = n if got.resident == 0 else m
    yield dict(RESIDENT=got.resident, TILES=2 if extent > got.tile else 1, ACC_TYPE="tl.float32", ALLOW_TF32=False,
               num_stages=1, num_warps=4, **blocks(m, n, k, got))
