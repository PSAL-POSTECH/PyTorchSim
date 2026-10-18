"""What the compiler is handed for one kernel: the module Inductor generated, and the KernelSpec file beside it.

The module is Inductor's own, imported the way Inductor imports it; the only rewrite is the mm/bmm
templates' modulo wrap turned into a load mask. `write` puts both files in the kernel's workdir.
"""

import os
import re

from pytorchsim import config
from pytorchsim import contract
from pytorchsim.contract import SpecIncomplete
from pytorchsim.driver import compiler_bridge

logger = config.setup_logger()

_WRAP_TRIPLES = (("rm", "M", "BLOCK_M"), ("rn", "N", "BLOCK_N"))


_ROW_MASK = "_torchsim_row_mask"


_COL_MASK = "_torchsim_col_mask"


_MASK_FOR = {"A": _ROW_MASK, "B": _COL_MASK}


def _literal_int(body, name):
    """`name`'s value if the kernel assigns it an integer literal, else None."""
    m = re.search(rf"^\s*{name}\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)\s*$",
                  body, re.M)
    return int(m.group(1)) if m else None


def _load_re(ptr):
    """`tl.load(<ptr>` in either spelling: mm builds the address, bmm the pointer."""
    return re.compile(rf"\btl\.load\({ptr}\b")


def _add_mask_to_loads(lines, ptr, mask_name):
    """Give every `tl.load(<ptr>...)` the bound `mask_name`. Returns the count.

    An existing mask is widened with `&` rather than matched by regex: it is an
    expression holding both a comma and a paren, so it ends at ` other=`.
    """
    rx = _load_re(ptr)
    n = 0
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not rx.search(stripped) or not stripped.endswith(")"):
            continue
        at = stripped.find("mask=")
        if at < 0:
            lines[i] = stripped[:-1] + f", mask={mask_name}, other=0.0)"
            n += 1
            continue
        start = at + len("mask=")
        end = stripped.find(", other=", start)
        if end < 0:
            end = len(stripped) - 1
        expr = stripped[start:end]
        lines[i] = (stripped[:start] + f"({expr}) & {mask_name}"
                    + stripped[end:])
        n += 1
    return n


def clamp_instead_of_wrap(body, kernel_name=""):
    """Replace the mm/bmm templates' `rm % M` / `rn % N` with a load mask.

    A modulo on a pointer index is not a stride, so the operand becomes a gather;
    clamping keeps the descriptor. A dividing block needs neither and gets none.
    """
    lines = body.splitlines()
    text = "\n".join(lines)
    needs = {}
    for idx, dim, block in _WRAP_TRIPLES:
        d, b = _literal_int(text, dim), _literal_int(text, block)
        if d is None or not b:
            continue
        needs[idx] = bool(d % b)

    if not needs:
        return body

    if any(needs.values()):
        anchor = next((i for i, l in enumerate(lines)
                       if l.strip().startswith(("offs_k = tl.arange(",
                                                "rk = tl.arange("))), None)
        if anchor is None or not any(_load_re(p).search(text) for p in _MASK_FOR):
            logger.warning(
                "[torchsim-compile] %s: a block does not divide its dimension and "
                "this does not recognise the loads to bound; leaving the wrap",
                kernel_name or "kernel")
            return body

        pad = " " * (len(lines[anchor]) - len(lines[anchor].lstrip()))
        inserted, masks, applied = [], set(), {}
        for idx, dim, _block in _WRAP_TRIPLES:
            if not needs.get(idx):
                continue
            name = _ROW_MASK if idx == "rm" else _COL_MASK
            slice_ = "[:, None]" if idx == "rm" else "[None, :]"
            inserted.append(f"{pad}{name} = {idx}{slice_} < {dim}")
            masks.add(name)
        lines[anchor + 1:anchor + 1] = inserted
        for ptr, name in _MASK_FOR.items():
            if name in masks:
                applied[ptr] = _add_mask_to_loads(lines, ptr, name)
        if any(v == 0 for v in applied.values()):
            logger.warning(
                "[torchsim-compile] %s: could not attach a bound to every load; "
                "leaving the wrap in place", kernel_name or "kernel")
            return body

    body = "\n".join(lines)
    for idx, dim, _block in _WRAP_TRIPLES:
        if idx not in needs:
            continue
        body, n = re.subn(rf"\b{idx}\s*%\s*{dim}\b", idx, body)
        if n:
            logger.info(
                "[torchsim-compile] %s: replaced %d `%s %% %s` with %s, so the "
                "operand stays a descriptor instead of becoming a gather",
                kernel_name or "kernel", n, idx, dim,
                "a load bound" if needs[idx] else "nothing (the block divides)")
    return body


_C_TYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float"}


def scalar_args(meta):
    """Scalar parameters in kernel order, as [(name, c_type, default)].

    A numel's value is known at codegen; an input Inductor unwrapped to a scalar (a 0-d CPU
    tensor, Adam's step) only at launch, so its default is None and the launch passes it.
    """
    numels = meta["numels"]
    out = []
    for name, token in meta["signature"].items():
        if token.startswith("*") or token == "constexpr":
            continue
        ctype = _C_TYPE.get(token)
        if ctype is None:
            raise SpecIncomplete(
                f"{meta['kernel_name']}: scalar '{name}' has type {token!r}, "
                f"which has no C mapping in _C_TYPE")
        v = numels.get(name)
        out.append((name, ctype, None if v is None else int(v)))
    return out


SPEC_TEMPLATE = '''\
"""Generated by pytorchsim/driver/compiler_input.py -- do not edit.

Inductor kernel {kernel_name!r}: the compiler takes its @triton.jit function, with the block
sizes pinned as constexprs, so the launch shape is static. See compiler_input.py for why.
"""
import importlib.util
import os
import sys

sys.path.insert(0, {compiler_dir!r})
from {spec_module} import KernelSpec, Arg  # noqa: E402

#: The kernel's module, beside this file. It must be a REAL file on
#: disk, not an exec'd string: triton's @jit reads the function back with
#: inspect.getsourcefile and rejects anything else ("@jit functions should be
#: defined in a Python file").
TRITON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           {triton_module!r})


def kernel():
    # As torch/_inductor/runtime/compile_tasks.py loads it.
    from torch._inductor.runtime.triton_helpers import skip_gpu_driver_setup
    spec = importlib.util.spec_from_file_location(
        {kernel_name!r} + "_triton", TRITON_PATH)
    mod = importlib.util.module_from_spec(spec)
    with skip_gpu_driver_setup():
        spec.loader.exec_module(mod)
    return getattr(mod, {kernel_name!r}){jit_fn}


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {{}}
{make_inputs_body}
    return out


def reference(inputs):
    # The Inductor route has no per-kernel torch reference: correctness is
    # checked at the graph level by the test that ran torch.compile. the compiler's
    # stage 7 is therefore not meaningful here and the pipeline is driven to
    # stage 6 (spike) instead.
    return {{}}


SPEC = KernelSpec(
    name={kernel_name!r},
    kernel=kernel,
    signature={signature!r},
    constexprs={constexprs!r},
    args=[
{args_body}
    ],
    grid={grid!r},
    parallel_axes={parallel_axes!r},
    reference=reference,
    make_inputs=make_inputs,
    extra={{"scalar_args": {scalar_decls!r},
           "scalar_values": {scalar_values!r}}},
)
'''


def write(src_code, meta, path, compiler_dir):
    """Write the kernel's module and the kernel file for this Inductor kernel. Returns `path`.

    A user-defined kernel's source (meta["user_defined"]) is written as it is: the rewriter reads
    Inductor's generated source, not a model's.

    A block Inductor fixed in the kernel BODY rather than taking as a parameter
    is skipped: passing it would not match the signature.
    """
    missing = [a["name"] for a in meta["args"] if not a["dtype"] or not a["numel"]]
    if missing:
        raise SpecIncomplete(
            f"{meta['kernel_name']}: no dtype/numel for {missing} -- "
            f"collect_meta could not resolve them from V.graph")

    signature = dict(meta["signature"])
    constexprs = dict(meta["constants"])
    for k, v in (meta.get("fixed_config") or {}).items():
        if k not in signature:
            continue
        if v is None:
            raise SpecIncomplete(
                f"{meta['kernel_name']}: block size {k} is unset "
                f"(fixed_config_for leaves reduction blocks unset on purpose)")
        constexprs[k] = v
        signature[k] = "constexpr"

    args_body = "\n".join(
        f"        Arg({a['name']!r}, {a['role']!r}, {a['dtype']!r}, ({a['numel']},)),"
        for a in meta["args"])
    make_inputs_body = "\n".join(
        f"    out[{a['name']!r}] = torch.randn({a['numel']}, generator=g)"
        f".to(torch.{a['dtype']})"
        for a in meta["args"] if a["role"] in ("in", "inout")) or "    pass"

    triton_module = f"{meta['kernel_name']}_triton.py"
    with open(os.path.join(os.path.dirname(path), triton_module), "w") as f:
        f.write(src_code if meta.get("user_defined") else clamp_instead_of_wrap(
            src_code, meta["kernel_name"]))

    scalars = scalar_args(meta)
    text = SPEC_TEMPLATE.format(
        kernel_name=meta["kernel_name"],
        compiler_dir=compiler_dir,
        spec_module=f"{compiler_bridge.COMPILER_PKG}.contract.kernel_spec",
        triton_module=triton_module,
        jit_fn="" if meta.get("user_defined") else ".fn",
        signature=signature,
        constexprs=constexprs,
        args_body=args_body,
        make_inputs_body=make_inputs_body,
        grid=contract.grid_xyz(meta),
        parallel_axes=tuple(contract.launch_axes(meta)),
        scalar_decls=[(n, c) for n, c, _ in scalars],
        scalar_values={n: v for n, _, v in scalars if v is not None},
    )
    with open(path, "w") as f:
        f.write(text)
    return path
