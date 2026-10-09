"""Inductor's Triton source -> source the compiler venv can compile.

Two rewrites: drop what a torch-free venv cannot import, and replace the mm/bmm
templates' modulo wrap with a load mask so the operand stays a descriptor.
"""

import re

from PyTorchSimFrontend import extension_config

from . import triton_helpers_src
from .errors import SpecIncomplete

logger = extension_config.setup_logger()

_HEURISTIC_RE = re.compile(r"^@triton_heuristics\.")
_DROP_IMPORT_RE = re.compile(
    r"^\s*(import torch|from torch\b|from __future__|import __main__)")
_DROP_CALL_RE = re.compile(r"^\s*triton_helpers\.set_driver_to_gpu\(\)")
_HELPER_USE_RE = re.compile(r"\btriton_helpers\.(\w+)")


def strip_for_compiler(src):
    """Remove everything the torch-free the compiler venv cannot import.

    Drops torch/inductor imports and the @triton_heuristics decorator, then
    re-adds the imports and vendored helpers the stripped body still needs.
    """
    lines = src.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if _HEURISTIC_RE.match(line.strip()) or _HEURISTIC_RE.match(line):
            while i < len(lines) and lines[i].strip() != "@triton.jit":
                i += 1
            continue
        if _DROP_IMPORT_RE.match(line) or _DROP_CALL_RE.match(line):
            i += 1
            continue
        out.append(line)
        i += 1
    body = "\n".join(out)

    used = sorted(set(_HELPER_USE_RE.findall(body)))
    unvendored = [h for h in used if h not in triton_helpers_src.VENDORED]
    if unvendored:
        raise SpecIncomplete(
            f"kernel uses triton_helpers.{{{','.join(unvendored)}}}, which lives "
            f"in torch and the compiler venv has no torch. Add it to "
            f"triton_helpers_src if it is pure triton, or lower it another way.")

    prefix = ""
    if "import triton.language as tl" not in body:
        prefix = "import triton\nimport triton.language as tl\n\n"
    if re.search(r"\btl_math\.", body):
        prefix += "from triton.language import math as tl_math\n"
    if re.search(r"\blibdevice\.", body):
        prefix += "from triton.language.extra import libdevice\n"
    if used:
        prefix += triton_helpers_src.SRC
    return prefix + body


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
        inserted, applied = [], {}
        for idx, dim, _block in _WRAP_TRIPLES:
            if not needs.get(idx):
                continue
            name = _ROW_MASK if idx == "rm" else _COL_MASK
            slice_ = "[:, None]" if idx == "rm" else "[None, :]"
            inserted.append(f"{pad}{name} = {idx}{slice_} < {dim}")
        lines[anchor + 1:anchor + 1] = inserted
        for ptr, name in _MASK_FOR.items():
            if name in "".join(inserted):
                applied[ptr] = _add_mask_to_loads(lines, ptr, name)
        if any(v == 0 for v in applied.values()):
            logger.warning(
                "[torchsim-compile] %s: could not attach a bound to every load; "
                "leaving the wrap in place", kernel_name or "kernel")
            return body

    body = "\n".join(lines)
    for idx, dim, block in _WRAP_TRIPLES:
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
