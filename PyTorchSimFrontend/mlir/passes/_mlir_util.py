"""Small, dependency-light helpers shared across the MLIR passes.

Every pass had its own copy of the same op-walk generator (named variously
`_iter_ops` / `_walk` / `_walk_ops`) and the same one-line attribute builders
(`_i32` / `_i64` / ...). This module is the single source for both.

Import-safety: `walk_ops` is pure block/op attribute access and needs no MLIR
bindings, so this module does NOT import `mlir.ir` at top level -- some passes
(e.g. lower_vlane_idx, decompose_transfer) are deliberately importable without
the bindings present and only touch `mlir.ir` inside their run functions. The
attribute builders therefore import `mlir.ir` lazily; they require an active
MLIR context (the caller's `with ctx:`), exactly as the per-pass copies did.
"""


def walk_ops(block):
    """Yield every op under `block` in program order, recursing into regions.

    Snapshots each block's operation list, so a caller may erase ops while
    iterating (the strictest of the former copies; a superset of the rest)."""
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from walk_ops(b)


def _ir():
    import mlir.ir as ir
    return ir


def i32(v):
    """`i32` IntegerAttr for `v` (uses the active MLIR context)."""
    ir = _ir()
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def i64(v):
    """`i64` IntegerAttr for `v`."""
    ir = _ir()
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(v))


def i64_array(vals):
    """ArrayAttr of `i64` IntegerAttrs for `vals`."""
    ir = _ir()
    i = ir.IntegerType.get_signless(64)
    return ir.ArrayAttr.get([ir.IntegerAttr.get(i, int(v)) for v in vals])


def str_attr(v):
    """StringAttr of `str(v)`."""
    ir = _ir()
    return ir.StringAttr.get(str(v))


# attribute readers -- accept an OpView or an Operation; `default` is returned when
# `key` is absent.
def _attrs(op):
    return getattr(op, "operation", op).attributes


def attr_int(op, key, default=None):
    """Integer value of `op`'s `key` attribute, or `default` if absent."""
    ir = _ir()
    a = _attrs(op)
    return ir.IntegerAttr(a[key]).value if key in a else default


def attr_bool(op, key, default=False):
    """Bool value of `op`'s `key` attribute, or `default` if absent."""
    ir = _ir()
    a = _attrs(op)
    return bool(ir.BoolAttr(a[key]).value) if key in a else default


def attr_i64_array(op, key, default=None):
    """`op`'s `key` ArrayAttr of integers as a Python list, or `default` if
    absent (pass `default=[]` for the "missing -> empty" convention)."""
    ir = _ir()
    a = _attrs(op)
    return ([ir.IntegerAttr(x).value for x in ir.ArrayAttr(a[key])]
            if key in a else default)
