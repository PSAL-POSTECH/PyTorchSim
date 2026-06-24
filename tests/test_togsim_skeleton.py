"""Tests for the C++ trace-generation front-end pieces (docs/design/togsim_cpp_trace.md).

Two layers:

* `test_togsim_ops_contract` runs anywhere (no MLIR bindings, no torch). It pins
  the skeleton+API vocabulary (`togsim_ops.py`) and checks it stays in lockstep
  with the runtime ABI header (`togsim_runtime.h`) -- the single thing most
  likely to silently drift.
* `test_build_skeleton_on_fixture` exercises the real `build_skeleton` pass, and
  is skipped unless the MLIR bindings are importable AND a post-vcix `.mlir`
  fixture is supplied via the `TOGSIM_SKELETON_FIXTURE` env var. (A valid
  build_tog-compatible fixture is hard to hand-write reliably; point this at a
  kernel dump from a real run.)
"""
import os
import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_OPS_PY = _ROOT / "PyTorchSimFrontend" / "mlir" / "passes" / "togsim_ops.py"
_HEADER = _ROOT / "TOGSim" / "include" / "togsim_runtime.h"


def _load_togsim_ops():
    spec = importlib.util.spec_from_file_location("togsim_ops", _OPS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_togsim_ops_contract():
    ts = _load_togsim_ops()
    header = _HEADER.read_text()

    # Every op maps to a callee, and every callee is the header's free function.
    assert set(ts.EMITC_CALLEE) == set(ts.OP_NAMES)
    for callee in ts.EMITC_CALLEE.values():
        assert callee in header, f"{callee} missing from togsim_runtime.h"

    # Entry point symbol agrees with the header.
    assert ts.ENTRY_SYMBOL == "togsim_kernel"
    assert ts.ENTRY_SYMBOL in header

    # Runtime callee emitted directly by lower_to_emitc: the work-item dispatch
    # wrapper. (The outlined tile fn TILE_SYMBOL is producer-generated.)
    assert ts.DISPATCH_CALLEE in header

    # Direction enum agrees with the header's togsim_dma_dir.
    assert (ts.DIR_LOAD, ts.DIR_STORE) == (0, 1)
    assert "TOGSIM_DMA_LOAD  = 0" in header
    assert "TOGSIM_DMA_STORE = 1" in header


def _mlir_available():
    return importlib.util.find_spec("mlir") is not None


@pytest.mark.skipif(not _mlir_available(), reason="MLIR Python bindings not installed")
def test_build_skeleton_on_fixture():
    fixture = os.environ.get("TOGSIM_SKELETON_FIXTURE")
    if not fixture or not os.path.isfile(fixture):
        pytest.skip("set TOGSIM_SKELETON_FIXTURE to a post-vcix kernel .mlir")

    import sys
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import build_skeleton

    import mlir.ir as ir
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(pathlib.Path(fixture).read_text(), ctx)
        report = build_skeleton.build_skeleton(module)
        out = str(module)

    # The data-movement ops are gone; the API ops took their place.
    assert "memref.dma_start" not in out
    assert "memref.dma_wait" not in out
    assert "togsim.dma" in out
    assert "togsim.memory_barrier" in out   # the explicit async-DMA sync (was dma_wait)
    assert "event_id" not in out            # static pairing replaced by the runtime tag
    # Loop skeleton is preserved.
    assert ("affine.for" in out) or ("scf.for" in out)
    assert module.operation.verify()
    print(report)


@pytest.mark.skipif(not _mlir_available(), reason="MLIR Python bindings not installed")
def test_strip_accum_terms_drops_reduction_marker():
    """Regression: the dma_wait tag index built by lower_to_vcix carries a `-d_i`
    term for each accumulation (reduction) loop var -- a sentinel marker, not an
    offset. build_skeleton must drop those so a memory_barrier waits on the same
    subtile slot the async load wrote; otherwise the producer evaluates `-acc_iv`
    to a negative slot at reduction iteration > 0, the recorded barrier slot
    diverges from the load slot, and TOGSim aborts with "Key does not exist in ...
    tag table" on subtile + multi-tile-K. See docs/design/togsim_cpp_trace.md and
    legacy TileGraphParser.cc (which skips stride -1 for the same reason)."""
    import sys
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import build_skeleton as bs

    import mlir.ir as ir
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(
            "func.func @k() {\n"
            "  %r = arith.constant 1 : index\n"   # stand-in reduction iv
            "  %a = arith.constant 0 : index\n"   # subtile dim 1
            "  %b = arith.constant 0 : index\n"   # subtile dim 2
            "  return\n"
            "}", ctx)
        block = module.body.operations[0].regions[0].blocks[0]
        consts = [op.results[0] for op in block.operations if op.name == "arith.constant"]
        anchor = [op for op in block.operations if op.name == "func.return"][0]
        r, a, b = consts

        def neg_dims(val):
            amap = ir.AffineMapAttr(val.owner.attributes["map"]).value
            return [p for p in (bs._neg_coeff_dim(s) for s in bs._flatten_add(amap.results[0]))
                    if p is not None]

        # #map8-style: -d0 (reduction) + d1 + d2 floordiv 2.
        d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
        expr = d0 * -1 + d1 + ir.AffineExpr.get_floor_div(d2, 2)
        with ir.InsertionPoint(anchor):
            apply = ir.Operation.create(
                "affine.apply", results=[ir.IndexType.get()], operands=[r, a, b],
                attributes={"map": ir.AffineMapAttr.get(ir.AffineMap.get(3, 0, [expr]))})
        tag_in = apply.results[0]
        assert neg_dims(tag_in) == [0]                       # the reduction marker is present

        tag_out = bs._strip_accum_terms(ctx, tag_in, anchor)
        assert tag_out is not tag_in                         # a new, reduced apply was emitted
        out_map = ir.AffineMapAttr(tag_out.owner.attributes["map"]).value
        assert out_map.n_dims == 2                           # the reduction dim was dropped
        assert neg_dims(tag_out) == []                       # no reduction marker remains
        assert list(tag_out.owner.operands) == [a, b]        # only the subtile operands survive

        # No-op: an index with no reduction marker is returned unchanged.
        plain = d0 + d1
        with ir.InsertionPoint(anchor):
            papply = ir.Operation.create(
                "affine.apply", results=[ir.IndexType.get()], operands=[a, b],
                attributes={"map": ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [plain]))})
        pin = papply.results[0]
        assert bs._strip_accum_terms(ctx, pin, anchor) is pin

        assert module.operation.verify()


@pytest.mark.skipif(not _mlir_available(), reason="MLIR Python bindings not installed")
def test_cycle_table_on_fixture():
    fixture = os.environ.get("TOGSIM_SKELETON_FIXTURE")
    if not fixture or not os.path.isfile(fixture):
        pytest.skip("set TOGSIM_SKELETON_FIXTURE to a post-vcix kernel .mlir")

    import sys
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import build_skeleton, cycle_table

    import mlir.ir as ir
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(pathlib.Path(fixture).read_text(), ctx)
        build_skeleton.build_skeleton(module)
        types = cycle_table._compute_types(module)
        # synthetic per-tile cycles (gem5 sample-mode is reused at P3 task 5).
        cyc = [10 * (i + 1) for i in range(len(types))]
        x_off, w_off = 4, 0
        table = cycle_table.build_cycle_table(module, cyc, x_off, w_off)

    assert len(table) == len(types) >= 1
    # cycle is carried verbatim; overlapping_cycle follows the legacy formula.
    for (cy, ov), t, raw in zip(table, types, cyc):
        assert cy == raw
        if t == cycle_table.VECTOR_COMPUTE:
            assert ov == 0
        else:
            off = w_off if t == cycle_table.MATMUL_PRELOAD else x_off
            assert ov == max(raw - off, 0)
