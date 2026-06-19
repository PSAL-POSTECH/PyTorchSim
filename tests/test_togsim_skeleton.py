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
    assert ts.ENTRY_SYMBOL == "togsim_emit"
    assert ts.ENTRY_SYMBOL in header

    # Runtime callee emitted directly by lower_to_emitc (core alloc).
    assert ts.CORE_ALLOC_CALLEE in header

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
