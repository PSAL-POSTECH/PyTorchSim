"""Drive the Triton codegen route as far as it currently goes.

This route is WIP (see PyTorchSimFrontend/triton_backend/README.md). The test is
written to report WHERE it stops rather than to assert success: the value right
now is a reproducible statement of the next gap, not a pass/fail gate. Register
it in .github/workflows/pytorchsim_test.yml only once the route runs end to end.

    TORCHSIM_TRITON_CODEGEN=1 python tests/system/test_triton_codegen.py
"""
import os
import sys
import traceback

# Must be set before torch_openreg registers the Inductor backend for `npu`.
os.environ.setdefault("TORCHSIM_TRITON_CODEGEN", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


N = 1024


def build():
    def fn(x, y):
        return x + y

    x = torch.randn(N)
    y = torch.randn(N)
    return fn, x, y


def check_multi_axis_grid():
    """A 2-D grid must nest one loop per axis and hand both indices to iv[].

    Guards the multi-axis path, which the add kernel does not reach: Inductor
    only uses y/z when x would overflow, so a 1-D grid exercises just the first
    iteration of the nest.
    """
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as l2e
    from PyTorchSimFrontend.mlir.passes.build_tog import ir

    src = """
    module {
      func.func @k(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32) {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : i32
        %a = arith.muli %arg1, %c8 : i32
        %b = arith.addi %a, %arg2 : i32
        %o = arith.index_cast %b : i32 to index
        "togsim.dma"(%o, %c0) {arg_id = 0 : i32, base = "arg0", dims = [128],
            dir = 0 : i32, elem_bits = 32 : i32, is_async = false, read_bufs = [],
            strides = [1], tag_id = 0 : i32, write_bufs = [0]} : (index, index) -> ()
        return
      }
    }
    """
    problems = []
    # Verify the IR the pass itself produces: a bound created after an outer loop
    # would not dominate an inner loop's use of it, which only shows at rank >= 2
    # and which the emitc lowering happens to paper over.
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(src, ctx)
        l2e._materialize_grid_loop(
            l2e._find_kernel(module),
            l2e.WorkItem(parallel_args=[1, 2], grid=[4, 3]), ctx)
        try:
            module.operation.verify()
        except Exception as e:  # noqa: BLE001
            problems.append(f"materialized IR does not verify: {e}")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(src, ctx)
        emitc = l2e.lower_to_emitc(
            module, work_item=l2e.WorkItem(parallel_args=[1, 2], grid=[4, 3]))
        cpp = l2e.emitc_to_cpp(emitc, include_dir=l2e._default_include_dir())

    entry = cpp.split("togsim_kernel(EmitCtx*")[-1]
    if entry.count("for (") != 2:
        problems.append(f"expected 2 nested loops, found {entry.count('for (')}")
    if "togsim_dispatch" not in entry:
        problems.append("no togsim_dispatch call")
    if ", 2);" not in entry:
        problems.append("dispatch does not pass 2 indices")
    for p in problems:
        print(f"  multi-axis grid: {p}")
    return not problems


def check_reduction_is_right():
    """A reduction must compute the right numbers, on BOTH axes.

    THIS CHECK USED TO REQUIRE THE OPPOSITE and said so: "A reduction must fail
    LOUDLY, not compile into wrong numbers ... tnpu has no lane-aware reduction
    ... When the lane path lands, this is the test to delete." It landed, from an
    unexpected direction: nothing in tnpu changed, but Inductor now emits a
    PERSISTENT reduction wherever this backend's block covers the extent
    (inductor_templates._persist_a_reduction_that_fits_one_tile), and a
    reduction that finishes inside one tile never crosses a lane -- which was the
    whole reason the old form could not be lowered.

    Deleting it outright would give up the thing it was really guarding, which is
    not "does this stop" but "are the numbers real". So it is turned around and
    asks that instead.

    AND IT ASKS IT OF dim 0 TOO, which used to be the axis that stopped. It did
    not stop for being dim 0: Inductor hands the reduction down correctly either
    way (dim 0 gives xnumel 64 against r0_numel 128, dim 1 the reverse) and
    `fixed_config_for` gave BOTH XBLOCK = 128 without looking at the numel, so
    dim 0 got a tile twice its iteration space and the surplus became a mask
    that stage 4 refused. Blocks are clamped to their numel now, and this checks
    the axis that clamping fixed as well as the one that never needed it.
    """
    x = torch.randn(128, 64)
    for dim in (1, 0):
        ref = x.sum(dim=dim)
        try:
            got = torch.compile(lambda t: t.sum(dim=dim))(x.to("npu:0")).to("cpu")
        except Exception as e:  # noqa: BLE001 - a stop is now a failure, and says so
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]
            print(f"  reduction dim={dim} stopped at: "
                  f"{type(e).__name__}: {first[:64]}")
            return False
        err = (got - ref).abs().max().item()
        print(f"  reduction dim={dim} max_abs_err: {err:g}")
        if not torch.allclose(got, ref, rtol=1e-4, atol=1e-4):
            return False
    return True


def main():
    from PyTorchSimFrontend import extension_config
    from PyTorchSimFrontend.triton_backend import tnpu_bridge

    print(f"multi-axis grid          = "
          f"{'ok' if check_multi_axis_grid() else 'FAILED'}")
    print(f"reduction is right       = "
          f"{'ok' if check_reduction_is_right() else 'FAILED'}")
    print(f"TORCHSIM_TRITON_CODEGEN = {extension_config.CONFIG_TRITON_CODEGEN}")
    print(f"TNPU_DIR                = {extension_config.CONFIG_TNPU_DIR}")
    ok, _out = tnpu_bridge.doctor()
    print(f"tnpu doctor             = {'ok' if ok else 'FAILED (see run.py doctor)'}")
    print()

    fn, x, y = build()
    expected = fn(x, y)

    opt = torch.compile(fn, backend="inductor")
    try:
        got = opt(x.to("npu:0"), y.to("npu:0"))
    except Exception as e:  # noqa: BLE001 - the point is to report the stop
        print(f"STOPPED AT: {type(e).__name__}")
        print()
        traceback.print_exc()
        print()
        print("The stage reached is what this test measures; see the traceback "
              "above and README.md's gap list.")
        return 1

    ok = torch.allclose(got.cpu(), expected, rtol=1e-4, atol=1e-4)
    if not ok:
        bad = (~torch.isclose(got.cpu(), expected, rtol=1e-4, atol=1e-4))
        print(f"VALUES WRONG: {int(bad.sum())}/{expected.numel()} elements")
        print(f"  got      {got.cpu()[:4].tolist()}")
        print(f"  expected {expected[:4].tolist()}")
        return 1
    print(f"values ok ({expected.numel()} elements through Spike)")

    import glob

    from PyTorchSimFrontend.triton_backend import timing

    dirs = glob.glob(os.path.join(extension_config.get_dump_path(), "triton_*"))
    if not dirs:
        print("no kernel directory was produced")
        return 1
    workdir = max(dirs, key=os.path.getmtime)
    if not extension_config.pytorchsim_timing_mode:
        # NOT A SKIP OF THE TEST, A SKIP OF THE HALF THAT WAS TURNED OFF.
        # `codecache` builds no trace and runs no TOGSim when timing mode is
        # off, so trace.so and trace_cycles.tsv do not exist and their absence
        # is the setting working. The values above still went through Spike,
        # which is what this test is mostly for.
        print(f"\ntiming mode is off, so no trace was built ({workdir})")
        return 0
    for name in (timing.TRACE_SO, timing.CYCLE_TSV):
        path = os.path.join(workdir, name)
        if not os.path.isfile(path):
            print(f"missing {name} in {workdir}")
            return 1
        print(f"  {name:18s} {os.path.getsize(path)} bytes")
    print(f"\ntiming path OK ({workdir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
