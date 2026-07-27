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


def main():
    from PyTorchSimFrontend import extension_config
    from PyTorchSimFrontend.triton_backend import tnpu_bridge

    print(f"multi-axis grid          = "
          f"{'ok' if check_multi_axis_grid() else 'FAILED'}")
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

    # Values are NOT checked: the launch simulates the kernel but does not
    # marshal tensors through Spike, so `got` is undefined. What is asserted is
    # that the timing path ran end to end -- the two artifacts TOGSim consumes.
    del got
    import glob

    from PyTorchSimFrontend.triton_backend import timing

    dirs = glob.glob(os.path.join(extension_config.get_dump_path(), "triton_*"))
    if not dirs:
        print("no kernel directory was produced")
        return 1
    workdir = max(dirs, key=os.path.getmtime)
    for name in (timing.TRACE_SO, timing.CYCLE_TSV):
        path = os.path.join(workdir, name)
        if not os.path.isfile(path):
            print(f"missing {name} in {workdir}")
            return 1
        print(f"  {name:18s} {os.path.getsize(path)} bytes")
    print(f"\ntiming path OK ({workdir})")
    print(f"values NOT verified -- torch would give {expected[:3].tolist()}...; "
          f"the functional launch is not wired (triton_backend/README.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
