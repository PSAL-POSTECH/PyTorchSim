"""scripts/ci/bundle.py --where: only the workloads the compiler's simulate_shapes.py --where keeps are built,
a column given twice takes either, and a column the suite has not is refused as the compiler refuses it."""
import importlib.util
import os

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
spec = importlib.util.spec_from_file_location("bundle", os.path.join(TORCHSIM_DIR, "scripts", "ci", "bundle.py"))
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)

MOE = ["section=op", "name=dispatch", "name=combine", "name=router_post", "name=gmm_gate_up", "name=gmm_down"]


def test_where_keeps_the_named_ops():
    names = [n for _, n in bundle.largest_first({"layer"}, MOE)]
    assert names and all(n.startswith(("layer_op_dispatch", "layer_op_combine", "layer_op_router_post",
                                       "layer_op_gmm_gate_up", "layer_op_gmm_down")) for n in names)
    assert len(names) < len(bundle.largest_first({"layer"}))


def test_where_refuses_an_unknown_column():
    try:
        bundle.largest_first({"layer"}, ["kind=op"])
    except SystemExit:
        return
    raise AssertionError("--where kind=op was not refused")


if __name__ == "__main__":
    test_where_keeps_the_named_ops()
    test_where_refuses_an_unknown_column()
    print("PASS")
