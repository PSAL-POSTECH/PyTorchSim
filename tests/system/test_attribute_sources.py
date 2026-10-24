"""The args a kernel's runtime attributes are read from keep their launch values in a bundle (the manifest names
them, the kernel file takes those and draws every other input, an unplaceable source stops by name); a kernel
that declares no attributes has no runtime attribute directory."""
import importlib.util
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)
os.environ.setdefault("TORCHSIM_FUNCTIONAL_MODE", "0")

import torch  # noqa: E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim import contract  # noqa: E402
from pytorchsim.backend import compiler_bridge, compiler_input  # noqa: E402

META = {"kernel_name": "k", "signature": {"ids_ptr": "*i32", "x_ptr": "*bf16", "o_ptr": "*bf16", "B": "constexpr"},
        "constants": {"B": 4}, "numels": {}, "template_grid": [1, 1, 1], "user_defined": True,
        "args": [{"name": "ids_ptr", "role": "in", "dtype": "int32", "numel": 4},
                 {"name": "x_ptr", "role": "in", "dtype": "bfloat16", "numel": 8},
                 {"name": "o_ptr", "role": "out", "dtype": "bfloat16", "numel": 8}]}
IDS = torch.tensor([3, 0, 2, 1], dtype=torch.int32)


def _manifest(sources):
    return {"name": "k", "args": META["args"],
            "runtime_attributes": {"attributes": [dict({"id": i}, **({"sources": s} if s is not None else {}))
                                                  for i, s in enumerate(sources)]}}


def _raises(kind, words, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except kind as e:
        assert all(w in str(e) for w in words), (words, str(e))
        return
    raise AssertionError(f"{fn.__name__} raised no {kind.__name__}")


def _inputs(path):
    spec = importlib.util.spec_from_file_location("k_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_inputs(torch)


def test_recorded_args_are_the_launch_values_and_the_rest_drawn():
    with tempfile.TemporaryDirectory() as d:
        path = compiler_input.write("", META, os.path.join(d, "k_spec.py"), recorded={"ids_ptr": IDS})
        got = _inputs(path)
        os.makedirs(os.path.join(d, "drawn"))
        drawn = _inputs(compiler_input.write("", META, os.path.join(d, "drawn", "k_spec.py")))
        assert torch.equal(got["ids_ptr"], IDS) and not torch.equal(drawn["ids_ptr"], IDS)
        assert got["x_ptr"].dtype == torch.bfloat16 and got["x_ptr"].numel() == 8 and "o_ptr" not in got
        os.remove(os.path.join(d, compiler_input.recorded_file("k")))
        _raises(FileNotFoundError, ["k_inputs.pt"], _inputs, path)


def test_a_recorded_arg_that_is_no_input_is_refused():
    with tempfile.TemporaryDirectory() as d:
        _raises(contract.SpecIncomplete, ["o_ptr"], compiler_input.write, "", META, os.path.join(d, "k_spec.py"),
                recorded={"o_ptr": IDS})


def test_the_manifest_names_the_sources_in_arg_order():
    assert contract.attribute_sources(_manifest([["x_ptr"], ["ids_ptr", "x_ptr"]])) == ["ids_ptr", "x_ptr"]
    assert contract.attribute_sources({"name": "k", "args": META["args"]}) == []


def test_a_source_the_manifest_does_not_place_is_refused():
    _raises(contract.KernelObjectMismatch, ["[1] name no sources"], contract.attribute_sources,
            _manifest([["ids_ptr"], None]))
    _raises(contract.KernelObjectMismatch, ["w_ptr"], contract.attribute_sources, _manifest([["w_ptr"]]))


def test_a_launch_with_no_manifest_stops_the_bundle_by_name():
    spec = importlib.util.spec_from_file_location("bundle", os.path.join(TORCHSIM_DIR, "scripts", "ci", "bundle.py"))
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
    with tempfile.TemporaryDirectory() as d:
        _raises(RuntimeError, ["k: ", "holds no kernel.json"], bundle.attribute_inputs,
                SimpleNamespace(kernel_name="k", workdir=d, meta=META), ())


def test_the_runtime_attribute_directory_is_the_declaring_kernels_only():
    with tempfile.TemporaryDirectory() as workdir:
        with mock.patch.object(contract, "kernel_object", return_value={"name": "k", "args": META["args"]}):
            assert compiler_bridge.runtime_attribute_dir(workdir) is None
        with mock.patch.object(contract, "kernel_object", return_value=_manifest([None])):
            got = compiler_bridge.runtime_attribute_dir(workdir)
        assert got is not None and got.startswith(contract.runtime_dir(workdir)), got


if __name__ == "__main__":
    test_the_runtime_attribute_directory_is_the_declaring_kernels_only()
    test_recorded_args_are_the_launch_values_and_the_rest_drawn()
    test_a_recorded_arg_that_is_no_input_is_refused()
    test_the_manifest_names_the_sources_in_arg_order()
    test_a_source_the_manifest_does_not_place_is_refused()
    test_a_launch_with_no_manifest_stops_the_bundle_by_name()
    print("PASS")
