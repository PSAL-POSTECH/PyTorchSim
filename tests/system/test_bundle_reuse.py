"""scripts/ci/bundle.py --reuse: a workload whose key the previous bundle holds is copied byte for byte
without building it, and every built one gets a key; a changed input or Inductor/Triton/PyTorchSim env,
an unpicked launch, no PTS_INPUTS or a row no longer measured is not reused; a per-run path is no input."""
import filecmp
import importlib.util
import json
import os
import shutil
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
os.environ.setdefault("TORCHSIM_FUNCTIONAL_MODE", "0")

spec = importlib.util.spec_from_file_location("bundle", os.path.join(TORCHSIM_DIR, "scripts", "ci", "bundle.py"))
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)

NAME = "vector_slice_8x128_bf16"
INPUTS = "tree-a device-a bundle-a"
_all = bundle.workloads


def _one(suites=None):
    return (w for w in _all(suites) if w[0] == NAME)


def _same(a, b):
    cmp = filecmp.dircmp(a, b)
    assert not (cmp.left_only or cmp.right_only or cmp.funny_files), (a, b)
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    assert not mismatch and not errors, (a, b, mismatch, errors)
    for d in cmp.common_dirs:
        _same(os.path.join(a, d), os.path.join(b, d))


def _run(root, prev=None, inputs=INPUTS, extra=None):
    """main() into root/out and root/keys, with env `extra` set: the workloads it built."""
    built, real = [], bundle.build

    def build(out, suite, name, keys=None):
        built.append(name)
        return real(out, suite, name, keys)
    env = dict({"PTS_INPUTS": inputs} if inputs else {}, **(extra or {}))
    with mock.patch.dict(os.environ, env), mock.patch.object(bundle, "workloads", _one), \
            mock.patch.object(bundle, "build", build):
        if not inputs:
            os.environ.pop("PTS_INPUTS", None)
        bundle.main(os.path.join(root, "out"), {"vector"}, 1, prev=prev)
    return built


def test_reuse():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c, d, e = (os.path.join(tmp, x) for x in "abcde")
        assert _run(a) == [NAME]
        with open(os.path.join(a, "keys", f"{NAME}.json")) as f:
            key = json.load(f)
        assert key["inputs"]["pts"] == INPUTS.split() and bundle.template_record.HEURISTIC not in key["tile_sources"]

        shutil.copytree(os.path.join(a, "out", NAME), os.path.join(a, "out", "vector_dropped_1x1_bf16"))
        shutil.copy(os.path.join(a, "keys", f"{NAME}.json"), os.path.join(a, "keys", "vector_dropped_1x1_bf16.json"))
        assert _run(b, prev=a) == []
        _same(os.path.join(a, "out", NAME), os.path.join(b, "out", NAME))
        assert not os.path.exists(os.path.join(b, "out", "vector_dropped_1x1_bf16"))
        with open(os.path.join(b, "keys", f"{NAME}.json")) as f:
            assert json.load(f)["key"] == key["key"]

        assert _run(c, prev=a, inputs="tree-b device-a bundle-a") == [NAME]
        _same(os.path.join(a, "out", NAME), os.path.join(c, "out", NAME))

        x, y = os.path.join(tmp, "x"), os.path.join(tmp, "y")
        assert _run(x, prev=a, extra={"TORCHINDUCTOR_PERSISTENT_REDUCTIONS": "0"}) == [NAME]
        assert _run(y, prev=a, extra={"TORCHSIM_DUMP_PATH": os.path.join(tmp, "dump"),
                                      "TORCHSIM_LOG_PATH": os.path.join(tmp, "log"),
                                      "TRITON_CACHE_DIR": os.path.join(tmp, "triton")}) == []

        shutil.copytree(a, d)
        with open(os.path.join(d, "keys", f"{NAME}.json"), "w") as f:
            json.dump(dict(key, tile_sources=[bundle.template_record.HEURISTIC]), f)
        assert _run(e, prev=d) == [NAME]

        shutil.rmtree(e)
        assert _run(e, prev=a, inputs=None) == [NAME]
        assert not os.path.exists(os.path.join(e, "keys", f"{NAME}.json"))


if __name__ == "__main__":
    test_reuse()
    print("PASS")
