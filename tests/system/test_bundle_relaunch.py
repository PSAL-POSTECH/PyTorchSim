"""scripts/ci/bundle.py's capture: a kernel a workload launches again on other values its kernel file takes from the
launch (a sort of 32768 per row merges in two steps, _merge_far at DIST 8192 then 16384 and 8192 at S 15) gets a
directory per set of values, each launch naming its own; launches on the same values share one."""
import importlib.util
import os
import sys
import tempfile

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
os.environ.setdefault("TORCHSIM_FUNCTIONAL_MODE", "0")
sys.path.insert(0, TORCHSIM_DIR)

spec = importlib.util.spec_from_file_location("bundle", os.path.join(TORCHSIM_DIR, "scripts", "ci", "bundle.py"))
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)
import torch  # noqa: E402


def test_a_kernel_launched_again_on_other_values_gets_a_directory_per_values():
    with tempfile.TemporaryDirectory() as out:
        launches, _, _ = bundle.capture("sort_1x32768", lambda x: torch.sort(x, dim=-1)[0],
                                        lambda g: [torch.randn(1, 32768, generator=g)], out)
        far = [la["dir"] for la in launches if la["kernel"] == "_merge_far"]
        assert len(far) == 3 and len(set(far)) == 3, launches
        specs = [open(os.path.join(out, "sort_1x32768", d, "_merge_far_spec.py")).read() for d in far]
        assert len(set(specs)) == 3, "each _merge_far directory takes its own launch's values"
        dirs = sorted(os.listdir(os.path.join(out, "sort_1x32768")))
        assert dirs == sorted({la["dir"] for la in launches}), (dirs, launches)


if __name__ == "__main__":
    test_a_kernel_launched_again_on_other_values_gets_a_directory_per_values()
    print("PASS")
