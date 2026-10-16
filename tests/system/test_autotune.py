"""How a kernel's build is chosen and reused: every candidate asks the compiler its options outright,
trip counts are offered and the compiler decides which it applies, and identical builds are
recognised by the kernel object's artifacts."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))
os.environ["TORCHSIM_TIMING_MODE"] = "1"
os.environ["TORCHSIM_CODEGEN_AUTOTUNE_KERNEL_TOPK"] = "64"

from torch._inductor import config as inductor_config

from pytorchsim import contract
from pytorchsim.driver import autotune

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got!r}" + ("" if ok else f", want {want!r}"))
    if not ok:
        FAILED.append(name)


def _softmax_meta(xblock=2048):
    return {"kernel_name": "triton_softmax", "numels": {"xnumel": 2048, "r0_numel": 2048},
            "fixed_config": {"XBLOCK": xblock, "R0_BLOCK": 2048}, "template_grid": None,
            "signature": {}}


def test_every_trip_count_is_offered_and_asked_outright():
    """A reduction is asked every PIPE_STAGES count, 1 included, each in its compile_env: which
    counts torchsim-pipeline-reductions applies is the compiler's to decide, not a front-end _GRAIN."""
    with inductor_config.patch(max_autotune=True):
        cands = autotune._kernel_candidates("", _softmax_meta())
    trips = [mt["compile_env"].get(autotune.PIPE_ENV) for label, _, mt in cands
             if mt["fixed_config"]["XBLOCK"] == 2048]
    check("TripsOffered", trips, list(autotune.PIPE_STAGES))


def test_a_dynamic_reduction_asks_no_trips():
    meta = _softmax_meta()
    meta["numels"]["r0_numel"] = None
    with inductor_config.patch(max_autotune=True):
        cands = autotune._kernel_candidates("", meta)
    check("DynamicNoTrips", sorted({mt["compile_env"][autotune.PIPE_ENV] for _, _, mt in cands}),
          [1])


def test_an_exported_option_never_reaches_a_build():
    """TORCHSIM_PIPE_STAGES exported in the shell does not leak into a build that did not ask it."""
    os.environ[autotune.PIPE_ENV] = "4"
    try:
        got = [autotune.compile_env({}), autotune.compile_env({"compile_env": {autotune.PIPE_ENV: 8}})]
    finally:
        os.environ.pop(autotune.PIPE_ENV)
    check("NoAmbientStages", got, [{autotune.PIPE_ENV: "1"}, {autotune.PIPE_ENV: "8"}])


def _kernel_object(blobs):
    d = tempfile.mkdtemp()
    arts = {}
    for kind, data in blobs.items():
        with open(os.path.join(d, f"{kind}.bin"), "wb") as f:
            f.write(data)
        arts[kind] = f"{kind}.bin"
    with open(os.path.join(d, contract.KERNEL_MANIFEST), "w") as f:
        json.dump({"format": contract.KERNEL_FORMAT, "artifacts": arts}, f)
    return d


def test_identical_builds_are_recognised_by_the_kernel_object():
    """The build key reads the artifacts the manifest names, whatever the files are called."""
    meta = _softmax_meta(256)
    same = {"elf": b"a", "cycle_elf": b"b", "tog_so": b"c"}
    k1 = autotune._build_key(_kernel_object(same), meta)
    k2 = autotune._build_key(_kernel_object(same), meta)
    k3 = autotune._build_key(_kernel_object(dict(same, cycle_elf=b"B")), meta)
    k4 = autotune._build_key(_kernel_object(same), _softmax_meta(512))
    k5 = autotune._build_key(_kernel_object({"elf": b"a", "tog_so": b"c"}), meta)
    check("BuildKey", [k1 is not None, k1 == k2, k1 == k3, k1 == k4, k5], [True, True, False, False,
                                                                        None])


if __name__ == "__main__":
    for test in (test_every_trip_count_is_offered_and_asked_outright,
                 test_a_dynamic_reduction_asks_no_trips,
                 test_an_exported_option_never_reaches_a_build,
                 test_identical_builds_are_recognised_by_the_kernel_object):
        try:
            test()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
            FAILED.append(test.__name__)
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        sys.exit(1)
