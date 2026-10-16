"""How a kernel's build is chosen and reused: every candidate asks the compiler its options outright,
trip counts are offered and the compiler decides which it applies, identical builds are recognised by
the kernel object's artifacts, and a cached build is reused only under the policy that chose it and
launched at the blocks it was built at."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))
os.environ["TORCHSIM_TIMING_MODE"] = "1"
os.environ["TORCHSIM_CODEGEN_AUTOTUNE_KERNEL_TOPK"] = "64"

from torch._inductor import config as inductor_config

from pytorchsim import contract
from pytorchsim.driver import autotune, codecache

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
    trips = [c.meta["compile_env"].get(autotune.PIPE_ENV) for c in cands
             if c.meta["fixed_config"]["XBLOCK"] == 2048]
    check("TripsOffered", trips, list(autotune.PIPE_STAGES))


def test_a_dynamic_reduction_asks_no_trips():
    meta = _softmax_meta()
    meta["numels"]["r0_numel"] = None
    with inductor_config.patch(max_autotune=True):
        cands = autotune._kernel_candidates("", meta)
    check("DynamicNoTrips", sorted({c.meta["compile_env"][autotune.PIPE_ENV] for c in cands}),
          [1])


def test_an_exported_option_never_reaches_a_build():
    """TORCHSIM_PIPE_STAGES exported in the shell does not leak into a build that did not ask it."""
    os.environ[autotune.PIPE_ENV] = "4"
    try:
        got = [autotune.compile_env({}), autotune.compile_env({"compile_env": {autotune.PIPE_ENV: 8}})]
    finally:
        os.environ.pop(autotune.PIPE_ENV)
    check("NoAmbientStages", got, [{autotune.PIPE_ENV: "1"}, {autotune.PIPE_ENV: "8"}])


def test_the_default_cap_keeps_the_largest_blocks_first():
    """Under the default codegen_autotune_kernel_topk (8), softmax 2048^2 is timed at XBLOCK 2048 down
    to the lanes and then trips 2, 4, 8 at 2048; trips 16..128 are cut (and logged)."""
    os.environ.pop("TORCHSIM_CODEGEN_AUTOTUNE_KERNEL_TOPK")
    try:
        with inductor_config.patch(max_autotune=True):
            labels = [c.label for c in autotune._kernel_candidates("", _softmax_meta())]
    finally:
        os.environ["TORCHSIM_CODEGEN_AUTOTUNE_KERNEL_TOPK"] = "64"
    lanes = int(autotune.config.CONFIG_MACHINE.vector_lanes)
    xs = [x for x in (2048, 1024, 512, 256, 128, 64, 32) if x >= lanes]
    want = ([f"x{x}-s1" for x in xs] + [f"x2048-s{n}" for n in autotune.PIPE_STAGES[1:]])[:8]
    check("DefaultTopK", labels, want)


class _Stop(Exception):
    pass


def _recording_pipeline(seen, then=_Stop):
    def run_pipeline(*args, **kwargs):
        seen.append(kwargs.get("env"))
        raise then("stop")
    return run_pipeline


def test_each_candidate_and_the_final_build_reach_the_compiler_with_their_options():
    """The compiler subprocess is handed each candidate's compile_env (TORCHSIM_PIPE_STAGES), and the
    final build its own: a trip count only in meta never reaches the compiler, so every trip
    candidate would build as 1."""
    from pytorchsim.driver import compiler_bridge, compiler_input
    seen, real = [], (compiler_bridge.run_pipeline, compiler_input.write, autotune.tune)
    compiler_input.write = lambda *a, **k: None
    try:
        compiler_bridge.run_pipeline = _recording_pipeline(seen, FileNotFoundError)
        for n in (4, None):
            meta = _softmax_meta()
            if n:
                meta["compile_env"] = {autotune.PIPE_ENV: n}
            autotune._time_tile("", meta, "triton_softmax", tempfile.mkdtemp(), 1)
        compiler_bridge.run_pipeline = _recording_pipeline(seen)
        os.environ["TORCHSIM_DUMP_PATH"] = tempfile.mkdtemp()
        autotune.tune = lambda s, m, k, w: (s, {"compile_env": {autotune.PIPE_ENV: 8}})
        try:
            codecache.torchsim_compile("def triton_softmax(): pass", _softmax_meta(),
                                       "triton_softmax")
        except _Stop:
            pass
    finally:
        compiler_bridge.run_pipeline, compiler_input.write, autotune.tune = real
    check("EnvReachesCompiler", seen, [{autotune.PIPE_ENV: "4"}, {autotune.PIPE_ENV: "1"},
                                       {autotune.PIPE_ENV: "8"}])


def test_a_retired_mapping_strategy_is_refused():
    """codegen_mapping_strategy=autotune (the old default) no longer turns autotune on, so it is
    refused instead of building heuristic tiles under a config that says autotune."""
    got = []
    for value in ("autotune", "external-then-autotune", "heuristic", "external"):
        os.environ["TORCHSIM_CODEGEN_MAPPING_STRATEGY"] = value
        try:
            autotune.build_policy()
            got.append("ok")
        except ValueError:
            got.append("refused")
        finally:
            os.environ.pop("TORCHSIM_CODEGEN_MAPPING_STRATEGY")
    check("RetiredStrategy", got, ["refused", "refused", "ok", "ok"])


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


def test_a_cached_build_launches_at_its_own_blocks():
    """A cache hit takes the blocks, grid and compiler options the build was made with: softmax 2048^2
    tuned to XBLOCK 256 is 8 programs, not codegen's one."""
    d = tempfile.mkdtemp()
    built = dict(_softmax_meta(256), compile_env={autotune.PIPE_ENV: 2})
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(built, f)
    meta = codecache.restore_tuned(d, _softmax_meta(2048))
    check("CachedGrid", [list(contract.grid_of(meta)), meta.get("compile_env")],
          [[8], {autotune.PIPE_ENV: 2}])


def test_the_cache_key_carries_the_build_policy():
    """A max-autotune run never reuses an as-made build of the same source, nor the reverse."""
    src = "def triton_softmax(): pass"
    off = codecache._write_path(src)
    with inductor_config.patch(max_autotune=True):
        on = codecache._write_path(src)
    with inductor_config.patch(max_autotune_pointwise=True):
        pointwise = codecache._write_path(src)
    check("PolicyInKey", [off != on, on != pointwise, codecache._write_path(src) == off],
          [True, True, True])


if __name__ == "__main__":
    for test in (test_every_trip_count_is_offered_and_asked_outright,
                 test_a_dynamic_reduction_asks_no_trips,
                 test_the_default_cap_keeps_the_largest_blocks_first,
                 test_each_candidate_and_the_final_build_reach_the_compiler_with_their_options,
                 test_a_retired_mapping_strategy_is_refused,
                 test_an_exported_option_never_reaches_a_build,
                 test_identical_builds_are_recognised_by_the_kernel_object,
                 test_a_cached_build_launches_at_its_own_blocks,
                 test_the_cache_key_carries_the_build_policy):
        try:
            test()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
            FAILED.append(test.__name__)
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        sys.exit(1)
