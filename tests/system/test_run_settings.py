"""Run settings come from their environment variable, else the machine file's key of the same name,
else their default; codegen_mapping_strategy picks a template's tile from the heuristic, TOGSim's
cycles or an external mapping file, which an mm also takes over the resident mm's plan. Each compile
runs in a child process with the settings unset; a bmm, which the resident mm never takes."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
sys.path.insert(0, TORCHSIM_DIR)

import torch  # noqa: E402,F401  the npu backend loads pytorchsim whole, before its parts are imported

from pytorchsim import config  # noqa: E402
from pytorchsim.frontend import tiling  # noqa: E402
from pytorchsim.backend import template_record as select  # noqa: E402

SHAPE = (128, 256, 256)
SETTING_ENVS = [env for env, _, _ in config.RUN_SETTINGS.values()]

CHILD = r"""
import sys
import torch
op = sys.argv[1]
m, k, n = map(int, sys.argv[2:5])
lead = (2,) if op == "bmm" else ()
a = torch.randn(*lead, m, k).to(torch.bfloat16).to("npu:0")
b = torch.randn(*lead, k, n).to(torch.bfloat16).to("npu:0")
torch.compile(dynamic=False)(lambda x, y: x @ y)(a, b)
"""


def _machine(root, name, **settings):
    """A copy of tpuv6e.yml under `root` that states `settings`, and its path; the copy's
    ramulator_config_path is made absolute, since a relative one is the file's directory's."""
    path = os.path.join(root, f"{name}.yml")
    with open(os.path.join(config.CONFIG_TOGSIM_CONFIG_DIR, "tpuv6e.yml")) as f:
        text = f.read()
    text = re.sub(r"^ramulator_config_path: (?!/)(.*)$",
                  lambda m: "ramulator_config_path: " + os.path.normpath(
                      os.path.join(config.CONFIG_TOGSIM_CONFIG_DIR, m.group(1))),
                  text, flags=re.M)
    with open(path, "w") as f:
        f.write(text)
    with open(path, "a") as f:
        for key, value in settings.items():
            f.write(f"\n{key}: {json.dumps(value)}")
        f.write("\n")
    return path


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in SETTING_ENVS}
    env.update(extra)
    return env


def test_env_then_machine_then_default():
    with tempfile.TemporaryDirectory() as root:
        path = _machine(root, "stated", pytorchsim_timing_mode=1, pytorchsim_functional_mode=0,
                        codegen_mapping_strategy="heuristic", codegen_autotune_template_topk=2)
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=path), clear=True):
            assert (config.pytorchsim_timing_mode, config.pytorchsim_functional_mode) == (1, 0)
            assert config.codegen_mapping_strategy == "heuristic"
            assert config.codegen_autotune_template_topk == 2
            assert config.run_setting_source("pytorchsim_timing_mode") == "machine"
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=path, TORCHSIM_TIMING_MODE="0",
                                              TORCHSIM_CODEGEN_MAPPING_STRATEGY="autotune"),
                             clear=True):
            assert config.pytorchsim_timing_mode == 0
            assert config.codegen_mapping_strategy == "autotune"
            assert config.run_setting_source("pytorchsim_timing_mode") == "env"
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET="tpuv6e"), clear=True):
            assert config.pytorchsim_timing_mode == 0
            assert config.codegen_mapping_strategy == "autotune"
            assert config.run_setting_source("codegen_mapping_strategy") == "default"


def test_a_strategy_or_mapping_file_that_cannot_be_used_is_refused():
    with tempfile.TemporaryDirectory() as root:
        bad = _machine(root, "bad", codegen_mapping_strategy="external")
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=bad), clear=True):
            try:
                config.codegen_mapping_strategy
            except ValueError as e:
                assert "is none of" in str(e)
            else:
                raise AssertionError("an unknown strategy was taken")
        subtile = os.path.join(root, "subtile.json")
        with open(subtile, "w") as f:
            json.dump({"1_2_3": {"TILE_M": 1, "TILE_N": 2, "TILE_K": 3, "SUB_TILE_M": 1}}, f)
        for file, said in (("", "names no codegen_external_mapping_file"),
                           ("missing.json", "No such file"), ("subtile.json", "an entry is exactly")):
            path = _machine(root, f"ext_{len(said)}", codegen_mapping_strategy="external-then-heuristic",
                            codegen_external_mapping_file=file)
            with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=path), clear=True):
                try:
                    select.external_table()
                except ValueError as e:
                    assert said in str(e), (file, e)
                else:
                    raise AssertionError(f"{file!r} was taken")


def test_a_mapping_file_is_found_and_digested():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "maps"))
        mapping = os.path.join(root, "maps", "tiles.json")
        with open(mapping, "w") as f:
            json.dump({"1_2_3": {"TILE_M": 4, "TILE_N": 5, "TILE_K": 6}}, f)
        path = _machine(root, "relative", codegen_mapping_strategy="external-then-autotune",
                        codegen_external_mapping_file="maps/tiles.json")
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=path), clear=True):
            assert config.codegen_external_mapping_file == mapping
            assert select.external_tile(1, 2, 3) == (4, 5, 6)
            assert select.external_tile(3, 2, 1) is None
            before = select.selection_digest()
            with open(mapping, "w") as f:
                json.dump({"1_2_3": {"TILE_M": 8, "TILE_N": 5, "TILE_K": 6}}, f)
            os.utime(mapping, ns=(os.stat(mapping).st_mtime_ns + 10**9,) * 2)
            assert select.selection_digest() != before
            assert select.external_tile(1, 2, 3) == (8, 5, 6)
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=path,
                                              TORCHSIM_CODEGEN_EXTERNAL_MAPPING_FILE="tiles.json"),
                             clear=True):
            assert config.codegen_external_mapping_file == os.path.abspath("tiles.json")
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET="tpuv6e"), clear=True):
            plain = select.selection_digest()
        with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET="tpuv6e",
                                              TORCHSIM_CODEGEN_MAPPING_STRATEGY="heuristic"),
                             clear=True):
            assert select.selection_digest() != plain
        one, two = _machine(root, "one"), _machine(root, "two")
        with open(two) as f:
            text = f.read().replace("num_cores: 1", "num_cores: 2")
        with open(two, "w") as f:
            f.write(text)
        digests = []
        for path in (one, two):
            with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET=path), clear=True):
                digests.append(select.graph_digest())
        assert digests[0] != digests[1]


def _compile(dump, machine, keep=False, op="bmm"):
    """The template kernels' {tile_source, blocks} of one compile, and whether anything was timed
    or recorded as timed."""
    if not keep:
        shutil.rmtree(dump, ignore_errors=True)
    os.makedirs(dump, exist_ok=True)
    env = _env(TORCHSIM_TARGET=machine, TORCHSIM_DUMP_PATH=dump,
               TORCHSIM_LOG_PATH=os.path.join(dump, "logs"))
    proc = subprocess.run([sys.executable, "-c", CHILD, op, *map(str, SHAPE)], env=env,
                          cwd=TORCHSIM_DIR, capture_output=True, text=True)
    assert proc.returncode == 0, f"{machine}: {proc.stderr[-4000:]}"
    built = []
    for name in sorted(os.listdir(dump)):
        meta = os.path.join(dump, name, "meta.json")
        if not (name.startswith("triton_") and os.path.isfile(meta)):
            continue
        with open(meta) as f:
            source = json.load(f).get("tile_source")
        if source is None:
            continue
        with open(os.path.join(dump, name, "kernel.py")) as f:
            src = f.read()
        blocks = tuple(int(re.search(rf"{b} : tl.constexpr = (\d+)", src).group(1))
                       for b in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        built.append((source, blocks))
    return built, os.path.isdir(os.path.join(dump, "select")) or \
        os.path.isfile(os.path.join(dump, select.RECORD_LOG))


def test_the_strategy_picks_the_tile():
    m, k, n = SHAPE
    with mock.patch.dict(os.environ, _env(TORCHSIM_TARGET="tpuv6e"), clear=True):
        tiles = [(c.block_m, c.block_n, c.block_k) for c in tiling.gemm_tiles(m, n, k, 2)]
    assert len(tiles) > 1, tiles
    user = tiles[-1]
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "tiles.json"), "w") as f:
            json.dump({f"{m}_{n}_{k}": dict(zip(select.EXTERNAL_KEYS, user))}, f)
        with open(os.path.join(root, "other.json"), "w") as f:
            json.dump({"1_1_1": dict(zip(select.EXTERNAL_KEYS, user))}, f)
        timed = dict(pytorchsim_timing_mode=1, pytorchsim_functional_mode=0)
        machine = lambda name, strategy, file="": _machine(
            root, name, codegen_mapping_strategy=strategy, codegen_external_mapping_file=file, **timed)
        dump = lambda name: os.path.join(root, "dump", name)

        built, was_timed = _compile(dump("heuristic"), machine("heuristic", "heuristic"))
        assert built == [(select.HEURISTIC, tiles[0])] and not was_timed, (built, was_timed)

        built, was_timed = _compile(dump("autotune"), machine("autotune", "autotune"))
        assert [s for s, _ in built] == [select.TIMED] and was_timed, (built, was_timed)

        external = machine("external", "external-then-heuristic", "tiles.json")
        built, was_timed = _compile(dump("external"), external)
        assert built == [(select.EXTERNAL, user)] and not was_timed, (built, was_timed)

        built, was_timed = _compile(dump("missing"),
                                    machine("missing", "external-then-autotune", "other.json"))
        assert [s for s, _ in built] == [select.TIMED] and was_timed, (built, was_timed)

        _compile(dump("cache"), machine("cache", "heuristic"))
        built, _ = _compile(dump("cache"), external, keep=True)
        assert (select.EXTERNAL, user) in built, built

        built, _ = _compile(dump("mm_heuristic"), machine("heuristic", "heuristic"), op="mm")
        assert [s for s, _ in built] == [select.PLANNED], built
        built, _ = _compile(dump("mm_external"), external, op="mm")
        assert built == [(select.EXTERNAL, user)], built


if __name__ == "__main__":
    test_env_then_machine_then_default()
    test_a_strategy_or_mapping_file_that_cannot_be_used_is_refused()
    test_a_mapping_file_is_found_and_digested()
    test_the_strategy_picks_the_tile()
    print("PASS")
