"""A kernel the compiler refused at its first tile is built at a smaller one, and a later cache hit
launches with that tile: the hit's caller holds the untuned fixed_config, and the grid it derives
from it would launch the wrong number of programs on the tuned ELF."""
import json
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: F401,E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim import contract  # noqa: E402
from pytorchsim.backend import codecache  # noqa: E402
from pytorchsim.backend import compiler_input  # noqa: E402

SRC = "def kernel(): pass\n"


def _meta(xblock):
    return {"kernel_name": "k", "numels": {"xnumel": 4096}, "fixed_config": {"XBLOCK": xblock},
            "signature": {"XBLOCK": "constexpr"}, "args": [], "graph_prefetch": []}


def test_a_hit_launches_the_tuned_tile():
    with tempfile.TemporaryDirectory() as dump, mock.patch.dict(os.environ, {"TORCHSIM_DUMP_PATH": dump}):
        untuned = _meta(1024)
        workdir = codecache._write_path(SRC, untuned)
        os.makedirs(workdir)
        compiler_input.store_meta(workdir, _meta(256))
        with mock.patch.object(codecache, "cached_elf", return_value=os.path.join(workdir, "k.elf")), \
                mock.patch.object(codecache.config, "pytorchsim_timing_mode", 0):
            launcher = codecache.torchsim_compile(SRC, _meta(1024), "k")
        assert launcher.meta["fixed_config"] == {"XBLOCK": 256}, launcher.meta["fixed_config"]
        assert contract.grid_of(launcher.meta)[0] == 4096 // 256, contract.grid_of(launcher.meta)


if __name__ == "__main__":
    test_a_hit_launches_the_tuned_tile()
    print("PASS")
