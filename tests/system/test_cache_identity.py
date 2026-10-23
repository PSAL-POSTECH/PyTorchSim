"""A cached kernel's identity is what the compiler itself resolves: the tools it would run, and the
TOGSim API and headers its tog.so is compiled against. A compiler that cannot state them stops the
run, rather than leaving every cached kernel guarded by its commit alone."""
import json
import os
import subprocess
import sys
from unittest import mock

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: F401,E402  the npu backend loads pytorchsim whole, before its parts are imported
from pytorchsim import config  # noqa: E402
from pytorchsim.driver import codecache, compiler_bridge  # noqa: E402


def _ask_compiler(expr):
    code = f"import json; from {compiler_bridge.COMPILER_PKG} import toolchain as t; print(json.dumps({expr}))"
    out = subprocess.run([config.CONFIG_TORCHSIM_COMPILE_PYTHON, "-c", code], capture_output=True,
                         text=True, check=True, cwd=compiler_bridge.compiler_dir(),
                         env=compiler_bridge.compiler_env()).stdout
    return json.loads(out.strip().splitlines()[-1])


def _fresh_identity():
    with mock.patch.object(codecache, "_identity", None):
        return codecache.identity()


def test_tools_are_the_compilers():
    tools = _fresh_identity()["tools"]
    want = _ask_compiler("[n for n, _ in t.TOOLS]")
    assert tools is not None and sorted(tools) == sorted(want), tools
    assert all(v is not None for v in tools.values()), tools


def test_tog_abi_is_the_compilers_togsim():
    abi = _fresh_identity()["tog_abi"]
    assert isinstance(abi, dict) and abi["api"] == _ask_compiler("t.togsim_api_version()"), abi
    assert abi["api"] is not None and abi["headers"]


def test_a_compiler_that_cannot_answer_stops_the_run():
    with mock.patch.object(config, "CONFIG_TORCHSIM_COMPILE_PYTHON", "/bin/false"):
        try:
            _fresh_identity()
        except RuntimeError as e:
            assert "did not state its toolchain" in str(e), e
            return
    raise AssertionError("identity() answered without the compiler")


if __name__ == "__main__":
    test_tools_are_the_compilers()
    test_tog_abi_is_the_compilers_togsim()
    test_a_compiler_that_cannot_answer_stops_the_run()
    print("PASS")
