"""Inductor's autotune tables (choices displayed, choice stats) are off unless TORCHSIM_DEBUG_MODE is
set, and Inductor's own variables for them win over both. Each case configures Inductor in a child
process, as both are read at import."""
import json
import os
import subprocess
import sys

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")

CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from torch._inductor import config as inductor_config
from pytorchsim.frontend import inductor
inductor._configure_inductor()
print(json.dumps([inductor_config.autotune_num_choices_displayed,
                  inductor_config.max_autotune_report_choices_stats]))
"""

TABLE_ENVS = ("TORCHSIM_DEBUG_MODE", "TORCHINDUCTOR_AUTOTUNE_NUM_CHOICES_DISPLAYED",
              "TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS")


def _tables(**env):
    """[choices displayed, choice stats] Inductor ends with under `env` and none of TABLE_ENVS else."""
    run_env = {k: v for k, v in os.environ.items() if k not in TABLE_ENVS}
    run_env.update(env, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    out = subprocess.run([sys.executable, "-c", CHILD, TORCHSIM_DIR], env=run_env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_off_unless_debug_mode():
    assert _tables() == [0, False]
    assert _tables(TORCHSIM_DEBUG_MODE="1") == [10, True]


def test_inductor_variables_win():
    got = _tables(TORCHINDUCTOR_AUTOTUNE_NUM_CHOICES_DISPLAYED="3",
                  TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS="1")
    assert got == [3, True], got
    got = _tables(TORCHSIM_DEBUG_MODE="1", TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS="0")
    assert got == [10, False], got


if __name__ == "__main__":
    test_off_unless_debug_mode()
    test_inductor_variables_win()
    print("PASS")
