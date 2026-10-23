"""The paper's artifact scripts name their machine the way this branch reads one, TORCHSIM_TARGET:
run_cycle.sh hands its workloads tpuv3 for the compiler to judge, and a speedup script whose
machine file is not shipped stops before it runs anything instead of reporting NA with rc 0."""
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
CYCLE = os.path.join(REPO, "experiments/artifact/cycle_validation/run_cycle.sh")
ILS = "experiments/artifact/speedup/scripts/run_speed_ils_matmul.sh"
RESOLVE = ("import torch; from pytorchsim.driver import compiler_bridge; "
           "print(compiler_bridge.target().file)")


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("TOGSIM_CONFIG", "TORCHSIM_TARGET")}
    env["TORCHSIM_DIR"] = REPO
    env.update(extra)
    return env


def test_cycle_script_names_tpuv3_through_torchsim_target():
    cmd = f'source "{CYCLE}" --only none && cd "{REPO}" && "{sys.executable}" -c "{RESOLVE}"'
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=_env())
    out = proc.stdout.strip().splitlines()
    assert "TOGSIM_CONFIG" not in proc.stderr, proc.stderr
    assert ((proc.returncode == 0 and os.path.basename(out[-1]) == "tpuv3.yml")
            or "NoTarget: TORCHSIM_TARGET=tpuv3:" in proc.stderr), proc.stdout + proc.stderr


def test_speedup_script_refuses_a_machine_that_is_not_shipped(tmp_path):
    root = tmp_path / "pts"
    shutil.copytree(os.path.join(REPO, "experiments"), root / "experiments")
    os.symlink(os.path.join(REPO, "pytorchsim"), root / "pytorchsim")
    proc = subprocess.run(["bash", str(root / ILS)], capture_output=True, text=True,
                          env=_env(TORCHSIM_DIR=str(root)), timeout=600)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "does not exist" in proc.stderr, proc.stderr
    assert not (root / "experiments/artifact/speedup/results").exists()
