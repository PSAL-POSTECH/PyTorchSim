"""scripts/setup_worktree.sh with no base ref branches off origin/pytorchsim-triton-backend, the
branch this repository develops on; it used to name origin/develop, which the remote does not have."""
import os
import shutil
import subprocess

REPO = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
BASE = "pytorchsim-triton-backend"


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_default_base_is_a_branch_the_remote_has(tmp_path):
    origin, clone = tmp_path / "origin", tmp_path / "PyTorchSim"
    _git("init", "-q", "-b", BASE, str(origin), cwd=tmp_path)
    shutil.copytree(os.path.join(REPO, "scripts"), origin / "scripts")
    _git("add", "scripts", cwd=origin)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base", cwd=origin)
    _git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    proc = subprocess.run(["bash", str(clone / "scripts/setup_worktree.sh"), "probe"],
                          capture_output=True, text=True, cwd=clone)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path / "PyTorchSim-probe",
                          capture_output=True, text=True, check=True).stdout
    assert head == subprocess.run(["git", "rev-parse", f"origin/{BASE}"], cwd=clone,
                                  capture_output=True, text=True, check=True).stdout
