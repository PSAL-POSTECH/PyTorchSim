"""--update-allowlist rewrites the allowlist's entries and keeps the comment block above them, and names
the entries it drops so a regression cannot leave the list unseen. Nothing runs."""
import os
import runpy
import sys
import tempfile

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
SWEEP = runpy.run_path(os.path.join(TORCHSIM_DIR, "scripts", "ci", "triton_route_sweep.py"))
HEAD = "# the header\n#\n# why a test is or is not here\n"


def test_update_keeps_the_comments_and_names_what_it_drops():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "passing.txt")
        with open(path, "w") as f:
            f.write(HEAD + "tests/a.py\ntests/b.py\n")
        dropped = SWEEP["write_allowlist"](["tests/a.py", "tests/c.py"], path)
        with open(path) as f:
            got = f.read()
    assert got == HEAD + "tests/a.py\ntests/c.py\n", got
    assert dropped == ["tests/b.py"], dropped


if __name__ == "__main__":
    test_update_keeps_the_comments_and_names_what_it_drops()
    print("PASS")
