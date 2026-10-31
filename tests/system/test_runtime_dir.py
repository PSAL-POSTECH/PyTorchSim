"""contract.runtime_dir under TORCHSIM_RUNTIME_DIR: launch-<pid>/runtime links into it, files written there land
in it, a second call keeps them, remove_launch_dir removes both, and without the variable it is a directory again."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pytorchsim import contract  # noqa: E402


def test_runtime_dir_links_into_the_local_root():
    with tempfile.TemporaryDirectory() as shared, tempfile.TemporaryDirectory() as local:
        os.environ[contract.RUNTIME_ROOT_ENV] = local
        try:
            d = contract.runtime_dir(shared)
            assert os.path.islink(d) and os.path.dirname(d) == contract.dir_for(shared)
            with open(os.path.join(d, "x.raw"), "wb") as f:
                f.write(b"1234")
            target = os.path.realpath(d)
            assert target.startswith(os.path.realpath(local)) and os.listdir(target) == ["x.raw"]
            assert contract.runtime_dir(shared) == d and os.listdir(d) == ["x.raw"]
            contract.remove_launch_dir(shared)
            assert not os.path.exists(target) and not os.path.exists(contract.dir_for(shared) + "/runtime")
        finally:
            del os.environ[contract.RUNTIME_ROOT_ENV]
        d = contract.runtime_dir(shared)
        assert os.path.isdir(d) and not os.path.islink(d)


def test_a_link_left_without_the_variable_becomes_a_directory():
    with tempfile.TemporaryDirectory() as shared, tempfile.TemporaryDirectory() as local:
        os.environ[contract.RUNTIME_ROOT_ENV] = local
        try:
            contract.runtime_dir(shared)
        finally:
            del os.environ[contract.RUNTIME_ROOT_ENV]
        d = contract.runtime_dir(shared)
        assert os.path.isdir(d) and not os.path.islink(d)


if __name__ == "__main__":
    test_runtime_dir_links_into_the_local_root()
    test_a_link_left_without_the_variable_becomes_a_directory()
    print("ok")
