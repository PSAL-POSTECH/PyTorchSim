"""A graph lowered with its Triton kernels not compiled: what a test of the lowering's decisions reads."""
import contextlib
from unittest import mock


class _NotCompiled:
    def run(self, *args, **kwargs):
        return None


@contextlib.contextmanager
def kernels_not_compiled():
    """Inductor's async_compile.triton hands back a kernel that launches nothing, so no kernel reaches PTB."""
    from torch._inductor.async_compile import AsyncCompile
    with mock.patch.object(AsyncCompile, "triton", lambda self, name, source, device_str="cuda": _NotCompiled()):
        yield
