"""A graph lowered with its Triton kernels not compiled: what a test of the lowering's decisions reads."""
import concurrent.futures
import contextlib
from unittest import mock


class _NotCompiled:
    def __init__(self):
        """A kernel that launches nothing, its precompile already done (Inductor's (_, elapsed_us) result)."""
        self.future = concurrent.futures.Future()
        self.future.set_result((None, 0))

    def run(self, *args, **kwargs):
        return None


@contextlib.contextmanager
def kernels_not_compiled():
    """Inductor's async_compile.triton hands back a kernel that launches nothing, so no kernel reaches PTB."""
    from torch._inductor.async_compile import AsyncCompile
    with mock.patch.object(AsyncCompile, "triton", lambda self, *args, **kwargs: _NotCompiled()):
        yield
