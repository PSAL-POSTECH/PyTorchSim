"""CustomFusedKernel, the form every fused op inherits, and the scratchpad budget its plan is
held to. Eager calls run the fallback; compiled ones the triton_op, or the fallback (logged) where
plan declines or a gradient has no backward. How to write one: docs/fused_kernel.md."""

import inspect

import torch
from torch.library import triton_op

from pytorchsim import config

logger = config.setup_logger()

_KEYS = ("AutogradPrivateUse1", "PrivateUse1")


class Decline(str):
    """Why a kernel does not take a call: `plan` returns it in place of the kernel's blocks."""


class CustomFusedKernel:
    """One fused op; launch's annotations after `plan` are the triton_op's schema."""

    #: The triton_op a compiled graph calls, "torchsim::<op>".
    name = None
    #: OpOverloads this is the npu kernel of; a qualified name only for an op defined after install.
    takes = ()
    #: (ctx, *grads) -> input grads, with setup_context; without it a gradient declines the kernel.
    backward = None

    def __init__(self):
        self.calls = {}
        self._op = None

    def arguments(self, op, *args, **kwargs):
        """A call of `op` as the triton_op's arguments."""
        return args

    def kernels(self):
        """The @triton.jit kernels launch may launch; each launch counts under its name."""
        return ()

    def plan(self, *args):
        """The kernel's blocks (a dict launch reads) for these arguments, or a Decline."""
        raise NotImplementedError

    def launch(self, plan, *args):
        raise NotImplementedError

    def fallback(self, *args):
        """launch's result as aten ops, differentiable, for every call the kernel does not run."""
        raise NotImplementedError

    def outputs(self, op, result, *args):
        """`op`'s outputs from `result`; the result itself unless the op returns more."""
        return result

    def setup_context(self, ctx, inputs, output):
        pass

    def install(self):
        """Build the triton_op, count launches and register every op in `takes`; once."""
        if self._op is not None:
            return
        from pytorchsim.backend import codecache
        self._op = self._triton_op()
        for kernel in self.kernels():
            codecache.observe_launches(kernel, lambda name=kernel.__name__: self._count(name))
        for op in self.takes:
            torch.library.impl(op if isinstance(op, str) else op.name(), _KEYS, self._entry(op))

    def _triton_op(self):
        sig = inspect.signature(self.launch)

        def body(*args):
            return self.launch(self.plan(*args), *args)

        body.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])
        op = triton_op(self.name, body, mutates_args=())
        op.register_kernel("npu")(self._eager)
        if self.backward is not None:
            op.register_autograd(self.backward, setup_context=self.setup_context)
        return op

    def _entry(self, op):
        def entry(*args, **kwargs):
            call = self.arguments(op, *args, **kwargs)
            return self.outputs(op, self._route(op, call), *call)
        return entry

    def _route(self, op, call):
        if not torch.compiler.is_compiling():
            return self._eager(*call)
        why = self._decline(call)
        if why is not None:
            logger.info("%s on the npu runs its fallback, not %s: %s (%s)", op, self.name,
                        _describe(call), why)
            return self.fallback(*call)
        return self._op(*call)

    def _decline(self, call):
        if (self.backward is None and torch.is_grad_enabled()
                and any(isinstance(a, torch.Tensor) and a.requires_grad for a in call)):
            return Decline("a gradient, and the kernel has no backward")
        plan = self.plan(*call)
        return plan if isinstance(plan, Decline) else None

    def _eager(self, *args):
        self._count("fallback")
        return self.fallback(*args)

    def _count(self, route):
        self.calls[route] = self.calls.get(route, 0) + 1


def _describe(call):
    return ", ".join(f"{tuple(a.shape)} {a.dtype}" if isinstance(a, torch.Tensor) else repr(a)
                     for a in call)


def next_pow2(n):
    """The least power of two at or above `n` (1 for 0)."""
    return 1 << max(0, (int(n) - 1).bit_length())


def lane_bytes(tiles):
    """The most bytes per lane the [rows, cols, bytes] tiles take: no buffer reuse, each
    banked along its shorter axis, at most the lane count."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    return sum(rows * cols * nbytes // min(rows, cols, lanes) for rows, cols, nbytes in tiles)


def fits_scratchpad(tiles):
    """Whether `tiles` fit one lane's scratchpad budget, and each is a tensor Triton builds."""
    from pytorchsim.frontend import tiling
    big = tiling.triton_max_numel()
    return (all(r * c <= big for r, c, _ in tiles)
            and lane_bytes(tiles) <= int(config.CONFIG_MACHINE.spad_budget))
