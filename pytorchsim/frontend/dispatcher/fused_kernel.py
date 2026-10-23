"""The common form of an op the npu runs as a Triton kernel of our own, and the scratchpad budget
its plan is held to. A subclass of CustomFusedKernel states the ops it takes, the kernel's plan, its
launch, an aten reference and, where the op returns more than the kernel, its outputs; `install`
registers them the same way for every op:

    the op's npu kernel, at AutogradPrivateUse1 and PrivateUse1 (where inference_mode lands)
        outside a compiled graph        -> reference, counted "reference"
        declined (plan says why, or a gradient with no backward)   -> reference, logged
        else                            -> the triton_op `name`, which Inductor traces into launch
    the triton_op's own npu kernel      -> reference, counted (a Triton kernel launches only
                                           inside a compiled graph)

`routed` counts the calls that ran each route: each kernel's launches under its key in `kernels()`,
and eager reference calls; a trace, or a reference compiled into a graph, is not a call.
"""

import inspect

import torch
from torch.library import triton_op

from pytorchsim import config

logger = config.setup_logger()

_KEYS = ("AutogradPrivateUse1", "PrivateUse1")


class Decline(str):
    """Why a kernel does not take a call: `plan` returns it in place of the kernel's blocks."""


class CustomFusedKernel:
    """One fused op. `launch(plan, *args)` carries the triton_op's schema in its annotations after
    `plan`, and launches the kernels through wrap_triton, their arguments by position."""

    #: The triton_op a compiled graph calls, "torchsim::<op>".
    name = None
    #: The ops this is the npu kernel of, by qualified name ("aten::topk").
    takes = ()
    #: (ctx, *grads) -> input grads, with setup_context; None makes a gradient decline the kernel.
    backward = None

    def __init__(self):
        self.routed = {}
        self._op = None

    def arguments(self, op, *args, **kwargs):
        """A call of `op` as the triton_op's arguments."""
        return args

    def kernels(self):
        """route -> @triton.jit kernel; each launch counts its route."""
        return {}

    def plan(self, *args):
        """The kernel's blocks (a dict launch reads) for these arguments, or a Decline."""
        raise NotImplementedError

    def launch(self, plan, *args):
        raise NotImplementedError

    def reference(self, *args):
        """What launch returns, as aten ops (differentiable), for every call the kernel does not
        run."""
        raise NotImplementedError

    def outputs(self, op, result, *args):
        """The outputs of `op` from the kernel's (or the reference's) `result` for `args`."""
        return result

    def setup_context(self, ctx, inputs, output):
        pass

    def install(self):
        """Build the triton_op, count each kernel's launches and register every op in `takes`;
        once."""
        if self._op is not None:
            return
        from pytorchsim.backend import codecache
        self._op = self._triton_op()
        for route, kernel in self.kernels().items():
            codecache.observe_launches(kernel, lambda route=route: self._count(route))
        for op in self.takes:
            torch.library.impl(op, _KEYS, self._entry(op))

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
            logger.info("%s on the npu runs its reference, not %s: %s (%s)", op, self.name,
                        _describe(call), why)
            return self.reference(*call)
        return self._op(*call)

    def _decline(self, call):
        if (self.backward is None and torch.is_grad_enabled()
                and any(isinstance(a, torch.Tensor) and a.requires_grad for a in call)):
            return Decline("a gradient, and the kernel has no backward")
        plan = self.plan(*call)
        return plan if isinstance(plan, Decline) else None

    def _eager(self, *args):
        self._count("reference")
        return self.reference(*args)

    def _count(self, route):
        self.routed[route] = self.routed.get(route, 0) + 1


def _describe(call):
    return ", ".join(f"{tuple(a.shape)} {a.dtype}" if isinstance(a, torch.Tensor) else repr(a)
                     for a in call)


def next_pow2(n):
    """The least power of two at or above `n` (1 for 0)."""
    return 1 << max(0, (int(n) - 1).bit_length())


def lane_bytes(tiles):
    """An upper bound on the bytes per lane the [rows, cols] tiles take: each in its own buffer (no
    reuse), banked across whichever axis leaves the most per lane. The compiler picks the axis
    (a decode's [GP, D] accumulator lands on its GP rows, its [GP, BKV] scores on BKV), so the
    bound takes the shorter one, at most the lane count."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    return sum(rows * cols * nbytes // min(rows, cols, lanes) for rows, cols, nbytes in tiles)


def fits_scratchpad(tiles):
    """Whether `tiles` fit one lane's scratchpad budget, and each is a tensor Triton builds."""
    from pytorchsim.frontend.inductor.codegen import tiling
    big = tiling.triton_max_numel()
    return (all(r * c <= big for r, c, _ in tiles)
            and lane_bytes(tiles) <= int(config.CONFIG_MACHINE.spad_budget))
