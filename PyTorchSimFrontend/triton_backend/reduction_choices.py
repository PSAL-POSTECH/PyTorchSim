"""Tell Inductor this target wants PERSISTENT reductions, through its own hook.

WHY IT IS NOT A PREFERENCE. A non-persistent reduction is what makes Inductor
emit Welford for `var_mean`:

    triton.py, reduction_type == "welford_reduce"
        cooperative     -> self.welford_reduce(...)      the real thing
        otherwise       -> self.welford_reduce_fallback  sum, mean, sum of dx^2

and the persistent path takes the fallback ("don't bother with welford's
algorithm since it uses more registers, and taking two reductions doesn't
increase memory usage"). The two forms are the same numbers; what differs is
what reaches this backend.

The Welford form does not reach it at all. Its combiner folds THREE tensors with
a six-argument body, which triton-shared does not convert, so it arrives at
tnpu as a `tt.reduce` over `(tensor<..>, tensor<..>, tensor<..>)` -- an
unregistered op holding tensors, which bufferize's inliner walks and dies in C++
on (triton-npu 3cd5f33 turned that crash into a refusal by dialect). The machine
has no warp-level Welford to lower it to either: the reduce this backend can
lower is one accumulator folded by a known combiner, in one lane's vector.

So "persistent" here is a statement about the target, not a tuning knob, which
is why it belongs in the choices handler rather than in a config flag.

WHAT THE THRESHOLD IS FOR. Inductor's own rule (choices.py) is 1024 elements for
`ReductionHint.INNER` and 64 for everything else. A LayerNorm over 768 lands on
either side of that depending on how the fusion left the layout -- INNER for the
transformer's, OUTER for ViT's, where the same normalisation is fused with a
patch convolution and a permute. 2048 covers both and stays finite: past that
the tile stops fitting a lane's scratchpad and the refusal moves to a place that
names it (fit_to_hardware), which is the honest failure.

THE HOOK IS INDUCTOR'S. `V.set_choices_handler` with a subclass of
`InductorChoices` is the documented way a backend states these decisions --
no private symbol, no monkeypatch (rule 16). Everything else defers to the base
class, so a shape Inductor already wants persistent stays persistent.
"""
import logging

logger = logging.getLogger(__name__)

#: Past this the tile no longer fits a lane's scratchpad, and the refusal that
#: says so is better than a Welford this backend cannot lower.
MAX_PERSISTENT_RNUMEL = 2048


def install():
    from torch._inductor.choices import InductorChoices
    from torch._inductor.virtualized import V

    class PyTorchSimChoices(InductorChoices):
        """Inductor's choices, with one of them made for this target."""

        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            if InductorChoices.should_use_persistent_reduction(
                    features, cooperative_reduction):
                return True
            try:
                rnumel = int(features.reduction_numel)
            except (TypeError, ValueError):
                # A dynamic extent: nothing to promise, so leave the base
                # class's answer alone rather than guess at a bound.
                return False
            return rnumel <= MAX_PERSISTENT_RNUMEL

    V.set_choices_handler(PyTorchSimChoices())
    logger.info("[triton-npu] persistent reductions preferred up to r=%d "
                "(Welford has no lowering on this target)",
                MAX_PERSISTENT_RNUMEL)
