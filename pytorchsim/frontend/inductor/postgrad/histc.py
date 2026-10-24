"""histc: aten.histc, which has no npu kernel, counted on the device."""

import functools

import torch
from torch._inductor.pattern_matcher import CallFunctionVarArgs

from pytorchsim.frontend.dispatcher.custom_fused_kernel import Decline
from pytorchsim.frontend.inductor.postgrad import fake_value, register_rewrite

aten = torch.ops.aten


# histc: aten.histc counted on the device

@register_rewrite(CallFunctionVarArgs(aten.histc.default))
def histc(node, x, *rest, **kwargs):
    """aten.histc counted on the device (counts). A floating input with min == max is left to aten:
    its range comes from its values, and aten refuses a range that is not finite."""

    def counts(x, *, bins, lo, hi):
        """aten's histc: bin (x - lo) * bins / (hi - lo), integral in int64 and floating in x's dtype, the top
        edge in the last bin, outside [lo, hi] dropped; lo == hi takes x's range, widened by 1 if one value."""
        integral = not x.is_floating_point()
        flat = x.reshape(-1).to(torch.int64 if integral else x.dtype)
        if lo == hi and x.numel():
            lo, hi = flat.amin(0), flat.amax(0)
            one = lo == hi
            lo, hi = torch.where(one, lo - 1, lo), torch.where(one, hi + 1, hi)
        else:
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            if integral:
                lo, hi = int(lo), int(hi)
            if hi <= lo:
                raise RuntimeError(f"torch.histc: max must be larger than min (min {lo}, max {hi})")
        scaled = (flat - lo) * bins
        if integral:
            pos = torch.div(scaled, hi - lo, rounding_mode="floor")
        else:
            pos = (scaled / (hi - lo)).to(torch.int64)
        pos = pos.clamp_max(bins - 1)
        inside = (flat >= lo) & (flat <= hi)
        seats = torch.arange(bins, dtype=torch.int64, device=x.device)
        hits = (pos.reshape(-1, 1) == seats.reshape(1, -1)) & inside.reshape(-1, 1)
        return hits.sum(0).to(x.dtype)

    if fake_value(x) is None:
        return Decline("its input has no fake value")
    said = dict(zip(("bins", "min", "max"), rest), **kwargs)
    lo, hi = said.get("min", 0), said.get("max", 0)
    if lo == hi and fake_value(x).is_floating_point():
        return Decline("min == max over a floating input: aten refuses a range of its values that is not finite")
    return functools.partial(counts, bins=int(said.get("bins", 100)), lo=lo, hi=hi), [x]
