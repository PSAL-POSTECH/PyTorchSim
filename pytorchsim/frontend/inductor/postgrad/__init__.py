"""The npu's post-grad pass (Inductor's device_custom_pass): rewrites that keep ops with no npu kernel on
the device, one module per op family; the one-op graph patterns run first, in import order:

    histc.py         histc                    aten.histc counted on the device
    gemm.py          mm_row, addmm_row        a 1 x K row times a matrix as a multiply and a sum: vector work
                     gemm_zero_length         a matmul with an empty output, or K = 0 and no bias, as zeros
    convolution.py   convolution_transposed   a transposed conv as its direct equivalent
                     convolution_1d           a 1-D conv as a 2-D one over a height of 1

then complex.py's complex_real_pairs: complex values as real [..., 2] pairs, by _REAL_PAIR's rules.
"""

import contextlib
import glob
import os

import torch
from torch import fx
from torch._inductor.custom_graph_pass import CustomGraphModulePass, get_hash_for_files
from torch._inductor.pattern_matcher import PatternMatcherPass, register_graph_pattern

from pytorchsim import config
from pytorchsim.frontend.dispatcher.custom_fused_kernel import Decline

logger = config.setup_logger()

# The pass: its graph patterns, then the complex values

PATTERNS = PatternMatcherPass(pass_name="pytorchsim_post_grad_pass")

#: rewrite name -> {"rewritten": n, "fallback": n}: the nodes each rewrite replaced or left.
CALLS = {}

#: The one-op rewrites by name, in the order they are registered; the module docstring lists each.
REWRITES = []


class _PostGradPass(CustomGraphModulePass):
    """The graph patterns, then complex values as real pairs; its uuid hashes every module of this
    package, so an edit to any rewrite is a new cache key."""

    def __call__(self, gm: fx.GraphModule) -> None:
        """Replacements trace under the graph's fake mode: Inductor sets it, a caller outside it may not."""
        from torch._inductor.virtualized import V

        fake_mode = _fake_mode(gm.graph)
        with V.set_fake_mode(fake_mode) if fake_mode is not None else contextlib.nullcontext():
            changed = PATTERNS.apply(gm) + complex_real_pairs(gm.graph, fake_mode)
        if changed:
            gm.graph.eliminate_dead_code()
            gm.recompile()

    def uuid(self):
        return get_hash_for_files(tuple(sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py")))))


pytorchsim_post_grad_pass = _PostGradPass()


# register_rewrite: the form every one-op rewrite takes, its fallback and its counts

def register_rewrite(pattern, extra_check=lambda match: True):
    """Register fn(node, *args, **kwargs) for `pattern` (`extra_check` picks the nodes it applies to). fn
    returns (replacement, args): plain torch code Inductor traces to take the node's place; or a Decline:
    the node stays, logged with where it runs instead (the fallback). Each counts in CALLS."""
    def register(fn):
        def handler(match, *args, **kwargs):
            node = match.output_node()
            got = fn(node, *args, **kwargs)
            if isinstance(got, Decline):
                count(fn.__name__, "fallback")
                logger.info("[post_grad_pass] %s stays: %s; it runs as %s", node.target, got,
                            fallback_route(node.target))
                return
            replacement, example = got
            match.replace_by_example(replacement, example)
            count(fn.__name__, "rewritten")
        register_graph_pattern(pattern, extra_check=extra_check, pass_dict=PATTERNS)(handler)
        if fn.__name__ not in REWRITES:
            REWRITES.append(fn.__name__)
        return fn
    return register


def fallback_route(target):
    """Where a node left as `target` runs: Inductor's own kernel, or an extern call aten makes on the
    CPU, outside the simulator, when Inductor has no lowering for it or falls back."""
    from torch._inductor.lowering import fallbacks, lowerings
    packet = getattr(target, "overloadpacket", None)
    lowered = target in lowerings or packet in lowerings
    if lowered and target not in fallbacks and packet not in fallbacks:
        return "Inductor's own kernel"
    return "an extern aten call on the CPU (not simulated)"


def count(rewrite, route):
    """One more node `rewrite` took down `route` ("rewritten" or "fallback"), in CALLS."""
    counts = CALLS.setdefault(rewrite, {})
    counts[route] = counts.get(route, 0) + 1


# Shared by the rewrites

def fake_value(node):
    """The fake value of a node, or the argument itself when it is not a node."""
    return node.meta.get("val", None) if isinstance(node, fx.Node) else node


def _fake_mode(graph):
    """The fake mode of the graph's tensors, or None."""
    for n in graph.nodes:
        v = fake_value(n)
        if isinstance(v, torch.Tensor) and hasattr(v, "fake_mode"):
            return v.fake_mode
    return None


# The op families, in registration order

from pytorchsim.frontend.inductor.postgrad import histc, gemm, convolution  # noqa: E402,F401
from pytorchsim.frontend.inductor.postgrad.complex import complex_real_pairs  # noqa: E402
