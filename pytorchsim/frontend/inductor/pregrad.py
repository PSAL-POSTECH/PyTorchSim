"""The npu's pre-grad pass (Inductor's pre_grad_custom_pass): an npu SDPA handed repeat_kv's copies of the key
and value heads takes the shared heads with enable_gqa, before AOT traces the dispatcher's kernel over the copies.
Attention reaches the npu as an SDPA call (or torchsim::attention); eager matmul attention is not rewritten."""

import os
import operator

import torch
from torch import fx
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

#: rewrite name -> {"rewritten": n}: the SDPA nodes each rewrite replaced.
CALLS = {}

SDPA = (torch._C._nn.scaled_dot_product_attention, torch.nn.functional.scaled_dot_product_attention)


class _PreGradPass(CustomGraphPass):
    """gqa_shared_kv over the graph; its uuid hashes this file, so an edit is a new cache key."""

    def __call__(self, graph: fx.Graph) -> None:
        gqa_shared_kv(graph)

    def uuid(self):
        return get_hash_for_files((os.path.abspath(__file__),))


pytorchsim_pre_grad_pass = _PreGradPass()


def gqa_shared_kv(graph):
    """Each npu SDPA whose key and value are repeat_kv copies [B, H*R, S, D] of [B, H, S, D] takes the shared
    heads and enable_gqa=True; the copies' nodes go when nothing else reads them. Returns how many it rewired."""
    changed = 0
    for node in list(graph.nodes):
        if not _is_npu_sdpa(node) or len(node.args) < 3:
            continue
        q, k, v = node.args[:3]
        got = [_get_repeated_kv(t) for t in (k, v)]
        if any(g is None for g in got) or got[0][1] != got[1][1]:
            continue
        if _get_shape(q)[1] != _get_shape(k)[1]:
            continue
        node.args = (q, got[0][0], got[1][0], *node.args[3:])
        node.kwargs = {**node.kwargs, "enable_gqa": True}
        _remove_unused_chain(graph, (k, v))
        _count("gqa_shared_kv")
        changed += 1
    return changed


def _is_npu_sdpa(node):
    if node.op != "call_function" or node.target not in SDPA:
        return False
    out = node.meta.get("example_value")
    return isinstance(out, torch.Tensor) and out.device.type == "npu"


def _get_repeated_kv(node):
    """(kv, R) when `node` is repeat_kv's reshape(expand(kv[:, :, None])) to [B, H*R, S, D], R > 1, else None."""
    reshape = node
    if not _is_call(reshape, ("reshape", "view"), (torch.reshape,)):
        return None
    expand = reshape.args[0]
    if not _is_call(expand, ("expand",), ()):
        return None
    unsqueeze = expand.args[0]
    if not _is_unsqueeze_2(unsqueeze):
        return None
    kv = unsqueeze.args[0]
    kvs, es, rs = _get_shape(kv), _get_shape(expand), _get_shape(reshape)
    if kvs is None or es is None or rs is None or len(kvs) != 4 or len(es) != 5:
        return None
    B, H, S, D = kvs
    R = es[2]
    if not isinstance(R, int) or R <= 1 or es != [B, H, R, S, D] or rs != [B, H * R, S, D]:
        return None
    return kv, R


def _is_call(node, methods, functions):
    if not isinstance(node, fx.Node):
        return False
    return ((node.op == "call_method" and node.target in methods)
            or (node.op == "call_function" and node.target in functions))


def _is_unsqueeze_2(node):
    """Whether `node` is x.unsqueeze(2) or x[:, :, None] (full slices and one None at 2) of a 4-D x."""
    if _is_call(node, ("unsqueeze",), (torch.unsqueeze,)):
        return len(node.args) > 1 and node.args[1] == 2
    if not (isinstance(node, fx.Node) and node.op == "call_function" and node.target is operator.getitem):
        return False
    index = node.args[1]
    full = slice(None, None, None)
    return (isinstance(index, tuple) and len(index) >= 3 and index[:3] == (full, full, None)
            and all(i == full for i in index[3:]))


def _get_shape(node):
    v = node.meta.get("example_value") if isinstance(node, fx.Node) else None
    return list(v.shape) if isinstance(v, torch.Tensor) else None


def _remove_unused_chain(graph, heads):
    for head in dict.fromkeys(heads):
        node = head
        while isinstance(node, fx.Node) and not node.users and node.op in ("call_method", "call_function"):
            parent = node.args[0]
            graph.erase_node(node)
            node = parent


def _count(rewrite):
    counts = CALLS.setdefault(rewrite, {})
    counts["rewritten"] = counts.get("rewritten", 0) + 1
