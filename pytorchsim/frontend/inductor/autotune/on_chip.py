"""A graph's intermediates kept in the scratchpad, as v6e XLA keeps a W8A8 Linear's int8 activation in VMEM (memory
space S(1)) for the matmul fusion that reads it: a tensor a kernel makes, every use of which (through views) is a
matmul operand, stays on chip from its maker to its last reader where it fits beside the others kept with it."""

import weakref

import torch

from pytorchsim import config
from . import graph_prefetch

logger = config.setup_logger()

READERS = {**graph_prefetch.MATMULS, torch.ops.aten._int_mm.default: (0, 1)}

_ELECTED = weakref.WeakKeyDictionary()


def elected(graph):
    """{FX node: (a lane's bytes, its position, its last reader's)} of one lowering: in graph order each node
    every use of which is a matmul operand (`readers`), kept where it fits copy_node.resident_budget beside the
    graph's resident inputs and the nodes kept over its life; once per graph, each skip logged."""
    if graph in _ELECTED:
        return _ELECTED[graph]
    from pytorchsim.backend import compiler_bridge
    fx = graph.module.graph
    position = {n: i for i, n in enumerate(fx.nodes)}
    held = graph_prefetch.elected(graph)
    budget = compiler_bridge.compiler_contract("copy_node").resident_budget(compiler_bridge.target()) - sum(
        graph_prefetch.lane_bytes(n) for n in fx.find_nodes(op="placeholder") if n.target in held)
    kept = {}
    for node in fx.nodes:
        if node.op != "call_function" or node.target in graph_prefetch.VIEWS:
            continue
        got, why = readers(node)
        if why is not None or not got:
            continue
        need, start, end = graph_prefetch.lane_bytes(node), position[node], max(position[u] for u in got)
        beside = sum(b for b, s, e in kept.values() if s <= end and e >= start)
        if beside + need > budget:
            logger.info("[on_chip] %s: not kept, %d bytes a lane beside %d of %d", node, need, beside, budget)
            continue
        kept[node] = (need, start, end)
    logger.info("[on_chip] the graph keeps %s", sorted(map(str, kept)) or "nothing")
    _ELECTED[graph] = kept
    return kept


def readers(node):
    """([the matmuls reading `node` through views], the first use that is no matmul operand or None)."""
    got = []
    for user in node.users:
        at = [i for i, a in enumerate(user.args) if a is node]
        if user.target in graph_prefetch.VIEWS and at == [0] and not user.kwargs:
            sub, why = readers(user)
            if why is not None:
                return got, why
            got += sub
        elif user.target in READERS and at and set(at) <= set(READERS[user.target]):
            got.append(user)
        else:
            return got, f"used by {user.target} (args {at})"
    return got, None


def statement(args):
    """The names of `args` (a kernel's meta args) whose buffer holds a node the graph keeps on chip: an output
    its kernel leaves there, an input its kernel reads there."""
    from torch._inductor.virtualized import V
    kept = elected(V.graph)
    if not kept:
        return []
    out = []
    for a in args:
        buf = V.graph.name_to_buffer.get(a.get("buffer"))
        if buf is not None and buf.get_origin_node() in kept:
            out.append(a["name"])
    return out
