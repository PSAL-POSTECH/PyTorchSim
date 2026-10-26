"""A graph's resident operands, by XLA's cross-program prefetch rule (its program is the whole graph): an
input every use of which (through views) is a matmul operand or a reduction input is a candidate; per graph,
the largest that fits (copy_node.resident_fits) is held, and only a resident kernel form states it."""

import importlib
import math
import weakref

import torch

from pytorchsim import config, contract

logger = config.setup_logger()

_A = torch.ops.aten
MATMULS = {_A.mm.default: (0, 1), _A.addmm.default: (1, 2), _A.bmm.default: (0, 1),
           _A.baddbmm.default: (1, 2),
           _A.scaled_dot_product_attention.default: (0, 1, 2),
           _A._scaled_dot_product_flash_attention.default: (0, 1, 2),
           _A._scaled_dot_product_efficient_attention.default: (0, 1, 2),
           _A._scaled_dot_product_flash_attention_for_cpu.default: (0, 1, 2)}
ATTENTIONS = tuple(t for t, at in MATMULS.items() if len(at) == 3)
REDUCTIONS = (_A.sum.default, _A.sum.dim_IntList, _A.mean.default, _A.mean.dim, _A.amax.default,
              _A.amin.default,
              _A.max.dim, _A.min.dim, _A.prod.default, _A.prod.dim_int, _A.argmax.default, _A.argmin.default)
VIEWS = (_A.permute.default, _A.t.default, _A.transpose.int, _A.view.default, _A.reshape.default,
         _A._unsafe_view.default, _A.alias.default, _A.detach.default, _A.expand.default, _A.squeeze.default,
         _A.squeeze.dim, _A.unsqueeze.default)
FORMS = {}
CHOICE_FORMS = []

_ELECTED = weakref.WeakKeyDictionary()


def form(*targets):
    """Register `fn(kernel)` as the resident form of `targets`' kernels: the buffer it loads whole to hold,
    None for a kernel of another form."""
    def register(fn):
        FORMS[fn] = targets
        return fn
    return register


def choice_form(fn):
    """Register `fn(ktc)` as the resident form of a template choice: the buffer its config loads whole, None
    for a choice of another template."""
    CHOICE_FORMS.append(fn)
    return fn


def count():
    """How many operands one graph holds on this machine, as the compiler states it (contract/copy_node)."""
    from pytorchsim.backend import compiler_bridge
    return compiler_bridge.compiler_contract("copy_node").graph_prefetches(compiler_bridge.target())


def fits(node):
    """Why the graph input `node` does not fit held whole, None where it does: the compiler's own rule
    (contract/copy_node.resident_fits) of the tensor as the kernel spec states it, its storage span."""
    from pytorchsim.backend import compiler_bridge
    cn, target = compiler_bridge.compiler_contract("copy_node"), compiler_bridge.target()
    val = node.meta["val"]
    shape, size = (contract.storage_span(sizes(val.shape), sizes(val.stride())),), val.dtype.itemsize
    if cn.resident_fits(target, shape, size):
        return None
    return (f"{cn.resident_lane_bytes(target, shape, size)} bytes a lane held whole, over "
            f"{cn.resident_budget(target)}")


def elected(graph):
    """{placeholder target: its consumers} of one lowering: of the candidates (`uses`) in XLA's order
    (copy_node.prefetch_order), the count() first that fit; once per graph, each skip logged."""
    if graph in _ELECTED:
        return _ELECTED[graph]
    fx = graph.module.graph
    position = {n: i for i, n in enumerate(fx.nodes)}
    ids = buffer_ids(fx)
    pool = []
    for node in fx.find_nodes(op="placeholder"):
        got, why = uses(node)
        if why is None and not got:
            why = "no use"
        if why is not None:
            logger.info("[graph_prefetch] %s: no candidate, %s", node.target, why)
            continue
        pool.append((node, got))
    out = {}
    nodes = dict(pool)
    for node in _copy_node().prefetch_order({n: candidate(n, got, position, ids) for n, got in pool}):
        got = nodes[node]
        if len(out) == count():
            break
        why = fits(node)
        if why is not None:
            logger.info("[graph_prefetch] %s: skipped, %s", node.target, why)
            continue
        out[node.target] = [u for u, _ in got]
        bare = sorted({str(u.target) for u, _ in got if not any(u.target in t for t in FORMS.values())})
        if bare:
            logger.info("[graph_prefetch] %s: held; %s have no resident form and stream it",
                        node.target, bare)
    logger.info("[graph_prefetch] the graph holds %s", sorted(out) or "nothing")
    _ELECTED[graph] = out
    return out


def uses(node):
    """([(consumer, operand position)] of `node` through views, the first use that makes it no candidate or
    None): XLA's IsCrossProgramPrefetchCandidate, each use a matmul operand or a reduction input."""
    got = []
    for user in node.users:
        at = [i for i, a in enumerate(user.args) if a is node]
        if user.target in VIEWS and at == [0] and not user.kwargs:
            sub, why = uses(user)
            got += sub
        elif user.target in MATMULS and at and set(at) <= set(MATMULS[user.target]):
            got += [(user, i) for i in at]
            why = None
        elif user.target in REDUCTIONS and at == [0]:
            got.append((user, 0))
            why = None
        else:
            why = f"used by {user.target} (args {at})"
        if why is not None:
            return got, why
    return got, None


def candidate(node, got, position, ids):
    """`node` as XLA's cross-program prefetch comparator reads it (copy_node.Candidate): its bytes, the
    elements its consumers make, its latest use, its buffer id (`buffer_ids`); prefetch_order sorts them."""
    return _copy_node().Candidate(nbytes(node), sum(consumer_numel(u, at) for u, at in got),
                                  max(position[u] for u, _ in got), ids.get(node, len(position)))


def _copy_node():
    """The compiler's contract/copy_node."""
    from pytorchsim.backend import compiler_bridge
    return compiler_bridge.compiler_contract("copy_node")


def buffer_ids(fx):
    """{placeholder: XLA's buffer id}: the order a post-order walk from the outputs, in output order, reaches
    them, each node's operands as XLA's fusion lists them (`fused_operands`), operand 0 first."""
    inputs = set(fx.find_nodes(op="placeholder"))
    roots = [n for out in fx.find_nodes(op="output") for n in _tensors(out.args)]
    ids, seen, stack = {}, set(), [(n, False) for n in reversed(roots)]
    while stack:
        node, done = stack.pop()
        if done:
            if node in inputs:
                ids.setdefault(node, len(ids))
        elif node not in seen:
            seen.add(node)
            stack.append((node, True))
            stack += [(n, False) for n in reversed(fused_operands(node))]
    return ids


def fused_operands(node):
    """`node`'s tensor operands in the order XLA's dot fusion takes them: the second dot operand first where
    the first has a fusible producer (`collapsed`), an attention's v before its scores' operands."""
    args = _tensors((node.args, node.kwargs))
    if node.target not in MATMULS:
        return args
    first, second = (node.args[i] for i in MATMULS[node.target][:2])
    dot = [second, first] if collapsed(first) else [first, second]
    rest = [a for a in args if a not in dot]
    return rest + dot if node.target in ATTENTIONS else dot + rest


def collapsed(node):
    """Whether a view node that lowers the rank (XLA's fusible bitcast producer) sits between the dot operand
    `node` and its source in the FX graph; a transpose does not."""
    while node.target in VIEWS:
        if node.meta["val"].dim() < node.args[0].meta["val"].dim():
            return True
        node = node.args[0]
    return False


def _tensors(tree):
    """The FX nodes in the nest of tuples, lists and dicts `tree`, in order."""
    if isinstance(tree, torch.fx.Node):
        return [tree]
    if isinstance(tree, dict):
        tree = list(tree.values())
    if isinstance(tree, (tuple, list)):
        return [n for t in tree for n in _tensors(t)]
    return []


def consumer_numel(user, at):
    """The elements of what `user` makes of its operand `at`: an attention's q and k feed the scores, its v
    the output; any other consumer its own output."""
    if user.target in ATTENTIONS:
        q, k, v = (user.args[i].meta["val"] for i in range(3))
        last = k.shape[-2] if at < 2 else v.shape[-1]
        return math.prod(sizes(q.shape[:-1])) * int(last)
    val = user.meta["val"]
    return math.prod(sizes((val[0] if isinstance(val, (tuple, list)) else val).shape))


def held_operand(graph, node):
    """The index, among `node`'s matmul operands, of the one the graph holds; None where it holds none."""
    held = elected(graph)
    for i, at in enumerate(MATMULS.get(node.target, ())):
        if storage(node.args[at]).target in held and node in held[storage(node.args[at]).target]:
            return i
    return None


def statement(kernel, args):
    """The names of `args` a kernel holds: the buffer its form loads whole (FORMS) where the graph elected
    it, read as an input argument; none for a kernel of any other form. Elects the graph's first."""
    importlib.import_module(f"{__package__}.mm_resident")
    return _stated([b for fn in FORMS for b in [fn(kernel)] if b is not None], args)


def choice_statement(ktc, args):
    """statement of a template choice timed before it is a kernel (its KernelTemplateChoice `ktc`), so its
    build holds what the kernel it becomes would."""
    importlib.import_module(f"{__package__}.mm_resident")
    return _stated([b for fn in CHOICE_FORMS for b in [fn(ktc)] if b is not None], args)


def _stated(held, args):
    """The names of `args` that read the buffers in `held`, checked against the graph's election."""
    from torch._inductor.virtualized import V
    elected(V.graph)
    if not held:
        return []
    names = [a["name"] for a in args if a["role"] == "in" and a["buffer"] in held]
    if [b for b in held if b not in elected(V.graph)] or not names:
        raise contract.SpecIncomplete(f"a resident form loads {held} whole, which the graph does not hold "
                                      f"({sorted(elected(V.graph))}) or the kernel does not read as an input")
    return names[:1]


def storage(node):
    """The node whose storage `node` views: its chain of view ops followed to the source."""
    while node.target in VIEWS:
        node = node.args[0]
    return node


def nbytes(node):
    """The bytes of the tensor FX node `node` makes."""
    val = node.meta["val"]
    return math.prod(sizes(val.shape)) * val.dtype.itemsize


def sizes(shape):
    """`shape` as integers: a static size as it is, else the graph's hint for it."""
    from torch._inductor.virtualized import V
    return [s if isinstance(s, int) else int(V.graph.sizevars.optimization_hint(s.node.expr)) for s in shape]
