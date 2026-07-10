"""dep_analysis.py -- dependency-edge analysis for the C++ trace pipeline (sec 10).

The current TOG pass does NO dependency analysis (it emits a lexical loop tree +
runtime tags). This module derives the producer->consumer edges that the explicit
dataflow trace needs, from two sources available on the post-vcix IR (before
build_skeleton collapses the compute regions):

  1. SRAM access: each DMA/compute's read/write SRAM buffer(s), recovered by
     following SSA (a vcix.iv's input vector -> its vector.transfer_read -> the
     memref -> @global), and the DMA's spad operand. Edge: a reader depends on
     the last node that wrote the same buffer.
  2. vcix preload/matmul pairing: a matmul (vcix opcode 0) consumes the weights a
     preceding preload (opcode 1) loaded into the systolic array -- an SA-internal
     dependency NOT visible as a memref access, so it comes from the opcode order.

This is a node-level analysis (one node per build_tog compute/DMA node); the loops
replay the nodes, so loop-carried edges (the Y_spad accumulator) are materialized
per iteration downstream. First cut: buffer granularity (slot-level value matching
is a later refinement). Output is an edge list for validation / to drive emit.
"""
import sys
import os

from .build_tog import TogBuilder, ir, _reset_ids
from . import build_skeleton as _bs


def _global_of(memref_val):
    """memref SSA value -> @global symbol name (e.g. 'X_spad'), or None."""
    owner = memref_val.owner
    op = owner if isinstance(owner, ir.Operation) else getattr(owner, "operation", None)
    if op is None:
        return None
    if op.name == "memref.get_global":
        return str(op.attributes["name"]).strip('@" ')
    # walk through view-like ops (subview/cast) to their source
    if op.operands:
        try:
            return _global_of(op.operands[0])
        except Exception:
            return None
    return None


# Ops that touch SRAM-buffer DATA, by category. A view op only computes an address,
# so it is skipped; the real access is the load/store using it. Anything else with a
# memref operand raises, catching a new fusion pattern at compile time.
_LOAD_OPS = {"vector.transfer_read", "affine.vector_load", "vector.load",
             "memref.load", "affine.load"}
_STORE_OPS = {"vector.transfer_write", "affine.vector_store", "vector.store",
              "memref.store", "affine.store"}
_IGNORE_OPS = {"memref.dealloc"}   # lifetime, not a data access


def _is_memref(v):
    try:
        return ir.MemRefType.isinstance(v.type)
    except Exception:
        return False


def _walk_compute_ops(cn):
    """Every op in the compute node, recursing into nested regions (loop bodies). A
    fused epilogue (BatchNorm/ReLU) keeps its ops inside an un-unrolled affine.for, so
    a top-level-only scan (cn.operations) sees just the loop and misses every access."""
    for top in cn.operations:
        stack = [top]
        while stack:
            op = stack.pop()
            yield op
            for region in op.operation.regions:
                for block in region.blocks:
                    stack.extend(block.operations)


def _rw_buffers_of_compute(cn):
    """(reads, writes): the @global SRAM buffers a compute node reads/writes, walking
    nested regions and classifying each op that touches a memref."""
    reads, writes = set(), set()
    def rd(v):
        b = _global_of(v)
        if b:
            reads.add(b)
    def wr(v):
        b = _global_of(v)
        if b:
            writes.add(b)
    for op in _walk_compute_ops(cn):
        if any(_is_memref(r) for r in op.results):
            continue                                   # view/cast/alloc -- address only
        mrefs = [v for v in op.operands if _is_memref(v)]
        if not mrefs:
            continue
        name = op.name
        if name in _LOAD_OPS:
            for v in mrefs:
                rd(v)
        elif name in _STORE_OPS:
            for v in mrefs:
                wr(v)                                  # the store target memref
        elif name == "memref.copy":
            rd(mrefs[0])
            wr(mrefs[-1])
        elif name.startswith("linalg."):               # DPS: ins read, outs read+write
            for v in op.inputs:
                if _is_memref(v):
                    rd(v)
            for v in op.outputs:
                if _is_memref(v):
                    rd(v)
                    wr(v)
        elif name in _IGNORE_OPS:
            continue
        else:
            raise RuntimeError(
                f"dep_analysis: unclassified memref op '{name}' in a compute node -- "
                f"it touches an SRAM buffer; classify it in _LOAD_OPS/_STORE_OPS")
    return reads, writes


def _dma_buffer(builder, dma_node):
    """The SRAM spad buffer a DMA touches (dst for load, src for store)."""
    try:
        f = builder._dma_start_fields(dma_node.op)
    except Exception:
        return None
    val = f["dst"] if not dma_node.is_write else f["src"]
    return _global_of(val)


# Virtual buffer for the systolic-array weight registers: a preload writes it,
# the following matmul reads it. This folds the SA-internal preload->matmul
# dependency (not a memref access) into the uniform "last-writer per buffer" rule.
SA_WEIGHTS = "__SA_WEIGHTS__"


def compute_buffers(cn):
    """(read_buffers, write_buffers) for one compute node, including the virtual
    SA_WEIGHTS edge (preload writes it, matmul reads it)."""
    reads, writes = _rw_buffers_of_compute(cn)
    if cn.compute_type == 1:      # MATMUL consumes the preloaded weights
        reads.add(SA_WEIGHTS)
    elif cn.compute_type == 2:    # PRELOAD loads them
        writes.add(SA_WEIGHTS)
    return reads, writes


def analyze(module):
    """Return (nodes, edges). nodes: list of dicts; edges: list of (consumer_idx,
    producer_idx, reason)."""
    _reset_ids()
    builder = TogBuilder()
    _bs._build(module, builder)

    nodes = []
    # DMA nodes only (the map also contains TOGDMAWaitNode; keep real DMAs).
    dma_nodes = [dn for dn in dict.fromkeys(_bs._collect_dma_nodes(builder).values())
                 if hasattr(dn, "is_write")]
    for dn in dma_nodes:
        buf = _dma_buffer(builder, dn)
        nodes.append({
            "kind": "STORE" if dn.is_write else "LOAD",
            "buf": buf, "arg": str(dn.base_addr),
            "reads": {buf} if dn.is_write else set(),
            "writes": {buf} if not dn.is_write else set(),
            "node": dn,
        })
    for cn in builder.compute_nodes:
        if not cn.operations:
            continue
        ct = {0: "VECTOR", 1: "MATMUL", 2: "PRELOAD"}.get(cn.compute_type, f"c{cn.compute_type}")
        creads, cwrites = _rw_buffers_of_compute(cn)
        nodes.append({
            "kind": ct,
            "reads": creads,
            "writes": cwrites,
            "node": cn,
            "compute_type": cn.compute_type,
        })

    # Order nodes by program position (last-writer needs program order: e.g. the
    # store reads Y_spad written by the matmul, which lexically precedes it).
    pos = {}
    idx = [0]
    def _index(op):
        pos[op] = idx[0]; idx[0] += 1
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    _index(o)
    _index(module.operation)
    def _key(n):
        node = n["node"]
        op = getattr(node, "op", None) or (node.operations[0] if getattr(node, "operations", None) else None)
        return pos.get(op, 1 << 30)
    nodes.sort(key=_key)

    # Edges: (1) buffer last-writer, (2) preload->matmul.
    edges = []
    last_writer = {}  # buffer -> node idx
    prev_preload = None
    for i, n in enumerate(nodes):
        for b in sorted(n["reads"]):
            if b in last_writer:
                edges.append((i, last_writer[b], f"reads {b}"))
        if n["kind"] == "MATMUL" and prev_preload is not None:
            edges.append((i, prev_preload, "uses preloaded weights (vcix op1->op0)"))
        for b in n["writes"]:
            last_writer[b] = i
        if n["kind"] == "PRELOAD":
            prev_preload = i
    return nodes, edges


def _main():
    path = sys.argv[1]
    ctx = ir.Context(); ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(path).read(), ctx)
        nodes, edges = analyze(module)
    print("=== nodes ===")
    for i, n in enumerate(nodes):
        r = ",".join(sorted(n["reads"])) or "-"
        w = ",".join(sorted(n["writes"])) or "-"
        print(f"  #{i:<2} {n['kind']:<8} reads[{r}] writes[{w}]")
    print("=== edges (consumer -> producer) ===")
    for c, p, why in edges:
        print(f"  #{c} ({nodes[c]['kind']}) -> #{p} ({nodes[p]['kind']})   [{why}]")


if __name__ == "__main__":
    _main()
