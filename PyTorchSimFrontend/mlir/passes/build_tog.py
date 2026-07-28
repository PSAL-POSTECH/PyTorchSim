"""Python port of the C++ `-test-tile-operation-graph` analysis pass.

Reads a post-vcix MLIR file, walks `func.func @kernel`, and prints the Tile
Operation Graph (TOG) as a Python `graph = { id: {dict}, ... }` literal to
stdout. The output is byte-exact with the upstream C++ pass
(mlir/test/lib/Analysis/TestTileOperationGraph.cpp + the display() methods in
mlir/include/mlir/Analysis/TileOperationGraph.h).

Usage:
    python3 build_tog.py <postvcix.mlir>

Requires the MLIR Python bindings on PYTHONPATH
(/riscv-llvm/python_packages/mlir_core).
"""

import os
import sys

# Allow running standalone without PYTHONPATH set.
_DEFAULT_BINDINGS = "/riscv-llvm/python_packages/mlir_core"
if os.path.isdir(_DEFAULT_BINDINGS) and _DEFAULT_BINDINGS not in sys.path:
    sys.path.insert(0, _DEFAULT_BINDINGS)

import mlir.ir as ir  # noqa: E402


# ---------------------------------------------------------------------------
# Node kinds / compute types (mirror TileOperationGraph.h enums).
# ---------------------------------------------------------------------------
BASE_NODE = 0
COMPUTE_NODE = 1
LOOP_NODE = 2
DMA_NODE = 3
DMA_WAIT_NODE = 4

VECTOR_COMPUTE = 0
MATMUL_COMPUTE = 1
MATMUL_PRELOAD = 2

# Printed compute_type string for each compute-node type (mirrors the C++
# inline-asm marker emission in TestTileOperationGraph.cpp).
_COMPUTE_TYPE_NAME = {
    VECTOR_COMPUTE: "VectorCompute",
    MATMUL_COMPUTE: "MatmulCompute",
    MATMUL_PRELOAD: "MatmulPreload",
}


# Unique-id counter (TOGNode::unique_id), incremented at construction. The C++
# pass keeps this as a process global, but each kernel runs in its own mlir-opt
# PROCESS so the counters never collide. Here the pass runs in-process and
# extension_codecache compiles kernels CONCURRENTLY in a thread pool, so a single
# module global would race across threads (one kernel's reset/increments interleave
# with another's, leaving nodes -- including the root -- with wrong ids). Keep the
# counter thread-local so each compile thread has an isolated counter, mirroring
# the per-process isolation of the C++ pass.
import threading  # noqa: E402

_ids = threading.local()


def _new_id():
    i = getattr(_ids, "counter", 0)
    _ids.counter = i + 1
    return i


def _reset_ids():
    _ids.counter = 0


# ---------------------------------------------------------------------------
# Node classes with display() mirroring the C++ header byte-for-byte.
# ---------------------------------------------------------------------------
class TOGNode:
    kind = BASE_NODE

    def __init__(self, name, node_id=None):
        self.node_id = _new_id() if node_id is None else node_id
        self.node_name = name
        self.parents = []
        self.children = []
        self.op = None

    def get_last_child(self):
        return self.children[-1] if self.children else None

    def add_child(self, c):
        self.children.append(c)

    def add_parent(self, p):
        self.parents.append(p)

    @staticmethod
    def _print_list(name, vec, out):
        out.append('\t"%s": [' % name)
        out.append(",".join(str(v) for v in vec))
        out.append("]")

    @staticmethod
    def _print_str_list(name, vec, out):
        out.append('\t"%s": [' % name)
        out.append(",".join('"%s"' % v for v in vec))
        out.append("]")

    def _print_node_list(self, name, vec, out):
        out.append('\t"%s": [' % name)
        out.append(",".join(str(n.node_id) for n in vec))
        out.append("]")

    def display(self, out):
        out.append("%d : {\n" % self.node_id)
        out.append('\t"node_id": %d,\n' % self.node_id)
        out.append('\t"node_name": "%s",\n' % self.node_name)
        out.append('\t"node_type": %d,\n' % self.kind)
        self._print_node_list("parents", self.parents, out)
        out.append(",\n")
        self._print_node_list("children", self.children, out)
        if self.kind == BASE_NODE:
            out.append("\n}\n")

    def bfs(self, out):
        if self.node_name == "root":
            out.append("graph = {\n")
        self.display(out)
        out.append(",")
        for child in self.children:
            child.bfs(out)
        if self.node_name == "root":
            out.append("}")


class TOGLoopNode(TOGNode):
    kind = LOOP_NODE

    def __init__(self, name, idx, start, end, step, loop_type):
        super().__init__(name)
        self.loop_idx = idx
        self.loop_start = start
        self.loop_end = end
        self.loop_step = step
        self.loop_type = loop_type

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"loop_index": "%s",\n' % self.loop_idx)
        out.append('\t"loop_start": %d,\n' % self.loop_start)
        out.append('\t"loop_end": %d,\n' % self.loop_end)
        out.append('\t"loop_step": %d,\n' % self.loop_step)
        out.append('\t"loop_type": "%s"\n' % self.loop_type)
        out.append("}\n")


class TOGDMANode(TOGNode):
    kind = DMA_NODE

    def __init__(self, name, addr, tile_size, tile_stride, elem_size, is_write,
                 is_async, tag_idx_list, tag_stride_list, loop_idx_list,
                 loop_stride_list, indirect_mode):
        super().__init__(name)
        self.base_addr = addr
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.element_size = elem_size
        self.is_write = is_write
        self.is_async = is_async
        self.tag_idx_list = tag_idx_list
        self.tag_stride_list = tag_stride_list
        self.loop_idx_list = loop_idx_list
        self.loop_stride_list = loop_stride_list
        self.indirect_mode = indirect_mode

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"is_write": %d,\n' % int(self.is_write))
        out.append('\t"is_async": %d,\n' % int(self.is_async))
        out.append('\t"base_address": "%s",\n' % self.base_addr)
        out.append('\t"indirect_mode": %d,\n' % int(self.indirect_mode))
        self._print_list("tile_size", self.tile_size, out)
        out.append(",\n")
        self._print_list("tile_stride", self.tile_stride, out)
        out.append(",\n")
        self._print_str_list("tag_idx_list", self.tag_idx_list, out)
        out.append(",\n")
        self._print_list("tag_stride_list", self.tag_stride_list, out)
        out.append(",\n")
        self._print_str_list("loop_idx_list", self.loop_idx_list, out)
        out.append(",\n")
        self._print_list("loop_stride_list", self.loop_stride_list, out)
        out.append(",\n")
        out.append('\t"element_size": %d\n' % self.element_size)
        out.append("}\n")


class TOGDMAWaitNode(TOGNode):
    kind = DMA_WAIT_NODE

    def __init__(self, name, idx_list, tag_stride_list, tag_divider_list, addr):
        super().__init__(name)
        self.tag_idx_list = idx_list
        self.tag_stride_list = tag_stride_list
        self.tag_divider_list = tag_divider_list
        self.base_addr = addr

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"base_address": "%s",\n' % self.base_addr)
        self._print_str_list("tag_idx_list", self.tag_idx_list, out)
        out.append(",\n")
        self._print_list("tag_stride_list", self.tag_stride_list, out)
        out.append(",\n")
        self._print_list("tag_divider_list", self.tag_divider_list, out)
        out.append("}\n")


class TOGComputeNode(TOGNode):
    kind = COMPUTE_NODE

    def __init__(self, name, cycle, ctype):
        super().__init__(name)
        self.compute_cycle = cycle
        self.compute_type = ctype
        self.operations = []

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"compute_cycle": %d,\n' % self.compute_cycle)
        out.append('\t"compute_type": %d\n' % self.compute_type)
        out.append("}\n")


# ---------------------------------------------------------------------------
# MLIR helpers.
# ---------------------------------------------------------------------------
def _op_name(value_or_op):
    return value_or_op.operation.name


def _is_block_arg(value):
    return ir.BlockArgument.isinstance(value)


def _defining_op_name(value):
    """Return the op name that defines `value`, or None if it is a block arg."""
    if _is_block_arg(value):
        return None
    return value.owner.name


def _const_index_value(value):
    """If `value` is an arith.constant of index type, return its int value."""
    if _is_block_arg(value):
        return None
    owner = value.owner
    if owner.name != "arith.constant":
        return None
    attr = owner.attributes["value"]
    try:
        return ir.IntegerAttr(attr).value
    except Exception:
        return None


def _value_key(value):
    """Identity key for a Value (induction vars map -> loop name)."""
    if _is_block_arg(value):
        ba = ir.BlockArgument(value)
        return ("ba", ba.owner, ba.arg_number)
    return ("res", value.owner, 0)


def _memref_space(memref_type):
    mt = ir.MemRefType(memref_type)
    sp = mt.memory_space
    if sp is None:
        return 0
    return ir.IntegerAttr(sp).value


def _int_array_attr(op, key):
    if key not in op.attributes:
        return None
    arr = ir.ArrayAttr(op.attributes[key])
    return [ir.IntegerAttr(x).value for x in arr]


# ----- affine expr utilities (mirror the C++ free functions) ---------------
def _is_function_of_dim(expr, dim):
    if ir.AffineDimExpr.isinstance(expr):
        return ir.AffineDimExpr(expr).position == dim
    if ir.AffineBinaryExpr.isinstance(expr):
        b = ir.AffineBinaryExpr(expr)
        return _is_function_of_dim(b.lhs, dim) or _is_function_of_dim(b.rhs, dim)
    return False


def _get_coefficient_from_dim(expr, dim):
    """Port of getCoefficientFromDim: coefficient of `dim` in expr, or -1."""
    if ir.AffineMulExpr.isinstance(expr):
        b = ir.AffineMulExpr(expr)
        lhs, rhs = b.lhs, b.rhs
        if ir.AffineConstantExpr.isinstance(lhs) and ir.AffineDimExpr.isinstance(rhs):
            if _is_function_of_dim(rhs, dim):
                return ir.AffineConstantExpr(lhs).value
        elif ir.AffineConstantExpr.isinstance(rhs) and ir.AffineDimExpr.isinstance(lhs):
            if _is_function_of_dim(lhs, dim):
                return ir.AffineConstantExpr(rhs).value
        return -1
    if ir.AffineAddExpr.isinstance(expr):
        b = ir.AffineAddExpr(expr)
        r = _get_coefficient_from_dim(b.lhs, dim)
        if r != -1:
            return r
        r = _get_coefficient_from_dim(b.rhs, dim)
        if r != -1:
            return r
        return -1
    if ir.AffineDimExpr.isinstance(expr):
        if _is_function_of_dim(expr, dim):
            return 1
    return -1


def _collect_coefficients(expr):
    """Port of collectCoefficientsFromAffineExpr."""
    out = []

    def rec(e):
        if ir.AffineMulExpr.isinstance(e):
            b = ir.AffineMulExpr(e)
            lhs, rhs = b.lhs, b.rhs
            if ir.AffineConstantExpr.isinstance(lhs):
                if ir.AffineDimExpr.isinstance(rhs):
                    out.append(ir.AffineConstantExpr(lhs).value)
                elif ir.AffineFloorDivExpr.isinstance(rhs):
                    out.append(ir.AffineConstantExpr(lhs).value)
            elif ir.AffineConstantExpr.isinstance(rhs):
                if ir.AffineDimExpr.isinstance(lhs):
                    out.append(ir.AffineConstantExpr(rhs).value)
                elif ir.AffineFloorDivExpr.isinstance(lhs):
                    out.append(ir.AffineConstantExpr(rhs).value)
        elif ir.AffineAddExpr.isinstance(e):
            b = ir.AffineAddExpr(e)
            rec(b.lhs)
            rec(b.rhs)
        elif ir.AffineFloorDivExpr.isinstance(e):
            b = ir.AffineFloorDivExpr(e)
            if ir.AffineConstantExpr.isinstance(b.rhs):
                out.append(ir.AffineConstantExpr(b.rhs).value)
        elif ir.AffineDimExpr.isinstance(e):
            out.append(1)

    rec(expr)
    return out


def _collect_dividers(expr):
    """Port of collectDividersFromAffineExpr."""
    out = []

    def rec(e):
        if ir.AffineMulExpr.isinstance(e):
            b = ir.AffineMulExpr(e)
            lhs, rhs = b.lhs, b.rhs
            if ir.AffineConstantExpr.isinstance(lhs):
                if ir.AffineDimExpr.isinstance(rhs):
                    out.append(1)
                elif ir.AffineFloorDivExpr.isinstance(rhs):
                    rec(rhs)
            elif ir.AffineConstantExpr.isinstance(rhs):
                if ir.AffineDimExpr.isinstance(lhs):
                    out.append(1)
                elif ir.AffineFloorDivExpr.isinstance(lhs):
                    rec(lhs)
        elif ir.AffineAddExpr.isinstance(e):
            b = ir.AffineAddExpr(e)
            rec(b.lhs)
            rec(b.rhs)
        elif ir.AffineFloorDivExpr.isinstance(e):
            b = ir.AffineFloorDivExpr(e)
            if ir.AffineConstantExpr.isinstance(b.rhs):
                out.append(ir.AffineConstantExpr(b.rhs).value)
        elif ir.AffineDimExpr.isinstance(e):
            out.append(1)

    rec(expr)
    return out


# ---------------------------------------------------------------------------
# The builder (mirrors TestTileOperationGraph members).
# ---------------------------------------------------------------------------
SKIP_OPS = {
    "affine.yield", "affine.apply", "memref.get_global", "arith.constant",
    "memref.alloc", "memref.reinterpret_cast",
}


class MatmulFsm:
    Idle = 0
    Preload = 1
    MMPush = 2
    MMVpop = 3
    MMEpilogue = 4


class TogBuilder:
    def __init__(self):
        self.nr_loop = 0
        self.loop_var_name = {}  # value-identity-key -> loop name
        self.compute_nodes = []
        self.loop_nodes = []
        # `_collect_dma_nodes` descends from the loop nodes, so a DMA hanging off
        # the root (no tile loop in the kernel) would be missed.
        self.dma_nodes = []
        self._reset_matmul_fsm()

    # ---- matmul FSM ----
    def _reset_matmul_fsm(self):
        self.matmul_fsm = MatmulFsm.Idle
        self.mm_iv_op0_count = 0
        self.mm_expected_vpop = 0
        self.mm_vpop_seen = 0
        self.mm_expected_transfer_writes = 0
        self.mm_transfer_writes_seen = 0
        self.current_preload_node = None
        self.current_matmul_compute_node = None

    def _finish_matmul_block(self):
        self.matmul_fsm = MatmulFsm.Idle
        self.current_matmul_compute_node = None
        self.mm_iv_op0_count = 0
        self.mm_expected_vpop = 0
        self.mm_vpop_seen = 0
        self.mm_expected_transfer_writes = 0
        self.mm_transfer_writes_seen = 0

    def _enter_epilogue_or_finish(self):
        if self.mm_expected_transfer_writes == 0:
            self._finish_matmul_block()
        else:
            self.matmul_fsm = MatmulFsm.MMEpilogue

    def _append_mm_with_write_count(self, op):
        cn = self.current_matmul_compute_node
        if cn is None:
            return
        cn.operations.append(op)
        if _op_name(op) != "vector.transfer_write":
            return
        if self.mm_expected_transfer_writes == 0:
            return
        self.mm_transfer_writes_seen += 1
        if self.mm_transfer_writes_seen >= self.mm_expected_transfer_writes:
            self._finish_matmul_block()

    def _steal_leading_transfer_read(self, parent):
        prepend = []
        if parent is None:
            return prepend
        last = parent.get_last_child()
        if last is None or not isinstance(last, TOGComputeNode):
            return prepend
        if last.compute_type != VECTOR_COMPUTE:
            return prepend
        ops = last.operations
        idx = None
        for i, o in enumerate(ops):
            if _op_name(o) == "vector.transfer_read":
                idx = i
                break
        if idx is None:
            return prepend
        prepend.append(ops[idx])
        del ops[idx]
        if not ops:
            if parent.children and parent.children[-1] is last:
                parent.children.pop()
            else:
                parent.children = [c for c in parent.children if c is not last]
            if last in self.compute_nodes:
                self.compute_nodes.remove(last)
            # node deleted; its id stays consumed (matches C++ unique_id).
        return prepend

    def _append_vector_compute(self, parent, op):
        if parent is None:
            return
        last = parent.get_last_child()
        if (isinstance(last, TOGComputeNode)
                and last.compute_type == VECTOR_COMPUTE):
            last.operations.append(op)
            return
        cn = TOGComputeNode("ComputeNode", 0, VECTOR_COMPUTE)
        cn.op = op
        cn.operations.append(op)
        parent.add_child(cn)
        cn.add_parent(parent)
        self.compute_nodes.append(cn)

    # ---- vcix classification ----
    @staticmethod
    def _vcix_iv_opcode(op):
        if _op_name(op) != "vcix.iv":
            return None
        if "opcode" not in op.operation.attributes:
            return None
        return ir.IntegerAttr(op.operation.attributes["opcode"]).value

    @staticmethod
    def _vcix_vi_opcode(op):
        if _op_name(op) != "vcix.v.i":
            return None
        if "opcode" not in op.operation.attributes:
            return None
        return ir.IntegerAttr(op.operation.attributes["opcode"]).value

    # ---- affine.for bounds ----
    @staticmethod
    def _affine_for_bounds(forop):
        oper = forop.operation
        step = ir.IntegerAttr(oper.attributes["step"]).value
        lb_map = ir.AffineMapAttr(oper.attributes["lowerBoundMap"]).value
        ub_map = ir.AffineMapAttr(oper.attributes["upperBoundMap"]).value
        # operandSegmentSizes: [lb operands, ub operands, iter operands]
        seg_attr = oper.attributes["operandSegmentSizes"]
        seg = [seg_attr[i] for i in range(len(seg_attr))]
        operands = list(oper.operands)

        def single_const(m):
            return len(m.results) == 1 and ir.AffineConstantExpr.isinstance(m.results[0])

        if single_const(lb_map):
            start = ir.AffineConstantExpr(lb_map.results[0]).value
        else:
            start = _const_index_value(operands[0])
        n_lb = seg[0] if seg else 0
        if single_const(ub_map):
            end = ir.AffineConstantExpr(ub_map.results[0]).value
        else:
            end = _const_index_value(operands[n_lb])
        return start, end, step

    # ---- DRAM index processing ----
    def _process_dram_indices(self, value, loop_index_list, indirect_box):
        if not _is_block_arg(value) and value.owner.name == "affine.apply":
            apply_op = value.owner
            amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
            operands = list(apply_op.operands)
            if "indirect_access" in apply_op.attributes:
                indirect_box[0] = True
            for op_v in operands:
                if _is_block_arg(op_v):
                    expr = amap.results[0]
                    index_pos = operands.index(op_v)
                    coeff = _get_coefficient_from_dim(expr, index_pos)
                    key = self.loop_var_name[_value_key(op_v)]
                    loop_index_list.append((key, coeff))
                else:
                    self._process_dram_indices(op_v, loop_index_list, indirect_box)
        elif _is_block_arg(value):
            key = self.loop_var_name[_value_key(value)]
            loop_index_list.append((key, 1))
        else:
            c = _const_index_value(value)
            if c is not None:
                loop_index_list.append(("c" + str(c), c))

    # ---- main recursion ----
    def visit_operation(self, op, node):
        """Walk `op` and attach the nodes it produces under `node`.

        Builds the graph; it does not print. (The C++ pass this is ported from
        does both in one method, `printOperation` -- here `bfs`/`display` own
        the printing.)
        """
        name = _op_name(op)
        if name in SKIP_OPS:
            return

        if name == "affine.for":
            oper = op.operation
            attrs = oper.attributes
            loop_type = ""

            def bool_true(k):
                return k in attrs and ir.BoolAttr(attrs[k]).value

            if bool_true("outer_loop"):
                loop_type = "outer_loop"
            elif bool_true("accumulation_loop"):
                loop_type = "accumulation_loop"
            elif bool_true("inner_loop"):
                loop_type = "inner_loop"

            if (bool_true("outer_loop") or bool_true("accumulation_loop")
                    or bool_true("inner_loop")):
                start, end, step = self._affine_for_bounds(op)
                loop_index = "loop_arg%03d" % self.nr_loop
                self.nr_loop += 1
                iter_var = oper.regions[0].blocks[0].arguments[0]
                self.loop_var_name[_value_key(iter_var)] = loop_index
                loop_node = TOGLoopNode("loopNode", loop_index, start, end, step,
                                        loop_type)
                loop_node.op = op
                self.loop_nodes.append(loop_node)
                if node is not None:
                    node.add_child(loop_node)
                    loop_node.add_parent(node)
                for region in oper.regions:
                    for block in region.blocks:
                        for inner in block.operations:
                            self.visit_operation(inner, loop_node)
                return

        if name == "togsim.transfer":
            self._handle_dma_start(op, node)
            return

        if name == "togsim.wait":
            self._handle_dma_wait(op, node)
            return

        if node is None:
            return

        self._handle_compute(op, node)

    # ---- compute / matmul FSM dispatch ----
    def _handle_compute(self, op, node):
        if _op_name(op) == "vcix.iv":
            if (self.matmul_fsm == MatmulFsm.MMEpilogue
                    and self.current_matmul_compute_node):
                self._finish_matmul_block()
            opc = self._vcix_iv_opcode(op)
            if opc is not None:
                if opc == 1:
                    if self.matmul_fsm == MatmulFsm.Idle:
                        prepend = self._steal_leading_transfer_read(node)
                        cn = TOGComputeNode("ComputeNode", 0, MATMUL_PRELOAD)
                        for o in prepend:
                            cn.operations.append(o)
                        cn.operations.append(op)
                        cn.op = cn.operations[0]
                        node.add_child(cn)
                        cn.add_parent(node)
                        self.compute_nodes.append(cn)
                        self.current_preload_node = cn
                        self.matmul_fsm = MatmulFsm.Preload
                    elif (self.matmul_fsm == MatmulFsm.Preload
                          and self.current_preload_node):
                        self.current_preload_node.operations.append(op)
                    else:
                        raise RuntimeError("vcix.iv opcode=1 invalid state")
                    return
                if opc == 0:
                    if (self.matmul_fsm == MatmulFsm.Preload
                            and self.current_preload_node):
                        cn = TOGComputeNode("ComputeNode", 0, MATMUL_COMPUTE)
                        cn.op = op
                        cn.operations.append(op)
                        node.add_child(cn)
                        cn.add_parent(node)
                        self.compute_nodes.append(cn)
                        self.current_preload_node = None
                        self.current_matmul_compute_node = cn
                        self.matmul_fsm = MatmulFsm.MMPush
                        self.mm_iv_op0_count = 1
                        return
                    if (self.matmul_fsm == MatmulFsm.MMPush
                            and self.current_matmul_compute_node):
                        self.current_matmul_compute_node.operations.append(op)
                        self.mm_iv_op0_count += 1
                        return
                    raise RuntimeError("vcix.iv opcode=0 invalid state")
            if (self.matmul_fsm == MatmulFsm.MMEpilogue
                    and self.current_matmul_compute_node):
                self._append_mm_with_write_count(op)
                return
            self._append_vector_compute(node, op)
            return

        if _op_name(op) == "vcix.v.i":
            viopc = self._vcix_vi_opcode(op)
            if viopc is not None and viopc == 2:
                if (self.matmul_fsm == MatmulFsm.MMPush
                        and self.current_matmul_compute_node):
                    self.mm_expected_vpop = self.mm_iv_op0_count
                    self.mm_expected_transfer_writes = self.mm_expected_vpop
                    self.mm_transfer_writes_seen = 0
                    self.matmul_fsm = MatmulFsm.MMVpop
                    self.mm_vpop_seen = 1
                    self.current_matmul_compute_node.operations.append(op)
                    if self.mm_vpop_seen >= self.mm_expected_vpop:
                        self._enter_epilogue_or_finish()
                    return
                if (self.matmul_fsm == MatmulFsm.MMVpop
                        and self.current_matmul_compute_node):
                    self.current_matmul_compute_node.operations.append(op)
                    self.mm_vpop_seen += 1
                    if self.mm_vpop_seen >= self.mm_expected_vpop:
                        self._enter_epilogue_or_finish()
                    return
            if (self.matmul_fsm == MatmulFsm.MMEpilogue
                    and self.current_matmul_compute_node):
                self._append_mm_with_write_count(op)
                return
            self._append_vector_compute(node, op)
            return

        if self.matmul_fsm == MatmulFsm.Preload and self.current_preload_node:
            self.current_preload_node.operations.append(op)
            return
        if (self.matmul_fsm == MatmulFsm.MMEpilogue
                and self.current_matmul_compute_node):
            self._append_mm_with_write_count(op)
            return
        if (self.matmul_fsm in (MatmulFsm.MMPush, MatmulFsm.MMVpop)
                and self.current_matmul_compute_node):
            self._append_mm_with_write_count(op)
            return
        self._append_vector_compute(node, op)

    # ---- transfer (formerly memref.dma_start) ----
    @staticmethod
    def _transfer_is_load(op, dma_type_operand):
        """True if a `togsim.transfer` is a load (DRAM -> SRAM), False for a store.

        Prefer the `dma_kind` attr ("MVIN"/"MVIN2"/"MVIN3" load, "MVOUT" store);
        fall back to the `dma_type` operand constant (MVIN=2/MVIN2=1/MVIN3=14 are
        loads, MVOUT=3 is a store)."""
        oper = op.operation
        if "dma_kind" in oper.attributes:
            try:
                kind = ir.StringAttr(oper.attributes["dma_kind"]).value
            except Exception:
                kind = str(oper.attributes["dma_kind"]).strip('"')
            if kind:
                return kind.upper().startswith("MVIN")
        c = _const_index_value(dma_type_operand)
        if c is not None:
            return c in (1, 2, 14)
        return True

    def _dma_start_fields(self, op):
        """Decode a `togsim.transfer` into the legacy src/dst view.

        togsim.transfer operand layout (mirrors build_skeleton._transfer_fields /
        lower_transfer_to_gemmini):
            dram, dram_idx, sram, sram_idx, tag, tag_idx, dma_type, vst[, offset]
        The DRAM side is always operand[0]/[1], the SRAM spad operand[2]/[3], the
        runtime tag slot operand[4] (tag memref) + operand[5] (tag_idx). The
        optional indirect-offset spad is operand[8].

        Direction (from dma_kind / dma_type) decides the src/dst mapping so the
        rest of build_tog keeps the old memref.dma_start convention: for a load
        the DRAM side is the SOURCE and the SRAM spad the DEST; a store reverses
        it. src/dst therefore still carry the right memory spaces (DRAM=0,
        SRAM=1) that `_handle_dma_start` recovers is_write from."""
        operands = list(op.operands)
        dram, dram_idx = operands[0], operands[1]
        sram, sram_idx = operands[2], operands[3]
        tag, tag_idx = operands[4], operands[5]
        dma_type = operands[6]
        offset = operands[8] if len(operands) > 8 else None

        if self._transfer_is_load(op, dma_type):   # DRAM -> SRAM
            src, src_idx = dram, dram_idx
            dst, dst_idx = sram, sram_idx
        else:                                       # SRAM -> DRAM
            src, src_idx = sram, sram_idx
            dst, dst_idx = dram, dram_idx
        return {
            "src": src, "src_type": src.type, "src_indices": [src_idx],
            "dst": dst, "dst_type": dst.type, "dst_indices": [dst_idx],
            "tag": tag, "tag_indices": [tag_idx],
            "dma_type": dma_type, "offset": offset,
        }

    def _handle_dma_start(self, op, node):
        oper = op.operation
        f = self._dma_start_fields(op)
        dma_async = bool(self._get_async(oper))

        dst_space = _memref_space(f["dst_type"])
        src_space = _memref_space(f["src_type"])

        subtile = _int_array_attr(oper, "subtile_size")

        if dst_space == 0 and src_space == 1:
            is_write = True
            tile_type = f["src_type"]
            dram_memref = f["dst"]
            dram_indices = f["dst_indices"]
        elif dst_space == 1 and src_space == 0:
            is_write = False
            tile_type = f["dst_type"]
            dram_memref = f["src"]
            dram_indices = f["src_indices"]
        else:
            raise RuntimeError("Unexpected memory space")

        tile_mt = ir.MemRefType(tile_type)
        tile_shape = subtile if subtile else list(tile_mt.shape)
        tile_size = [int(x) for x in tile_shape]

        tile_stride = []
        ds = _int_array_attr(oper, "dram_stride")   # TOGDMANode.tile_stride keeps the DRAM stride (as before)
        if ds:
            tile_stride = list(ds)

        loop_index_map = []
        indirect_box = [False]
        self._process_dram_indices(dram_indices[0], loop_index_map, indirect_box)

        # std::map<string,...> => dedup + lexicographic sort by key.
        reordered = {}
        for key, stride in loop_index_map:
            if key not in reordered:
                reordered[key] = stride
        loop_idx_list = []
        loop_stride_list = []
        for key in sorted(reordered.keys()):
            loop_idx_list.append(key)
            loop_stride_list.append(reordered[key])

        # base address: which tensor this DMA touches. The operand is the block
        # argument itself in PyTorchSim's codegen; when it is a view of one
        # instead, only the producer knows which -- so it says so (`dram_arg`)
        # rather than the consumer guessing its way back through view ops.
        address = "arg"
        if "dram_arg" in oper.attributes:
            address += str(ir.IntegerAttr(oper.attributes["dram_arg"]).value)
        elif _is_block_arg(dram_memref):
            address += str(ir.BlockArgument(dram_memref).arg_number)

        # element size
        et = tile_mt.element_type
        if ir.IntegerType.isinstance(et):
            element_size = ir.IntegerType(et).width
        elif ir.FloatType.isinstance(et):
            element_size = ir.FloatType(et).width
        else:
            raise RuntimeError("Unsupported element type")

        # tag indices
        tag_index_list = []
        tag_stride_list = []
        for tag_idx in f["tag_indices"]:
            c = _const_index_value(tag_idx)
            if c is not None:
                tag_index_list.append(str(c))
                continue
            if not _is_block_arg(tag_idx) and tag_idx.owner.name == "affine.apply":
                apply_op = tag_idx.owner
                for operand in apply_op.operands:
                    if _is_block_arg(operand):
                        tag_index_list.append(
                            self.loop_var_name[_value_key(operand)])
                    elif (not _is_block_arg(operand)
                          and operand.owner.name == "affine.apply"):
                        nested = operand.owner
                        for nop in nested.operands:
                            if _is_block_arg(nop):
                                tag_index_list.append(
                                    self.loop_var_name[_value_key(nop)])
                            else:
                                nc = _const_index_value(nop)
                                if nc is not None:
                                    tag_index_list.append(str(nc))
                    else:
                        oc = _const_index_value(operand)
                        if oc is not None:
                            tag_index_list.append(str(oc))
                amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
                tag_stride_list = _collect_coefficients(amap.results[0])

        if len(tag_index_list) == 0:
            tag_index_list.append("0")
        if len(tag_stride_list) == 0:
            tag_stride_list.append(1)

        dma_node = TOGDMANode("DMANode", address, tile_size, tile_stride,
                              element_size, is_write, dma_async, tag_index_list,
                              tag_stride_list, loop_idx_list, loop_stride_list,
                              indirect_box[0])
        dma_node.op = op
        self.dma_nodes.append(dma_node)
        node.add_child(dma_node)
        dma_node.add_parent(node)

    # ---- dma_wait ----
    def _handle_dma_wait(self, op, node):
        oper = op.operation
        # togsim.wait operands: (tag[0], tag_idx[1], num_elements[2]). tag_idx is
        # now a single operand (memref.dma_wait had a variadic [tag_idx]).
        operands = list(oper.operands)
        tag = operands[0]
        tag_indices = operands[1:2]

        tag_index_list = []
        tag_stride_list = []
        tag_divider_list = []
        for tag_idx in tag_indices:
            if not _is_block_arg(tag_idx) and tag_idx.owner.name == "affine.apply":
                apply_op = tag_idx.owner
                for operand in apply_op.operands:
                    if _is_block_arg(operand):
                        tag_index_list.append(
                            self.loop_var_name[_value_key(operand)])
                    else:
                        c = _const_index_value(operand)
                        if c is not None:
                            tag_index_list.append(str(c))
                amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
                tag_stride_list = _collect_coefficients(amap.results[0])
                tag_divider_list = _collect_dividers(amap.results[0])

        # base address: scan users of tag memref for a togsim.transfer.
        address = "arg"
        for use in tag.uses:
            user = use.owner
            if user.name == "togsim.transfer":
                f = self._dma_start_fields(user)
                dst_space = _memref_space(f["dst_type"])
                src_space = _memref_space(f["src_type"])
                dram_memref = None
                if dst_space == 0 and src_space == 1:
                    dram_memref = f["dst"]
                elif dst_space == 1 and src_space == 0:
                    dram_memref = f["src"]
                if "dram_arg" in user.attributes:
                    address += str(ir.IntegerAttr(user.attributes["dram_arg"]).value)
                elif dram_memref is not None and _is_block_arg(dram_memref):
                    address += str(ir.BlockArgument(dram_memref).arg_number)

        if len(tag_stride_list) == 0:
            tag_stride_list.append(1)
            tag_divider_list.append(1)

        wait_node = TOGDMAWaitNode("DMAWaitNode", tag_index_list, tag_stride_list,
                                   tag_divider_list, address)
        wait_node.op = op
        self.dma_nodes.append(wait_node)
        node.add_child(wait_node)
        wait_node.add_parent(node)

    @staticmethod
    def _get_async(oper):
        if "async" not in oper.attributes:
            return 0
        attr = oper.attributes["async"]
        try:
            return ir.IntegerAttr(attr).value
        except Exception:
            pass
        try:
            return 1 if ir.BoolAttr(attr).value else 0
        except Exception:
            return 1


# ---------------------------------------------------------------------------
# Empty affine.for erasure (fixed point), mirroring eraseEmptyAffineForLoops.
# ---------------------------------------------------------------------------
def _is_empty_affine_for(op):
    if op.operation.name != "affine.for":
        return False
    body = op.operation.regions[0].blocks[0]
    ops = list(body.operations)
    # body holds exactly the terminator (affine.yield) and nothing else.
    if len(ops) != 1:
        return False
    return ops[0].operation.name == "affine.yield"


def _erase_empty_affine_for(func_op):
    changed = True
    while changed:
        changed = False
        to_erase = []

        def walk(block):
            for op in block.operations:
                for region in op.operation.regions:
                    for b in region.blocks:
                        walk(b)
                if _is_empty_affine_for(op):
                    to_erase.append(op)

        walk(func_op.regions[0].blocks[0])
        for op in to_erase:
            op.operation.erase()
            changed = True


# ---------------------------------------------------------------------------
# IR mutation (mirrors the side effects of -test-tile-operation-graph).
# ---------------------------------------------------------------------------
def _make_inline_asm(ctx, ip, asm_string, compute_type=None):
    """Build an llvm.inline_asm compute marker identical to the C++ pass."""
    i64 = ir.IntegerType.get_signless(64)
    attrs = {
        "has_side_effects": ir.UnitAttr.get(),
        "asm_dialect": ir.IntegerAttr.get(i64, 0),
        "asm_string": ir.StringAttr.get(asm_string),
        "constraints": ir.StringAttr.get("~{a0},~{memory}"),
    }
    if compute_type is not None:
        attrs["compute_type"] = ir.StringAttr.get(compute_type)
    return ir.Operation.create(
        "llvm.inline_asm", results=[], operands=[], attributes=attrs,
        loc=ir.Location.unknown(ctx), ip=ip)


def _containing_block(op):
    """Return the Block that directly contains `op`."""
    parent = op.operation.parent
    for region in parent.regions:
        for block in region.blocks:
            for sib in block.operations:
                if sib.operation == op.operation:
                    return block
    return None


def _next_sibling(op):
    """Return the op immediately following `op` in its block, or None."""
    block = _containing_block(op)
    siblings = list(block.operations)
    for i, sib in enumerate(siblings):
        if sib.operation == op.operation:
            return siblings[i + 1] if i + 1 < len(siblings) else None
    return None


_ASM_START = ".insn r CUSTOM_3, 0, 0x40, x0, x0, x0"
_ASM_END = ".insn r CUSTOM_3, 0, 0x41, x0, x0, x0"


def _rewrite_loop_steps(builder):
    """Sample mode: rewrite each loop node's affine.for step to its end.

    Done after the graph is built so the printed graph keeps the original step.
    """
    index_t = ir.IndexType.get()
    for node in builder.loop_nodes:
        forop = node.op.operation
        new_step = ir.IntegerAttr.get(index_t, node.loop_end)
        forop.attributes["step"] = new_step


def _insert_compute_markers(builder):
    """Insert inline-asm start/end markers around each compute node's ops."""
    for node in builder.compute_nodes:
        ops = node.operations
        if not ops:
            continue
        front = ops[0]
        back = ops[-1]
        ctx = front.operation.context
        ctype_name = _COMPUTE_TYPE_NAME[node.compute_type]
        # End marker immediately after `back`: insert before back's next sibling
        # (or append to the block when back is the last op). The bindings have no
        # "insert after" point, and move_after on a detached op crashes.
        after = _next_sibling(back)
        if after is not None:
            end_ip = ir.InsertionPoint(after)
        else:
            end_ip = ir.InsertionPoint(_containing_block(back))
        _make_inline_asm(ctx, end_ip, _ASM_END)
        # Start marker immediately before `front`. Done after the end marker so
        # that the single-op case (front == back) still brackets the op.
        _make_inline_asm(ctx, ir.InsertionPoint(front), _ASM_START, ctype_name)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def _find_kernel(module):
    """The kernel function: named `kernel` in PyTorchSim's codegen, else the
    module's only func.func (triton-npu carries the Triton kernel's own name).
    Declines when there is more than one -- the intent would be a guess."""
    funcs = [op for op in module.body.operations
             if op.operation.name == "func.func"]
    for op in funcs:
        if ir.StringAttr(op.operation.attributes["sym_name"]).value == "kernel":
            return op
    return funcs[0] if len(funcs) == 1 else None


#: The loop roles (sec 9.1). Without one, a loop is a micro-loop the compute FSM
#: folds into a single node, not a tile loop.
_LOOP_ROLE_ATTRS = ("outer_loop", "accumulation_loop", "inner_loop")


def _has_loop_role(op):
    attrs = op.operation.attributes
    return any(k in attrs and ir.BoolAttr(attrs[k]).value for k in _LOOP_ROLE_ATTRS)


def _is_address_plumbing(op):
    """Scalar index/integer math (DMA offsets, mask extents) and the terminator.

    Only consulted on the no-top-level-loop path. PyTorchSim's codegen puts this
    math in `affine.apply`, which SKIP_OPS drops; triton-npu emits an
    arith/index_cast chain that would otherwise count as vector compute.

    Keyed on result type: tile data here is always vector- or float-typed. A
    top-level SCALAR arithmetic kernel would be misread, but no path emits one.
    """
    name = _op_name(op)
    if name in ("func.return", "memref.cast"):
        return True
    if not name.startswith("arith."):
        return False
    results = list(op.operation.results)
    if not results:
        return False
    return all(ir.IndexType.isinstance(r.type) or ir.IntegerType.isinstance(r.type)
               for r in results)


def _build(module, builder):
    """Build the graph and return its display string, populating `builder`."""
    func_op = _find_kernel(module)
    if func_op is None:
        return ""

    _erase_empty_affine_for(func_op)

    block = func_op.regions[0].blocks[0]
    out = []
    # A root is a top-level TILE loop, identified by its role attribute (sec
    # 9.1) -- not by being an affine.for: bank_vectorize leaves a bare one for
    # the tile's vector work, and rooting there orphans every DMA.
    roots = [op for op in block.operations
             if op.operation.name == "affine.for" and _has_loop_role(op)]
    if roots:
        for op in roots:
            root = TOGNode("root")
            builder._reset_matmul_fsm()
            builder.visit_operation(op, root)
            root.bfs(out)
        return "".join(out)

    # No top-level loop: the body is ONE work-item -- the shape a Triton kernel
    # arrives in, its grid becoming the trace producer's dispatch loop (sec 9.3).
    # PyTorchSim's codegen keeps the tile loops in the kernel and never lands here.
    root = TOGNode("root")
    builder._reset_matmul_fsm()
    for op in block.operations:
        if _is_address_plumbing(op):
            continue
        builder.visit_operation(op, root)
    root.bfs(out)
    return "".join(out)


def build_tog_string(module):
    # The C++ unique_id is a process global; each mlir-opt invocation starts at
    # 0. When this runs in-process per kernel, reset so node ids match byte-exact.
    _reset_ids()
    return _build(module, TogBuilder())


def build_tog_and_mutate(module, sample_mode=True):
    """Build the TOG and apply the C++ pass IR mutation in place.

    Returns the graph display string (byte-exact with build_tog_string). The
    caller is responsible for serializing the mutated module (str(module)).
    """
    _reset_ids()
    builder = TogBuilder()
    result = _build(module, builder)
    if sample_mode:
        _rewrite_loop_steps(builder)
    _insert_compute_markers(builder)
    return result


def run_tog(in_path, tog_out_path, custom_out_path, sample_mode=True, vectorlane=128):
    """In-process replacement for the C++ -test-tile-operation-graph pass.

    Reads the post-vcix MLIR at `in_path`, builds the TOG (written as the
    `graph = {...}` Python literal to `tog_out_path`) and applies the pass IR
    mutation (sample-mode step rewrite + compute markers), writing the mutated
    module to `custom_out_path`. `vectorlane` is accepted for parity with the
    C++ option; it does not affect the output. Requires the MLIR bindings.
    """
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(in_path).read(), ctx)
        graph = build_tog_and_mutate(module, sample_mode=bool(sample_mode))
        with open(custom_out_path, "w") as fh:
            fh.write(str(module))
    with open(tog_out_path, "w") as fh:
        fh.write(graph)


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="build_tog.py")
    parser.add_argument("input")
    parser.add_argument("--sample-mode", type=int, default=1, choices=(0, 1))
    parser.add_argument("--vectorlane", type=int, default=128)
    parser.add_argument("--mutate-out", default=None)
    args = parser.parse_args(argv[1:])

    with open(args.input) as fh:
        src = fh.read()
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(src, ctx)
        if args.mutate_out is not None:
            result = build_tog_and_mutate(module, sample_mode=bool(args.sample_mode))
            with open(args.mutate_out, "w") as fh:
                fh.write(str(module))
        else:
            result = build_tog_string(module)
    sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
