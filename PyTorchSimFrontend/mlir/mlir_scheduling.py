import os
import math
import sympy
from functools import reduce
import operator
from sympy import symbols, sympify
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

from torch.utils._ordered_set import OrderedSet
from torch._inductor import config
from torch._inductor.scheduler import BaseScheduling, FusedSchedulerNode, SchedulerNode, BaseSchedulerNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._inductor.ir import LoopBody
from torch._inductor import dependencies
from torch._inductor.codegen.common import BackendFeature

from . import mlir_common
from . import mlir_lowering # DO NOT REMOVE THIS LINE, it is used for lowering
from . import mlir_decomposition # DO NOT REMOVE THIS LINE, it is used for decomposition

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        if scheduler is not None:
            self.scheduler.can_fuse_origin = self.scheduler.can_fuse
            self.scheduler.can_fuse = self.can_fuse_with_exceptions # FIXME. Monkey patch: For prolouge fusion
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()
        self._ready_to_flush = False
        self.outer_function = set()
        config.inplace_buffers = False # FIXME. inout kernel makes trouble.. So disabled it!
        self.max_fusion_size = 5

    def can_fuse_with_exceptions(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        if not extension_config.CONFIG_FUSION_PROLOGUE:
            return self.scheduler.can_fuse_origin(node1, node2)

        # Extract base template node
        base_template_node1 = [node for node in node1.get_nodes() if node.is_template()]
        base_template_node2 = [node for node in node2.get_nodes() if node.is_template()]

        # Case 3: Prologue(Pointwise) + Tempalte
        if len(base_template_node1) == 0 and len(node1.get_nodes())==1 and len(node2.get_nodes())==1 and not node1.is_reduction() and len(base_template_node2) == 1 and extension_config.CONFIG_FUSION_PROLOGUE:
            target_node = base_template_node2[0].node

            # Check if template supports prologue fusion
            if not getattr(target_node.template, 'support_prologue_fusion', False):
                return False

            if len(node1.read_writes.writes) != 1:
                return False
            if node1.node not in target_node.inputs or any(["view" in str(ori) for ori in node1.node.origins]): #FIXME
                return False

            # We don't fuse this edge case...
            if base_template_node2[0].group[1][0][0] == 1:
                return False

            if list(node1.read_writes.writes)[0].name in [dep.name for dep in node2.read_writes.reads]:
                node1 = self.revert_group(node1)
                return True
        return self.scheduler.can_fuse_origin(node1, node2)


    def _set_flush_status(self, status: bool):
        self._ready_to_flush = status

    def reset_kernel_group(self):
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()

    def get_backend_features(self, device):
        """Return a set of .codegen.common.BackendFeature()"""
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT])

    def can_fuse_vertical(self, node1, node2):
        return self.can_fuse_horizontal(node1, node2)

    def can_fuse_multi_outputs_template(self, node1, node2):
        return self.can_fuse_horizontal(node1, node2)

    def can_fuse_horizontal(self, node1, node2):
        if not extension_config.CONFIG_FUSION:
            return False

        if (len(node1.get_nodes())+ len(node2.get_nodes())) > self.max_fusion_size:
            return False

        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        # For input/dependency checks
        reads1 = {dep.name for dep in node1.read_writes.reads}
        reads2 = {dep.name for dep in node2.read_writes.reads}
        writes1 = {dep.name for dep in node1.read_writes.writes}
        writes2 = {dep.name for dep in node2.read_writes.writes}

        # Can't fuse two template node
        if node1.is_template() and node2.is_template():
            return False

        if '_unsafe_index' in node1.get_nodes()[0].node.origins or "_unsafe_index" in node2.get_nodes()[0].node.origins:
            return False

        # Extract base template node
        base_template_node1 = [node for node in node1.get_nodes() if node.is_template()]
        base_template_node2 = [node for node in node2.get_nodes() if node.is_template()]

        # Case 0: Reduction fusion
        if (
            node1.is_reduction()
            and node2.is_reduction()
            and not node1.is_template()
            and not node2.is_template()
            and extension_config.CONFIG_FUSION_REDUCTION_REDUCTION
        ):
            # 1) Same loop/iteration domain
            same_iter = vars1 == vars2 and reduce1 == reduce2
            # 2) No data dependency between the two reductions
            no_dependency = not (
                writes1 & (reads2 | writes2) or writes2 & (reads1 | writes1)
            )
            return same_iter and no_dependency

        # Case 1: Template + Pointwise fusion
        if len(base_template_node1) == 1 and len(node1.get_nodes())==1 and len(node2.get_nodes())==1 and len(base_template_node2) == 0 and not node2.is_reduction():
            # Don't fuse maxpool template code
            from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate

            template_node = base_template_node1[0]
            epilogue_node = node2

            # Check if template supports epilogue fusion
            if not getattr(template_node.node.template, 'support_epilogue_fusion', False):
                return False

            if isinstance(template_node.node.template, MLIRMaxPoolTemplate):
                return False

            # Pointwise check
            v1_total = math.prod(vars1) if len(vars1) else 0
            v2_total = math.prod(vars2) if len(vars2) else 0
            if v1_total != v2_total:
                return False

            # Pattern check: check data dependency between act_node and template_node
            template_sched_nodes = list(template_node.get_nodes())
            # Buffers produced by the template (its outputs)
            template_writes = {
                dep
                for n in template_sched_nodes
                for dep in n.read_writes.writes
            }
            # Buffers still required by the activation node (unmet) or read by it
            epilogue_unmet = { dep for dep in epilogue_node.unmet_dependencies }
            has_dependency = bool(template_writes) and epilogue_unmet.issubset(template_writes) and not bool(reads1 & writes2)
            if not has_dependency:
                return False

            # Revert act_node.group : simplify_and_reorder() modified _body, _size, group
            if template_node.group != epilogue_node.group:
                # We don't fuse this case...
                if getattr(template_node.node.template, 'support_prologue_fusion', False) and template_node.group[1][0][0] == 1:
                    return False

                if list(template_node.group[1][0]) != list(epilogue_node.get_nodes()[0].node.data.get_size()):
                    return False
                self.revert_group(epilogue_node)
            return True

        # Case 2: Tempalte + Reduction fusion
        if len(base_template_node1) == 1 and len(node1.get_nodes())==1 and len(node2.get_nodes())==1 and len(base_template_node2) == 0 and node2.is_reduction() and extension_config.CONFIG_FUSION_REDUCTION_EPILOGUE:
            target_node = base_template_node1[0].node

            # Check if template supports reduction fusion
            if not getattr(target_node.template, 'support_reduction_fusion', False):
                return False

            size_match = node1.get_nodes()[0].node.get_numel() == reduce(operator.mul, node2.get_nodes()[0].node.get_size(), 1) * reduce(operator.mul, node2.get_nodes()[0].node.get_reduction_size(), 1)
            target_symbol = symbols("r0_0")
            try:
                stride = [i.strip()[:-1].split(",")[-1].strip() for i in str(node2.get_nodes()[0].node).split("\n") if "r0" in i][1]
                stride = int(sympify(stride).coeff(target_symbol))
            except:
                return False

            # We can't fuse dim=-1 & N == 1
            layout_possible = stride != 1 and (1 not in node1.node.get_size())
            # Directed linked?
            dependency_check = writes1 & reads2
            dependency_size = all([i.get_numel() == node1.get_nodes()[0].node.get_numel() for i in node2.read_writes.reads])
            return size_match and layout_possible and dependency_check and dependency_size

        # Case 3: Prologue(Pointwise) + Tempalte
        # if len(base_template_node1) == 0 and len(node1.get_nodes())==1 and not node1.is_reduction() and len(base_template_node2) == 1 and extension_config.CONFIG_FUSION_PROLOGUE:
        #     from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
        #     from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate

        #    target_node = base_template_node2[0].node
        #    # Currently only BMM, MM support prologue fusion
        #    if not isinstance(target_node.template, (MLIRBMMTemplate, MLIRGemmTemplate)):
        #        return False

        #    if len(node1.read_writes.writes) != 1:
        #        return False
        #    if node1.node not in target_node.inputs or any(["view" in str(ori) for ori in node1.node.origins]): #FIXME
        #        return False

        #    # We don't fuse this edge case...
        #    if base_template_node2[0].group[1][0][0] == 1:
        #        return False

        #    if list(node1.read_writes.writes)[0].name in [dep.name for dep in node2.read_writes.reads]:
        #        node1 = self.revert_group(node1)
        #        return True
        return False

    def revert_group(self, act_nodes, args=None, var_ranges=None):
        for act_node in act_nodes.get_nodes():
            if args is None or var_ranges is None:
                args, var_ranges = dependencies.index_vars_no_squeeze(
                        act_node.node.data.get_size(), act_node.node.data.get_reduction_size(), prefix="q"
                )
            body = LoopBody(
                act_node.node.get_store_function(),
                (args if act_node.node.get_reduction_type() else args[:1]),
                var_ranges,
                args[0],
                args[1]
            )
            index_size = []
            reduce_size = []
            for v, s in var_ranges.items():
                if v in args[0]:
                    index_size.append(s)
                else:
                    reduce_size.append(s)
            node_device = act_node.get_device()
            ranges = (index_size, reduce_size)
            act_node._sizes, act_node._body, act_node.group = (ranges), body, (node_device, self.group_fn(ranges))

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_node(self, _node):
        nodes = _node.get_nodes()
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        # Note: We assume that there is at least one loop in the nodes
        # But, inductor simplifies the group, there could be no loop
        # In that case, we add dummy loop(size=1) to the group
        if len(group) == 0:
            for idx, node in enumerate(nodes):
                if len(node.node.data.get_size()) == 0:
                    continue
                if len(reduction_group) != 0:
                    sym0, sym1 = sympy.Symbol("q0"), sympy.Symbol("q1")
                    args = [[sym0] + [sympy.Number(0)] * (len(node.node.data.get_size())-1), [sym1]]
                    var_ranges = {sym0: sympy.Number(1), sym1: reduction_group[0]}
                else:
                    sym0 = sympy.Symbol("q0")
                    args = [[sym0] + [sympy.Number(0)] * (len(node.node.data.get_size())-1), []]
                    var_ranges = {sym0: sympy.Number(1)}
                self.revert_group(node, args, var_ranges)
            _, (group, reduction_group) = max(
                nodes, key=lambda x: int(x.is_reduction())
            ).group

        ex_kernel = self.target_kernel(kernel_group=self.kernel_group)
        ex_kernel.kernel_group = self.kernel_group

        kernel_name_candidate = f"extension_kernel_{MLIRScheduling.count}"
        MLIRScheduling.count += 1
        src_code, meta_code = ex_kernel.codegen_nodes(nodes, kernel_name_candidate)
        kernel_name = self.define_kernel(src_code, meta_code, kernel_name_candidate, ex_kernel.vector_lane,
                           ex_kernel.spad_info, origins={str(i) for node in nodes for i in node.node.origins})
        ex_kernel.call_kernel(kernel_name)
        _, args, _, _ = ex_kernel.args.mlir_argdefs()
        args = ", ".join(args)
        self._set_flush_status(True)

    def ready_to_flush(self):
        return self._ready_to_flush

    def codegen_sync(self):
        pass

    def flush(self):
        src_code = self.kernel_group.codegen_group()
        if src_code:
            kernel_name = self.define_kernel(
                src_code, self.kernel_group.scheduled_nodes
            )
            self.kernel_group.call_kernel(V.graph.wrapper_code, kernel_name)
        self.reset_kernel_group()
        self._set_flush_status(False)

    def define_function(self, kernel):
        partial_code, function_name = kernel.def_function()
        if partial_code is not None and function_name not in self.outer_function:
            with V.set_kernel_handler(kernel):
                code = partial_code.finalize_all()
                wrapper = V.graph.wrapper_code
                wrapper.header.writeline(code)
                self.outer_function.add(function_name)

    def define_kernel(self, src_code, meta_code, kernel_name, vector_lane, spad_info, loop_size=None, origins={}):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            wrapper.src_to_kernel[src_code] = kernel_name
            codecache_def = IndentedBuffer()
            codecache_def.writeline(f"custom_async_compile.mlir('''{src_code}''', ")
            codecache_def.writeline(f"vectorlane_size={vector_lane},")
            codecache_def.writeline(f"loop_size={loop_size},")
            codecache_def.writeline(f"spad_info={spad_info},")
            codecache_def.writeline(f"origins={origins},")
            codecache_def.writeline(f"arg_attributes={meta_code},")
            codecache_def.writeline(f"vlen={extension_config.vpu_vector_length_bits})")
            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), gpu=False)
        return kernel_name

    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes):
        # Generate template code
        template_buffer = template_node.node
        kernel, tile_candidates, render = template_buffer.make_kernel_render(template_buffer, prologue_nodes=prologue_nodes, epilogue_nodes=epilogue_nodes, kernel_group=self.kernel_group)
        _, _, _, kernel.buffer_types = self.kernel_group.args.mlir_argdefs()
        src_code, meta_code = kernel.codegen_nodes(tile_candidates, render, template_node, prologue_nodes, epilogue_nodes)

        with kernel:
            all_nodes = [template_node] + (epilogue_nodes or []) + (prologue_nodes or [])
            origins = {str(i) for n in all_nodes for i in n.node.origins}
            kernel_name = self.define_kernel(src_code, meta_code, kernel.kernel_name, kernel.vector_lane, kernel.spad_info,
                                             kernel.loop_size, origins=origins)
            self.define_function(kernel)

        kernel.call_kernel(kernel_name)
        V.graph.removed_buffers |= kernel.removed_buffers
        _, args, _, _ = self.kernel_group.args.mlir_argdefs()
        self._set_flush_status(True)

    def enter_context_fixed(self, node):
        def get_order(n):
            if n not in self.scheduler.origin_to_index:
                self.scheduler.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
            return self.scheduler.origin_to_index[n]

        origins = [(get_order(e), idx, e) for n in node.get_nodes() for idx, e in enumerate(n.node.origins)]
        if origins:
            _, _, last = max(origins)
            V.graph.wrapper_code.enter_context(last)
