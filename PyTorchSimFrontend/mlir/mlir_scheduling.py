import os
import math
import sympy
from functools import reduce
import operator
from sympy import symbols, sympify, Symbol
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

from torch._inductor import config
from torch._inductor.scheduler import BaseScheduling, FusedSchedulerNode, SchedulerNode, BaseSchedulerNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._inductor.ir import LoopBody
from torch._inductor import dependencies

from . import mlir_common
from . import mlir_lowering # DO NOT REMOVE THIS LINE, it is used for lowering

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.can_fuse_origin = self.scheduler.can_fuse
        self.scheduler.can_fuse = self.can_fuse_with_exceptions
        #self.scheduler.enter_context = self.enter_context_fixed # FIXME. Monkey patch: For fixing the inductor bug
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()
        self._ready_to_flush = False
        self.outer_function = set()
        config.inplace_buffers = False # FIXME. inout kernel makes trouble.. So disabled it!
        self.max_fusion_size = 5

    def can_fuse_with_exceptions(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        # Extract base template node
        base_template_node1 = [node for node in node1.get_nodes() if node.is_template()]
        base_template_node2 = [node for node in node2.get_nodes() if node.is_template()]
        if node1.get_device() != node2.get_device():
            return False
        if not (isinstance(node1, (SchedulerNode, FusedSchedulerNode)) and isinstance(node2, (SchedulerNode, FusedSchedulerNode))):
            return False

        if len(base_template_node1) == 1 and len(base_template_node2) == 0 and extension_config.CONFIG_FUSION_REDUCTION_EPILOGUE:
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate
            if (isinstance(base_template_node1[0].node.template, MLIRGemmTemplate) or isinstance(base_template_node1[0].node.template, MLIRBMMTemplate)) and node2.is_reduction():
                # For matmul/bmm+reduction case
                size_match = node1.get_nodes()[0].node.get_numel() == reduce(operator.mul, node2.get_nodes()[0].node.get_size(), 1) * reduce(operator.mul, node2.get_nodes()[0].node.get_reduction_size(), 1)
                target_symbol = symbols("r0")
                try:
                    stride = [i.strip()[:-1].split(",")[-1].strip() for i in str(node2.get_nodes()[0].node).split("\n") if "r0" in i][1]
                    stride = int(sympify(stride).coeff(target_symbol))
                except sympy.core.SympifyError:
                    return False

                # We can't fuse dim=-1
                layout_possible = int(sympify(stride).coeff(target_symbol)) != 1
                # Directed linked?
                dependency_check = node2.get_nodes()[0] in [node.node for node in base_template_node1[0].users]# and len(node2.read_writes.reads)==1
                dependency_size = all([i.get_numel() == node1.get_nodes()[0].node.get_numel() for i in node2.read_writes.reads])
                return size_match and layout_possible and dependency_check and dependency_size

        # For prologue fusion case
        if extension_config.CONFIG_FUSION_PROLOGUE and len(base_template_node1) == 0 and len(node1.get_nodes())==1 and len(base_template_node2) == 1:
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate
            target_node = base_template_node2[0].node
            if target_node.origin_node is not None and hasattr(target_node.origin_node.target, "_name") and target_node.origin_node.target._name == 'aten::convolution':
                return False
            if node1.is_reduction():
                return False
            if len(node1.read_writes.writes) != 1:
                return False
            if node1.node not in target_node.inputs or any(["view" in str(ori) for ori in node1.node.origins]): #FIXME
                return False

            # Currently only BMM, MM support prologue fusion
            if not isinstance(target_node.template, (MLIRBMMTemplate, MLIRGemmTemplate)):
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

    def can_fuse_vertical(self, node1, node2):
        return self.can_fuse_horizontal(node1, node2)

    def can_fuse_horizontal(self, node1, node2):
        if (len(node1.get_nodes())+ len(node2.get_nodes())) > self.max_fusion_size:
            return False
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        # Reduction is currently not supported
        if node1.is_reduction() and node2.is_reduction() and not node1.is_template() and not node2.is_template() and extension_config.CONFIG_FUSION_REDUCTION_REDUCTION:
            return vars1 == vars2 and reduce1 == reduce2 and node1.inverse_users == node2.inverse_users
        if node1.is_reduction() or node2.is_reduction():
            return False

        # Can't fuse two template node
        if node1.is_template() and node2.is_template():
            return False

        # Check template node fusion
        if node1.is_template() or node2.is_template():
            # Don't fuse maxpool template code
            from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
            template_node1 = next((n for n in node1.get_nodes() if n.is_template()), None)
            template_node2 = next((n for n in node2.get_nodes() if n.is_template()), None)

            if template_node1 and len(node1.get_nodes()) == 1 and isinstance(template_node1.node.template, MLIRMaxPoolTemplate) or \
               template_node2 and len(node2.get_nodes()) == 1 and isinstance(template_node2.node.template, MLIRMaxPoolTemplate):
                return False

            # Pointwise check
            v1_total = math.prod(vars1) if len(vars1) else 0
            v2_total = math.prod(vars2) if len(vars2) else 0
            if v1_total != v2_total:
                return False

            # Pattern check
            template_node, act_node = (template_node1, node2) if template_node1 else (template_node2, node1)
            has_depedency = set(act_node.inverse_users) <= set(template_node.get_nodes())
            if not has_depedency:
                return False

            # Revert act_node.group : simplify_and_reorder() modified _body, _size, group
            if template_node.group != act_node.group:
                # We don't fuse this case...
                if (isinstance(template_node.node.template, MLIRBMMTemplate) or isinstance(template_node.node.template, MLIRGemmTemplate)) and template_node.group[1][0][0] == 1:
                    return False

                if list(template_node.group[1][0]) != list(act_node.get_nodes()[0].node.data.get_size()):
                    return False
                self.revert_group(act_node)
            return True

        # Check elementwise fusion
        if vars1 == vars2 and reduce1 == reduce2:
            return True
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

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        # Note: We assume that ther is at least one loop in the nodes
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
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name_candidate)
        kernel_name = self.define_kernel(src_code, kernel_name_candidate, ex_kernel.vector_lane,
                           ex_kernel.spad_info, origins= {str(i) for i in nodes[0].node.origins})
        ex_kernel.call_kernel(kernel_name)
        _, args, _, _ = ex_kernel.args.mlir_argdefs()
        args = ", ".join(args)
        eager_mode = int(os.environ.get('BACKENDSIM_EAGER_MODE', default=False))
        if (eager_mode):
            V.graph.wrapper_code.writeline(
                f"yield ({kernel_name}, ({args}))"
            )
        self._set_flush_status(True)

    def ready_to_flush(self):
        return self._ready_to_flush

    def codegen_sync(self):
        pass

    def flush(self):
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()
        self._set_flush_status(False)

    def define_function(self, kernel):
        partial_code, function_name = kernel.def_function()
        if partial_code is not None and function_name not in self.outer_function:
            with V.set_kernel_handler(kernel):
                code = partial_code.finalize()
                wrapper = V.graph.wrapper_code
                wrapper.header.writeline(code)
                self.outer_function.add(function_name)

    def define_kernel(self, src_code, kernel_name, vector_lane, spad_info, loop_size=None, origins={}):
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
            codecache_def.writeline("arg_attributes=arg_attributes,")
            codecache_def.writeline(f"vlen={extension_config.CONFIG_VLEN})")
            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
        return kernel_name

    def codegen_template_code(self, kernel, render, template_node, prologue_nodes, epilogue_nodes):
        with kernel:
            _, _, _, kernel.buffer_types = self.kernel_group.args.mlir_argdefs()
            for node in [template_node, *prologue_nodes, *epilogue_nodes]:
                node.mark_run()
            # Partial codgen template nodes
            partial_code = render()

            # Swap load/store functions
            kernel.load = kernel.load_epilogue
            kernel.store = kernel.store_epilogue
            kernel.store_reduction = kernel.store_reduction_epilogue
            kernel.reduction = kernel.reduction_epilogue

            # Codegen prologue nodes
            if prologue_nodes:
                # Flush created varaibles, since template fusion doen't share variable
                with kernel.prologue_buffer_group.as_local():
                    _, (group, reduction_group) = max(
                        [prologue_nodes[-1]], key=lambda x: int(x.is_reduction())
                    ).group
                    prologue_tile_desc = kernel.set_tile_size(kernel.prologue_info, prologue=True)
                    kernel.kernel_group.set_tile_info(prologue_tile_desc)
                    vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                    for node in prologue_nodes:
                        # Reuse created spad
                        read_list = sorted([i.name for i in node.read_writes.reads])
                        candidate_found = False
                        # Why? There is a case that memdep.get_size() != data.get_size()
                        buf_dict = {}
                        buf_dict.update({val.name : val for val in V.graph.buffers})
                        buf_dict.update(V.graph.graph_inputs)
                        for candidate_read in read_list:
                            if candidate_read in buf_dict and reduce(operator.mul, buf_dict[candidate_read].get_size(), 1) == node.node.get_numel():
                                prologue_input_arg = candidate_read
                                candidate_found = True
                                break
                        assert(candidate_found)
                        assert(len(node.read_writes.writes)==1)
                        prologue_output_arg = list(node.read_writes.writes)[0].name
                        template_buf = self.kernel_group.args.input_buffers[prologue_output_arg]
                        target_buf = f"{template_buf}_buffer" # FIXME. How to pass spad buffer name?

                        # To skip the dma code gen
                        kernel.buffer_names[prologue_input_arg] = target_buf
                        kernel.buffer_names[prologue_output_arg] = target_buf

                        # Edge delete
                        kernel.kernel_group.args.input_buffers = {
                            (arg if buf != template_buf else prologue_input_arg): buf
                            for arg, buf in kernel.kernel_group.args.input_buffers.items()
                        }
                        node.codegen((vars, reduction_vars))

            # Codegen epilogue nodes
            tile_desc = kernel.set_tile_size(kernel.epilogue_info)
            kernel.kernel_group.set_tile_info(tile_desc)
            kernel.call_ranges = None
            if epilogue_nodes:
                with kernel.epilogue_buffer_group.as_local():
                    _, (group, reduction_group) = max(
                        epilogue_nodes, key=lambda x: int(x.is_reduction())
                    ).group
                    vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                    for node in epilogue_nodes:
                        node.codegen((vars, reduction_vars))

        with V.set_kernel_handler(kernel):
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize()
            )

        # For consistency, white space could make wrong write_path
        buffer = IndentedBuffer()
        buffer.splice(src_code)
        return buffer.getvalue()

    def codegen_template(self, template_node, epilogue_nodes):
        # Handle prologue pattern
        prologue_nodes = []
        if not template_node.is_template():
            epilogue_nodes = [template_node] + epilogue_nodes
            for i, node in enumerate(epilogue_nodes):
                if node.is_template():
                    template_node = node
                    prologue_nodes = epilogue_nodes[:i]
                    epilogue_nodes = epilogue_nodes[i+1:]
                    break

        _, (numel, rnumel) = template_node.group
        template_buffer = template_node.node
        kernel, render, codegen_header = template_buffer.make_kernel_render(template_buffer, prologue_nodes=prologue_nodes, epilogue_nodes=epilogue_nodes, kernel_group=self.kernel_group)
        _, _, _, kernel.buffer_types = self.kernel_group.args.mlir_argdefs()

        src_code = self.codegen_template_code(kernel, render, template_node, prologue_nodes, epilogue_nodes)
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel: # [CONV] check inner function is already defined
            kernel_name = wrapper.src_to_kernel[src_code]
            kernel, render, codegen_header = template_buffer.make_kernel_render(template_buffer, prologue_nodes=prologue_nodes, epilogue_nodes=epilogue_nodes, kernel_name=kernel_name) # update kernel name
            src_code = self.codegen_template_code(kernel, render, template_node, prologue_nodes, epilogue_nodes)

        with V.set_kernel_handler(kernel):
            spad_end_symbol = f"int spad_end[0] __attribute__ ((section(\".spad\")));\n"
            spad_section_end_symbol = f"int spad_section_end[0] __attribute__ ((section(\".spad\"), aligned({kernel.spad_info['spad_size']*kernel.vector_lane})));"
            codegen_header(src_code, (kernel.header.getvalue()+spad_end_symbol+spad_section_end_symbol, kernel.gem5_header.getvalue()))
            kernel.meta_kernel()
            kernel_name = self.define_kernel(src_code, kernel.kernel_name, kernel.vector_lane, kernel.spad_info,
                                             kernel.loop_size, origins={str(i) for i in template_node.node.origins})
            self.define_function(kernel)

        kernel.call_kernel(kernel_name)
        V.graph.removed_buffers |= kernel.removed_buffers
        _, args, _, _ = self.kernel_group.args.mlir_argdefs()
        eager_mode = int(os.environ.get('BACKENDSIM_EAGER_MODE', default=False))
        if (eager_mode):
            target_kernel_name = kernel_name if kernel.outer_func_name is None else kernel.outer_func_name + f"_{len(args)}"
            args = ", ".join(args)
            V.graph.wrapper_code.writeline(
                f"yield ({target_kernel_name}, ({args}))"
            )
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
