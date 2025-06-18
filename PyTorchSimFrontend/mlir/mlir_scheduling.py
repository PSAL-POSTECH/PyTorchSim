import os
import math
from functools import reduce
import operator
from sympy import symbols, sympify
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

from torch._inductor import config
from torch._inductor.scheduler import BaseScheduling, FusedSchedulerNode, SchedulerNode, BaseSchedulerNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V

from . import mlir_common
from . import mlir_lowering

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.can_fuse_origin = self.scheduler.can_fuse
        self.scheduler.can_fuse = self.can_fuse_with_exceptions
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()
        self._ready_to_flush = False
        self.outer_function = set()
        config.inplace_buffers = False # FIXME. inout kernel makes trouble.. So disabled it!
        self.max_fusion_size = 5

    def can_fuse_with_exceptions(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        if node1.get_device() == node2.get_device():
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate
            if (node1.is_template() and len(node1.get_nodes())==1 and \
                (isinstance(node1.node.template, MLIRGemmTemplate) or isinstance(node1.node.template, MLIRBMMTemplate)) and \
                node2.is_reduction() and len(node2.get_nodes())==1):
                # For matmul/bmm+reduction case
                size_match = node1.node.get_size() == node2.node.get_size() + node2.node.get_reduction_size()
                if len(node1.node.get_size()) == len(node2.node.get_size()):
                    size_match = node1.node.get_size() == [dim for dim in node2.node.get_size() if dim!=1] + node2.node.get_reduction_size()
                stride = [i.strip()[:-1].split(",")[-1].strip() for i in str(node2.node).split("\n") if "r0" in i][1]
                target_symbol = symbols("r0")
                # We can't fuse dim=-1
                possible = int(sympify(stride).coeff(target_symbol)) != 1
                return size_match and possible

            # For prologue fusion case
            if not node1.is_template() and len(node1.get_nodes())==1 and node2.is_template():
                # Return false if node2 is Convolution template
                # if node2.get_nodes()[0].node.origin_node.target._name == 'aten::mm' or \
                #     node2.get_nodes()[0].node.origin_node.target._name == 'aten::addmm':
                #     return False
                if node2.get_nodes()[0].node.origin_node is not None and hasattr(node2.get_nodes()[0].node.origin_node.target, "_name") and node2.get_nodes()[0].node.origin_node.target._name == 'aten::convolution':
                    return False
                if node1.is_reduction():
                    return False
                if len(node1.read_writes.writes) != 1:
                    return False
                if list(node1.read_writes.writes)[0].name in [dep.name for dep in node2.read_writes.reads]:
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
        if node1.is_reduction() or node2.is_reduction():
            return False

        # Can't fuse two template node
        nr_template = 0
        for node in node1.get_nodes() + node2.get_nodes():
            if node.is_template():
                nr_template += 1

        if nr_template > 1:
            return False

        # Check template node fusion
        if node1.is_template() or node2.is_template():
            # Don't fuse maxpool template code
            from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate
            if node1.is_template() and len(node1.get_nodes())==1 and isinstance(node1.node.template, MLIRMaxPoolTemplate) or \
                node2.is_template() and len(node1.get_nodes())==1 and isinstance(node2.node.template, MLIRMaxPoolTemplate):
                return False

            # Different layout is not supported
            if node1.get_nodes()[0].node.layout.dtype != node2.get_nodes()[0].node.layout.dtype:
                return False

            # Convolution is currently not supported
            # if node1.is_template() and node1.get_nodes()[0].node.origin_node is not None and hasattr(node1.get_nodes()[0].node.origin_node.target, "_name") and node1.get_nodes()[0].node.origin_node.target._name == 'aten::convolution':
            #     return False

            # if node2.is_template() and node2.get_nodes()[0].node.origin_node is not None and hasattr(node2.get_nodes()[0].node.origin_node.target, "_name") and node2.get_nodes()[0].node.origin_node.target._name == 'aten::convolution':
            #     return False

            v1_total = math.prod(vars1) if len(vars1) else 0
            v2_total = math.prod(vars2) if len(vars2) else 0
            if v1_total != v2_total:
                return False

            has_depedency = False
            template_node = node1 if node1.is_template() else node2
            act_node = node2 if node1.is_template() else node1
            for write_buf in template_node.read_writes.writes:
                has_depedency = has_depedency or (write_buf in act_node.read_writes.reads)
            return has_depedency

        # Check elementwise fusion
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        ex_kernel = self.target_kernel(kernel_group=self.kernel_group)
        ex_kernel.kernel_group = self.kernel_group

        kernel_name = f"extension_kernel_{MLIRScheduling.count}"
        MLIRScheduling.count += 1
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name)
        self.define_kernel(src_code, kernel_name, ex_kernel.vector_lane,
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
            for node in [template_node, *prologue_nodes, *epilogue_nodes]:
                node.mark_run()
            partial_code = render()
            tile_desc = kernel.set_tile_size(kernel.epilogue_info)
            kernel.kernel_group.set_tile_info(tile_desc)
            if prologue_nodes:
                _, (group, reduction_group) = max(
                    prologue_nodes, key=lambda x: int(x.is_reduction())
                ).group
                tile_desc = kernel.set_tile_size(kernel.prologue_info)
                kernel.kernel_group.set_prologue_tile_info(tile_desc)
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            # Flush created varaibles, since template fusion doen't share variable
            kernel.cse.cache.clear()
            kernel.prologue_buffer_group.set_buffers()
            kernel.call_ranges = None
            kernel.load = kernel.load_prologue
            kernel.store = kernel.store_prologue
            for node in prologue_nodes:
                # Reuse created spad
                read_list = sorted(list(node.read_writes.reads))
                if reduce(operator.mul, read_list[-1].size, 1) == template_node.node.get_numel():
                    prologue_input_arg = read_list[-1].name
                else:
                    prologue_input_arg = read_list[0].name
                prologue_output_arg = list(node.read_writes.writes)[0].name
                template_buf = self.kernel_group.args.input_buffers[prologue_output_arg]
                if template_node.get_nodes()[0].node.origin_node.target._name == 'aten::bmm':
                    target_buf = f"{template_buf}_buffer2D"
                else:
                    target_buf = f"{template_buf}_buffer"

                # To skip the dma code gen
                kernel.buffer_names[prologue_input_arg] = target_buf
                kernel.buffer_names[prologue_output_arg] = target_buf

                # Edge delete
                kernel.kernel_group.args.input_buffers = {
                    (arg if buf != template_buf else prologue_input_arg): buf
                    for arg, buf in kernel.kernel_group.args.input_buffers.items()
                }
                node.codegen((vars, reduction_vars))

            if epilogue_nodes:
                _, (group, reduction_group) = max(
                    epilogue_nodes, key=lambda x: int(x.is_reduction())
                ).group
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            # Flush created varaibles, since template fusion doen't share variable
            kernel.cse.cache.clear()
            kernel.epilogue_buffer_group.set_buffers()
            kernel.load = kernel.load_epilogue
            kernel.store = kernel.store_epilogue
            for node in epilogue_nodes:
                if template_node.node.name in [dep[0] for dep in list(node.read_writes.reads)]:
                    kernel.epilogue_info['dependent_buf'].append(node.node.name)
                node.codegen((vars, reduction_vars))
        with V.set_kernel_handler(kernel):
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize()
            )
        return src_code

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