import math
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

from torch._inductor import config
from torch._inductor.scheduler import BaseScheduling, FusedSchedulerNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V

from . import mlir_common
from . import mlir_lowering

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()
        self._ready_to_flush = False
        self.outer_function = set()
        config.inplace_buffers = False # FIXME. inout kernel makes trouble.. So disabled it!

    def _set_flush_status(self, status: bool):
        self._ready_to_flush = status

    def can_fuse_vertical(self, node1, node2):
        return self.can_fuse_horizontal(node1, node2) and not node1.is_reduction()

    def can_fuse_horizontal(self, node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        # Convolution is currently not supported
        if node1.node.origin_node.target._name == 'aten::convolution' or node2.node.origin_node.target._name == 'aten::convolution':
            return False

        # Reduction is currently not supported
        if node1.is_reduction() or node2.is_reduction():
            return False

        if not isinstance(node1, FusedSchedulerNode) and not isinstance(node2, FusedSchedulerNode):
            # Different layout is not supported
            if node1.node.layout.dtype != node2.node.layout.dtype:
                return False

            # Different size is not supported for non-template node
            if  not node1.is_template() and (node1._sizes[0] != node2._sizes[0]):
                return False

        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True

        #TODO: Temporary solution determining the fusion condition similar to CPP/OpenMP
        v1_total = math.prod(vars1) if len(vars1) else 0
        v2_total = math.prod(vars2) if len(vars2) else 0
        r1_total = math.prod(reduce1) if len(reduce1) else 0
        r2_total = math.prod(reduce2) if len(reduce2) else 0
        if reduce1 == () \
            and v1_total == (v2_total + r2_total):
            return True

        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        ex_kernel = self.target_kernel()
        ex_kernel.kernel_group = self.kernel_group

        kernel_name = f"extension_kernel_{MLIRScheduling.count}"
        MLIRScheduling.count += 1
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name)
        self.define_kernel(src_code, kernel_name, ex_kernel.vector_lane,
                           ex_kernel.spad_info, origins= {str(i) for i in nodes[0].node.origins})
        ex_kernel.call_kernel(kernel_name)
        _, args, _, _ = ex_kernel.args.mlir_argdefs()
        args = ", ".join(args)
        if (extension_config.CONFIG_BACKENDSIM_EAGER_MODE):
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
        code, function_name = kernel.def_function()
        if code is not None and function_name not in self.outer_function:
            wrapper = V.graph.wrapper_code
            wrapper.header.writeline(code)
            self.outer_function.add(function_name)

    def define_kernel(self, src_code, kernel_name, vector_lane, spad_info, tile_size=[1, 1, 1], loop_size=None, origins={}):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            wrapper.src_to_kernel[src_code] = kernel_name

            codecache_def = IndentedBuffer()
            codecache_def.writeline(f"custom_async_compile.mlir('''{src_code}''', ")
            codecache_def.writeline(f"vectorlane_size={vector_lane},")
            codecache_def.writeline(f"tile_size={tile_size},")
            codecache_def.writeline(f"loop_size={loop_size},")
            codecache_def.writeline(f"spad_info={spad_info},")
            codecache_def.writeline(f"origins={origins},")
            codecache_def.writeline("arg_attributes=arg_attributes)")
            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
        return kernel_name

    def codegen_src_code(self, kernel, render, template_node, epilogue_nodes):
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                    node.mark_run()
            partial_code = render()
            for node in epilogue_nodes:
                ranges = node.get_ranges()
                node.codegen(kernel.set_ranges(ranges[0], ranges[1], None))
        with V.set_kernel_handler(kernel):
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize()
            )
            src_code = kernel.add_extra_global_vars(src_code)
            src_code = kernel.add_extra_local_vars(src_code)
        return src_code

    def codegen_template(self, template_node, epilogue_nodes):
        _, (numel, rnumel) = template_node.group
        template_buffer = template_node.node
        kernel, render, codegen_header = template_buffer.make_kernel_render(template_buffer, epilogue_nodes=epilogue_nodes)
        _, _, _, kernel.buffer_types = kernel.args.mlir_argdefs()

        src_code = self.codegen_src_code(kernel, render, template_node, epilogue_nodes)
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel: # [CONV] check inner function is already defined
            kernel_name = wrapper.src_to_kernel[src_code]
            kernel, render, codegen_header = template_buffer.make_kernel_render(template_buffer, epilogue_nodes=epilogue_nodes, kernel_name=kernel_name) # update kernel name
            src_code = self.codegen_src_code(kernel, render, template_node, epilogue_nodes)

        with V.set_kernel_handler(kernel):
            codegen_header(src_code, (kernel.header.getvalue(), kernel.gem5_header.getvalue()))
            # node_schedule = [template_node, *epilogue_nodes]
            kernel.meta_kernel()
            kernel_name = self.define_kernel(src_code, kernel.kernel_name, kernel.vector_lane, kernel.spad_info,
                                             kernel.tile_size, kernel.loop_size, origins={str(i) for i in template_node.node.origins})
            self.define_function(kernel)

        kernel.call_kernel(kernel_name)
        V.graph.removed_buffers |= kernel.removed_buffers
        _, args, _, _ = kernel.args.mlir_argdefs()
        args = ", ".join(args)
        if (extension_config.CONFIG_BACKENDSIM_EAGER_MODE):
            target_kernel_name = kernel_name if kernel.outer_func_name is None else kernel.outer_func_name
            V.graph.wrapper_code.writeline(
                f"yield ({target_kernel_name}, ({args}))"
            )
        self._set_flush_status(True)