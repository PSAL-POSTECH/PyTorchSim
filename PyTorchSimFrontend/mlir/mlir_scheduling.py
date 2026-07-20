import os
import sympy
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend import extension_codecache
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

from torch.utils._ordered_set import OrderedSet
from torch._inductor import config
from torch._inductor.scheduler import BaseScheduling, FusedSchedulerNode, SchedulerNode, BaseSchedulerNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
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
        # Monkey patch: Inductor's own can_fuse never considers prologue fusion into a
        # template, so intercept that one shape and ask the template about it.
        if not extension_config.CONFIG_FUSION_PROLOGUE:
            return self.scheduler.can_fuse_origin(node1, node2)
        if self._single_prologue_template_pair(node1, node2) is not None:
            return self._prologue_plan_for(node1, node2) is not None
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

    def _single_template_epilogue_pair(self, node1, node2):
        """node1 is a lone template node and node2 a lone non-template node -> that template node."""
        templates = [n for n in node1.get_nodes() if n.is_template()]
        if len(templates) != 1 or len(node1.get_nodes()) != 1:
            return None
        if len(node2.get_nodes()) != 1 or any(n.is_template() for n in node2.get_nodes()):
            return None
        return templates[0]

    def _single_prologue_template_pair(self, node1, node2):
        """node1 is a lone non-template node feeding a lone template node2 -> that template node."""
        if len(node1.get_nodes()) != 1 or any(n.is_template() for n in node1.get_nodes()):
            return None
        if node1.is_reduction():
            return None
        templates = [n for n in node2.get_nodes() if n.is_template()]
        if len(templates) != 1 or len(node2.get_nodes()) != 1:
            return None
        if not ({d.name for d in node1.read_writes.writes} & {d.name for d in node2.read_writes.reads}):
            return None
        return templates[0]

    def _epilogue_plan_for(self, node1, node2):
        """The FusionPlan the template approved for this pair, or None. Pure."""
        template_node = self._single_template_epilogue_pair(node1, node2)
        if template_node is None:
            return None
        return template_node.node.template.try_fuse_epilogue(node1, [], node2)

    def _prologue_plan_for(self, node1, node2):
        """The FusionPlan the template approved for prologue-fusing node1 into node2. Pure."""
        template_node = self._single_prologue_template_pair(node1, node2)
        if template_node is None:
            return None
        return template_node.node.template.try_fuse_prologue(node2, node1)

    def fuse(self, node1, node2):
        # Commit hook: templates defer their node mutations into the plan so that
        # can_fuse stays a pure predicate. Recomputing the plan here (it is pure and
        # cheap) avoids caching it across the many speculative can_fuse calls.
        plan = self._epilogue_plan_for(node1, node2) or self._prologue_plan_for(node1, node2)
        if plan is not None and plan.remap is not None:
            plan.remap()
        return super().fuse(node1, node2)

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

        # Template + epilogue (pointwise or reduction): every condition is the template's.
        if self._single_template_epilogue_pair(node1, node2) is not None:
            return self._epilogue_plan_for(node1, node2) is not None

        return False

    def revert_group(self, act_nodes, args=None, var_ranges=None):
        # Used by axis-split to re-trace a node over split ranges. Fusion reaches the same
        # re-derivation through FusionPlan.remap, so it never runs from a can_fuse predicate.
        from PyTorchSimFrontend.mlir.mlir_template import realign_node_group
        realign_node_group(act_nodes, args, var_ranges)

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_node(self, _node):
        nodes = _node.get_nodes()
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        # axis-split: linearize compatible floor/mod radices at the scheduling layer.
        from . import axis_split
        plan = axis_split.find_split_plan(nodes)
        if plan:
            for _n in nodes:
                if getattr(_n, "_body", None) is None:
                    continue
                _body, _ranges = axis_split.build_split_body(_n, plan)
                _n._sizes, _n._body, _n.group = _ranges, _body, (_n.get_device(), self.group_fn(_ranges))
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

    @staticmethod
    def _literalize_meta(obj):
        """Render meta (arg_attributes) as a valid Python literal for the generated
        wrapper. Dynamic shapes put sympy symbols (e.g. s52) in the shape/stride
        fields; emitted bare they are undefined at module scope -> NameError on
        import. Stringify them ('s52'); the real extent arrives as a runtime kernel
        arg (see the wrapper's call() body), so the compile-time descriptor only
        needs to be import-safe and shape-agnostic."""
        if isinstance(obj, sympy.Expr):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return type(obj)(MLIRScheduling._literalize_meta(x) for x in obj)
        return obj

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
            codecache_def.writeline(f"arg_attributes={self._literalize_meta(meta_code)},")
            headers = extension_codecache.get_header(src_code)
            if headers is not None:
                codecache_def.writeline(f"global_var_header='''{headers[0]}''',")
                codecache_def.writeline(f"gem5_global_var_header='''{headers[1]}''',")
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


# Install the graph-copy (incompatible-radix relayout) lowering hook once at import.
# See graph_copy.py.
from . import graph_copy as _graph_copy
_graph_copy.install()
