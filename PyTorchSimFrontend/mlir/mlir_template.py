import functools
import itertools
import textwrap
from typing import List, Optional
from unittest.mock import patch

from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.codegen.common import ChoiceCaller
from torch._inductor.codegen.common import Kernel
from torch._inductor.codegen.common import OpOverrides
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import TemplateBuffer
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V

from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel

class MLIRTemplateKernel(MLIRKernel, BaseMLIRHardwareInfo):
    def __init__(self,
                 kernel_name,
                 input_nodes,
                 call_size,
                 outer_func_name=None,
                 outer_func_render=None,
                 kernel_arg_attributes=None) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.call_size = call_size
        self.template_buf = None
        self.named_nodes = {}
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}
        self.outer_func_name = outer_func_name
        self.outer_func_render = outer_func_render
        self.kernel_arg_attributes = kernel_arg_attributes
        self.render_hooks = dict()

    def add_loop_info(self, mat_size, tile_size):
        for idx, (loop_size, stride) in enumerate(zip(mat_size, tile_size)):
            self.loop_info[f"index{idx}"] = [0, loop_size, stride]

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        arg_attributes = self.kernel_arg_attributes
        if arg_attributes is None:
            _, _, arg_attributes, _ = self.args.mlir_argdefs()
        if self.template_buf is not None:
            arg_attributes = [attribute for attribute in arg_attributes if attribute[0] != self.template_buf.name]
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"load_tile_info = {self.load_desc}")
        wrapper.add_import_once(f"store_tile_info = {self.store_desc}")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.args.mlir_argdefs()
        if self.template_buf is not None:
            call_args.remove(self.template_buf.name)
       # generate the code to call this
        wrapper.generate_kernel_call(
            kernel_name if self.outer_func_name is None else self.outer_func_name,
            call_args, cuda=False)

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        extra_node = {}
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name
                extra_node[node.get_name()] = node

        arg_defs, *_ = self.args.mlir_argdefs(extra_node=extra_node)

        def hook():
            return f"({', '.join(arg_defs)})"

        assert "<DEF_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def def_global_vars(self):
        def hook():
            return textwrap.indent(self.global_vars.getvalue(), "").strip()
        assert "<GLOBAL_VARS>" not in self.render_hooks
        self.render_hooks["<GLOBAL_VARS>"] = hook
        return "<GLOBAL_VARS>"

    def store_output(self):

        def hook():
            self.codegen_body()
            return textwrap.indent(self.body.getvalue(), "      ").strip()  #TODO: First line is not indented

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        return "<STORE_OUTPUT>"

    def def_function(self):
        _, call_args, _ = self.args.python_argdefs()
        if self.outer_func_render is not None:
            return self.outer_func_render(input_args=call_args)
        else:
            return None, None

    def render(self, template, kwargs):
        # self.render_hooks = {}
        return PartialRender(
            template.render(**kwargs),
            self.render_hooks,
        )

class MLIRTemplateCaller(CUDATemplateCaller):
    def __str__(self):
        return f"MLIRTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"mlir_template_kernels.{self.name}"

class MLIRTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self, name, input_nodes, layout, input_reorder = None):
        """
        Baseclass for MLIR Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = [node for node in input_nodes if node is not None]
        self.output_node: Buffer = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(self, **kwargs) -> ChoiceCaller:
        kernel_name = f"mlir_{self.name}"
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            kernel  = MLIRTemplateKernel(kernel_name=kernel_name, input_nodes=self.input_nodes, call_size=self.layout.size,
                                         outer_func_name=self.function_name if hasattr(self, 'function_name') else None,
                                         outer_func_render=self.outer_func_render if hasattr(self, 'outer_func_render') else None,
                                         kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None)
            code = self.render(kernel=kernel, **kwargs)

        kernel_hash_name = f"mlir_{self.name}_{next(self.index_counter)}"
        extra_args = []
        # create the BenchmarkRequest
        bmreq = MLIRBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: TemplateBuffer,
            epilogue_nodes: Optional[List[IRNode]] = None,
            kernel_name: str = kernel_hash_name,
        ):
            kernel = MLIRTemplateKernel(
                kernel_name=kernel_name,
                input_nodes=self.input_nodes,
                call_size=self.layout.size,
                outer_func_name=self.function_name if hasattr(self, 'function_name') else None,
                outer_func_render=functools.partial(
                    self.outer_func_render,
                    kernel_name=kernel_name
                ) if hasattr(self, 'outer_func_render') else None,
                kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None
            )
            # render = functools.partial(
            #     self.render,
            #     kernel=kernel,
            #     template_buffer_node=template_node,
            #     epilogue_nodes=epilogue_nodes,
            #     **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            # )
            kwargs = {
                'kernel': kernel,
                'template_buffer_node': template_node,
                'epilogue_nodes': epilogue_nodes
            }
            render = functools.partial(
                kernel.render,
                template=self,
                kwargs=kwargs
            )
            return kernel, render, self.codegen_header

        return MLIRTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def render(self, **kwargs) -> str:
        raise NotImplementedError