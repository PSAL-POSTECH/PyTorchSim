import functools
import itertools
from typing import List, Optional
from unittest.mock import patch

from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.codegen.common import ChoiceCaller
from torch._inductor.codegen.common import Kernel
from torch._inductor.codegen.common import OpOverrides
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import TemplateBuffer
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V

from PyTorchSimFrontend.llvm_autotune import LLVMBenchmarkRequest
from PyTorchSimFrontend.llvm_common import LLVMKernelArgs

class LLVMTemplateKernel(Kernel):
    overrides = OpOverrides
    def __init__(self, kernel_name,
                 kernel_caller_function=None,
                 kernel_function_render=None,
                 kernel_arg_attributes=None) -> None:
        super().__init__(LLVMKernelArgs())
        self.kernel_name = kernel_name
        self.named_nodes = {}
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}
        self.kernel_caller_function = kernel_caller_function
        self.kernel_function_render = kernel_function_render
        self.kernel_arg_attributes = kernel_arg_attributes

    def load_matrix(self, row, col, stride, dtype, stype, ptr, base_addr, data_size):
        suffix = f"v{row*col}{stype}.p0{stype}"
        argument = f"(ptr {ptr}, i64 {stride}, i1 0, i32 {row}, i32 {col})"
        code = f"<{row*col} x {dtype}> @llvm.matrix.column.major.load.{suffix} {argument}"

        self.add_desc(True, base_addr, data_size, [col, 1], [row, col])
        return f"call {code}"

    def store_matrix(self, row, col, stride, dtype, stype, ptr, vec, base_addr, data_size):
        suffix = f"v{row*col}{stype}.p0{stype}"
        argument = f"(<{row*col} x {dtype}> {vec}, ptr {ptr}, i64 {stride}, i1 0, i32 {row}, i32 {col})"
        code = f"void @llvm.matrix.column.major.store.{suffix} {argument}"

        self.add_desc(False, base_addr, data_size, [col, 1], [row, col])
        return f"call {code}"

    def store_output(self, row, col, stride, dtype, stype, ptr, vec, base_addr, data_size):
        code = ""
        if len(self.args.input_buffers) > 2:
            indexes = [f"i32 {i%col}" for i in range(row * col)]
            mask = ", ".join(indexes)
            code += f"%add.ptr23 = getelementptr inbounds {dtype}, ptr %Bias, i64 %indvars.iv47\n  "
            code += f"%call19 = " + self.load_matrix(1, col, 1, dtype, stype, "%add.ptr23", "Bias", data_size) + "\n  " #FIXME: Hardcoded %call19
            code += f"%call20 = shufflevector <{col} x {dtype}> %call19, <{col} x {dtype}> undef, <{row*col} x i32> <{mask}>\n  "
            code += f"%call21 = fadd <{row*col} x {dtype}> %call18, %call20\n  "
            vec = "%call21"
        code += self.store_matrix(row, col, stride, dtype, stype, ptr, vec, base_addr, data_size)
        return code

    def add_desc(self, is_load, base_addr, element_size, stride_list, tile_size):
        if is_load:
            key = f"load{len(self.load_desc)}"
            self.load_desc[key] = {
                "base_addr": base_addr,
                "element_size": element_size,
                "stride_list": stride_list,
                "tile_size": tile_size,
                "tile_stride": stride_list[-2:]
            }
        else:
            key = f"store{len(self.store_desc)}"
            self.store_desc[key] = {
                "base_addr": base_addr,
                "element_size": element_size,
                "stride_list": stride_list,
                "tile_size": tile_size,
                "tile_stride": stride_list[-2:]
            }

    def add_loop_info(self, mat_size, tile_size):
        for idx, (loop_size, stride) in enumerate(zip(mat_size, tile_size)):
            self.loop_info[f"index{idx}"] = [0, loop_size, stride]

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        arg_attributes = self.kernel_arg_attributes
        if arg_attributes is None:
            _, _, arg_attributes = self.args.llvm_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"load_tile_info = {self.load_desc}")
        wrapper.add_import_once(f"store_tile_info = {self.store_desc}")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")

    def call_kernel(self, kernel_name):
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.WrapperCodeGen
        """
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        wrapper.generate_kernel_call(
            kernel_name if self.kernel_caller_function is None else self.kernel_caller_function,
            call_args,
            cuda=False,
        )

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

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.llvm_argdefs(only_args=True)
        return f"({', '.join(arg_defs)})"

    def def_function(self):
        _, call_args, _ = self.args.python_argdefs()
        if self.kernel_function_render is not None:
            return self.kernel_function_render(input_args=call_args)

class LLVMTemplateCaller(CUDATemplateCaller):
    def __str__(self):
        return f"LLVMTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"llvm_template_kernels.{self.name}"

class LLVMTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self, name, input_nodes, layout, input_reorder = None):
        """
        Baseclass for LLVM Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(self, **kwargs) -> ChoiceCaller:
        kernel_name = f"llvm_{self.name}"
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            kernel  = LLVMTemplateKernel(kernel_name=kernel_name,
                                         kernel_caller_function=self.function_name if hasattr(self, 'function_name') else None,
                                         kernel_function_render=self.function_render if hasattr(self, 'function_render') else None,
                                         kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None)
            code = self.render(kernel=kernel, **kwargs)

        kernel_hash_name = f"llvm_{self.name}_{next(self.index_counter)}"
        extra_args = []
        # create the BenchmarkRequest
        bmreq = LLVMBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: TemplateBuffer,
            epilogue_nodes: Optional[List[IRNode]] = None,
        ):
            kernel = LLVMTemplateKernel(
                kernel_name=kernel_hash_name,
                kernel_function_render=functools.partial(
                    self.function_render,
                    kernel_name=kernel_hash_name
                ) if hasattr(self, 'function_render') else None,
                kernel_caller_function=self.function_name if hasattr(self, 'function_name') else None,
                kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return LLVMTemplateCaller(
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
