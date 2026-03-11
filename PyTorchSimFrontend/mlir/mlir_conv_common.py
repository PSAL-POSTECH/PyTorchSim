import os
import math
from typing import List, Optional

from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import IRNode
from PyTorchSimFrontend import extension_config

class MLIRConvCommonTemplate(MLIRTemplate):
    WRAPPER_TEMPLATE = None
    def __init__(self, input_nodes, layout, input_reorder=None, **kwargs):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.support_epilogue_fusion = True
        self.support_prologue_fusion = False
        self.support_reduction_fusion = False
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]
        self.dilation = kwargs["dilation"]
        self.weight_shape = [str(i) for i in input_nodes[1].layout.size]
        self.input_shape = [str(i) for i in input_nodes[0].layout.size]
        self.function_name = "Conv2D_" + "_".join(self.input_shape) + "_".join(self.weight_shape)+ "_" \
            + "_".join([str(i) for i in self.stride]) \
            + "_" + "_".join([str(i) for i in self.padding]) \
            + "_" + "_".join([str(i) for i in self.dilation])
        self.kernel_args = ['X', 'W', 'Bias', 'Y']

    def get_padded_input_size(self, X):
        input_padded = list(X.layout.size)
        input_padded[2] += 2 * self.padding[0]
        input_padded[3] += 2 * self.padding[1]
        return math.prod(input_padded)

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
        raise NotImplementedError()

    def select_tile(self, kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W):
        raise NotImplementedError()

    def extract_info(self, kernel, template_buffer_node, epilogue_nodes):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        self.kernel = kernel
        self.epilogue_nodes = epilogue_nodes

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
        dtype_infos = [("X", X.get_dtype()), ("W", W.get_dtype()), ("Y", Y.get_dtype())]
        if Bias is not None:
            dtype_infos.append(("Bias", Bias.get_dtype()))
        if len({dtype for _, dtype in dtype_infos}) != 1:
            dtype_desc = ", ".join(f"{name}={dtype}" for name, dtype in dtype_infos)
            raise NotImplementedError(f"Mixed dtype Conv is not implemented yet ({dtype_desc})")

        if epilogue_nodes is not None:
            extra_node_rw = {
                item.name for epilogue_node in epilogue_nodes
                for item in epilogue_node.read_writes.reads | epilogue_node.read_writes.writes
                if item.name != Y.name
            }
        n_extra_node = len(extra_node_rw) if epilogue_nodes is not None else 0

        BATCH, I_C, I_H, I_W = X.layout.size
        O_C, _, K_H, K_W = W.layout.size
        O_H = Y.layout.size[2] if template_buffer_node is None else template_buffer_node.layout.size[2]
        O_W = Y.layout.size[3] if template_buffer_node is None else template_buffer_node.layout.size[3]
        PADDING_H=self.padding[0]
        PADDING_W=self.padding[1]
        STRIDE_H=self.stride[0]
        STRIDE_W=self.stride[1]
        return X,W,Y,Bias,n_extra_node,BATCH,I_C,I_H,I_W,O_C,K_H,K_W,O_H,O_W,PADDING_H,PADDING_W,STRIDE_H,STRIDE_W

    def get_tile_candidates(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        # Extract input arguments info
        X, W, Y, Bias, n_extra_node, BATCH, I_C, I_H, I_W, O_C, K_H, K_W, O_H, O_W, PADDING_H, PADDING_W, STRIDE_H, STRIDE_W = self.extract_info(kernel, template_buffer_node, epilogue_nodes)
        return self.select_tile(kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W)

    def outer_func_render(self, kernel_name, input_args):
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        options = dict(
            kernel=self.kernel,
            KERNEL_NAME=kernel_name,
            FUNC_NAME="wrapper_" + kernel_name,
            INPUT=X,
            WEIGHT=W,
            BIAS=Bias,
            OUTPUT=Y,
            PADDING_H=self.padding[0],
            PADDING_W=self.padding[1],
            VALIDATION_MODE=extension_config.pytorchsim_functional_mode,
            input_reorder=self.input_reorder
        )
        code = self._template_from_string(self.WRAPPER_TEMPLATE).render(**options)
        return code, "wrapper_" + kernel_name

    def get_arg_attributes(self):
        arg_attributes = []

        X = self.input_nodes[0]
        X_shape = [X.get_size()[i] for i in (2, 3, 0, 1)]
        X_shape[0] += 2 * self.padding[0]
        X_shape[1] += 2 * self.padding[1]

        def compute_stride(shape):
            stride = [1] * len(shape)
            for i in range(len(shape)-2, -1, -1):
                stride[i] = stride[i+1] * shape[i+1]
            return stride

        X_stride = compute_stride(X_shape)
        arg_attributes.append([X.data.data.name, [MLIRKernelArgs.MLIR_ARGS_IN, X.layout.dtype, math.prod(X_shape), X_shape, X_stride]])

        return arg_attributes
