import os
import math
from sympy import divisors, Range
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import PyTorchSimFrontend.extension_codecache as extension_codecache
from torch._inductor.codecache import get_hash
from PyTorchSimFrontend import extension_config

GEMM_TEMPLATE = r"""
%map1 = affine_map<(d0, d1, d2, d3) -> ()>
memref.global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

func.func @{{ KERNEL_NAME }}({{ KERNEL_DEF }}) {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %c_set = arith.constant 2 : index
  %c0 = arith.constant 0 : index

  // 1x1 convolution loop
  affine.for %k_h = 0 to {{ K_H }} {
    affine.for %k_w = 0 to {{ K_W }} {
      // 1x1 convolution tiling loop
        affine.for %tile_m = 0 to {{ M * N }} step {{ TILE_M }} {
          affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
            %index2 = affine.apply #map1(%k_h, %k_w, %tile_m, %tile_n)
            affine.for %tile_k = 0 to {{ I_C }} step {{ TILE_K }} {
            
            }
          }
        }
      }
  } { outer_loop=true }
  return
}
"""

CONV2D_FUNC_TEMPLATE = r"""
def {{ FUNC_NAME }}({{ INPUT }}, {{ WEIGHT }}, {{ OUT }}):
    # Tanspose tensors
    t_{{ INPUT }} = {{ INPUT }}.permute(0, 2, 3, 1)
    t_{{ WEIGHT }} = {{ WEIGHT }}.permute(0, 2, 3, 1)
    t_{{ OUT }} = {{ OUT }}.permute(0, 2, 3, 1)

    {{ KERNEL_NAME }}(t_{{ INPUT }}, t_{{ WEIGHT }}, t_{{ OUT }})

    # Transpose back
    {{ OUT }} = t_{{ OUT }}.permute(0, 3, 1, 2)

    print("Print OUTPUT ")
    print("out > ")
    print({{ OUT }}.shape)
    print({{ OUT }}.cpu())
"""


class MLIRConvTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None, **kwargs):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]
        self.dilation = kwargs["dilation"]
        weight_shape = [str(i) for i in input_nodes[1].layout.size]
        self.function_name = "Conv2D_" + "_".join(weight_shape)+ "_" \
            + "_".join([str(i) for i in self.stride]) \
            + "_" + "_".join([str(i) for i in self.padding]) \
            + "_" + "_".join([str(i) for i in self.dilation])
        self.gemm_args = ['input', 'weight', 'output']

        self.calculate_gemm_shape()

    def is_transposed(self, node):
        if isinstance(node, ReinterpretView):
            if node.layout.stride != node.data.layout.stride:
                if node.layout.stride[-2] == node.data.layout.stride[-1] and node.layout.stride[-1] == node.data.layout.stride[-2]:
                    return True
                else:
                  raise NotImplementedError("If the stride is not equal to the original stride, it should have been transposed.")
        return False

    def calculate_gemm_shape(self):
        input_shape = self.input_nodes[0].get_size()
        weight_shape = self.input_nodes[1].get_size()
        gemm_h = int((input_shape[2] + 2*self.padding[0] - (weight_shape[2]-1) - 1) / self.stride[0]) + 1
        gemm_w = int((input_shape[3] + 2*self.padding[1] - (weight_shape[3]-1) - 1) / self.stride[1]) + 1

        self.gemm_input_shape = [input_shape[0],input_shape[1],gemm_h, gemm_w]
        self.gemm_weight_shape = [weight_shape[0],weight_shape[1],1,1]
        self.gemm_output_shape = [self.gemm_input_shape[2]*self.gemm_input_shape[3], self.gemm_weight_shape[0]] # Consider Batch size 1

    def def_kernel(self) ->str:
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node

        def flatten(shape):
            r = 1
            for i in shape:
                r *= i
            return r

        input_size = flatten(X.layout.size)
        weight_size = flatten(W.layout.size)
        output_size = flatten(Y.layout.size)
        return f"%X: memref<{input_size}xf32>, %W: memref<{weight_size}xf32>, %Y: memref<{output_size}xf32>"

    def get_tile_option(self):
        I_C, H, W = self.input_nodes[0].layout.size[1], self.input_nodes[0].layout.size[2], self.input_nodes[0].layout.size[3]
        O_C = self.input_nodes[1].layout.size[0]
        
        tile_k_options = divisors(I_C)
        tile_n_options = divisors(O_C)

        H_divisors = divisors(H)
        H_multiples = list(Range(H, H * W, H))
        tile_m_options = sorted(set(H_divisors) | set(H_multiples))
        
        return tile_m_options, tile_n_options, tile_k_options
        

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])
            self.function_name += f"_fused_{epilogue_nodes[0].node.origin_node.name}"

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        M = self.gemm_input_shape[2] * self.gemm_input_shape[3]
        N = self.gemm_weight_shape[0]
        K = self.gemm_weight_shape[1]
        # TILE_M, TILE_N, TILE_K = kernel.gemm_combination_mapping(M, N, K)
        # kernel.tile_size = [TILE_M, TILE_N, TILE_K]
        # kernel.loop_size = [M, N, K]

        tile_m_options, tile_n_options, tile_k_options = self.get_tile_option()
        print("tile_m_options: ", tile_m_options)
        print("tile_n_options: ", tile_n_options)
        print("tile_k_options: ", tile_k_options)

        TILE_M = tile_m_options[3]
        TILE_N = tile_n_options[3]
        TILE_K = tile_k_options[3]

        kernel.tile_size = [TILE_M, TILE_N, TILE_K]
        kernel.loop_size = [M, N, K]

        # W_transposed = self.is_transposed(W)
        # X_transposed = self.is_transposed(X)

        options = dict(
            KERNEL_NAME=self.name,
            KERNEL_DEF=self.def_kernel(),
            kernel=kernel,
            I_C=X.layout.size[1],
            I_H=X.layout.size[2],
            I_W=X.layout.size[3],
            O_C=W.layout.size[0],
            K_H=W.layout.size[2],
            K_W=W.layout.size[3],
            M=M,
            N=N,
            K=K,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            PADDING_H=self.padding[0],
            PADDING_W=self.padding[1],
            STRIDE_H=self.stride[0],
            STRIDE_W=self.stride[1],
            DILATION_H=self.dilation[0],
            DILATION_W=self.dilation[1],
            DATA_STYPE="f32",
            DATA_SIZE=4,
        )
        code = self._template_from_string(GEMM_TEMPLATE).render(**options)

        self.header = f"float X_spad[{TILE_M * TILE_K // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float W_spad[{TILE_K * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{TILE_M * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header = f"float X_spad[{TILE_M * TILE_K}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float W_spad[{TILE_K * TILE_N}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float Y_spad[{TILE_M * TILE_N}] __attribute__ ((section(\".spad\")));\n"

        kernel.add_loop_info([options["M"], options["N"], options["K"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
        kernel.def_kernel(inputs=[X, W], outputs=[Y], names_str="X, W, Y", input_reorder=self.input_reorder)

        return code

    def outer_func_render(self, kernel_name, input_args):
        options = dict(
            KERNEL_NAME=kernel_name,
            FUNC_NAME=self.function_name,
            INPUT=input_args[0],
            WEIGHT=input_args[1],
            OUT=input_args[3] if len(input_args) == 4 else input_args[2],
            VALIDATION_MODE=extension_config.CONFIG_TORCHSIM_VALIDATION_MODE,
            BACKENDSIM_EAGER_MODE=extension_config.CONFIG_BACKENDSIM_EAGER_MODE,
            HASH_VALUE=self.hash_value
        )
        code = self._template_from_string(CONV2D_FUNC_TEMPLATE).render(**options)
        return code, self.function_name

    def get_arg_attributes(self):
        arg_attributes = []

        input_shape = self.input_nodes[0].get_size()
        weight_shape = self.input_nodes[1].get_size()
        gemm_h = int((input_shape[2] + 2*self.padding[0] - (weight_shape[2]-1) - 1) / self.stride[0]) + 1
        gemm_w = int((input_shape[3] + 2*self.padding[1] - (weight_shape[3]-1) - 1) / self.stride[1]) + 1

        gemm_input_shape = [input_shape[0],input_shape[1],gemm_h, gemm_w]
        gemm_weight_shape = [weight_shape[0],weight_shape[1],1,1]
        gemm_output_shape = [gemm_input_shape[2]*gemm_input_shape[3], gemm_weight_shape[0]] # Consider Batch size 1

        arg_attributes.append([self.gemm_args[0], [MLIRKernelArgs.MLIR_ARGS_IN, self.input_nodes[0].layout.dtype, math.prod(gemm_input_shape)]])
        arg_attributes.append([self.gemm_args[1], [MLIRKernelArgs.MLIR_ARGS_IN, self.input_nodes[1].layout.dtype, math.prod(gemm_weight_shape)]])
        # arg_attributes.append([self.gemm_args[2], [MLIRKernelArgs.MLIR_ARGS_IN, self.input_nodes[0].layout.dtype, math.prod(gemm_output_shape)]])
        arg_attributes.append([self.gemm_args[2], [MLIRKernelArgs.MLIR_ARGS_OUT, self.input_nodes[0].layout.dtype, math.prod(gemm_output_shape)]])

        return arg_attributes

    def codegen_header(self, code, extra_headers):
        write_path = extension_codecache.get_write_path(code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        spike_write_path = os.path.join(write_path, "global_var.h")
        gem5_write_path = os.path.join(write_path, "gem5_global_var.h")
        if not os.path.exists(spike_write_path):
            write_atomic(spike_write_path, self.header+extra_headers[0])
        if not os.path.exists(gem5_write_path):
            write_atomic(gem5_write_path, self.gem5_header+extra_headers[1])
        self.hash_value = get_hash(code.strip())