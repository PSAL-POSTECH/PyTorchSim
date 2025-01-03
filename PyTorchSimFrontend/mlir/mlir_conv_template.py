import os
import math
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
#map0 = affine_map<(d0, d1) -> (d0 * {{ K }} + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * {{ N }} + d1)>
memref.global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @B_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

func.func @{{ KERNEL_NAME }}({{ KERNEL_DEF }}) {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %c_set = arith.constant 2 : index
  %c0 = arith.constant 0 : index

  %N = arith.constant {{ N }} : index
  %K = arith.constant {{ K }} : index
  %X_buffer = memref.get_global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>

  affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {
    affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
        %index2 = affine.apply #map1(%t_m, %t_n)
        affine.dma_start %B[%index2], %Y_buffer[%c0, %c0], %tag[0], %c_mvin3, %N, %c_set : memref<{{ M * N }}xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ kernel.vector_lane }}, {{ kernel.vector_lane }}], async=1 }
        affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
            %index0 = affine.apply #map0(%t_m, %t_k)
            %index1 = affine.apply #map1(%t_k, %t_n)
            affine.dma_start %X[%index0], %X_buffer[%c0, %c0], %tag[0], %c_mvin, %K, %c_set : memref<{{ M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ kernel.vector_lane }}, {{ TILE_K }}], async=1 }
            affine.dma_start %W[%index1], %W_buffer[%c0, %c0], %tag[0], %c_mvin2, %N, %c_set : memref<{{ K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ TILE_K }}, {{ kernel.vector_lane }}], async=1 }
            linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                    outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true }
        affine.dma_start %Y_buffer[%c0, %c0], %Y[%index2], %tag[0], %c_mvout, %N, %c_set : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<{{ M * N }}xf32>, memref<1xi32>  { subtile_size=[{{ kernel.vector_lane }}, {{ kernel.vector_lane }}], async=1 }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""


CONV2D_FUNC_TEMPLATE = r"""
def {{ FUNC_NAME }}({{ INPUT }}, {{ WEIGHT }}{% if BIAS %}, {{ BIAS }}{% endif %}, {{ OUT }}):
    {{ INPUT }}_cpu = {{ INPUT }}.cpu()
    {{ WEIGHT }}_cpu = {{ WEIGHT }}.cpu(){% if BIAS %}
    {{ BIAS }}_cpu = {{ BIAS }}.cpu(){% endif %}
    {{ OUT }}_cpu = {{ OUT }}.cpu()

    # Torch support NCHW, so we need to transpose for now
    {{ INPUT }}_cpu = {{ INPUT }}_cpu.permute(0, 2, 3, 1)
    {{ WEIGHT }}_cpu = {{ WEIGHT }}_cpu.permute(0, 2, 3, 1)
    {{ OUT }}_cpu = {{ OUT }}_cpu.permute(0, 2, 3, 1)
    {{ OUT }}_cpu.zero_()

    input_shape = {{ INPUT }}_cpu.shape
    weight_shape = {{ WEIGHT }}_cpu.shape
    output_shape = {{ OUT }}_cpu.shape
    {{ OUT }}_cpu = {{ OUT }}_cpu.reshape(-1, output_shape[3]).contiguous()

    input_pad_shape = (input_shape[0], input_shape[1]+2*{{ PADDING_H }}, input_shape[2]+2*{{ PADDING_W }}, input_shape[3])
    input_pad = torch.zeros(input_pad_shape)

    if {{ PADDING_H }} != 0 and {{ PADDING_W }} != 0:
        input_pad[:, {{ PADDING_H }}:-{{ PADDING_H }}, {{ PADDING_W }}:-{{ PADDING_W }}, :] = {{ INPUT }}_cpu
    elif {{ PADDING_H }} != 0:
        input_pad[:, {{ PADDING_H }}:-{{ PADDING_H }}, :, :] = {{ INPUT }}_cpu
    elif {{ PADDING_W }} != 0:
        input_pad[:,:, {{ PADDING_W }}:-{{ PADDING_W }}, :] = {{ INPUT }}_cpu
    else:
        input_pad = {{ INPUT }}_cpu

    {% if VALIDATION_MODE %}
    {% endif %}

    for kh in range(weight_shape[1]):
        for kw in range(weight_shape[2]):
            input_tile = input_pad[:, kh:input_pad_shape[1]-(weight_shape[1]-1)+kh, kw:input_pad_shape[2]-(weight_shape[2]-1)+kw, :]
            input_tile = input_tile[:,::{{ STRIDE_H }},::{{ STRIDE_W }}, :]
            kernel_tile = {{ WEIGHT }}_cpu[:, kh, kw, :].t()
            input_tile = input_tile.reshape(-1, input_pad_shape[3])

            {% if VALIDATION_MODE %}
            if kh == 0 and kw == 0:
                {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu, intermediate_op=0b01)
            elif kh == weight_shape[1]-1 and kw == weight_shape[2]-1:
                {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu, intermediate_op=0b10)
            else:
                {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu, intermediate_op=0b11)
            {% else %}
            {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu)  # input, weight, bias, out
            {% endif %}
            {% if BACKENDSIM_EAGER_MODE %}
            yield ({{KERNEL_NAME}}, (input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu))
            {% endif %}

    {{ OUT }}_cpu = {{ OUT }}_cpu.reshape(output_shape)
    {{ OUT }}_cpu = {{ OUT }}_cpu.permute(0, 3, 1, 2){% if BIAS %}
    {{ OUT }}_cpu += {{ BIAS }}_cpu.reshape(-1, 1, 1) #TODO: BIAS should be added in the kernel{% endif %}
    {{ OUT }}.copy_({{ OUT }}_cpu)
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
        self.gemm_args = ['input', 'weight', 'bias', 'output']

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
        input_size = self.gemm_input_shape[1]*self.gemm_input_shape[2]*self.gemm_input_shape[3]
        weight_size = self.gemm_weight_shape[0]*self.gemm_weight_shape[1]
        output_size = self.gemm_output_shape[0]*self.gemm_output_shape[1]
        return f"%X: memref<{input_size}xf32>, %W: memref<{weight_size}xf32>, %B: memref<{output_size}xf32>, %Y: memref<{output_size}xf32>"

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
        TILE_M, TILE_N, TILE_K = kernel.gemm_combination_mapping(M, N, K)
        kernel.tile_size = [TILE_M, TILE_N, TILE_K]
        kernel.loop_size = [M, N, K]

        W_transposed = self.is_transposed(W)
        X_transposed = self.is_transposed(X)

        options = dict(
            KERNEL_NAME=self.name,
            KERNEL_DEF=self.def_kernel(),
            kernel=kernel,
            M=M,
            N=N,
            K=K,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
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
        if Bias is not None:
            self.header += f"float B_spad[{TILE_M * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
            self.gem5_header += f"float B_spad[{TILE_M * TILE_N}] __attribute__ ((section(\".spad\")));\n"

        kernel.add_loop_info([options["M"], options["N"], options["K"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
        kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=self.input_reorder)

        return code

    def outer_func_render(self, kernel_name, input_args):
        options = dict(
            KERNEL_NAME=kernel_name,
            FUNC_NAME=self.function_name,
            INPUT=input_args[0],
            WEIGHT=input_args[1],
            BIAS=input_args[2] if len(input_args) == 4 else None,
            OUT=input_args[3] if len(input_args) == 4 else input_args[2],
            PADDING_H=self.padding[0],
            PADDING_W=self.padding[1],
            STRIDE_H=self.stride[0],
            STRIDE_W=self.stride[1],
            DILATION_H=self.dilation[0],
            DILATION_W=self.dilation[1],
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
        arg_attributes.append([self.gemm_args[2], [MLIRKernelArgs.MLIR_ARGS_IN, self.input_nodes[0].layout.dtype, math.prod(gemm_output_shape)]])
        arg_attributes.append([self.gemm_args[3], [MLIRKernelArgs.MLIR_ARGS_OUT, self.input_nodes[0].layout.dtype, math.prod(gemm_output_shape)]])

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