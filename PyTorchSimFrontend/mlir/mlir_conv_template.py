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

CONV_TEMPLATE = r"""
// Conv2D kernel
// BATCH = {{ BATCH }}
// I_C = {{ I_C }}
// I_H = {{ I_H }}
// I_W = {{ I_W }}
// O_C = {{ O_C }}
// K_H = {{ K_H }}
// K_W = {{ K_W }}
// O_H = {{ O_H }}
// O_W = {{ O_W }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DILATION_H = {{ DILATION_H }}
// DILATION_W = {{ DILATION_W }}
// DATA_STYPE = {{ DATA_STYPE }}
// DATA_SIZE = {{ DATA_SIZE }}

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ O_W * BATCH * O_C }} + d1 * {{ BATCH * O_C }} + d2 * {{ O_C }} + d3)> // output (O_H, O_W, BATCH, O_C)
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ (I_W + 2 * PADDING_W) * BATCH * I_C }} + d1 * {{ BATCH * I_C }} + d2 * {{ I_C }} + d3)> // input (I_H, I_W, BATCH, I_C)
#map2 = affine_map<(d0, d1, d2) -> (d0 * {{ I_C * O_C }} + d1 * {{ O_C }} + d2)> // weight (K_H * K_W, I_C, O_C)
#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#map_I_W = affine_map<(d0, d1) -> (d0 * {{ STRIDE_W }} + d1)>
#map_K_HW = affine_map<(d0, d1) -> (d0 * {{ K_W }} + d1)>

memref.global @X_spad : memref<{{TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

func.func @{{ KERNEL_NAME }}({{ KERNEL_DEF }}) {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %input_axis = arith.constant 3 : index
  %weight_axis = arith.constant 2 : index
  %X_buffer = memref.get_global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %v0 = arith.constant dense<0.0> : vector<{{ TILE_N }}xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %K_W = arith.constant {{ K_W }} : index
  %stride_h = arith.constant {{ STRIDE_H }} : index
  %stride_w = arith.constant {{ STRIDE_W }} : index

  // 1x1 convolution loop
  affine.for %k_h = 0 to {{ K_H }} {
    affine.for %k_w = 0 to {{ K_W }} {
      // 1x1 convolution tiling loop
      affine.for %o_h = 0 to {{ O_H }} {
        affine.for %o_w = 0 to {{ O_W }} {
          affine.for %tile_m = 0 to {{ BATCH }} step {{ TILE_M }} {
            affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
              // Init output matrix
              %index0 = affine.apply #map0(%o_h, %o_w, %tile_m, %tile_n)
              %cond_h = arith.cmpi eq, %k_h, %c0 : index
              %cond_w = arith.cmpi eq, %k_w, %c0 : index
              %cond_hw = arith.andi %cond_h, %cond_w : i1
              scf.if %cond_hw {
                {%- if BIAS %}
                memref.dma_start %Bias[%tile_n], %Y_buffer[%c0, %c0], %c_mvin, %tag[%c0], %c0, %vstride
                    : memref<{{ O_C }}xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { async=1, sram_stride=[1, {{ TILE_M }}]}
                {%- else %}
                affine.vector_store %v0, %Y_buffer[%c0, %c0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ TILE_N }}xf32>
                {%- endif %}
              } else {
                memref.dma_start %Y[%index0], %Y_buffer[%c0, %c0], %c_mvin, %tag[%c0], %input_axis, %vstride
                    : memref<{{ BATCH * O_C * O_H * O_W }}xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { padding=0, sram_stride=[1, {{ TILE_M }}]}
              }
              affine.for %tile_k = 0 to {{ I_C }} step {{ TILE_K }} {
                %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
                %index_i_w = affine.apply #map_I_W(%o_w, %k_w)
                %index1 = affine.apply #map1(%index_i_h, %index_i_w, %tile_m, %tile_k) // input index
                %index_k_hw = affine.apply #map_K_HW(%k_h, %k_w)
                %index2 = affine.apply #map2(%index_k_hw, %tile_k, %tile_n) // weight index
                // Load input matrix
                memref.dma_start %X[%index1], %X_buffer[%c0, %c0], %c_mvin, %tag[%c0], %input_axis, %vstride
                    : memref<{{ BATCH * I_C * (I_H + 2 * PADDING_H) * (I_W + 2 * PADDING_W) }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ kernel.vector_lane }}, {{ TILE_K }}], async=1, sram_stride=[1, {{ TILE_M }}]}
                // Load kernel matrix
                memref.dma_start %W[%index2], %W_buffer[%c0, %c0], %c_mvin, %tag[%c0], %weight_axis, %vstride
                    : memref<{{ O_C * I_C * K_H * K_W }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ TILE_K }}, {{ kernel.vector_lane }}], async=1, sram_stride=[1, 1]}
                // matmul
                linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>)
                      outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>)
              } { accumulation_loop=true }
              // Store output matrix
              memref.dma_start %Y_buffer[%c0, %c0], %Y[%index0], %c_mvout, %tag[%c0], %input_axis, %vstride
                  : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<{{ BATCH * O_C * O_H * O_W }}xf32>, memref<1xi32> {padding=0, sram_stride=[1, {{ TILE_M }}]}
            } { outer_loop=true }
          } { outer_loop=true }
        } { outer_loop=true }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

WRAPPER_TEMPLATE = r"""
def {{ FUNC_NAME }}({{ INPUT }}, {{ WEIGHT }}{% if BIAS %}, {{ BIAS }} {% endif %}, {{ OUT }}):
    # Padding input
    padded_shape = list({{ INPUT }}.shape)
    padded_shape[2] += 2 * {{ PADDING_H }}
    padded_shape[3] += 2 * {{ PADDING_W }}
    {{ INPUT }}_padding = torch.zeros(padded_shape, device={{ INPUT }}.device)
    {{ INPUT }}_padding[:, :, {{ PADDING_H }}:{{ INPUT }}.shape[2] + {{ PADDING_H }}, {{ PADDING_W }}:{{ INPUT }}.shape[3] + {{ PADDING_W }}] = {{ INPUT }}

    # Tanspose tensors
    t_{{ INPUT }} = {{ INPUT }}_padding.permute(2, 3, 0, 1).contiguous() # (BATCH, I_C, I_H, I_W) -> (I_H, I_W, BATCH, I_C)
    t_{{ WEIGHT }} = {{ WEIGHT }}.permute(2, 3, 1, 0).contiguous() # (O_C, I_C, K_H, K_W) -> (K_H, K_W, I_C, O_C)
    t_{{ OUT }} = {{ OUT }}.permute(2, 3, 0, 1).contiguous() # (BATCH, O_C, O_H, O_W) -> (O_H, O_W, BATCH, O_C)

    {{ KERNEL_NAME }}(t_{{ INPUT }}, t_{{ WEIGHT }}{% if BIAS %}, {{ BIAS }} {% endif %}, t_{{ OUT }})

    # Transpose back
    {{ OUT }}.copy_(t_{{ OUT }}.permute(2, 3, 0, 1).contiguous()) # (O_H, O_W, BATCH, O_C) -> (BATCH, O_C, O_H, O_W)
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
        self.kernel_args = ['X', 'W', 'Bias', 'Y']

    def is_transposed(self, node):
        if isinstance(node, ReinterpretView):
            if node.layout.stride != node.data.layout.stride:
                if node.layout.stride[-2] == node.data.layout.stride[-1] and node.layout.stride[-1] == node.data.layout.stride[-2]:
                    return True
                else:
                  raise NotImplementedError("If the stride is not equal to the original stride, it should have been transposed.")
        return False

    # Can use math.multi ?
    def def_kernel(self) ->str:
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        if len(self.input_nodes) == 3:
          Bias = self.input_nodes[2]
        else:
          Bias = None

        input_padded = list(X.layout.size)
        input_padded[2] += 2 * self.padding[0]
        input_padded[3] += 2 * self.padding[1]
        input_size = math.prod(input_padded)
        weight_size = math.prod(W.layout.size)
        if Bias is not None:
          bias_size = math.prod(Bias.layout.size)
        output_size = math.prod(Y.layout.size)

        if Bias is None:
          return f"%{self.kernel_args[0]}: memref<{input_size}xf32>, %{self.kernel_args[1]}: memref<{weight_size}xf32>, %{self.kernel_args[3]}: memref<{output_size}xf32>"
        else:
          return f"%{self.kernel_args[0]}: memref<{input_size}xf32>, %{self.kernel_args[1]}: memref<{weight_size}xf32>, %{self.kernel_args[2]}: memref<{bias_size}xf32>, %{self.kernel_args[3]}: memref<{output_size}xf32>"

    def get_tile_options(self):
        BATCH = self.input_nodes[0].layout.size[0]
        I_C = self.input_nodes[0].layout.size[1]
        O_C = self.input_nodes[1].layout.size[0]

        tile_m_options = divisors(BATCH)
        tile_k_options = divisors(I_C)
        tile_n_options = divisors(O_C)

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

        BATCH = X.layout.size[0]
        I_C = X.layout.size[1]
        O_C = W.layout.size[0]
        K_H = W.layout.size[2]
        K_W = W.layout.size[3]
        O_H = Y.layout.size[2]
        O_W = Y.layout.size[3]

        # FIXME: fixed tile size
        TILE_M = kernel.vector_lane
        TILE_N = kernel.vector_lane
        TILE_K = kernel.vector_lane

        kernel.tile_size = [TILE_M, TILE_N, TILE_K]
        kernel.loop_size = [K_H, K_W, O_H, O_W, BATCH, O_C, I_C]

        # FIXME: transposed inputs not supported
        # W_transposed = self.is_transposed(W)
        # X_transposed = self.is_transposed(X)

        options = dict(
            KERNEL_NAME=self.name,
            KERNEL_DEF=self.def_kernel(),
            kernel=kernel,
            BATCH=X.layout.size[0],
            I_C=X.layout.size[1],
            I_H=X.layout.size[2],
            I_W=X.layout.size[3],
            O_C=W.layout.size[0],
            K_H=W.layout.size[2],
            K_W=W.layout.size[3],
            O_H=O_H,
            O_W=O_W,
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
            BIAS=Bias
        )
        code = self._template_from_string(CONV_TEMPLATE).render(**options)

        self.header = f"float X_spad[{TILE_M * TILE_K // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float W_spad[{TILE_K * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{TILE_M * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header = f"float X_spad[{TILE_M * TILE_K}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float W_spad[{TILE_K * TILE_N}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float Y_spad[{TILE_M * TILE_N}] __attribute__ ((section(\".spad\")));\n"

        kernel.add_loop_info([options["K_H"], options["K_W"], options["O_H"], options["O_W"], options["BATCH"], options["O_C"], options["I_C"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
        kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=self.input_reorder)

        return code

    def outer_func_render(self, kernel_name, input_args):
        options = dict(
            KERNEL_NAME=kernel_name,
            FUNC_NAME=self.function_name,
            INPUT=input_args[0],
            WEIGHT=input_args[1],
            BIAS=0 if len(input_args) == 3 else input_args[2],
            OUT=input_args[3] if len(input_args) == 4 else input_args[2],
            PADDING_H=self.padding[0],
            PADDING_W=self.padding[1],
            VALIDATION_MODE=extension_config.CONFIG_TORCHSIM_VALIDATION_MODE,
            BACKENDSIM_EAGER_MODE=extension_config.CONFIG_BACKENDSIM_EAGER_MODE,
            HASH_VALUE=self.hash_value
        )
        code = self._template_from_string(WRAPPER_TEMPLATE).render(**options)
        return code, self.function_name

    def get_arg_attributes(self):
        arg_attributes = []

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        X_shape = [X.get_size()[i] for i in (2, 3, 0, 1)]
        X_shape[0] += 2 * self.padding[0]
        X_shape[1] += 2 * self.padding[1]
        W_shape = [W.get_size()[i] for i in (2, 3, 1, 0)]
        Y_shape = [Y.get_size()[i] for i in (2, 3, 0, 1)]

        if Bias is not None:
          Bias_shape = [Bias.get_size()]

        def compute_stride(shape):
            stride = [1] * len(shape)
            for i in range(len(shape)-2, -1, -1):
                stride[i] = stride[i+1] * shape[i+1]
            return stride

        X_stride = compute_stride(X_shape)
        W_stride = compute_stride(W_shape)
        Y_stride = compute_stride(Y_shape)
        if Bias is not None:
          Bias_stride = compute_stride(Bias_shape)

        arg_attributes.append([self.kernel_args[0], [MLIRKernelArgs.MLIR_ARGS_IN, X.layout.dtype, math.prod(X_shape), X_shape, X_stride]])
        arg_attributes.append([self.kernel_args[1], [MLIRKernelArgs.MLIR_ARGS_IN, W.layout.dtype, math.prod(W_shape), W_shape, W_stride]])
        if Bias is not None:
          arg_attributes.append([self.kernel_args[2], [MLIRKernelArgs.MLIR_ARGS_IN, Bias.layout.dtype, math.prod(Bias_shape), Bias_shape, Bias_stride]])
        arg_attributes.append([self.kernel_args[3], [MLIRKernelArgs.MLIR_ARGS_OUT, Y.layout.dtype, math.prod(Y_shape), Y_shape, Y_stride]])

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