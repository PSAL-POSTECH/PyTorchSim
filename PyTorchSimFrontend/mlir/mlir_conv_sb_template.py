from sympy import  Symbol, Number
from typing import List, Optional

from PyTorchSimFrontend.mlir.mlir_conv_common import MLIRConvCommonTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import IRNode
from PyTorchSimFrontend.mlir import mlir_common

CONV_TEMPLATE = r"""
// Single Batch Conv2D kernel
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
// TILE_I_H={{ TILE_I_H }},
// TILE_I_W={{ TILE_I_W }},
// TILE_O_H={{ TILE_O_H }},
// TILE_O_W={{ TILE_O_W }},
// TILE_K_H={{ TILE_K_H }},
// TILE_K_W={{ TILE_K_W }},
// SUB_TILE_M={{ SUB_TILE_M }},
// SUB_TILE_N={{ SUB_TILE_N }},
// SUB_TILE_I_W={{ SUB_TILE_I_W }},
// SUB_TILE_K_H={{ SUB_TILE_K_H }},
// SUB_TILE_K_W={{ SUB_TILE_K_W }},
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DATA_STYPE = {{ DATA_STYPE }}

#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#map_I_W = affine_map<(d0, d1) -> (d0 * {{ STRIDE_W }} + d1)>
#offset_w_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_K_W * TILE_K, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_K, TILE_N) }})>
#offset_x_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_I_W, TILE_K) }} + d1)>
#offset_y_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }})>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_conv_kernel(inputs=[X, W, BIAS], outputs=[Y], names_str="X, W, Bias, Y", padded_input_size=PADDED_INPUT_SIZE, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}x{{DATA_STYPE}}>
  %c0 = arith.constant 0 : index
  {{- kernel.def_local_vars(indent_size=2) }}
  affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
    affine.for %o_h = 0 to {{ O_H }} step {{ TILE_O_H }} {
      affine.for %tile_m = 0 to {{ O_W }} step {{ TILE_M }} {
        // Initialize output
        {%- if BIAS %}
        {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Bias_tile_desc, subtile_size=[1, SUB_TILE_N, TILE_O_H, SUB_TILE_M], indent_size=8) }}
        {%- else %}
        affine.vector_store %v0, %output_buffer[%c0, %c0, %c0, %c0] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }}, vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}x{{DATA_STYPE}}>
        {%- endif %}
        affine.for %k_h = 0 to {{ K_H }} step {{ TILE_K_H }} {
          affine.for %k_w = 0 to {{ K_W }} step {{ TILE_K_W }} {
            affine.for %tile_k = 0 to {{ I_C }} step {{ TILE_K }} {
              %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
              %index_i_w = affine.apply #map_I_W(%tile_m, %k_w)
              // Load input & weight matrix
              {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, subtile_size=[1, SUB_TILE_I_H, SUB_TILE_M, SUB_TILE_K], indent_size=14) }}
              {{ kernel.def_dma_op("MVIN", "W", W_idx, W_tile_desc, subtile_size=[SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_K, SUB_TILE_N], indent_size=14) }}
              // Compute body part
              affine.for %tile_k_h = 0 to {{ TILE_K_H }} { // loop order should be fixed for timing simulation. Do not change this order.
                affine.for %tile_k_w = 0 to {{ TILE_K_W }} {
                  %offset_w = affine.apply #offset_w_map(%tile_k_h, %tile_k_w)
                  %W_buffer = memref.reinterpret_cast %weight_buffer to offset: [%offset_w], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ W_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_K }}x{{ TILE_N }}x{{DATA_STYPE}}, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                  affine.for %tile_o_h = 0 to {{ TILE_O_H }} {
                    affine.for %tile_o_w = 0 to {{ 1 }} { // TILE_O_W
                      %tile_i_h = affine.apply #map_I_H(%tile_o_h, %tile_k_h)
                      %offset_x = affine.apply #offset_x_map(%tile_i_h, %tile_k_w)
                      %offset_y = affine.apply #offset_y_map(%tile_o_h, %tile_o_w)
                      %X_buffer = memref.reinterpret_cast %input_buffer to offset: [%offset_x], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : {{ X_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_K }}x{{DATA_STYPE}}, strided<[{{ TILE_K }}, 1], offset: ?>, 1>
                      %Y_buffer = memref.reinterpret_cast %output_buffer to offset: [%offset_y], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_N }}x{{DATA_STYPE}}, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                      linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}x{{DATA_STYPE}}, strided<[{{ TILE_K }}, 1], offset: ?>, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{DATA_STYPE}}, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                            outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}x{{DATA_STYPE}}, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                    } { inner_loop=true }
                  } { inner_loop=true }
                } { inner_loop=true }
              } { inner_loop=true }
            } { accumulation_loop=true, subtile_loop="k" }
          } { accumulation_loop=true }
        } { accumulation_loop=true }
        // Store output matrix
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true, subtile_loop="m" }
    } { outer_loop=true }
  } { outer_loop=true, subtile_loop="n" }
  return
}
"""

class MLIRConvSingleBatchTemplate(MLIRConvCommonTemplate):
    WRAPPER_TEMPLATE = r"""
def {{ FUNC_NAME }}{{kernel.def_wrapper()}}:
    # Padding input
    padded_shape = list(X.shape)
    padded_shape[2] += 2 * {{ PADDING_H }}
    padded_shape[3] += 2 * {{ PADDING_W }}
    X_padding = torch.zeros(padded_shape).to(device=X.device)
    X_padding[:, :, {{ PADDING_H }}:X.shape[2] + {{ PADDING_H }}, {{ PADDING_W }}:X.shape[3] + {{ PADDING_W }}] = X

    # Tanspose inputs
    {%- for buf, name in kernel.get_conv_inputs().items() %}
      {%- if name == "X" %}
    {{ name }} = {{ name }}_padding.permute(0, 2, 3, 1).contiguous() # (BATCH, I_C, I_H, I_W) -> (BATCH, I_H, I_W, I_C)
      {%- elif name == "W" %}
    {{ name }} = {{ name }}.permute(2, 3, 1, 0).contiguous() # (O_C, I_C, K_H, K_W) -> (K_H, K_W, I_C, O_C)
      {%- elif name == "Bias" %}
    {{ name }} = {{ name }}
      {%- endif %}
    {%- endfor %}

    # Launch kernel
    {{ KERNEL_NAME }}<DEF_CONV_WRAPPER>
"""
    def __init__(self, input_nodes, layout, input_reorder=None, **kwargs):
        super().__init__(input_nodes, layout, input_reorder, **kwargs)

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
        # Extract input arguments info
        X, W, Y, Bias, n_extra_node, BATCH, I_C, I_H, I_W, O_C, K_H, K_W, O_H, O_W, PADDING_H, PADDING_W, STRIDE_H, STRIDE_W, precision_bytes = self.extract_info(kernel, template_buffer_node, epilogue_nodes)

        # Select tile size adn template
        conv_template = CONV_TEMPLATE
        if tile_info is None:
            TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K, TILE_I_H, TILE_I_W, SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = self.select_tile(kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W, precision_bytes)[0]
        else:
            TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K, TILE_I_H, TILE_I_W, SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = tile_info
        SUB_TILE_N = TILE_N if TILE_N > 512 else SUB_TILE_N
        TOG_latency = O_W if TILE_M > O_W else TILE_M
        TOG_latency = 8 if TOG_latency < 8 else TOG_latency
        kernel.loop_size = [TOG_latency, TILE_N, TILE_K]
        # Prepare tile descriptors
        vlane_stride = 1
        vlane_split_axis = 1
        X_tile_size = [1, TILE_I_H, TILE_I_W, TILE_K]
        X_tile_stride = [TILE_I_H * TILE_I_W * TILE_K , TILE_I_W * TILE_K, 1, TILE_I_W]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, 3, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("input_buffer")
        X_dim = [Symbol("c0"), Symbol("index_i_h"), Symbol("index_i_w"), Symbol("tile_k")]
        X_idx = [X_dim[0]*((I_W+2*PADDING_W)*(I_H+2*PADDING_H)*I_C), X_dim[1]*((I_W+2*PADDING_W)*I_C), X_dim[2]*I_C, X_dim[3]]

        W_tile_size = [TILE_K_H, TILE_K_W, TILE_K, TILE_N]
        W_tile_stride = [TILE_K_W * TILE_K * TILE_N, TILE_K * TILE_N, 1, TILE_K]
        W_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, 3, vlane_stride)
        W_tile_desc.set_tile_size_stride(W_tile_size, W_tile_stride)
        W_tile_desc.set_name("weight_buffer")
        W_dim = [Symbol("k_h"), Symbol("k_w"), Symbol("tile_k"), Symbol("tile_n")]
        W_idx = [W_dim[0]*K_W*I_C*O_C , W_dim[1]*I_C*O_C, W_dim[2]*O_C, W_dim[3]]

        Y_tile_size = [1, TILE_N, TILE_O_H, TILE_M]
        Y_tile_stride = [TILE_O_H * TILE_M * TILE_N, TILE_M, TILE_M * TILE_N, 1] # N, C, H, W
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("output_buffer")
        Y_idx = [Number(0), Symbol("tile_n")*O_H*O_W, Symbol("o_h")*O_W, Symbol("tile_m")]

        # Extract Bias info
        Bias_idx = [Number(0), Symbol("tile_n"), Number(0), Number(0)]
        Bias_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Bias_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Bias_tile_desc.set_name("output_buffer")
        if Bias is not None:
          Bias_tile_desc.offset = Bias.get_layout().offset

        data_stype = mlir_common.DTYPE_TO_MLIR[X.get_dtype()]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            X=X, W=W, Y=Y, BIAS=Bias,
            PADDED_INPUT_SIZE=self.get_padded_input_size(X),
            BATCH=BATCH,
            I_C=I_C,
            I_H=I_H,
            I_W=I_W,
            O_C=O_C,
            K_H=K_H,
            K_W=K_W,
            O_H=O_H,
            O_W=O_W,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            TILE_I_H=TILE_I_H,
            TILE_I_W=TILE_I_W,
            TILE_O_H=TILE_O_H,
            TILE_O_W=TILE_O_W,
            TILE_K_H=TILE_K_H,
            TILE_K_W=TILE_K_W,
            SUB_TILE_M=SUB_TILE_M,
            SUB_TILE_N=SUB_TILE_N,
            SUB_TILE_K=SUB_TILE_K,
            SUB_TILE_I_H=SUB_TILE_I_H,
            SUB_TILE_I_W=SUB_TILE_I_W,
            SUB_TILE_K_H=SUB_TILE_K_H,
            SUB_TILE_K_W=SUB_TILE_K_W,
            PADDING_H=PADDING_H,
            PADDING_W=PADDING_W,
            STRIDE_H=STRIDE_H,
            STRIDE_W=STRIDE_W,
            X_tile_desc = X_tile_desc,
            W_tile_desc = W_tile_desc,
            Y_tile_desc = Y_tile_desc,
            Bias_tile_desc = Bias_tile_desc,
            X_idx = X_idx,
            W_idx = W_idx,
            Bias_idx = Bias_idx,
            DATA_STYPE=data_stype,
            input_reorder=self.input_reorder
        )

        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            sram_var = "output_buffer",
            dram_var = "Y",
            dram_idx = Y_idx,
            dram_tile_desc = Y_tile_desc,
            dim_aliasing = {"index0":"c0", "index1":"tile_n", "index2":"o_h", "index3":"tile_m"}
        )
        kernel.exception_nodes["X"] = {"numel" : (I_W+2*PADDING_W)*(I_H+2*PADDING_H)*I_C*BATCH}
        code = self._template_from_string(conv_template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["K_H"], kernel.render_options["K_W"], kernel.render_options["O_H"], kernel.render_options["O_W"], kernel.render_options["BATCH"], kernel.render_options["O_C"], kernel.render_options["I_C"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])
        return code

    def select_tile(self, kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W, precision_bytes):
        tile_candidates = kernel.conv_single_batch_mapping(BATCH, O_C, I_C, K_H, 1, O_H, O_W, self.stride, self.dilation, n_extra_node, precision_bytes=precision_bytes) # TODO: implement K_W
        for idx, (TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K) in enumerate(tile_candidates):
            TILE_I_H = 1 + (TILE_O_H - 1) * self.stride[0] + (TILE_K_H - 1) * self.dilation[0]
            TILE_I_W = 1 + (TILE_O_W - 1) * self.stride[1] + (TILE_K_W - 1) * self.dilation[1]
            SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W = 1, 1, 1, 1
            SUB_TILE_M = TILE_I_W if TILE_I_W < kernel.vector_lane else kernel.vector_lane
            SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
            SUB_TILE_K = TILE_K
            tile_candidates[idx] = TILE_K_H,TILE_K_W,TILE_O_H,TILE_O_W,TILE_M,TILE_N,TILE_K,TILE_I_H,TILE_I_W,SUB_TILE_I_H,SUB_TILE_I_W,SUB_TILE_K_H,SUB_TILE_K_W,SUB_TILE_M,SUB_TILE_N,SUB_TILE_K
        return tile_candidates
