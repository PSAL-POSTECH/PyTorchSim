import os
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import PyTorchSimFrontend.extension_codecache as extension_codecache

GEMM_TEMPLATE = r"""
{% if X_transposed %}#map0 = affine_map<(d0, d1) -> (d1 * {{ M }} + d0)>{% else %}#map0 = affine_map<(d0, d1) -> (d0 * {{ K }} + d1)>{% endif %}
{% if W_transposed %}#map1 = affine_map<(d0, d1) -> (d1 * {{ K }} + d0)>{% else %}#map1 = affine_map<(d0, d1) -> (d0 * {{ N }} + d1)>{% endif %}
#map2 = affine_map<(d0, d1) -> (d0 * {{ N }} + d1)>
memref.global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index{% if Bias %}
  %c_mvin3 = arith.constant 14 : index{% endif %}
  %c_mvout = arith.constant 3 : index
  %c_set = arith.constant 2 : index
  %x_chunk = arith.constant {% if X_transposed %} {{ kernel.vector_lane * 2 + 0 }} {% else %} {{ 2 }} {% endif %} : index
  %w_chunk = arith.constant {% if W_transposed %} {{ TILE_K * 2 + 0 }} {% else %} {{ 2 }} {% endif %} : index
  %M = arith.constant {{ M }} : index
  %N = arith.constant {{ N }} : index
  %K = arith.constant {{ K }} : index
  %X_buffer = memref.get_global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>{% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ TILE_M * TILE_N // kernel.vector_lane }}xf32>{% endif %}
  %c0 = arith.constant 0 : index

  affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {
    affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
      %index2 = affine.apply #map2(%t_m, %t_n)
      {% if Bias -%}
      affine.dma_start %Bias[
        {%- if Bias_rank == 2 -%} %index2 {%- else -%} %t_n {%- endif -%}
        ], %Y_buffer[%c0, %c0], %tag[0], %c_mvin3, %
        {%- if Bias_rank == 2 -%} N {%- else -%} c0 {%- endif -%}
        , %c_set : memref<
        {%- if Bias_rank == 2 -%}  {{ M * N }} {%- else -%} {{ N }} {%- endif -%}
        xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32>  { subtile_size=[{{ kernel.vector_lane }}, {{ kernel.vector_lane }}], async=1 }
      {%- else -%}
      affine.vector_store %v0, %Y_buffer[0, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ TILE_M * TILE_N // kernel.vector_lane }}xf32>
      {%- endif %}
      affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
        %index0 = affine.apply #map0(%t_m, %t_k)
        %index1 = affine.apply #map1(%t_k, %t_n)
        affine.dma_start %X[%index0], %X_buffer[%c0, %c0], %tag[0], %c_mvin,
        {%- if X_transposed -%} %M, %x_chunk {%- else -%} %K, %x_chunk {%- endif -%}
           : memref<{{ M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ kernel.vector_lane }}, {{ TILE_K }}], async=1{% if X_transposed %}, transpose=1{% endif %} }
        affine.dma_start %W[%index1], %W_buffer[%c0, %c0], %tag[0], %c_mvin2,
        {%- if W_transposed -%} %K, %w_chunk {%- else -%} %N, %w_chunk {%- endif -%}
           : memref<{{ K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ TILE_K }}, {{ kernel.vector_lane }}], async=1{% if W_transposed %}, transpose=1{% endif %} }
        linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
      } { accumulation_loop=true }
      {{kernel.store_output()}}
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

EMPTY_TEMPLATE = r"""
func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
    return
}
"""

class MLIRGemmTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def is_transposed(self, node):
        if isinstance(node, ReinterpretView):
            if 0 in node.layout.stride: # [MoE] Temporary solution
                if node.layout.stride[1] == 0:
                    return True
            if node.layout.stride != node.data.layout.stride:
                if node.layout.stride[-2] == node.data.layout.stride[-1] and node.layout.stride[-1] == node.data.layout.stride[-2]:
                    return True
                else:
                  raise NotImplementedError("If the stride is not equal to the original stride, it should have been transposed.")
        return False

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        # if epilogue_nodes is not None and len(epilogue_nodes) > 0:
        #     self.output_node = cast(Buffer, epilogue_nodes[-1]) #FIXME: Temperary solution

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        M, N, K = X.get_size()[0], W.get_size()[1], X.get_size()[1]
        if (M == 0) or (N == 0) or (K == 0):
            TILE_M, TILE_N, TILE_K = 0, 0, 0
            template = EMPTY_TEMPLATE
        else:
            TILE_M, TILE_N, TILE_K = kernel.gemm_combination_mapping(M, N, K)
            template = GEMM_TEMPLATE
        kernel.tile_size = [TILE_M, TILE_N, TILE_K]
        kernel.loop_size =[M, N, K]

        W_transposed = self.is_transposed(W)
        X_transposed = self.is_transposed(X)

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            M=M,
            N=N,
            K=K,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            DATA_STYPE="f32",
            DATA_SIZE=4,
            X = X,
            W = W,
            Y = Y,
            Bias = Bias,
            Bias_rank = len(Bias.data.get_size()) if Bias is not None else 0,
            W_transposed = W_transposed,
            X_transposed = X_transposed,
            epilogue_nodes = epilogue_nodes,
            input_reorder = self.input_reorder
        )
        code = self._template_from_string(template).render(**kernel.render_options)

        self.header = f"float X_spad[{TILE_M * TILE_K // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float W_spad[{TILE_K * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{TILE_M * TILE_N // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header = f"float X_spad[{TILE_M * TILE_K}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float W_spad[{TILE_K * TILE_N}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float Y_spad[{TILE_M * TILE_N}] __attribute__ ((section(\".spad\")));\n"

        kernel.add_loop_info([kernel.render_options["M"], kernel.render_options["N"], kernel.render_options["K"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])

        return code

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
