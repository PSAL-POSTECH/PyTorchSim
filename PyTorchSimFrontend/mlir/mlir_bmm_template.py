import os
from torch import empty_strided
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import PyTorchSimFrontend.extension_codecache as extension_codecache

BMM_TEMPLATE = r"""
// BMM kernel
// BATCH = {{ B }}
// M = {{ M }}
// N = {{ N }}
// K = {{ K }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// SUB_TILE_M = {{ SUB_TILE_M }}
// SUB_TILE_N = {{ SUB_TILE_N }}
#map0 = affine_map<(d0, d1, d2) -> ({{ X_map }})>
#map1 = affine_map<(d0, d1, d2) -> ({{ W_map }})>
#map2 = affine_map<(d0, d1, d2) -> (d0 * {{ M * N }} + d1 * {{ N }} + d2)>
memref.global @X_spad : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index{% if Bias %}
  %c_mvin3 = arith.constant 14 : index{% endif %}
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %axis = arith.constant 2 : index
  %X_buffer = memref.get_global @X_spad : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>{% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
  %c0 = arith.constant 0 : index
{{ kernel.def_local_vars() }}
  affine.for %b=0 to {{ B }} {
    affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {
      affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
        %X_buffer2D = memref.reinterpret_cast %X_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
        %W_buffer2D = memref.reinterpret_cast %W_buffer to offset: [0], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
        %Y_buffer2D = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

        %index2 = affine.apply #map2(%b, %t_m, %t_n)
        {% if Bias -%}
        memref.dma_start %Bias[
        {%- if Bias_rank == 2 -%} %index2 {%- else -%} %t_n {%- endif -%}
          ], %Y_buffer2D[0, 0], %c_mvin3, %tag0[%c0], %
        {%- if Bias_rank == 2 -%} axis {%- else -%} c0 {%- endif -%}
          , %vstride : memref<
        {%- if Bias_rank == 2 -%} {{ M * N }} {%- else -%} {{ N }} {%- endif -%}
          xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1 , {{ TILE_M }}] }
        {%- else -%}
        affine.vector_store %v0, %Y_buffer2D[0, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
        affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
          %index0 = affine.apply #map0(%b, %t_m, %t_k)
          %index1 = affine.apply #map1(%b, %t_k, %t_n)
          memref.dma_start %X[%index0], %X_buffer2D[%c0, %c0], %c_mvin, %tag1[%c0], %axis, %vstride
             : memref<{{ B * M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[1, {{ TILE_M }}]}
          memref.dma_start %W[%index1], %W_buffer2D[%c0, %c0], %c_mvin2, %tag2[%c0], %axis, %vstride
             : memref<{{ B * K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1, {{ TILE_K }}]}

          linalg.matmul ins(%X_buffer2D, %W_buffer2D : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                  outs(%Y_buffer2D : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true }
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

BMM_PROLOGUE_TEMPLATE = r"""
// BMM Prologue kernel
// BATCH = {{ B }}
// M = {{ M }}
// N = {{ N }}
// K = {{ K }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// SUB_TILE_M = {{ SUB_TILE_M }}
// SUB_TILE_N = {{ SUB_TILE_N }}
#map0 = affine_map<(d0, d1, d2) -> ({{ X_map }})>
#map1 = affine_map<(d0, d1, d2) -> ({{ W_map }})>
#map2 = affine_map<(d0, d1, d2) -> (d0 * {{ M * N }} + d1 * {{ N }} + d2)>
memref.global @X_spad : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index{% if Bias %}
  %c_mvin3 = arith.constant 14 : index{% endif %}
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %axis = arith.constant 2 : index
  %X_buffer = memref.get_global @X_spad : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>{% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
  %c0 = arith.constant 0 : index
{{ kernel.def_local_vars() }}
  affine.for %b=0 to {{ B }} {
    affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {
      affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
        %X_buffer2D = memref.reinterpret_cast %X_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
        %W_buffer2D = memref.reinterpret_cast %W_buffer to offset: [0], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
        %Y_buffer2D = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

        %index2 = affine.apply #map2(%b, %t_m, %t_n)
        {% if Bias -%}
        memref.dma_start %Bias[
        {%- if Bias_rank == 2 -%} %index2 {%- else -%} %t_n {%- endif -%}
          ], %Y_buffer2D[0, 0], %c_mvin3, %tag0[%c0], %
        {%- if Bias_rank == 2 -%} axis {%- else -%} c0 {%- endif -%}
          , %vstride : memref<
        {%- if Bias_rank == 2 -%} {{ M * N }} {%- else -%} {{ N }} {%- endif -%}
          xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1 , {{ TILE_M }}] }
        {%- else -%}
        affine.vector_store %v0, %Y_buffer2D[0, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
        affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
          %index0 = affine.apply #map0(%b, %t_m, %t_k)
          %index1 = affine.apply #map1(%b, %t_k, %t_n)
          {{kernel.prepare_input(indent_size=10)}}
          linalg.matmul ins(%X_buffer2D, %W_buffer2D : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                  outs(%Y_buffer2D : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true }
        memref.dma_start %Y_buffer[%c0, %c0, %c0], %Y[%index2], %c_mvout, %tag[%c0], %axis, %vstride : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<{{ B * M * N }}xf32>, memref<1xi32> { padding=0, sram_stride=[1, 1, {{ TILE_M }}] }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

BMM_REDUCTION_TEMPLATE = r"""
// BMM Reduction kernel
// BATCH = {{ B }}
// M = {{ M }}
// N = {{ N }}
// K = {{ K }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// SUB_TILE_M = {{ SUB_TILE_M }}
// SUB_TILE_N = {{ SUB_TILE_N }}
#map0 = affine_map<(d0, d1, d2) -> ({{ X_map }})>
#map1 = affine_map<(d0, d1, d2) -> ({{ W_map }})>
#map2 = affine_map<(d0, d1, d2) -> (d0 * {{ M * N }} + d1 * {{ N }} + d2)>
memref.global @X_spad : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index{% if Bias %}
  %c_mvin3 = arith.constant 14 : index{% endif %}
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %axis = arith.constant 2 : index
  %X_buffer = memref.get_global @X_spad : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>{% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
  %c0 = arith.constant 0 : index
{{ kernel.def_local_vars() }}
  affine.for %b=0 to {{ B }} {
    affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
      %red_idx = affine.apply affine_map<(d0, d1) -> ({{M}}*d0 + d1)>(%b, %t_n)
      {{kernel.reduction_acc()}} affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {{kernel.reduction_iter_arg()}} {
        %X_buffer2D = memref.reinterpret_cast %X_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
        %W_buffer2D = memref.reinterpret_cast %W_buffer to offset: [0], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
        %Y_buffer2D = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

        %index2 = affine.apply #map2(%b, %t_m, %t_n)
        {% if Bias -%}
        memref.dma_start %Bias[
        {%- if Bias_rank == 2 -%} %index2 {%- else -%} %t_n {%- endif -%}
          ], %Y_buffer2D[0, 0], %c_mvin3, %tag0[%c0], %
        {%- if Bias_rank == 2 -%} axis {%- else -%} c0 {%- endif -%}
          , %vstride : memref<
        {%- if Bias_rank == 2 -%} {{ M * N }} {%- else -%} {{ N }} {%- endif -%}
          xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1 , {{ TILE_M }}] }
        {%- else -%}
        affine.vector_store %v0, %Y_buffer2D[0, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
        affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
          %index0 = affine.apply #map0(%b, %t_m, %t_k)
          %index1 = affine.apply #map1(%b, %t_k, %t_n)
          memref.dma_start %X[%index0], %X_buffer2D[%c0, %c0], %c_mvin, %tag1[%c0], %axis, %vstride
             : memref<{{ B * M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[1, {{ TILE_M }}]}
          memref.dma_start %W[%index1], %W_buffer2D[%c0, %c0], %c_mvin2, %tag2[%c0], %axis, %vstride
             : memref<{{ B * K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1, {{ TILE_K }}]}

          linalg.matmul ins(%X_buffer2D, %W_buffer2D : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                  outs(%Y_buffer2D : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true, loop_k=true }
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true, loop_m=true }
      {{kernel.reduction_output(indent_size=6)}}
    } { outer_loop=true, loop_n=true}
  } { outer_loop=true }
  return
}
"""

class MLIRBMMTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        #if epilogue_nodes is not None and len(epilogue_nodes) > 0:
        #    self.output_node = cast(Buffer, epilogue_nodes[-1])

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        W_tensor =  empty_strided(W.layout.size, W.layout.stride)
        X_tensor =  empty_strided(X.layout.size, X.layout.stride)
        if len(W_tensor.size()) > 3:
          W_tensor = W_tensor.view([-1, W_tensor.shape[-2], W_tensor.shape[-1]])
        if len(X_tensor.size()) > 3:
          X_tensor = X_tensor.view([-1, X_tensor.shape[-2], X_tensor.shape[-1]])
        W_stride = W_tensor.stride()
        X_stride = X_tensor.stride()
        W_map = " + ".join([f"d{idx}*{s}" for idx, s in enumerate(W_stride)])
        X_map = " + ".join([f"d{idx}*{s}" for idx, s in enumerate(X_stride)])

        B, M, N, K = X_tensor.size()[0], X_tensor.size()[1], W_tensor.size()[2], X_tensor.size()[2]
        n_extra_node = len(epilogue_nodes) if epilogue_nodes is not None else 0
        TILE_M, TILE_N, TILE_K = kernel.gemm_combination_mapping(M, N, K, n_extra_node=n_extra_node)
        TOG_latency = M if TILE_M > M else TILE_M
        kernel.loop_size = [TOG_latency, TILE_N, TILE_K]
        TILE_K = TILE_K // 2 if prologue_nodes else TILE_K
        SUB_TILE_M = TILE_M if (TILE_M < kernel.vector_lane) or prologue_nodes else kernel.vector_lane
        SUB_TILE_N = TILE_N # if (TILE_N < kernel.vector_lane) or prologue_nodes else kernel.vector_lane
        SUB_TILE_K = TILE_K # if (TILE_K < kernel.vector_lane) or prologue_nodes else kernel.vector_lane

        if n_extra_node==1 and epilogue_nodes[0].is_reduction():
          template = BMM_REDUCTION_TEMPLATE
          nr_rdim = 1
        elif prologue_nodes:
          template = BMM_PROLOGUE_TEMPLATE
          nr_rdim = 0
        else:
          template = BMM_TEMPLATE
          nr_rdim = 0

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            B=B,
            M=M,
            N=N,
            K=K,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            SUB_TILE_M=SUB_TILE_M,
            SUB_TILE_N=SUB_TILE_N,
            SUB_TILE_K=SUB_TILE_K,
            DATA_STYPE="f32",
            DATA_SIZE=4,
            X = X,
            W = W,
            Y = Y,
            Bias = Bias,
            Bias_rank = len(Bias.data.get_size()) if Bias is not None else 0,
            X_map = X_map,
            W_map = W_map,
            Y_numel = B * M * N,
            input_reorder = self.input_reorder
        )

        kernel.prologue_info = dict (
            input_sram_var = "X_buffer2D",
            input_dram_var = "X",
            input_index_var = "index0",
            input_tag_var = "tag1",
            input_numel = B * M * K,
            input_tile_size = (TILE_M, TILE_K),
            input_sram_stride = [1, TILE_M],
            input_subtile_size = (SUB_TILE_M, SUB_TILE_K),
            weight_sram_var = "W_buffer2D",
            weight_dram_var = "W",
            weight_index_var = "index1",
            weight_tag_var = "tag2",
            weight_numel = B * K * N,
            weight_tile_size = (TILE_K, TILE_N),
            weight_sram_stride = [1, TILE_K],
            weight_subtile_size = (SUB_TILE_K, SUB_TILE_N),
            tile_size = (TILE_M, TILE_K),
            vlane_split_axis = 1,
            vlane_stride = 1,
            is_bmm = True,
        )
        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            dependent_buf = [],
            sram_var = "Y_buffer",
            dram_var = "Y",
            index_var = "index2",
            tag_var = "tag",
            vlane_split_axis = 2,
            vlane_stride = 1,
            mlir_dtype = kernel.render_options['DATA_STYPE'],
            dram_shape = f"memref<{kernel.render_options['Y_numel']}x{kernel.render_options['DATA_STYPE']}>",
            tile_size = (1, TILE_M, TILE_N),
            tile_stride = [1, 1, TILE_M],
            nr_rdim = nr_rdim,
            reduction_idx = "red_idx"
        )
        code = self._template_from_string(template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["M"], kernel.render_options["N"], kernel.render_options["K"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])

        self.header = f"float X_spad[{kernel.get_spad_size_per_lane(TILE_M, TILE_K)}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float W_spad[{kernel.get_spad_size_per_lane(TILE_K, TILE_N)}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{kernel.get_spad_size_per_lane(TILE_M, TILE_N)}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header = f"float X_spad[{TILE_M * TILE_K}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float W_spad[{TILE_K * TILE_N}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float Y_spad[{TILE_M * TILE_N}] __attribute__ ((section(\".spad\")));\n"

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