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
from PyTorchSimFrontend import extension_config

GEMM_TEMPLATE = r"""
// GEMM {% if prologue_nodes -%}prologue fused{%- endif %} {% if epilogue_nodes -%}eilogue fused{%- endif %} kernel
// M = {{ M }}
// N = {{ N }}
// K = {{ K }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// SUB_TILE_M = {{ SUB_TILE_M }}
// SUB_TILE_N = {{ SUB_TILE_N }}
#map0 = affine_map<(d0, d1) -> ({{ X_map }})>
#map1 = affine_map<(d0, d1) -> ({{ W_map }})>
#map2 = affine_map<(d0, d1) -> (d0 * {{ N }} + d1)>
#map3 = affine_map<(d0, d1) -> (d0 * {{ N }})>
memref.global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index{% if Bias %}
  %c_mvin3 = arith.constant 14 : index{% endif %}
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %axis = arith.constant 1 : index
  %X_buffer = memref.get_global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>{% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
  %c0 = arith.constant 0 : index
{{ kernel.def_local_vars() }}

  affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {
    affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
      %index2 = affine.apply #map2(%t_m, %t_n)
      %index3 = affine.apply #map2(%t_m, %t_n)
      {%- if Bias %}
      memref.dma_start %Bias[{{ Bias_idx }}], %Y_buffer[%c0, %c0], %c_mvin3, %tag0[%c0], %
        {%- if Bias_rank == 2 -%} axis {%- else -%} c0 {%- endif -%}
        , %vstride : memref<{{ Bias.data.get_numel() }}xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32>  { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1, {{ TILE_M }}] }
      {%- else %}
      affine.vector_store %v0, %Y_buffer[0, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>
      {%- endif %}
      affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
        %index0 = affine.apply #map0(%t_m, %t_k)
        %index1 = affine.apply #map1(%t_k, %t_n)
        {% if prologue_nodes -%}
        // prologue nodes
        {{kernel.prepare_input(indent_size=8)}}
        {%- else -%}
        memref.dma_start %X[%index0], %X_buffer[%c0, %c0], %c_mvin, %tag1[%c0], %axis, %vstride
           : memref<{{ M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[1, {{ TILE_M }}]}
        memref.dma_start %W[%index1], %W_buffer[%c0, %c0], %c_mvin2, %tag2[%c0], %axis, %vstride
           : memref<{{ K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1, {{ TILE_K }}]}
        {%- endif %}
        linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
      } { accumulation_loop=true }
      {{kernel.store_output(indent_size=6)}}
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

GEMM_REDUCTION_TEMPLATE = r"""
// GEMM reduction kernel
// M = {{ M }}
// N = {{ N }}
// K = {{ K }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// SUB_TILE_M = {{ SUB_TILE_M }}
// SUB_TILE_N = {{ SUB_TILE_N }}
#map0 = affine_map<(d0, d1) -> ({{ X_map }})>
#map1 = affine_map<(d0, d1) -> ({{ W_map }})>
#map2 = affine_map<(d0, d1) -> (d0 * {{ N }} + d1)>
#map3 = affine_map<(d0, d1) -> (d0 * {{ N }})>
memref.global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index{% if Bias %}
  %c_mvin3 = arith.constant 14 : index{% endif %}
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %axis = arith.constant 1 : index
  %X_buffer = memref.get_global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>{% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>{% endif %}
  %c0 = arith.constant 0 : index
{{ kernel.def_local_vars() }}

  affine.for %t_n = 0 to {{ N }} step {{ TILE_N }} {
    {{kernel.reduction_acc()}} affine.for %t_m = 0 to {{ M }} step {{ TILE_M }} {{kernel.reduction_iter_arg()}} {
      %index2 = affine.apply #map2(%t_m, %t_n)
      %index3 = affine.apply #map2(%t_m, %t_n)
      {%- if Bias %}
      memref.dma_start %Bias[{{ Bias_idx }}], %Y_buffer[%c0, %c0], %c_mvin3, %tag0[%c0], %
        {%- if Bias_rank == 2 -%} axis {%- else -%} c0 {%- endif -%}
        , %vstride : memref<{{ Bias.data.get_numel() }}xf32>, memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32>  { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1, {{ TILE_M }}] }
      {%- else %}
      affine.vector_store %v0, %Y_buffer[0, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}xf32>
      {%- endif %}
      affine.for %t_k = 0 to {{ K }} step {{ TILE_K }} {
        %index0 = affine.apply #map0(%t_m, %t_k)
        %index1 = affine.apply #map1(%t_k, %t_n)
        memref.dma_start %X[%index0], %X_buffer[%c0, %c0], %c_mvin, %tag1[%c0], %axis, %vstride
           : memref<{{ M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[1, {{ TILE_M }}]}
        memref.dma_start %W[%index1], %W_buffer[%c0, %c0], %c_mvin2, %tag2[%c0], %axis, %vstride
           : memref<{{ K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[1, {{ TILE_K }}]}
        linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
      } { accumulation_loop=true, loop_k=true }
      {{kernel.store_output(indent_size=6)}}
    } { outer_loop=true, loop_m=true}
    {{kernel.reduction_output(indent_size=4)}}
  } { outer_loop=true, loop_n=true }
  return
}
"""

class MLIRGemmTemplate(MLIRTemplate):
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
        # if epilogue_nodes is not None and len(epilogue_nodes) > 0:
        #     self.output_node = cast(Buffer, epilogue_nodes[-1]) #FIXME: Temperary solution

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node

        W_tensor =  empty_strided(W.layout.size, W.layout.stride)
        X_tensor =  empty_strided(X.layout.size, X.layout.stride)
        if len(W_tensor.size()) > 2 or len(X_tensor.size()) > 2:
            raise NotImplementedError("Please report this case to us...")
        W_stride = W_tensor.stride()
        X_stride = X_tensor.stride()
        W_map = " + ".join([f"d{idx}*{s}" for idx, s in enumerate(W_stride)])
        X_map = " + ".join([f"d{idx}*{s}" for idx, s in enumerate(X_stride)])

        M, N, K = X_tensor.size()[0], W_tensor.size()[1], X_tensor.size()[1]
        n_extra_node = len(epilogue_nodes) if epilogue_nodes is not None else 0
        # Caculate extra reads
        n_extra_read = set()
        if epilogue_nodes is not None:
          for enode in epilogue_nodes:
            n_extra_read.update(enode.node.get_read_names())
          if self.output_node.name in n_extra_read:
            n_extra_read.remove(self.output_node.name)

        n_prologue_node = len(prologue_nodes) if prologue_nodes is not None else 0
        nr_rdim = 0
        if (M == 0) or (N == 0) or (K == 0):
            TILE_M, TILE_N, TILE_K = 1, 1, 1
            template = EMPTY_TEMPLATE
        elif n_extra_node>=1 and epilogue_nodes[0].is_reduction():
            TILE_M, TILE_N, TILE_K = kernel.gemm_combination_mapping(M, N, K, len(n_extra_read), n_prologue_node, min_tile=True)
            template = GEMM_REDUCTION_TEMPLATE
            nr_rdim = 1
        else:
            TILE_M, TILE_N, TILE_K = kernel.gemm_combination_mapping(M, N, K, len(n_extra_read), n_prologue_node, min_tile=True)
            template = GEMM_TEMPLATE
        TILE_M = min(extension_config.CONFIG_FORCE_TILE_M, TILE_M)
        TILE_N = min(extension_config.CONFIG_FORCE_TILE_N, TILE_N)
        TILE_K = min(extension_config.CONFIG_FORCE_TILE_K, TILE_K)
        SUB_TILE_M = TILE_M if TILE_M < kernel.vector_lane else kernel.vector_lane
        if (TILE_M == M and TILE_N == N):
            SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
        else: # Avoid Row Conflict of weights
            SUB_TILE_N = TILE_N
        SUB_TILE_N = TILE_N if TILE_N > 512 else SUB_TILE_N # FIXME: hardcoded & 126 line has same feature
        SUB_TILE_K = TILE_K
        TOG_latency = M if SUB_TILE_M > M else SUB_TILE_M
        kernel.loop_size =[TOG_latency, SUB_TILE_N, SUB_TILE_K]

        # Extract Bias info
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
        if Bias is not None:
          if Bias.data.get_numel() == M*N:
            Bias_idx = "%index2"
          elif Bias.data.get_numel() == M:
            Bias_idx = "%index3"
          else:
            Bias_idx = "%t_n"
        else:
          Bias_idx = None

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
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
            Bias_idx = Bias_idx,
            Bias_rank = len(Bias.data.get_size()) if Bias is not None else 0,
            X_map = X_map,
            W_map = W_map,
            Y_numel = M * N,
            epilogue_nodes = epilogue_nodes,
            prologue_nodes = prologue_nodes,
            input_reorder = self.input_reorder
        )
        kernel.prologue_info = dict (
            input_sram_var = "X_buffer",
            input_dram_var = "X",
            input_index_var = "index0",
            input_tag_var = "tag1",
            input_numel = M * K,
            input_tile_size = (TILE_M, TILE_K),
            input_sram_stride = [1, TILE_M],
            vector_sram_stride = [TILE_M, 1],
            input_subtile_size = (SUB_TILE_M, SUB_TILE_K),
            weight_sram_var = "W_buffer",
            weight_dram_var = "W",
            weight_index_var = "index1",
            weight_tag_var = "tag2",
            weight_numel = K * N,
            weight_tile_size = (TILE_K, TILE_N),
            weight_sram_stride = [1, TILE_K],
            weight_subtile_size = (SUB_TILE_K, SUB_TILE_N),
            tile_size = (TILE_M, TILE_K),
            vlane_split_axis = 1,
            vlane_stride = 1,
            is_bmm = False,
        )
        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            dependent_buf = [],
            sram_var = "Y_buffer",
            dram_var = "Y",
            index_var = "index2",
            tag_var = "tag",
            vlane_split_axis = 1,
            vlane_stride = 1,
            mlir_dtype = kernel.render_options['DATA_STYPE'],
            dram_shape = f"memref<{kernel.render_options['Y_numel']}x{kernel.render_options['DATA_STYPE']}>",
            tile_size = (TILE_M, TILE_N),
            tile_stride = [1, TILE_M],
            nr_rdim = nr_rdim,
            reduction_idx = "t_n"
        )
        code = self._template_from_string(template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["M"], kernel.render_options["N"], kernel.render_options["K"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])

        self.header = f"float X_spad[{kernel.get_spad_size_per_lane(TILE_M, TILE_K)}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float W_spad[{kernel.get_spad_size_per_lane(TILE_K, TILE_N)}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{kernel.get_spad_size_per_lane(TILE_M, TILE_N)}] __attribute__ ((section(\".spad\")));\n"
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
