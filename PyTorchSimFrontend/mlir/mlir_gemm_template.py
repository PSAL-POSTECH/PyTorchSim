import json
from pathlib import Path
from torch import empty_strided
from typing import List, Optional
import sympy

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import IRNode
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir import mlir_common

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
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  {% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>{% endif %}
  {{ kernel.def_local_vars(indent_size=2) }}
  affine.for %index0 = 0 to {{ M }} step {{ TILE_M }} {
    affine.for %index1 = 0 to {{ N }} step {{ TILE_N }} {
      {%- if Bias %}
      {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Bias_tile_desc, subtile_size=[SUB_TILE_M, SUB_TILE_N], indent_size=6) }}
      {%- else %}
      affine.vector_store %v0, %Y_buffer[0, 0] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }}, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
      {%- endif %}
      affine.for %index2 = 0 to {{ K }} step {{ TILE_K }} {
        {% if prologue_nodes -%}
        // prologue nodes
        {{kernel.load_input(indent_size=8)}}
        {%- else -%}
        {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, subtile_size=[SUB_TILE_M, SUB_TILE_K], indent_size=8) }}
        {{ kernel.def_dma_op("MVIN", "W", W_idx, W_tile_desc, subtile_size=[SUB_TILE_K, SUB_TILE_N], indent_size=8) }}
        {%- endif %}
        linalg.matmul ins(%X_buffer, %W_buffer : {{ X_tile_desc.get_mlir_shape(DATA_STYPE) }}, {{ W_tile_desc.get_mlir_shape(DATA_STYPE) }})
                outs(%Y_buffer : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }})
      } { accumulation_loop=true, subtile_loop="k" }
      {{kernel.store_output(indent_size=6)}}
    } { outer_loop=true, subtile_loop="n"  }
  } { outer_loop=true, subtile_loop="m" }
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
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  {% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
  {% endif %}
  {{ kernel.def_local_vars(indent_size=2) }}
  affine.for %index1 = 0 to {{ N }} step {{ TILE_N }} {
    affine.for %index0 = 0 to {{ M }} step {{ TILE_M }} {
      %Y_bufferT = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_N }}x{{DATA_STYPE}}, 1>
      {%- if Bias %}
      {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Bias_tile_desc, subtile_size=[SUB_TILE_M, SUB_TILE_N], indent_size=6) }}
      {%- else %}
      affine.vector_store %v0, %Y_buffer[0, 0] : memref<{{ TILE_N }}x{{ TILE_M }}x{{DATA_STYPE}}, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
      {%- endif %}
      affine.for %index2 = 0 to {{ K }} step {{ TILE_K }} {
        {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, subtile_size=[SUB_TILE_M, SUB_TILE_K], indent_size=8) }}
        {{ kernel.def_dma_op("MVIN", "W", W_idx, W_tile_desc, subtile_size=[SUB_TILE_K, SUB_TILE_N], indent_size=8) }}
        linalg.matmul ins(%X_buffer, %W_buffer : {{ X_tile_desc.get_mlir_shape(DATA_STYPE) }}, {{ W_tile_desc.get_mlir_shape(DATA_STYPE) }})
                outs(%Y_bufferT : memref<{{TILE_M}}x{{TILE_N}}x{{DATA_STYPE}}, 1>)
      } { accumulation_loop=true, subtile_loop="k" }
      {{kernel.store_output(indent_size=6)}}
    } { outer_loop=true, subtile_loop="m" }
    {{kernel.reduction_output(indent_size=4)}}
  } { outer_loop=true, subtile_loop="n" }
  return
}
"""

class MLIRGemmTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.support_epilogue_fusion = True
        self.support_prologue_fusion = True
        self.support_reduction_fusion = True

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
        X, W, Y, M, N, K, n_epilogue_node, n_prologue_node, n_extra_read = self.extract_info(template_buffer_node, epilogue_nodes, prologue_nodes)
        if tile_info is None:
            TILE_M, TILE_N, TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = self.select_tile(kernel, M, N, K, n_epilogue_node, n_extra_read, n_prologue_node)[0]
        else:
            TILE_M, TILE_N, TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = tile_info

        # Select template code
        if (M == 0) or (N == 0) or (K == 0): # exception for MoE
            template = EMPTY_TEMPLATE
            nr_rdim = 0
            epilogue_dim_aliasing = {}
        elif n_epilogue_node>=1 and epilogue_nodes[0].is_reduction():
            template = GEMM_REDUCTION_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index1", "index1":"index0"}
            nr_rdim = 1
        else:
            template = GEMM_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index1"}
            nr_rdim = 0

        TOG_latency = M if SUB_TILE_M > M else SUB_TILE_M
        kernel.loop_size =[TOG_latency, SUB_TILE_N, SUB_TILE_K]

        # Prepare tile descriptors
        vlane_stride = 1
        vlane_split_axis = 1
        X_tile_size = [TILE_M, TILE_K]
        X_tile_stride = [1, TILE_M]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("X_buffer")
        X_tile_desc.offset = X.get_layout().offset
        X_stride = X.get_layout().stride
        X_idx = [sympy.Symbol("index0") * X_stride[0], sympy.Symbol("index2") * X_stride[1]] # To keep index arguemnt order, we used index_list

        W_tile_size = [TILE_K, TILE_N]
        W_tile_stride = [1, TILE_K]
        W_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        W_tile_desc.set_tile_size_stride(W_tile_size, W_tile_stride)
        W_tile_desc.set_name("W_buffer")
        W_tile_desc.offset = W.get_layout().offset
        W_stride = W.get_layout().stride if N>1 else [Y.get_layout().stride[0], 0]
        W_idx = [sympy.Symbol("index2") * W_stride[0], sympy.Symbol("index1") * W_stride[1]]

        vlane_split_axis = vlane_split_axis if nr_rdim==0 else 0
        Y_tile_size = [TILE_M, TILE_N] if nr_rdim == 0 else [TILE_N, TILE_M]
        Y_tile_stride=[1, TILE_M] if nr_rdim == 0 else [TILE_M, 1]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("Y_buffer")
        Y_stride = Y.get_layout().stride if N>1 else [Y.get_layout().stride[0], 0]
        if nr_rdim == 0:
            Y_idx = [sympy.Symbol("index0") * Y_stride[0], sympy.Symbol("index1") * Y_stride[1]]
        else:
            Y_idx = [sympy.Symbol("index1") * Y_stride[1], sympy.Symbol("index0") * Y_stride[0]]

        # Extract Bias info
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
        Bias_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Bias_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Bias_tile_desc.set_name("Y_buffer")
        if Bias is not None:
          Bias_stride = Bias.get_layout().stride
          Bias_tile_desc.offset = Bias.get_layout().offset
          if nr_rdim == 0:
            Bias_idx = [sympy.Symbol("index0") * Bias_stride[0], sympy.Symbol("index1") * Bias_stride[1]]
          else:
            Bias_idx = [sympy.Symbol("index1") * Bias_stride[1], sympy.Symbol("index0") * Bias_stride[0]]
        else:
          Bias_idx = None

        data_stype = mlir_common.DTYPE_TO_MLIR[X.get_dtype()]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            M=M, N=N, K=K,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            SUB_TILE_M=SUB_TILE_M,
            SUB_TILE_N=SUB_TILE_N,
            SUB_TILE_K=SUB_TILE_K,
            DATA_STYPE=data_stype,
            X = X, W = W, Y = Y,
            Bias = Bias,
            X_idx = X_idx,
            W_idx = W_idx,
            Bias_idx = Bias_idx,
            X_tile_desc = X_tile_desc,
            W_tile_desc = W_tile_desc,
            Y_tile_desc = Y_tile_desc,
            Bias_tile_desc = Bias_tile_desc,
            epilogue_nodes = epilogue_nodes,
            prologue_nodes = prologue_nodes,
            input_reorder = self.input_reorder
        )
        if prologue_nodes:
            prologue_output_name = list(prologue_nodes[0].read_writes.writes)[0].name
            if prologue_output_name == X.get_name():
                # Input fusion case
                prologue_var = "X"
                prologue_sram_var = "X_buffer"
                prologue_tile_desc = X_tile_desc
                prologue_dim_aliasing = {"index0":"index0", "index1":"index2"}
                is_input_fused = True
            else:
                # Weight fusion case
                prologue_var = "W"
                prologue_sram_var = "W_buffer"
                prologue_tile_desc = W_tile_desc
                prologue_dim_aliasing = {"index0":"index2", "index1":"index1"}
                is_input_fused = False

            kernel.prologue_info = dict (
                input_dram_var = "X",
                input_sram_var = "X_buffer",
                input_tile_desc = X_tile_desc,
                input_idx = X_idx,
                input_subtile_size = [TILE_M, TILE_K],
                input_dim_aliasing = {"index0":"index0", "index1":"index2"},

                weight_dram_var = "W",
                weight_sram_var = "W_buffer",
                weight_tile_desc = W_tile_desc,
                weight_idx = W_idx,
                weight_subtile_size = [TILE_K, TILE_N],
                weight_dim_aliasing = {"index0":"index2", "index1":"index1"},

                # Descriptor for fusion
                dram_var = prologue_var,
                sram_var = prologue_sram_var,
                dram_tile_desc = prologue_tile_desc,
                dim_aliasing = prologue_dim_aliasing,
                is_bmm = False,
                is_input_fused = is_input_fused
            )
        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            dram_var = "Y",
            sram_var = "Y_buffer",
            dram_idx = Y_idx,
            dram_tile_desc = Y_tile_desc,
            nr_rdim = nr_rdim,
            r_dim_size = M,
            dim_aliasing = epilogue_dim_aliasing
        )
        code = self._template_from_string(template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["M"], kernel.render_options["N"], kernel.render_options["K"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])
        return code

    def get_tile_candidates(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        X, W, Y, M, N, K, n_epilogue_node, n_prologue_node, n_extra_read = self.extract_info(template_buffer_node, epilogue_nodes, prologue_nodes)
        return self.select_tile(kernel, M, N, K, n_epilogue_node, n_extra_read, n_prologue_node)

    def extract_info(self, template_buffer_node, epilogue_nodes, prologue_nodes):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node

        # Extract input arguments info
        X, W, Y = self.input_nodes[0], self.input_nodes[1], self.output_node
        dtype_infos = [("X", X.get_dtype()), ("W", W.get_dtype()), ("Y", Y.get_dtype())]
        if len(self.input_nodes) > 2:
            dtype_infos.append(("Bias", self.input_nodes[2].get_dtype()))
        if len({dtype for _, dtype in dtype_infos}) != 1:
            dtype_desc = ", ".join(f"{name}={dtype}" for name, dtype in dtype_infos)
            raise NotImplementedError(f"Mixed dtype GEMM is not implemented yet ({dtype_desc})")
        X_tensor = empty_strided(X.layout.size, X.layout.stride)
        W_tensor = empty_strided(W.layout.size, W.layout.stride)
        if len(W_tensor.size()) > 2 or len(X_tensor.size()) > 2:
            raise NotImplementedError("Please report this case to us...")

        # Extract fusion info
        n_epilogue_node = len(epilogue_nodes) if epilogue_nodes is not None else 0
        n_prologue_node = len(prologue_nodes) if prologue_nodes is not None else 0
        n_extra_read = set()
        if epilogue_nodes is not None:
            for enode in epilogue_nodes:
                n_extra_read.update(enode.node.get_read_names())
            if self.output_node.name in n_extra_read:
                n_extra_read.remove(self.output_node.name)

        # Select tile size
        M, N, K = X_tensor.size()[0], W_tensor.size()[1], X_tensor.size()[1]
        return X,W,Y,M,N,K,n_epilogue_node,n_prologue_node,len(n_extra_read)

    def select_tile(self, kernel, M, N, K, n_extra_node, n_extra_read, n_prologue_node):
        data = {}
        gemm_shape = f"{M}_{N}_{K}"
        if "external" in extension_config.codegen_mapping_strategy:
            # case 1: use manual tile size
            path = Path(extension_config.codegen_external_mapping_file)
            with path.open("r") as f:
                data = json.load(f)
        if gemm_shape in data:
            tile_info = data[gemm_shape]
            if len(tile_info) == 3:
                TILE_M, TILE_N, TILE_K = tile_info.values()
                tile_candidates = [[TILE_M, TILE_N, TILE_K]]
            elif len(tile_info) == 6:
                TILE_M, TILE_N, TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = tile_info.values()
                full_tile_candidates = [[TILE_M, TILE_N, TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K]]
                return full_tile_candidates
        else:
            # case 2: use heuristic mapping
            min_tile = (n_extra_node + n_prologue_node) == 0
            tile_candidates = kernel.gemm_combination_mapping(M, N, K, max(n_extra_read-2, 0), n_prologue_node, min_tile=True)

        # Edge case
        if (M == 0) or (N == 0) or (K == 0):
            TILE_M, TILE_N, TILE_K = 1, 1, 1
            tile_candidates = [[TILE_M, TILE_N, TILE_K]]

        full_tile_candidates = []
        for idx, (TILE_M, TILE_N, TILE_K) in enumerate(tile_candidates):
            # Case 1: calculate sub tile size for fine-grained DMA
            if extension_config.CONFIG_SUBTILE:
                SUB_TILE_M = TILE_M if (TILE_M < kernel.vector_lane or n_prologue_node) else kernel.vector_lane
                if (TILE_M == M and TILE_N == N and TILE_N <= 512):
                    SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
                else: # Avoid Row Conflict of weights
                    SUB_TILE_N = TILE_N
                SUB_TILE_K = TILE_K
            # Case 2: None Subtile
            else:
                SUB_TILE_M = TILE_M
                SUB_TILE_N = TILE_N
                SUB_TILE_K = TILE_K
            full_tile_candidates.append([TILE_M,TILE_N,TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K])
        return full_tile_candidates
