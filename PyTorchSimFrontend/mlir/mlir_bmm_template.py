import os
from torch import empty_strided
from typing import List, Optional
import sympy

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import IRNode
from PyTorchSimFrontend.mlir import mlir_common

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
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  {% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
  {% endif %}
  %c0 = arith.constant 0 : index
  {{ kernel.def_local_vars(indent_size=2) }}
  affine.for %index0 = 0 to {{ B }} {
    affine.for %index1 = 0 to {{ M }} step {{ TILE_M }} {
      affine.for %index2 = 0 to {{ N }} step {{ TILE_N }} {
        %X_buffer2D = memref.reinterpret_cast %X_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : {{ X_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>
        %W_buffer2D = memref.reinterpret_cast %W_buffer to offset: [0], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ W_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>
        %Y_buffer2D = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>
        {% if Bias -%}
        {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Y_tile_desc, subtile_size=[1, SUB_TILE_M, SUB_TILE_N], indent_size=8) }}
        {%- else -%}
        affine.vector_store %v0, %Y_buffer[0, 0, 0] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }}, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
        {% endif %}

        affine.for %index3 = 0 to {{ K }} step {{ TILE_K }} {
          {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, subtile_size=[1, SUB_TILE_M, SUB_TILE_K], indent_size=10) }}
          {{ kernel.def_dma_op("MVIN", "W", W_idx, W_tile_desc, subtile_size=[1, SUB_TILE_K, SUB_TILE_N], indent_size=10) }}
          linalg.matmul ins(%X_buffer2D, %W_buffer2D : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                  outs(%Y_buffer2D : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true, subtile_loop="k" }
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true, subtile_loop="n" }
    } { outer_loop=true, subtile_loop="m" }
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
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  {% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
  {% endif %}
  %c0 = arith.constant 0 : index
  {{ kernel.def_local_vars(indent_size=2) }}
  affine.for %index0 = 0 to {{ B }} {
    affine.for %index1 = 0 to {{ M }} step {{ TILE_M }} {
      affine.for %index2 = 0 to {{ N }} step {{ TILE_N }} {
        %X_buffer2D = memref.reinterpret_cast %X_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_K }}x{{DATA_STYPE}}, 1> to memref<{{ TILE_M }}x{{ TILE_K }}x{{DATA_STYPE}}, 1>
        %W_buffer2D = memref.reinterpret_cast %W_buffer to offset: [0], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_K }}x{{ TILE_N }}x{{DATA_STYPE}}, 1> to memref<{{ TILE_K }}x{{ TILE_N }}x{{DATA_STYPE}}, 1>
        %Y_buffer2D = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_N }}x{{DATA_STYPE}}, 1> to memref<{{ TILE_M }}x{{ TILE_N }}x{{DATA_STYPE}}, 1>
        {% if Bias -%}
        {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Y_tile_desc, subtile_size=[1, SUB_TILE_M, SUB_TILE_N], indent_size=8) }}
        {%- else -%}
        affine.vector_store %v0, %Y_buffer[0, 0, 0] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }}, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
        {% endif %}
        affine.for %index3 = 0 to {{ K }} step {{ TILE_K }} {
          {{kernel.load_input(indent_size=10)}}
          linalg.matmul ins(%X_buffer2D, %W_buffer2D : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                  outs(%Y_buffer2D : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true, subtile_loop="k" }
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true, subtile_loop="n" }
    } { outer_loop=true, subtile_loop="m" }
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
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  {% if not Bias %}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
  {% endif %}
  %c0 = arith.constant 0 : index
  {{ kernel.def_local_vars(indent_size=2) }}
  affine.for %index0=0 to {{ B }} {
    affine.for %index2 = 0 to {{ N }} step {{ TILE_N }} {
      affine.for %index1 = 0 to {{ M }} step {{ TILE_M }} {
        %X_buffer2D = memref.reinterpret_cast %X_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<1x{{ TILE_M }}x{{ TILE_K }}x{{DATA_STYPE}}, 1> to memref<{{ TILE_M }}x{{ TILE_K }}x{{DATA_STYPE}}, 1>
        %W_buffer2D = memref.reinterpret_cast %W_buffer to offset: [0], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_K }}x{{ TILE_N }}x{{DATA_STYPE}}, 1> to memref<{{ TILE_K }}x{{ TILE_N }}x{{DATA_STYPE}}, 1>
        %Y_buffer2D = memref.reinterpret_cast %Y_buffer to offset: [0], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<1x{{ TILE_N }}x{{ TILE_M }}x{{DATA_STYPE}}, 1> to memref<{{ TILE_M }}x{{ TILE_N }}x{{DATA_STYPE}}, 1>

        {% if Bias -%}
        {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Y_tile_desc, subtile_size=[1, SUB_TILE_M, SUB_TILE_N], indent_size=8) }} // Why not N,M? Currently, dma-fine-grained pass assume M->N order...
        {%- else -%}
        affine.vector_store %v0, %Y_buffer[0, 0, 0] : memref<1x{{ TILE_N }}x{{ TILE_M }}x{{DATA_STYPE}}, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }}x{{DATA_STYPE}}>
        {% endif %}
        affine.for %index3 = 0 to {{ K }} step {{ TILE_K }} {
          {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, subtile_size=[1, SUB_TILE_M, SUB_TILE_K], indent_size=10) }}
          {{ kernel.def_dma_op("MVIN", "W", W_idx, W_tile_desc, subtile_size=[1, SUB_TILE_K, SUB_TILE_N], indent_size=10) }}
          linalg.matmul ins(%X_buffer2D, %W_buffer2D : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                  outs(%Y_buffer2D : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        } { accumulation_loop=true, subtile_loop="k" }
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true, subtile_loop="m" }
      {{kernel.reduction_output(indent_size=6)}}
    } { outer_loop=true, subtile_loop="n" }
  } { outer_loop=true }
  return
}
"""

class MLIRBMMTemplate(MLIRTemplate):
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
        X, W, Y, Bias, W_tensor, X_tensor, B, M, N, K, n_extra_node, n_prologue_node = self.extract_info(template_buffer_node, epilogue_nodes, prologue_nodes)
        precision_bytes = mlir_common.get_dtype_nbytes(X.get_dtype())
        if tile_info is None:
            TILE_M, TILE_N, TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = self.select_tile(kernel, M, N, K, n_extra_node, 0, n_prologue_node, precision_bytes)[0]
        else:
            TILE_M, TILE_N, TILE_K, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = tile_info

        TOG_latency = M if TILE_M > M else TILE_M
        kernel.loop_size = [TOG_latency, TILE_N, TILE_K]

        # Select template code
        nr_reduction_nodes = [node for node in epilogue_nodes if node.is_reduction()] if epilogue_nodes is not None else []
        if nr_reduction_nodes:
            template = BMM_REDUCTION_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index2", "index2": "index1"}
            nr_rdim = 1
        elif prologue_nodes:
            template = BMM_PROLOGUE_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index1", "index2": "index2"}
            nr_rdim = 0
        else:
            template = BMM_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index1", "index2": "index2"}
            nr_rdim = 0

        # Prepare tile descriptors
        vlane_stride = 1
        vlane_split_axis = 2
        loop_dim = [sympy.Symbol("index0"), sympy.Symbol("index1"), sympy.Symbol("index2"), sympy.Symbol("index3")]
        X_tile_size = [1, TILE_M, TILE_K]
        X_tile_stride = [0, 1, TILE_M]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("X_buffer")
        X_tile_desc.offset = X.get_layout().offset
        X_stride = X_tensor.stride()
        X_idx = [loop_dim[0]*X_stride[0], loop_dim[1]*X_stride[1], loop_dim[3]*X_stride[2]] # To keep index arguemnt order, we used index_list

        W_tile_size = [1, TILE_K, TILE_N]
        W_tile_stride = [0, 1, TILE_K]
        W_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        W_tile_desc.set_tile_size_stride(W_tile_size, W_tile_stride)
        W_tile_desc.set_name("W_buffer")
        W_tile_desc.offset = W.get_layout().offset
        W_stride = W_tensor.stride()
        W_idx = [loop_dim[0]*W_stride[0], loop_dim[3]*W_stride[1], loop_dim[2]*W_stride[2]]

        vlane_split_axis = vlane_split_axis if nr_rdim==0 else 1
        Y_tile_size = [1, TILE_M, TILE_N] if nr_rdim == 0 else [1, TILE_N, TILE_M]
        Y_tile_stride=[0, 1, TILE_M] if nr_rdim == 0 else [0, TILE_M, 1]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("Y_buffer")
        Y_stride = Y.get_layout().stride
        if nr_rdim == 0:
          Y_idx = [loop_dim[0]*Y_stride[0], loop_dim[1]*Y_stride[1], loop_dim[2]*Y_stride[2]]
        else:
          Y_idx = [loop_dim[0]*Y_stride[0], loop_dim[2]*Y_stride[2], loop_dim[1]*Y_stride[1]]

        # Extract Bias info
        Bias_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Bias_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Bias_tile_desc.set_name("Y_buffer")
        if Bias is not None:
          Bias_stride = Bias.get_layout().stride
          Bias_tile_desc.offset = Bias.get_layout().offset
          if nr_rdim == 0:
            Bias_idx = [loop_dim[0]*Bias_stride[0], loop_dim[1]*Bias_stride[1], loop_dim[2]*Bias_stride[2]]
          else:
            Bias_idx = [loop_dim[0]*Bias_stride[0], loop_dim[2]*Bias_stride[2], loop_dim[1]*Bias_stride[1]]
        else:
          Bias_idx = None

        data_stype = mlir_common.DTYPE_TO_MLIR[X.get_dtype()]
        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            B=B, M=M, N=N, K=K,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            SUB_TILE_M=SUB_TILE_M,
            SUB_TILE_N=SUB_TILE_N,
            SUB_TILE_K=SUB_TILE_K,
            DATA_STYPE=data_stype,
            X = X, W = W,Y = Y, Bias = Bias,
            X_idx = X_idx,
            W_idx = W_idx,
            Bias_idx = Bias_idx,
            X_tile_desc = X_tile_desc,
            W_tile_desc = W_tile_desc,
            Y_tile_desc = Y_tile_desc,
            input_reorder = self.input_reorder
        )

        if prologue_nodes:
          prologue_output_name = list(prologue_nodes[0].read_writes.writes)[0].name
          if prologue_output_name == X.get_name():
            # Input fusion case
            prologue_var = "X"
            prologue_sram_var = "X_buffer"
            prologue_tile_desc = X_tile_desc
            prologue_dim_aliasing = {"index0":"index0", "index1":"index1", "index2":"index3"}
            is_input_fused = True
          else:
            # Weight fusion case
            prologue_var = "W"
            prologue_sram_var = "W_buffer"
            prologue_tile_desc = W_tile_desc
            prologue_dim_aliasing = {"index0":"index0", "index1":"index3", "index2":"index2"}
            is_input_fused = False
 
          kernel.prologue_info = dict (
              input_dram_var = "X",
              input_sram_var = "X_buffer",
              input_tile_desc = X_tile_desc,
              input_idx = X_idx,
              input_subtile_size = [1, TILE_M, TILE_K], # TODO. Curently, Subtiling is not supported for prologue template
              input_dim_aliasing = {"index0":"index0", "index1":"index1", "index2":"index3"},

              weight_dram_var = "W",
              weight_sram_var = "W_buffer",
              weight_tile_desc = W_tile_desc,
              weight_idx = W_idx,
              weight_subtile_size = [1, TILE_K, TILE_N], # TODO. Curently, Subtiling is not supported for prologue template
              weight_dim_aliasing = {"index0":"index0", "index1":"index3", "index2":"index2"},

              # Descriptor for fusion
              dram_var = prologue_var,
              sram_var = prologue_sram_var,
              dram_tile_desc = prologue_tile_desc,
              dim_aliasing = prologue_dim_aliasing,
              is_bmm = True,
              is_input_fused = is_input_fused
          )

        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            sram_var = "Y_buffer",
            dram_var = "Y",
            dram_idx = Y_idx,
            dram_tile_desc = Y_tile_desc,
            nr_rdim = nr_rdim,
            r_dim_size = M,
            dim_aliasing = epilogue_dim_aliasing
        )
        code = self._template_from_string(template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["M"], kernel.render_options["N"], kernel.render_options["K"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])
        return code

    def extract_info(self, template_buffer_node, epilogue_nodes, prologue_nodes):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node

        # Extract input arguments info
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
        dtype_infos = [("X", X.get_dtype()), ("W", W.get_dtype()), ("Y", Y.get_dtype())]
        if Bias is not None:
            dtype_infos.append(("Bias", Bias.get_dtype()))
        if len({dtype for _, dtype in dtype_infos}) != 1:
            dtype_desc = ", ".join(f"{name}={dtype}" for name, dtype in dtype_infos)
            raise NotImplementedError(f"Mixed dtype BMM is not implemented yet ({dtype_desc})")

        W_tensor =  empty_strided(W.layout.size, W.layout.stride)
        X_tensor =  empty_strided(X.layout.size, X.layout.stride)
        if len(W_tensor.size()) > 3 or len(W_tensor.size()) == 2:
          W_tensor = W_tensor.view([-1, W_tensor.shape[-2], W_tensor.shape[-1]])
        if len(X_tensor.size()) > 3 or len(X_tensor.size()) == 2:
          X_tensor = X_tensor.view([-1, X_tensor.shape[-2], X_tensor.shape[-1]])
        B, M, N, K = X_tensor.size()[0], X_tensor.size()[1], W_tensor.size()[2], X_tensor.size()[2]

        W_stride = W_tensor.stride()
        X_stride = X_tensor.stride()

        # Select tile size
        n_extra_node = len(epilogue_nodes) if epilogue_nodes is not None else 0
        n_prologue_node = len(prologue_nodes) if prologue_nodes is not None else 0
        return X,W,Y,Bias,W_tensor,X_tensor,B,M,N,K,n_extra_node, n_prologue_node

    def get_tile_candidates(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        X, W, Y, Bias, W_tensor, X_tensor, B, M, N, K, n_extra_node, n_prologue_node = self.extract_info(template_buffer_node, epilogue_nodes, prologue_nodes)
        precision_bytes = mlir_common.get_dtype_nbytes(X.get_dtype())
        return self.select_tile(kernel, M, N, K, n_extra_node, 0, n_prologue_node, precision_bytes)

    def select_tile(self, kernel, M, N, K, n_extra_node, n_extra_read, n_prologue_node, precision_bytes):
        tile_candidates = kernel.gemm_combination_mapping(M, N, K, n_extra_node=n_extra_node, precision_bytes=precision_bytes)
        for idx, (TILE_M, TILE_N, TILE_K) in enumerate(tile_candidates):
            SUB_TILE_M = TILE_M if (TILE_M < kernel.vector_lane) or n_prologue_node else kernel.vector_lane
            SUB_TILE_N = TILE_N # if (TILE_N < kernel.vector_lane) or prologue_nodes else kernel.vector_lane
            SUB_TILE_K = TILE_K # if (TILE_K < kernel.vector_lane) or prologue_nodes else kernel.vector_lane
            TILE_K = TILE_K // 2 if n_prologue_node else TILE_K
            tile_candidates[idx] = TILE_M,TILE_N,TILE_K,SUB_TILE_M,SUB_TILE_N,SUB_TILE_K
        return tile_candidates
