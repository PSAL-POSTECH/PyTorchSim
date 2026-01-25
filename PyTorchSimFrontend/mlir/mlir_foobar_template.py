from typing import List, Optional
import sympy
from torch import empty_strided
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel
from torch._inductor.ir import IRNode, Buffer
from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend import extension_config
from pathlib import Path
import json


TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}  
  %M_const = arith.constant {{ M }} : index
  %N_const = arith.constant {{ N }} : index
  affine.for %index0 = 0 to {{ M }} step {{ TILE_M }} {
    affine.for %index1 = 0 to {{ N }} step {{ TILE_N }} {
      {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, indent_size=6) }}
      linialg.copy {{ X_tile_desc.get_mlir_shape(DATA_STYPE) }} to {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }} 
      {{ kernel.def_dma_op("MVOUT", "Y", Y_idx, Y_tile_desc, indent_size=6) }
    } {outer_loop=true }
  } { outer_loop=true }
  return
}
"""

class MLIRFoobarTemplate(MLIRTemplate):

    def __init__(self, input_nodes, layout, input_reorder=None):
        # Initialize the MLIR template with the kernel name and input/output nodes.
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):

        if template_buffer_node is not None:
            self.output_node = template_buffer_node  
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = epilogue_nodes[-1]  

        X = self.input_nodes[0] 
        Y = self.output_node 

        X_tensor = empty_strided(X.layout.size, X.layout.stride) 

        M = X_tensor.size()[0]
        N = X_tensor.size()[1]
        # path = Path(extension_config.codegen_external_mapping_file)
        # with path.open("r") as f:
        #     data = json.load(f)
        # tile_info = data[f"{M}x{N}"]
        # TILE_M, TILE_N = tile_info.values()
        TILE_M = 64
        TILE_N = 64

        TILE = kernel.vector_lane 

        vlane_stride = 1
        vlane_split_axis = 0
        X_tile_size = [TILE_M,TILE_N]
        X_tile_stride = [1, TILE_M]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("X_buffer")
        X_tile_desc.offset = X.get_layout().offset
        X_stride = X.get_layout().stride
        X_idx = [sympy.Symbol("index0") * X_stride[0], sympy.Symbol("index1") * X_stride[1]]

        Y_tile_size = [TILE_M,TILE_N]
        Y_tile_stride = [1, TILE_M]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("Y_buffer")
        Y_stride = Y.get_layout().stride
        Y_idx = [sympy.Symbol("index0") * Y_stride[0], sympy.Symbol("index1") * Y_stride[1]]

        # X_flat_mlir_shape = f"memref<{M}x{{DATA_STYPE}}>".replace('{DATA_STYPE}', 'f32')
        # Y_flat_mlir_shape = f"memref<{M}x{{DATA_STYPE}}>".replace('{DATA_STYPE}', 'f32')

        kernel.render_options = dict(
            KERNEL_NAME=self.name,  
            kernel=kernel,  
            M=M, N=N,
            TILE=TILE,  
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            X=X, 
            Y=Y,  
            X_idx=X_idx,  
            Y_idx=Y_idx,  
            X_tile_desc=X_tile_desc, 
            Y_tile_desc=Y_tile_desc,  
            #X_flat_mlir_shape=X_flat_mlir_shape,  
            #Y_flat_mlir_shape=Y_flat_mlir_shape,  
            DATA_STYPE="f32", 
            input_reorder=self.input_reorder,  
        )

        kernel.epilogue_info = dict(
            output_node=self.output_node.name, 
            sram_var="Y_buffer",  
            dram_var="Y",  
            dram_tile_desc=Y_tile_desc, 
        )

        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["M"], kernel.render_options["N"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"]])
        return code 