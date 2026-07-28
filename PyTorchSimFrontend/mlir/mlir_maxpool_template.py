from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from PyTorchSimFrontend.mlir import mlir_common
import sympy

# This template only represents the DMA operations
TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  {{- kernel.def_local_vars(indent_size=2) }}
  affine.for %index0 = 0 to {{ BCH }} step {{ out_tile }} {
    affine.for %index1 = 0 to {{ W }} step {{ out_tile }} {
      {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, indent_size=6) }}
      {{ kernel.def_dma_op("MVOUT", "Y", Y_idx, Y_tile_desc, indent_size=6) }}
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

class MLIRMaxPoolTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, kernel_size, stride, padding, dilation, ceil_mode, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])
        X = self.input_nodes[0]
        Y = self.output_node
        out_tile = kernel.vector_lane
        in_tile = self.stride[0] * (out_tile - 1) + self.dilation[0] * (self.kernel_size[0] - 1) + 1 # padding should be considered? - 2 * self.padding
        B = Y.get_size()[0]
        C = Y.get_size()[1]
        H = Y.get_size()[2]
        W = Y.get_size()[3]
        BCH = B * C * H
        kernel.loop_size = None

        # Prepare tile descriptors
        vlane_stride = 1 # Used dummy value
        vlane_split_axis = 1
        X_tile_size = [in_tile, in_tile]
        X_tile_stride = [1, in_tile]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("X_buffer")
        X_idx = [sympy.Symbol("index0"), sympy.Symbol("index1")*W] # To keep index arguemnt order, we used index_list

        Y_tile_size = [out_tile, out_tile]
        Y_tile_stride = [1, out_tile]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("W_buffer")
        Y_idx = [sympy.Symbol("index0"), sympy.Symbol("index1")*W]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            X=X,
            Y=Y,
            BCH=BCH,
            W=W,
            out_tile=out_tile,
            X_idx = X_idx,
            Y_idx = Y_idx,
            X_tile_desc = X_tile_desc,
            Y_tile_desc = Y_tile_desc,
            input_reorder = self.input_reorder
        )
        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            sram_var = "Y_buffer",
            dram_var = "Y",
            dram_tile_desc = Y_tile_desc,
        )
        kernel.exception_nodes["Y"] = {"numel" : Y.get_numel()}
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        kernel.add_loop_info([X.get_numel()], [kernel.vector_lane, kernel.vector_lane])
        return code
