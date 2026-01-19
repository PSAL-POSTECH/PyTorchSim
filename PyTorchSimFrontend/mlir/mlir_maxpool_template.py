from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from PyTorchSimFrontend.mlir import mlir_common
import sympy

# This template only represents the DMA operations
# TEMPLATE defines the MLIR code for the max-pooling operation.
# The MLIR dialect used here includes `func.func` for function definitions,
# `affine.for` for loop constructs, and `memref` for memory operations.
TEMPLATE = r"""
{{kernel.def_global_vars()}}  # Define global variables for the kernel.

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}  # Define SRAM buffer for input X.
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}  # Define SRAM buffer for output Y.
  {{- kernel.def_local_vars(indent_size=2) }}  # Define local variables for the kernel.
  affine.for %index0 = 0 to {{ BCH }} step {{ out_tile }} {  # Outer loop over batch-channel-height.
    affine.for %index1 = 0 to {{ W }} step {{ out_tile }} {  # Inner loop over width.
      {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, indent_size=6) }}  # DMA operation to move input data into SRAM.
      {{ kernel.def_dma_op("MVOUT", "Y", Y_idx, Y_tile_desc, indent_size=6) }}  # DMA operation to move output data from SRAM to DRAM.
    } { outer_loop=true }  # Mark the inner loop as an outer loop for parallelization.
  } { outer_loop=true }  # Mark the outer loop as an outer loop for parallelization.
  return  # Return from the function.
}
"""

class MLIRMaxPoolTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, kernel_size, stride, padding, dilation, ceil_mode, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.kernel_size = kernel_size  # Size of the pooling kernel.
        self.stride = stride  # Stride of the pooling operation.
        self.padding = padding  # Padding applied to the input.
        self.dilation = dilation  # Dilation factor for the pooling kernel.
        self.ceil_mode = ceil_mode  # Whether to use ceil or floor for output size calculation.

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node  # Set the output node if provided.
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])  # Use the last epilogue node as the output.
        X = self.input_nodes[0]  # Input tensor.
        Y = self.output_node  # Output tensor.
        out_tile = kernel.vector_lane  # Tile size for the output.
        in_tile = self.stride[0] * (out_tile - 1) + self.dilation[0] * (self.kernel_size[0] - 1) + 1  # Calculate input tile size.

        B = Y.get_size()[0]  # Batch size.
        C = Y.get_size()[1]  # Number of channels.
        H = Y.get_size()[2]  # Height of the output tensor.
        W = Y.get_size()[3]  # Width of the output tensor.
        BCH = B * C * H  # Combined batch, channel, and height size.
        kernel.loop_size = None  # No specific loop size set.

        # Prepare tile descriptors
        vlane_stride = 1  # Stride for vector lanes (dummy value).
        vlane_split_axis = 1  # Axis to split vector lanes.
        X_tile_size = [in_tile, in_tile]  # Tile size for input tensor.
        X_tile_stride = [1, in_tile]  # Stride for input tile.
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)  # Set tile size and stride.
        X_tile_desc.set_name("X_buffer")  # Name the tile descriptor for input.
        X_idx = [sympy.Symbol("index0"), sympy.Symbol("index1")*W]  # Indexing for input tensor.

        Y_tile_size = [out_tile, out_tile]  # Tile size for output tensor.
        Y_tile_stride = [1, out_tile]  # Stride for output tile.
        Y_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)  # Set tile size and stride.
        Y_tile_desc.set_name("W_buffer")  # Name the tile descriptor for output.
        Y_idx = [sympy.Symbol("index0"), sympy.Symbol("index1")*W]  # Indexing for output tensor.

        kernel.render_options = dict(
            KERNEL_NAME=self.name,  # Kernel name.
            kernel=kernel,  # Kernel object.
            X=X,  # Input tensor.
            Y=Y,  # Output tensor.
            BCH=BCH,  # Combined batch, channel, and height size.
            W=W,  # Width of the output tensor.
            out_tile=out_tile,  # Tile size for the output.
            X_idx = X_idx,  # Indexing for input tensor.
            Y_idx = Y_idx,  # Indexing for output tensor.
            X_tile_desc = X_tile_desc,  # Tile descriptor for input tensor.
            Y_tile_desc = Y_tile_desc,  # Tile descriptor for output tensor.
            input_reorder = self.input_reorder  # Input reorder option.
        )
        kernel.epilogue_info = dict(
            output_node = self.output_node.name,  # Name of the output node.
            sram_var = "Y_buffer",  # SRAM variable for output.
            dram_var = "Y",  # DRAM variable for output.
            dram_tile_desc = Y_tile_desc,  # Tile descriptor for output in DRAM.
        )
        kernel.exception_nodes["Y"] = {"numel" : Y.get_numel()}  # Exception handling for output tensor.
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)  # Render the MLIR code.
        kernel.add_loop_info([X.get_numel()], [kernel.vector_lane, kernel.vector_lane])  # Add loop information.
        return code  # Return the rendered MLIR code.
