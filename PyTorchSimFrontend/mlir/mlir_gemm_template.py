from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView

GEMM_TEMPLATE = r"""
func.func @{{ KERNEL_NAME }}({{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}}) {
  %M = arith.constant {{ M }} : index
  %N = arith.constant {{ N }} : index
  %K = arith.constant {{ K }} : index

  affine.for %t_m = 0 to %M step {{ TILE_M }} {
    affine.for %t_n = 0 to %N step {{ TILE_N }} {
      affine.for %t_k = 0 to %K step {{ TILE_K }} {
        %A_tile = memref.subview %A[%t_m, %t_k] [{{ TILE_M }}, {{ TILE_K }}] [1, 1] : memref<?x?x{{ DATA_STYPE }}> to memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, strided<[?, 1], offset: ?>>
        %B_tile = memref.subview %B[%t_k, %t_n] [{{ TILE_K }}, {{ TILE_N }}] [1, 1] : memref<?x?x{{ DATA_STYPE }}> to memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, strided<[?, 1], offset: ?>>
        %C_tile = memref.subview %C[%t_m, %t_n] [{{ TILE_M }}, {{ TILE_N }}] [1, 1] : memref<?x?x{{ DATA_STYPE }}> to memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, strided<[?, 1], offset: ?>>

        linalg.matmul ins(%A_tile, %B_tile : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, strided<[?, 1], offset: ?>>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, strided<[?, 1], offset: ?>>)
                outs(%C_tile : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, strided<[?, 1], offset: ?>>)
      }
    }
  }
  return
}
"""

class MLIRGemmTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def is_transposed(self, node):
        if isinstance(node, ReinterpretView):
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
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        TILE_M = min(16, X.get_size()[0]) # TODO:: This should be determined by the size of the SRAM
        TILE_N = min(16, W.get_size()[1]) # FIXME: 16 is hard-coded
        TILE_K = min(16, X.get_size()[1])

        W_transposed = self.is_transposed(W)
        X_transposed = self.is_transposed(X)

        options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            M=X.get_size()[0],
            N=W.get_size()[1],
            K=X.get_size()[1],
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            DATA_STYPE="f32",
            DATA_SIZE=4,
            X = X,
            W = W,
            Y = Y,
            Bias = Bias,
            W_transposed = W_transposed,
            X_transposed = X_transposed,
            input_reorder = self.input_reorder
        )
        code = self._template_from_string(GEMM_TEMPLATE).render(**options)
        kernel.add_loop_info([options["M"], options["N"], options["K"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
        return code