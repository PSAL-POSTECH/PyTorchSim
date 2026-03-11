from typing import List, Optional

import torch
from torch._inductor import ir
from torch._inductor.ir import Buffer, IRNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.codegen import common

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel


TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[YV], names_str=NAMES_STR, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X",  X_TILE_DESC,  id=0, indent_size=2) }}
  {{ kernel.def_sram_buffer("YV", YV_TILE_DESC, id=1, indent_size=2) }}
  {{ kernel.def_local_vars(indent_size=2) }}
  {{ kernel.def_dma_op("MVIN", "X", [], X_TILE_DESC, indent_size=2, dram_stride=[1, 1]) }}

  // Bitonic setup:
  //   even chunk  -> {{ EVEN_DIR_NAME }}
  //   odd  chunk  -> {{ ODD_DIR_NAME }}
  affine.for %chunk = 0 to {{ SORT_SIZE }} step {{ BITONIC_SETUP_STEP }} {
    {{- LOCAL_SORT_BODY_EVEN }}
  } { inner_loop=true }
  affine.for %chunk = {{ VECTOR_SIZE }} to {{ SORT_SIZE }} step {{ BITONIC_SETUP_STEP }} {
    {{- LOCAL_SORT_BODY_ODD }}
  } { inner_loop=true }

  {{ kernel.def_dma_op("MVOUT", "YV", [], YV_TILE_DESC, indent_size=2, dram_stride=[1, 1]) }}
  return
}
"""


class MLIRSortTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, dim, descending=False, stable=False, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.dim = dim
        self.descending = descending
        self.stable = stable
        self.output_nodes = [
            Buffer(name="buf_out_values", layout=layout),
        ]
        self.output_node = self.output_nodes[0]

    def render(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        tile_info=None,
        **kwargs,
    ):
        if template_buffer_node is not None:
            self.output_nodes[0] = template_buffer_node
            self.output_node = template_buffer_node

        x = self.input_nodes[0]
        yv = self.output_nodes[0]
        sort_size = int(x.get_size()[self.dim])
        max_width = min(kernel.vector_lane, sort_size) if sort_size > 0 else 1
        vector_size = 16

        vlane_stride = 1
        vlane_split_axis = 0
        x_tile_desc = mlir_common.MLIRMultiDimTile([1, sort_size], kernel.vector_lane, vlane_split_axis, vlane_stride)
        x_tile_desc.set_tile_size_stride([1, sort_size], [1, 1])
        x_tile_desc.set_name("X_buffer")
        x_tile_desc.offset = x.get_layout().offset

        yv_tile_desc = mlir_common.MLIRMultiDimTile([1, sort_size], kernel.vector_lane, vlane_split_axis, vlane_stride)
        yv_tile_desc.set_tile_size_stride([1, sort_size], [1, 1])
        yv_tile_desc.set_name("YV_buffer")
        yv_tile_desc.offset = yv.get_layout().offset

        data_stype = mlir_common.DTYPE_TO_MLIR[x.get_dtype()]

        # Generate local sort body in the same style as mlir_ops.bitonic_sort().
        # For bitonic setup, even/odd chunks are sorted in opposite directions.
        def _emit_local_sort_body(descending: bool, indent_size: int = 2):
            local_sort_code = IndentedBuffer(indent_size)
            temp_cse = common.CSE(kernel.newvar_prefix, kernel.suffix, name_prefix="sort")
            with kernel, kernel.override_buffer_cse(buffer=local_sort_code, cse=temp_cse):
                x_chunk = ops._load(
                    vector_size,
                    data_stype,
                    "X_buffer",
                    "%t_const0, %chunk",
                    x_tile_desc.get_mlir_shape(data_stype),
                )
                yv_chunk = ops.bitonic_sort(x_chunk, descending=descending)
                ops._store(
                    yv_chunk,
                    "YV_buffer",
                    "%t_const0, %chunk",
                    yv_tile_desc.get_mlir_shape(data_stype),
                )
            return local_sort_code.getvalue().rstrip()

        even_descending = self.descending
        odd_descending = not self.descending
        local_sort_body_even = _emit_local_sort_body(even_descending, indent_size=1)
        local_sort_body_odd = _emit_local_sort_body(odd_descending, indent_size=1)

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            NAMES_STR="X, YV",
            kernel=kernel,
            X=x,
            YV=yv,
            X_TILE_DESC=x_tile_desc,
            YV_TILE_DESC=yv_tile_desc,
            SORT_SIZE=sort_size,
            VECTOR_SIZE=vector_size,
            BITONIC_SETUP_STEP=vector_size * 2,
            DATA_STYPE=data_stype,
            IDX_STYPE="i64",
            EVEN_DIR_NAME="DESC" if even_descending else "ASC",
            ODD_DIR_NAME="DESC" if odd_descending else "ASC",
            LOCAL_SORT_BODY_EVEN=local_sort_body_even,
            LOCAL_SORT_BODY_ODD=local_sort_body_odd,
            input_reorder=self.input_reorder,
        )
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        kernel.exception_nodes["YI"] = {"numel" : sort_size}
        return code
