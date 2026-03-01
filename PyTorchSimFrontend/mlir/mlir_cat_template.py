from typing import List, Optional, cast

import sympy
from torch._inductor.ir import Buffer, IRNode
from torch._inductor.virtualized import V

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel


TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X0, X1], outputs=[Y], names_str=NAMES_STR, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X0", X0_TILE_DESC, id=0, indent_size=2) }}
  {{ kernel.def_sram_buffer("X1", X1_TILE_DESC, id=1, indent_size=2) }}
  {{ kernel.def_sram_buffer(OUT_DVAR, Y_TILE_DESC, id=2, indent_size=2) }}
  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %cat_block = 0 to 1 step 1 {
{% if DIM == 0 %}
    affine.for %index0 = 0 to {{ X0_ROWS }} step 1 {
      affine.for %index1 = 0 to {{ COLS }} step 1 {
        {{ kernel.def_dma_op("MVIN", "X0", X0_IDX, X0_TILE_DESC, indent_size=8) }}
        {{ kernel.def_dma_op("MVOUT", OUT_DVAR, Y0_IDX, X0_TILE_DESC, indent_size=8) }}
      }
    }

    affine.for %index2 = 0 to {{ X1_ROWS }} step 1 {
      affine.for %index3 = 0 to {{ COLS }} step 1 {
        {{ kernel.def_dma_op("MVIN", "X1", X1_IDX, X1_TILE_DESC, indent_size=8) }}
        {{ kernel.def_dma_op("MVOUT", OUT_DVAR, Y1_IDX, X1_TILE_DESC, indent_size=8) }}
      }
    }
{% else %}
    affine.for %index0 = 0 to {{ ROWS }} step 1 {
      affine.for %index1 = 0 to {{ X0_COLS }} step 1 {
        {{ kernel.def_dma_op("MVIN", "X0", X0_IDX, X0_TILE_DESC, indent_size=8) }}
        {{ kernel.def_dma_op("MVOUT", OUT_DVAR, Y0_IDX, X0_TILE_DESC, indent_size=8) }}
      }
      affine.for %index3 = 0 to {{ X1_COLS }} step 1 {
        {{ kernel.def_dma_op("MVIN", "X1", X1_IDX, X1_TILE_DESC, indent_size=8) }}
        {{ kernel.def_dma_op("MVOUT", OUT_DVAR, Y1_IDX, X1_TILE_DESC, indent_size=8) }}
      }
    }
{% endif %}
  } { outer_loop=true }
  return
}
"""


class MLIRCatTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, dim, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.dim = dim

    def render(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        tile_info=None,
        **kwargs,
    ):
        is_out_variant = template_buffer_node is not None
        if is_out_variant:
            self.output_node = template_buffer_node
        # cat template currently emits a single output buffer and does not
        # support epilogue output remapping.

        def _unwrap_node(n):
            return n.node if hasattr(n, "node") else n

        x0 = _unwrap_node(self.input_nodes[0])
        x1 = _unwrap_node(self.input_nodes[1])
        y = _unwrap_node(self.output_node)

        def _as_int(v):
            try:
                return int(v)
            except Exception:
                return int(V.graph.sizevars.size_hint(v))

        x0_rows = _as_int(x0.get_size()[0])
        x1_rows = _as_int(x1.get_size()[0])
        x0_cols = _as_int(x0.get_size()[1])
        x1_cols = _as_int(x1.get_size()[1])
        y_cols = _as_int(y.get_size()[1])
        kernel.loop_size = None

        # 2D cat template with contiguous layout.
        x0_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        x0_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        x0_tile_desc.set_name("x0_cat_tile")
        x1_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        x1_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        x1_tile_desc.set_name("x1_cat_tile")
        y_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        y_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        y_tile_desc.set_name("y_cat_tile")

        if self.dim == 0:
            # Flattened offsets for dim=0 cat.
            x0_idx = [sympy.Symbol("index0") * x0_cols, sympy.Symbol("index1")]
            x1_idx = [sympy.Symbol("index2") * x1_cols, sympy.Symbol("index3")]
            y0_idx = [sympy.Symbol("index0") * y_cols, sympy.Symbol("index1")]
            y1_idx = [(sympy.Symbol("index2") + x0_rows) * y_cols, sympy.Symbol("index3")]
        else:
            # Flattened offsets for dim=1 cat.
            x0_idx = [sympy.Symbol("index0") * x0_cols, sympy.Symbol("index1")]
            x1_idx = [sympy.Symbol("index0") * x1_cols, sympy.Symbol("index3")]
            y0_idx = [sympy.Symbol("index0") * y_cols, sympy.Symbol("index1")]
            y1_idx = [sympy.Symbol("index0") * y_cols, sympy.Symbol("index3") + x0_cols]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            X0=x0,
            X1=x1,
            Y=y,
            OUT_DVAR="out_ptr1" if is_out_variant else "Y",
            NAMES_STR="X0, X1, out_ptr1" if is_out_variant else "X0, X1, Y",
            DIM=self.dim,
            X0_ROWS=x0_rows,
            X1_ROWS=x1_rows,
            ROWS=x0_rows,
            X0_COLS=x0_cols,
            X1_COLS=x1_cols,
            COLS=x0_cols,
            X0_TILE_DESC=x0_tile_desc,
            X1_TILE_DESC=x1_tile_desc,
            Y_TILE_DESC=y_tile_desc,
            X0_IDX=x0_idx,
            X1_IDX=x1_idx,
            Y0_IDX=y0_idx,
            Y1_IDX=y1_idx,
            input_reorder=self.input_reorder,
        )
        # Needed when epilogue fusion requests set_ranges().
        kernel.dim_aliasing = {"index0": "index0", "index1": "index1"}

        if hasattr(self.output_node, "node") and hasattr(self.output_node.node, "get_name"):
            output_node_name = self.output_node.node.get_name()
        elif hasattr(self.output_node, "get_name"):
            output_node_name = self.output_node.get_name()
        else:
            output_node_name = self.output_node.name

        if hasattr(y, "get_numel"):
            y_numel = y.get_numel()
        elif hasattr(y, "node") and hasattr(y.node, "get_numel"):
            y_numel = y.node.get_numel()
        else:
            y_numel = None

        kernel.epilogue_info = dict(
            output_node=output_node_name,
            sram_var="y_cat_tile",
            dram_var=kernel.render_options["OUT_DVAR"],
            dram_tile_desc=y_tile_desc,
        )
        if y_numel is not None:
            kernel.exception_nodes[kernel.render_options["OUT_DVAR"]] = {"numel": y_numel}

        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        return code
