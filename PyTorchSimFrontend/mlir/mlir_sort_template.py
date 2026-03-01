from typing import List, Optional

import sympy
from torch._inductor.ir import IRNode
from torch._inductor.virtualized import V

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel


TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X, YI], outputs=[YV], names_str=NAMES_STR, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("YI", YI_TILE_DESC, id=1, indent_size=2) }}
  {{ kernel.def_sram_buffer(OUT_DVAR, YV_TILE_DESC, id=2, indent_size=2) }}
  {{ kernel.def_local_vars(indent_size=2) }}

  %c0 = arith.constant 0 : index
  %c_cols = arith.constant {{ COLS }} : index

  affine.for %sort_block = 0 to 1 step 1 {
    // Initialize output value/index buffers.
    affine.for %row = 0 to {{ ROWS }} step 1 {
      affine.for %col = 0 to {{ COLS }} step 1 {
        {{ kernel.def_dma_op("MVIN", "X", INIT_X_IDX, X_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=8) }}
        {{ kernel.def_dma_op("MVOUT", OUT_DVAR, INIT_YV_IDX, X_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=8) }}
{% if DIM == 1 %}
        %idx_i64 = arith.index_cast %col : index to {{ YI_ELEM_TYPE }}
{% else %}
        %idx_i64 = arith.index_cast %row : index to {{ YI_ELEM_TYPE }}
{% endif %}
        memref.store %idx_i64, %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}
        {{ kernel.def_dma_op("MVOUT", "YI", INIT_YI_IDX, YI_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=8) }}
      }
    }

{% if DIM == 1 %}
    // Stable bubble sort on each row (dim=1).
    affine.for %row = 0 to {{ ROWS }} step 1 {
      affine.for %pass = 0 to {{ COLS }} step 1 {
        affine.for %j = 0 to {{ COLS_MINUS1 }} step 1 {
          {{ kernel.def_dma_op("MVIN", OUT_DVAR, D1_S0_IDX, YV_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=10) }}
          %lhs = memref.load %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}

          {{ kernel.def_dma_op("MVIN", OUT_DVAR, D1_S1_IDX, YV_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=10) }}
          %rhs = memref.load %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}

{% if DESCENDING %}
          %need_swap = arith.cmpf olt, %lhs, %rhs : {{ YV_ELEM_TYPE }}
{% else %}
          %need_swap = arith.cmpf ogt, %lhs, %rhs : {{ YV_ELEM_TYPE }}
{% endif %}
          scf.if %need_swap {
            memref.store %rhs, %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", OUT_DVAR, D1_S0_IDX, YV_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}

            memref.store %lhs, %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", OUT_DVAR, D1_S1_IDX, YV_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}

            {{ kernel.def_dma_op("MVIN", "YI", D1_S0_IDX, YI_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}
            %li = memref.load %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}

            {{ kernel.def_dma_op("MVIN", "YI", D1_S1_IDX, YI_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}
            %ri = memref.load %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}

            memref.store %ri, %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", "YI", D1_S0_IDX, YI_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}

            memref.store %li, %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", "YI", D1_S1_IDX, YI_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}
          }
        }
      }
    }
{% else %}
    // Stable bubble sort on each column (dim=0).
    affine.for %col = 0 to {{ COLS }} step 1 {
      affine.for %pass = 0 to {{ ROWS }} step 1 {
        affine.for %i = 0 to {{ ROWS_MINUS1 }} step 1 {
          {{ kernel.def_dma_op("MVIN", OUT_DVAR, D0_S0_IDX, YV_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=10) }}
          %lhs = memref.load %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}

          {{ kernel.def_dma_op("MVIN", OUT_DVAR, D0_S1_IDX, YV_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=10) }}
          %rhs = memref.load %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}

{% if DESCENDING %}
          %need_swap = arith.cmpf olt, %lhs, %rhs : {{ YV_ELEM_TYPE }}
{% else %}
          %need_swap = arith.cmpf ogt, %lhs, %rhs : {{ YV_ELEM_TYPE }}
{% endif %}
          scf.if %need_swap {
            memref.store %rhs, %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", OUT_DVAR, D0_S0_IDX, YV_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}

            memref.store %lhs, %yv_sort_tile[%c0, %c0] : {{ YV_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", OUT_DVAR, D0_S1_IDX, YV_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}

            {{ kernel.def_dma_op("MVIN", "YI", D0_S0_IDX, YI_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}
            %li = memref.load %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}

            {{ kernel.def_dma_op("MVIN", "YI", D0_S1_IDX, YI_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}
            %ri = memref.load %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}

            memref.store %ri, %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", "YI", D0_S0_IDX, YI_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}

            memref.store %li, %yi_sort_tile[%c0, %c0] : {{ YI_TILE_MEMREF_TYPE }}
            {{ kernel.def_dma_op("MVOUT", "YI", D0_S1_IDX, YI_S1_TILE_DESC, subtile_size=[1, 1], async_type=0, indent_size=12) }}
          }
        }
      }
    }
{% endif %}
  } { outer_loop=true }
  return
}
"""


class MLIRSortTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, dim, descending=False, stable=False, indices_node=None, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.dim = dim
        self.descending = descending
        self.stable = stable
        self.indices_node = indices_node

    def render(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        tile_info=None,
        **kwargs,
    ):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if self.indices_node is None:
            raise RuntimeError("MLIRSortTemplate requires indices output node")

        x = self.input_nodes[0]
        yv = self.output_node
        yi = self.indices_node

        def _as_int(v):
            try:
                return int(v)
            except Exception:
                return int(V.graph.sizevars.size_hint(v))

        x_size = x.get_size()
        if len(x_size) != 2:
            raise RuntimeError("MLIRSortTemplate currently supports rank-2 input only")
        if self.dim not in (0, 1):
            raise RuntimeError(f"MLIRSortTemplate currently supports dim in {{0,1}} only, got dim={self.dim}")

        rows = _as_int(x_size[0])
        cols = _as_int(x_size[1])
        cols_minus1 = max(0, cols - 1)
        rows_minus1 = max(0, rows - 1)

        x_dtype = x.get_dtype()
        yv_dtype = yv.get_dtype()
        yi_dtype = yi.get_dtype()
        if x_dtype != yv_dtype:
            raise RuntimeError("sort template requires input/value dtype match")

        yi_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        yi_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        yi_tile_desc.set_name("yi_sort_tile")
        yv_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        yv_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        yv_tile_desc.set_name("yv_sort_tile")
        # Neighbor element descriptors use DRAM offset to preserve affine stride metadata.
        yv_s1_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        yv_s1_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        yv_s1_tile_desc.set_name("yv_sort_tile")
        yi_s1_tile_desc = mlir_common.MLIRMultiDimTile([1, 1], kernel.vector_lane, vlane_split_axis=1, vlane_stride=1)
        yi_s1_tile_desc.set_tile_size_stride([1, 1], [1, 1])
        yi_s1_tile_desc.set_name("yi_sort_tile")
        if int(self.dim) == 1:
            yv_s1_tile_desc.offset = sympy.Integer(1)
            yi_s1_tile_desc.offset = sympy.Integer(1)
        else:
            yv_s1_tile_desc.offset = sympy.Integer(cols)
            yi_s1_tile_desc.offset = sympy.Integer(cols)

        row = sympy.Symbol("row")
        col = sympy.Symbol("col")
        i = sympy.Symbol("i")
        j = sympy.Symbol("j")

        init_x_idx = [row * cols, col]
        init_yv_idx = [row * cols, col]
        init_yi_idx = [row * cols, col]

        d1_s0_idx = [row * cols, j]
        d1_s1_idx = [row * cols, j]

        d0_s0_idx = [i * cols, col]
        d0_s1_idx = [i * cols, col]

        kernel.loop_size = None
        numel = rows * cols
        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            X=x,
            YV=yv,
            YI=yi,
            OUT_DVAR="YV",
            NAMES_STR="X, YI, YV",
            ROWS=rows,
            COLS=cols,
            COLS_MINUS1=cols_minus1,
            ROWS_MINUS1=rows_minus1,
            DIM=int(self.dim),
            DESCENDING=bool(self.descending),
            YI_TILE_DESC=yi_tile_desc,
            YV_TILE_DESC=yv_tile_desc,
            YI_S1_TILE_DESC=yi_s1_tile_desc,
            YV_S1_TILE_DESC=yv_s1_tile_desc,
            INIT_X_IDX=init_x_idx,
            INIT_YV_IDX=init_yv_idx,
            INIT_YI_IDX=init_yi_idx,
            D1_S0_IDX=d1_s0_idx,
            D1_S1_IDX=d1_s1_idx,
            D0_S0_IDX=d0_s0_idx,
            D0_S1_IDX=d0_s1_idx,
            YV_ELEM_TYPE=mlir_common.DTYPE_TO_MLIR[yv_dtype],
            YI_ELEM_TYPE=mlir_common.DTYPE_TO_MLIR[yi_dtype],
            X_MEMREF_TYPE=f"memref<{numel}x{mlir_common.DTYPE_TO_MLIR[x_dtype]}>",
            YV_MEMREF_TYPE=f"memref<{numel}x{mlir_common.DTYPE_TO_MLIR[yv_dtype]}>",
            YI_MEMREF_TYPE=f"memref<{numel}x{mlir_common.DTYPE_TO_MLIR[yi_dtype]}>",
            YV_TILE_MEMREF_TYPE=yv_tile_desc.get_mlir_shape(mlir_common.DTYPE_TO_MLIR[yv_dtype]),
            YI_TILE_MEMREF_TYPE=yi_tile_desc.get_mlir_shape(mlir_common.DTYPE_TO_MLIR[yi_dtype]),
            X_TILE_DESC=yv_tile_desc,
            input_reorder=self.input_reorder,
        )

        output_node_name = yv.get_name() if hasattr(yv, "get_name") else yv.name
        kernel.epilogue_info = dict(
            output_node=output_node_name,
            sram_var="yv_sort_tile",
            dram_var=kernel.render_options["OUT_DVAR"],
            dram_tile_desc=yv_tile_desc,
        )
        kernel.exception_nodes[kernel.render_options["OUT_DVAR"]] = {"numel": yv.get_numel()}
        kernel.exception_nodes["YI"] = {"numel": yi.get_numel()}

        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        return code
