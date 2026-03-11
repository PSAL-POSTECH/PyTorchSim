from typing import List, Optional
import contextlib

import torch
from torch._inductor import ir
from torch._inductor.ir import Buffer, IRNode
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.codegen import common

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel
from PyTorchSimFrontend.mlir.mlir_common import LoopLevel


TEMPLATE = r"""
{{kernel.def_global_vars()}}
// chunk index -> element index
#map_chunk_to_elem = affine_map<(d0) -> (d0 * {{ VECTOR_SIZE }})>

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[YV], names_str=NAMES_STR, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X",  X_TILE_DESC,  id=0, indent_size=2) }}
  {{ kernel.def_sram_buffer("YV", YV_TILE_DESC, id=1, indent_size=2) }}
  {{ kernel.def_local_vars(indent_size=2) }}


  affine.for %cat_block = 0 to 1 step 1 {
  {%- for d in range(RANK-1) %}
    affine.for %index{{ OUTPUT_DIM[d] }} = 0 to {{ OUTPUT_SIZES[d] }} step {{ STEP_SIZES[d] }} {
  {%- endfor %}

    %x_dram_offset = affine.apply {{ X_OFFSET_MAP }}({{ OUTER_VARS }})
    %yv_dram_offset = affine.apply {{ YV_OFFSET_MAP }}({{ OUTER_VARS }})
    {{ kernel.def_dma_op("MVIN", "X", [], X_TILE_DESC, indent_size=INDENT_SIZE, dram_stride=X_DRAM_STRIDE, dram_offset="x_dram_offset") }}

    // SIMD local sort + loop-based chunk merge.
{{ BITONIC_BODY }}

    {{ kernel.def_dma_op("MVOUT", "YV", [], YV_TILE_DESC, indent_size=INDENT_SIZE, dram_stride=YV_DRAM_STRIDE, dram_offset="yv_dram_offset") }}
  {%- for d in range(RANK-1) %}
    } { outer_loop=true }
  {%- endfor %}
  } { outer_loop=true }
  return
}
"""


def _make_offset_map(outer_dims, all_strides, layout_offset):
    """Build an affine_map over outer-dim loop variables that computes the flat DRAM offset."""
    terms = []
    for j, d in enumerate(outer_dims):
        s = int(all_strides[d])
        if s == 1:
            terms.append(f"d{j}")
        elif s != 0:
            terms.append(f"d{j} * {s}")
    try:
        off = int(layout_offset)
    except (TypeError, ValueError):
        off = 0
    if off:
        terms.append(str(off))
    nd = len(outer_dims)
    dim_str = ", ".join(f"d{j}" for j in range(nd))
    expr = " + ".join(terms) if terms else "0"
    return f"affine_map<({dim_str}) -> ({expr})>"


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
        vector_size = 16
        if sort_size <= 0:
            raise NotImplementedError("Sort size must be > 0")
        if sort_size < vector_size or sort_size % vector_size != 0:
            raise NotImplementedError(
                f"Sort size must be a multiple of vector size (sort_size={sort_size}, vector_size={vector_size})"
            )
        num_chunks = sort_size // vector_size
        if num_chunks & (num_chunks - 1):
            raise NotImplementedError(
                f"Loop-based bitonic chunk merge requires power-of-two chunk count (num_chunks={num_chunks})"
            )

        # --- N-D generalization: outer loops over all non-sort dims ---
        rank = len(x.get_size())
        x_layout = x.get_layout()
        yv_layout = yv.get_layout()

        output_sizes = [sz for d, sz in enumerate(yv.get_size()) if d != self.dim]
        output_dim   = [d  for d, _ in enumerate(yv.get_size()) if d != self.dim]
        step_sizes = [1] * len(output_sizes)

        tile_dim = max(output_dim, key=lambda d: int(yv.get_size()[d]))
        tile_sizes = [min(128, int(yv.get_size()[tile_dim])), sort_size]
        step_sizes[tile_dim] = tile_sizes[0]

        x_dram_stride = [int(x_layout.stride[tile_dim]), int(x_layout.stride[self.dim])]
        yv_dram_stride = [int(yv_layout.stride[tile_dim]), int(yv_layout.stride[self.dim])]

        x_offset_map  = _make_offset_map(output_dim, x_layout.stride,  x_layout.offset)
        yv_offset_map = _make_offset_map(output_dim, yv_layout.stride, yv_layout.offset)
        outer_vars = ", ".join(f"%index{d}" for d in output_dim)

        # indent for DMA ops = 2 (inside func) + 2 per outer loop
        indent_size = 2 + (rank - 1) * 2 + 4

        vlane_stride = 1
        vlane_split_axis = 0
        x_tile_desc = mlir_common.MLIRMultiDimTile(tile_sizes, kernel.vector_lane, vlane_split_axis, vlane_stride)
        x_tile_desc.set_tile_size_stride(tile_sizes, [sort_size, 1])
        x_tile_desc.set_name("X_buffer")
        x_tile_desc.offset = x_layout.offset

        yv_tile_desc = mlir_common.MLIRMultiDimTile(tile_sizes, kernel.vector_lane, vlane_split_axis, vlane_stride)
        yv_tile_desc.set_tile_size_stride(tile_sizes, [sort_size, 1])
        yv_tile_desc.set_name("YV_buffer")
        yv_tile_desc.offset = yv_layout.offset

        data_stype = mlir_common.DTYPE_TO_MLIR[x.get_dtype()]

        elem_memref_t = f"memref<1x{sort_size}x{data_stype}, 1>"
        rev_indices = list(range(vector_size - 1, -1, -1))

        bitonic_body = mlir_common.ParallelLoopBuffer(initial_indent=2)
        bitonic_body.tabwidth = 2
        # 1) Local SIMD sort per chunk.
        init_cse = common.CSE(kernel.newvar_prefix, kernel.suffix, name_prefix="sort_init")
        with kernel, kernel.override_buffer_cse(buffer=bitonic_body, cse=init_cse):
            bitonic_body.writelines(LoopLevel("chunk", num_chunks).lines())
            with bitonic_body.indent(attribute="{inner_loop=true}"):
                bitonic_body.writeline("%elem = affine.apply #map_chunk_to_elem(%chunk)")
                x_chunk = ops._load(
                    vector_size,
                    data_stype,
                    "X_buffer",
                    "%t_const0, %elem",
                    x_tile_desc.get_mlir_shape(data_stype),
                )
                yv_chunk = ops.bitonic_sort(x_chunk, descending=self.descending)
                ops._store(
                    yv_chunk,
                    "YV_buffer",
                    "%t_const0, %elem",
                    yv_tile_desc.get_mlir_shape(data_stype),
                )

        # 2) Chunk-level bitonic merge (loop form).
        stage = 0
        k = 2
        while k <= num_chunks:
            j = k // 2
            while j >= 1:
                for block_start, is_even_block in ((0, True), (k, False)):
                    if block_start >= num_chunks:
                        continue
                    asc_dir = is_even_block if not self.descending else (not is_even_block)
                    stage_cse = common.CSE(kernel.newvar_prefix, kernel.suffix, name_prefix=f"sort_stage_{stage}")
                    with kernel, kernel.override_buffer_cse(buffer=bitonic_body, cse=stage_cse):
                        stage_loops = [
                            LoopLevel("base", num_chunks, start=block_start, step=2 * k),
                            LoopLevel("p", k, step=2 * j),
                            LoopLevel("q", j),
                        ]
                        with contextlib.ExitStack() as stack:
                            for idx, loop in enumerate(stage_loops):
                                bitonic_body.writelines(loop.lines())
                                attr = "{inner_loop=true}"
                                stack.enter_context(bitonic_body.indent(attribute=attr))

                            bitonic_body.writeline(
                                f"%left_elem = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1 + d2) * {vector_size})>(%base, %p, %q)"
                            )
                            bitonic_body.writeline(
                                f"%right_elem = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1 + d2 + {j}) * {vector_size})>(%base, %p, %q)"
                            )

                            left_vec = ops._load(
                                vector_size,
                                data_stype,
                                "YV_buffer",
                                "%t_const0, %left_elem",
                                yv_tile_desc.get_mlir_shape(data_stype),
                            )
                            right_vec = ops._load(
                                vector_size,
                                data_stype,
                                "YV_buffer",
                                "%t_const0, %right_elem",
                                yv_tile_desc.get_mlir_shape(data_stype),
                            )
                            if asc_dir:
                                left_norm = ops.bitonic_sort(left_vec, descending=False)
                                right_norm = ops.bitonic_sort(right_vec, descending=False)
                                right_rev = ops.vector_shuffle(right_norm, rev_indices, right_norm)
                                vmin = ops.minimum(left_norm, right_rev)
                                vmax = ops.maximum(left_norm, right_rev)
                                left_new = ops.bitonic_sort(vmin, descending=False)
                                right_new = ops.bitonic_sort(vmax, descending=False)
                            else:
                                left_norm = ops.bitonic_sort(left_vec, descending=True)
                                right_norm = ops.bitonic_sort(right_vec, descending=True)
                                right_rev = ops.vector_shuffle(right_norm, rev_indices, right_norm)
                                vmin = ops.minimum(left_norm, right_rev)
                                vmax = ops.maximum(left_norm, right_rev)
                                left_new = ops.bitonic_sort(vmax, descending=True)
                                right_new = ops.bitonic_sort(vmin, descending=True)
                            ops._store(
                                left_new,
                                "YV_buffer",
                                "%t_const0, %left_elem",
                                yv_tile_desc.get_mlir_shape(data_stype),
                            )
                            ops._store(
                                right_new,
                                "YV_buffer",
                                "%t_const0, %right_elem",
                                yv_tile_desc.get_mlir_shape(data_stype),
                            )
                    stage += 1
                j //= 2
            k *= 2

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
            DATA_STYPE=data_stype,
            IDX_STYPE="i64",
            ELEM_MEMREF_T=elem_memref_t,
            BITONIC_BODY=bitonic_body.getvalue().rstrip(),
            input_reorder=self.input_reorder,
            # N-D generalization
            RANK                  = rank,
            OUTPUT_SIZES          = output_sizes,
            OUTPUT_DIM            = output_dim,
            STEP_SIZES            = step_sizes,
            OUTER_VARS            = outer_vars,
            X_OFFSET_MAP          = x_offset_map,
            YV_OFFSET_MAP         = yv_offset_map,
            X_DRAM_STRIDE         = x_dram_stride,
            YV_DRAM_STRIDE        = yv_dram_stride,
            INDENT_SIZE           = indent_size,
        )
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        kernel.exception_nodes["YI"] = {"numel" : sort_size}
        return code
