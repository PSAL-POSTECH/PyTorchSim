from typing import List, Optional
import contextlib

from torch._inductor.ir import Buffer, IRNode
from torch._inductor.virtualized import _ops as ops
from torch._inductor.codegen import common

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel
from PyTorchSimFrontend.mlir.mlir_common import LoopLevel

VECTOR_SIZE = 16


TEMPLATE = r"""
{{kernel.def_global_vars()}}
// chunk index -> element index
#map_chunk_to_elem = affine_map<(d0) -> (d0 * {{ VECTOR_SIZE }})>

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X, XI], outputs=[YV], names_str=NAMES_STR, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X",  X_TILE_DESC,  id=0, indent_size=2) }}
  {{ kernel.def_sram_buffer("XI", XI_TILE_DESC, id=1, indent_size=2) }}
  {{ kernel.def_sram_buffer("YV", YV_TILE_DESC, id=2, indent_size=2) }}
  {{ kernel.def_local_vars(indent_size=2) }}


  affine.for %sort_block = 0 to 1 step 1 {
  {%- for d in range(RANK-1) %}
    affine.for %index{{ OUTPUT_DIM[d] }} = 0 to {{ OUTPUT_SIZES[d] }} step {{ STEP_SIZES[d] }} {
  {%- endfor %}

    %x_dram_offset = affine.apply {{ X_OFFSET_MAP }}({{ OUTER_VARS }})
    %xi_dram_offset = affine.apply {{ XI_OFFSET_MAP }}({{ OUTER_VARS }})
    %yv_dram_offset = affine.apply {{ YV_OFFSET_MAP }}({{ OUTER_VARS }})
    {{ kernel.def_dma_op("MVIN", "X", [], X_TILE_DESC, indent_size=INDENT_SIZE, dram_stride=X_DRAM_STRIDE, dram_offset="x_dram_offset") }}

    // SIMD local sort + loop-based chunk merge.
{{ BITONIC_BODY }}

    {{ kernel.def_dma_op("MVOUT", "XI", [], XI_TILE_DESC, indent_size=INDENT_SIZE, dram_stride=XI_DRAM_STRIDE, dram_offset="xi_dram_offset") }}
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


def _compute_bitonic_stages(n: int, descending: bool):
    stages = []
    size = 2
    while size <= n:
        stride = size // 2
        while stride >= 1:
            merged_shuffle = list(range(n))
            merged_mask = [None] * n
            for start in range(0, n, size):
                blk_dir = "ASCENDING" if (start // size) % 2 == 0 else "DESCENDING"
                for i in range(start, start + size - stride, stride * 2):
                    for j2 in range(stride):
                        a, b = i + j2, i + j2 + stride
                        merged_shuffle[a] = b
                        merged_shuffle[b] = a
                        if blk_dir == "ASCENDING":
                            merged_mask[a] = True
                            merged_mask[b] = False
                        else:
                            merged_mask[a] = False
                            merged_mask[b] = True
            select_min = [bool(x) if x is not None else False for x in merged_mask]
            if descending:
                select_min = [not x for x in select_min]
            stages.append({"shuffle": merged_shuffle, "select_min": select_min})
            stride //= 2
        size *= 2
    return stages


def _pair_less_equal(left_v, right_v, left_i, right_i):
    cmp_val = ops.lt(left_v, right_v)
    cmp_eq = ops.eq(left_v, right_v)
    cmp_idx = ops.le(left_i, right_i)
    return ops.or_(cmp_val, ops.and_(cmp_eq, cmp_idx))


def _pair_greater_equal(left_v, right_v, left_i, right_i):
    cmp_val = ops.gt(left_v, right_v)
    cmp_eq = ops.eq(left_v, right_v)
    cmp_idx = ops.le(left_i, right_i)
    return ops.or_(cmp_val, ops.and_(cmp_eq, cmp_idx))


def _bitonic_sort_pair(values, indices, vector_size: int, descending: bool, stable_sort: bool):
    cur_v = values
    cur_i = indices
    for stage_desc in _compute_bitonic_stages(vector_size, descending):
        mask = ops.constant_mask(stage_desc["select_min"], vector_size)
        shuf_v = ops.vector_shuffle(cur_v, stage_desc["shuffle"])
        shuf_i = ops.vector_shuffle(cur_i, stage_desc["shuffle"])
        if stable_sort:
            # `cmp` drives the "min side" selection in the bitonic network.
            # For descending stable sort, tie elements with smaller original index
            # must stay earlier, so the min side should treat larger index as smaller.
            if descending:
                cmp_val = ops.lt(cur_v, shuf_v)
                cmp_eq = ops.eq(cur_v, shuf_v)
                cmp_idx = ops.ge(cur_i, shuf_i)
                cmp = ops.or_(cmp_val, ops.and_(cmp_eq, cmp_idx))
            else:
                cmp = _pair_less_equal(cur_v, shuf_v, cur_i, shuf_i)
        else:
            cmp = ops.le(cur_v, shuf_v)
        min_v = ops.where(cmp, cur_v, shuf_v)
        min_i = ops.where(cmp, cur_i, shuf_i)
        max_v = ops.where(cmp, shuf_v, cur_v)
        max_i = ops.where(cmp, shuf_i, cur_i)
        cur_v = ops.where(mask, min_v, max_v)
        cur_i = ops.where(mask, min_i, max_i)
    return cur_v, cur_i


def _merge_sorted_pair_vectors(
    left_norm,
    left_idx_norm,
    right_norm,
    right_idx_norm,
    ascending: bool,
    stable_sort: bool,
    vector_size: int,
    rev_indices,
):
    right_pair = ops.vector_shuffle(right_norm, rev_indices, right_norm)
    right_idx_pair = ops.vector_shuffle(right_idx_norm, rev_indices, right_idx_norm)
    if ascending:
        cmp = (
            _pair_less_equal(left_norm, right_pair, left_idx_norm, right_idx_pair)
            if stable_sort
            else ops.le(left_norm, right_pair)
        )
    else:
        cmp = (
            _pair_greater_equal(left_norm, right_pair, left_idx_norm, right_idx_pair)
            if stable_sort
            else ops.ge(left_norm, right_pair)
        )
    left_merge = ops.where(cmp, left_norm, right_pair)
    left_idx_merge = ops.where(cmp, left_idx_norm, right_idx_pair)
    right_merge = ops.where(cmp, right_pair, left_norm)
    right_idx_merge = ops.where(cmp, right_idx_pair, left_idx_norm)
    return left_merge, left_idx_merge, right_merge, right_idx_merge


class MLIRSortTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, dim, descending=False, stable=False, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.dim = dim
        self.descending = descending
        self.stable = stable
        self.use_stable_sort = False
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
        xi = self.input_nodes[1]
        yv = self.output_nodes[0]
        # XI is updated in-place by the sort kernel, so mark it as an inout arg.
        kernel.kernel_group.args.make_inplace(xi.get_name(), xi.get_name())
        sort_size = int(x.get_size()[self.dim])
        vector_size = VECTOR_SIZE
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
        sort_dim = self.dim if self.dim >= 0 else self.dim + rank
        if sort_dim < 0 or sort_dim >= rank:
            raise NotImplementedError(f"Invalid sort dim for rank-{rank} tensor (dim={self.dim})")
        x_layout = x.get_layout()
        xi_layout = xi.get_layout()
        yv_layout = yv.get_layout()

        if rank == 1:
            # Edge case for 1D tensor
            output_sizes = [1]
            output_dim = [0]
            step_sizes = [1]
            tile_sizes = [1, sort_size]
            x_dram_stride = [int(x_layout.stride[sort_dim]), int(x_layout.stride[sort_dim])]
            xi_dram_stride = [int(xi_layout.stride[sort_dim]), int(xi_layout.stride[sort_dim])]
            yv_dram_stride = [int(yv_layout.stride[sort_dim]), int(yv_layout.stride[sort_dim])]
            template_rank = 2
        else:
            output_sizes = [sz for d, sz in enumerate(yv.get_size()) if d != sort_dim]
            output_dim = [d for d, _ in enumerate(yv.get_size()) if d != sort_dim]
            step_sizes = [1] * len(output_sizes)

            tile_dim = max(output_dim, key=lambda d: int(yv.get_size()[d]))
            tile_sizes = [min(kernel.vector_lane, int(yv.get_size()[tile_dim])), sort_size]
            step_sizes[output_dim.index(tile_dim)] = tile_sizes[0]

            x_dram_stride = [int(x_layout.stride[tile_dim]), int(x_layout.stride[sort_dim])]
            xi_dram_stride = [int(xi_layout.stride[tile_dim]), int(xi_layout.stride[sort_dim])]
            yv_dram_stride = [int(yv_layout.stride[tile_dim]), int(yv_layout.stride[sort_dim])]
            template_rank = rank

        x_offset_map  = _make_offset_map(output_dim, x_layout.stride,  x_layout.offset)
        xi_offset_map = _make_offset_map(output_dim, xi_layout.stride, xi_layout.offset)
        yv_offset_map = _make_offset_map(output_dim, yv_layout.stride, yv_layout.offset)
        outer_vars = ", ".join(f"%index{d}" for d in output_dim)

        # indent for DMA ops = 2 (inside func) + 2 per outer loop
        indent_size = 2 + len(output_dim) * 2 + 4

        vlane_stride = 1
        vlane_split_axis = 0
        x_tile_desc = mlir_common.MLIRMultiDimTile(tile_sizes, kernel.vector_lane, vlane_split_axis, vlane_stride)
        x_tile_desc.set_tile_size_stride(tile_sizes, [sort_size, 1])
        x_tile_desc.set_name("X_buffer")
        x_tile_desc.offset = x_layout.offset

        xi_tile_desc = mlir_common.MLIRMultiDimTile(tile_sizes, kernel.vector_lane, vlane_split_axis, vlane_stride)
        xi_tile_desc.set_tile_size_stride(tile_sizes, [sort_size, 1])
        xi_tile_desc.set_name("XI_buffer")
        xi_tile_desc.offset = xi_layout.offset

        yv_tile_desc = mlir_common.MLIRMultiDimTile(tile_sizes, kernel.vector_lane, vlane_split_axis, vlane_stride)
        yv_tile_desc.set_tile_size_stride(tile_sizes, [sort_size, 1])
        yv_tile_desc.set_name("YV_buffer")
        yv_tile_desc.offset = yv_layout.offset

        data_stype = mlir_common.DTYPE_TO_MLIR[x.get_dtype()]
        idx_stype = mlir_common.DTYPE_TO_MLIR[xi.get_dtype()]

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
                idx_step_index = kernel.register_var_cse("idx_step_index", vector_size, "index")
                bitonic_body.writeline(f"%{idx_step_index} = vector.step : vector<{vector_size}xindex>")
                idx_step = ops.index_cast(idx_step_index, idx_stype)
                idx_base = kernel.register_var_cse("idx_base", 1, idx_stype)
                bitonic_body.writeline(f"%{idx_base} = arith.index_cast %elem : index to {idx_stype}")
                idx_base_vec = ops.broadcast(idx_base, vector_size)
                idx_chunk = ops.add(idx_base_vec, idx_step)
                yv_chunk, yi_chunk = _bitonic_sort_pair(
                    x_chunk, idx_chunk, vector_size, descending=self.descending, stable_sort=self.use_stable_sort
                )
                ops._store(
                    yv_chunk,
                    "YV_buffer",
                    "%t_const0, %elem",
                    yv_tile_desc.get_mlir_shape(data_stype),
                )
                ops._store(
                    yi_chunk,
                    "XI_buffer",
                    "%t_const0, %elem",
                    xi_tile_desc.get_mlir_shape(idx_stype),
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
                            for loop in stage_loops:
                                bitonic_body.writelines(loop.lines())
                                stack.enter_context(bitonic_body.indent(attribute="{inner_loop=true}"))

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
                            left_idx = ops._load(
                                vector_size,
                                idx_stype,
                                "XI_buffer",
                                "%t_const0, %left_elem",
                                xi_tile_desc.get_mlir_shape(idx_stype),
                            )
                            right_idx = ops._load(
                                vector_size,
                                idx_stype,
                                "XI_buffer",
                                "%t_const0, %right_elem",
                                xi_tile_desc.get_mlir_shape(idx_stype),
                            )
                            norm_desc = not asc_dir
                            left_norm, left_idx_norm = _bitonic_sort_pair(
                                left_vec, left_idx, vector_size, descending=norm_desc, stable_sort=self.use_stable_sort
                            )
                            right_norm, right_idx_norm = _bitonic_sort_pair(
                                right_vec, right_idx, vector_size, descending=norm_desc, stable_sort=self.use_stable_sort
                            )
                            left_merge, left_idx_merge, right_merge, right_idx_merge = _merge_sorted_pair_vectors(
                                left_norm,
                                left_idx_norm,
                                right_norm,
                                right_idx_norm,
                                ascending=asc_dir,
                                stable_sort=self.use_stable_sort,
                                vector_size=vector_size,
                                rev_indices=rev_indices,
                            )
                            left_new, left_idx_new = _bitonic_sort_pair(
                                left_merge, left_idx_merge, vector_size, descending=norm_desc, stable_sort=self.use_stable_sort
                            )
                            right_new, right_idx_new = _bitonic_sort_pair(
                                right_merge, right_idx_merge, vector_size, descending=norm_desc, stable_sort=self.use_stable_sort
                            )
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
                            ops._store(
                                left_idx_new,
                                "XI_buffer",
                                "%t_const0, %left_elem",
                                xi_tile_desc.get_mlir_shape(idx_stype),
                            )
                            ops._store(
                                right_idx_new,
                                "XI_buffer",
                                "%t_const0, %right_elem",
                                xi_tile_desc.get_mlir_shape(idx_stype),
                            )
                    stage += 1
                j //= 2
            k *= 2

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            NAMES_STR="X, XI, YV",
            kernel=kernel,
            X=x,
            XI=xi,
            YV=yv,
            X_TILE_DESC=x_tile_desc,
            XI_TILE_DESC=xi_tile_desc,
            YV_TILE_DESC=yv_tile_desc,
            SORT_SIZE=sort_size,
            VECTOR_SIZE=vector_size,
            DATA_STYPE=data_stype,
            IDX_STYPE=idx_stype,
            ELEM_MEMREF_T=elem_memref_t,
            BITONIC_BODY=bitonic_body.getvalue().rstrip(),
            input_reorder=self.input_reorder,
            # N-D generalization
            RANK                  = template_rank,
            OUTPUT_SIZES          = output_sizes,
            OUTPUT_DIM            = output_dim,
            STEP_SIZES            = step_sizes,
            OUTER_VARS            = outer_vars,
            X_OFFSET_MAP          = x_offset_map,
            XI_OFFSET_MAP         = xi_offset_map,
            YV_OFFSET_MAP         = yv_offset_map,
            X_DRAM_STRIDE         = x_dram_stride,
            XI_DRAM_STRIDE        = xi_dram_stride,
            YV_DRAM_STRIDE        = yv_dram_stride,
            INDENT_SIZE           = indent_size,
        )
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        return code


class MLIRStableSortTemplate(MLIRSortTemplate):
    def __init__(self, input_nodes, layout, dim, descending=False, stable=True, input_reorder=None):
        super().__init__(
            input_nodes=input_nodes,
            layout=layout,
            dim=dim,
            descending=descending,
            stable=stable,
            input_reorder=input_reorder,
        )
        self.use_stable_sort = True
