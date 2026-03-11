from typing import List, Optional, Set
import math
import itertools

import sympy
from torch._inductor.ir import IRNode

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel


TEMPLATE = r"""
{{kernel.def_global_vars()}}
func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=INPUT_NAMES, outputs=[Y], names_str=NAMES_STR, input_reorder=input_reorder)}} {
{%- for buffer_name, tile_desc in UNIQUE_BUFFER_TILE_DESCS.items() %}
  {{ kernel.def_sram_buffer(buffer_name, tile_desc, indent_size=2) }}
{%- endfor %}
  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %cat_block = 0 to 1 step 1 {
{%- for d in range(RANK-1) %}
    affine.for %index{{ OUTPUT_DIM[d] }} = 0 to {{ OUTPUT_SIZES[d] }} step {{ TILE_SIZES[d] }} {
{%- endfor %}
{%- for i in range(NUM_INPUTS) %}
      // Input tensor{{ i }}
      affine.for %index_local{{ DIM }}_{{ i }} = 0 to {{ INPUTS[i].sizes[DIM] }} step {{ INPUTS[i].tile_size_dim }} {
        %index{{ DIM }}_{{ i }} = affine.apply affine_map<(d0) -> (d0 + {{ INPUTS[i].cum_offset }})> (%index_local{{ DIM }}_{{ i }})
        %input_dram_offset_{{ i }} = affine.apply {{ INPUTS[i].offset_map }}({{ INPUTS[i].offset_vars }})
        %output_dram_offset_{{ i }} = affine.apply {{ OUTPUTS[i].offset_map }}({{ OUTPUTS[i].offset_vars }})
        {{ kernel.def_dma_op("MVIN", INPUTS[i].dram_name, [], INPUTS[i].tile_desc, indent_size=INDENT_SIZE, dram_stride=INPUTS[i].dram_strides, dram_offset="input_dram_offset_" ~ i) }}
        {{ kernel.def_dma_op("MVOUT", "Y", [], OUTPUTS[i].tile_desc, indent_size=INDENT_SIZE, dram_stride=OUTPUTS[i].dram_strides, dram_offset="output_dram_offset_" ~ i) }}
      } { inner_loop=true }
{%- endfor %}

{%- for d in range(RANK-1) %}
    } { outer_loop=true }
{%- endfor %}
  } { outer_loop=true }
  return
}
"""


class MLIRCatTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, dim):
        super().__init__("kernel", input_nodes, layout)
        self.dim = dim

    def render(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        tile_info=None,
        **kwargs,
    ):
        input_nodes = self.input_nodes
        y = self.output_node
        dtype_infos = [("Y", y.get_dtype())] + [(f"X{i}", x.get_dtype()) for i, x in enumerate(input_nodes)]
        if len({dtype for _, dtype in dtype_infos}) != 1:
            dtype_desc = ", ".join(f"{name}={dtype}" for name, dtype in dtype_infos)
            raise NotImplementedError(f"Mixed dtype Cat is not implemented yet ({dtype_desc})")
        precision_bytes = mlir_common.get_dtype_nbytes(y.get_dtype())
        num_inputs = len(input_nodes)
        rank = len(y.get_size())

        input_sizes = [x.get_size() for x in input_nodes]
        output_sizes = [sz for d, sz in enumerate(y.get_size()) if d != self.dim]
        output_dim   = [d  for d, _ in enumerate(y.get_size()) if d != self.dim]
        output_strides = y.get_layout().stride

        tile_sizes = list(tile_info) if tile_info is not None else [1] * len(output_sizes)
        excluded_dims = self._compute_excluded_dims(tile_sizes)

        input_tile_sizes_dim = self._calculate_input_tile_sizes(
            kernel, input_sizes, tile_sizes, num_inputs, rank, precision_bytes
        )
        buffer_name_to_template_name, input_dram_names = self._build_buffer_mapping(input_nodes)
        input_tile_descs, output_tile_descs, unique_tile_descs = self._build_tile_descriptors(
            kernel, input_nodes, input_sizes, input_tile_sizes_dim, tile_sizes, rank,
            input_dram_names, y, excluded_dims=excluded_dims
        )
        (input_offset_maps, input_offset_var_strs, input_dram_strides,
         output_offset_maps, output_offset_var_strs, output_dram_strides,
         cumulative_offsets) = self._build_dma_info(
            input_nodes, input_sizes, output_strides, input_tile_descs, output_tile_descs,
            rank, num_inputs, excluded_dims=excluded_dims
        )

        unique_buffer_tile_descs = {
            buffer_name_to_template_name[name]: desc
            for name, desc in unique_tile_descs.items()
        }
        names_str = ", ".join(input_dram_names + ["Y"])
        indent_size = 2 + (rank - 1) * 2 + 4

        inputs_info = [
            dict(
                dram_name    = input_dram_names[i],
                sizes        = input_sizes[i],
                tile_size_dim= input_tile_sizes_dim[i],
                tile_desc    = input_tile_descs[i],
                offset_map   = input_offset_maps[i],
                offset_vars  = input_offset_var_strs[i],
                dram_strides = input_dram_strides[i],
                cum_offset   = cumulative_offsets[i],
            )
            for i in range(num_inputs)
        ]
        outputs_info = [
            dict(
                tile_desc    = output_tile_descs[i],
                offset_map   = output_offset_maps[i],
                offset_vars  = output_offset_var_strs[i],
                dram_strides = output_dram_strides[i],
            )
            for i in range(num_inputs)
        ]

        kernel.render_options = dict(
            KERNEL_NAME           = self.name,
            kernel                = kernel,
            NUM_INPUTS            = num_inputs,
            NAMES_STR             = names_str,
            Y                     = y,
            INPUT_NAMES           = input_nodes,
            RANK                  = rank,
            DIM                   = self.dim,
            OUTPUT_SIZES          = output_sizes,
            OUTPUT_DIM            = output_dim,
            TILE_SIZES            = tile_sizes,
            UNIQUE_BUFFER_TILE_DESCS = unique_buffer_tile_descs,
            INPUTS                = inputs_info,
            OUTPUTS               = outputs_info,
            INDENT_SIZE           = indent_size,
            input_reorder         = self.input_reorder,
        )

        return self._template_from_string(TEMPLATE).render(**kernel.render_options)

    def get_tile_candidates(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        **kwargs,
    ):
        """Generate tile candidates for cat operation. Concat dimension always has tile size 1."""
        if template_buffer_node is not None:
            self.output_node = template_buffer_node

        y = self.output_node
        dtype_infos = [("Y", y.get_dtype())] + [(f"X{i}", x.get_dtype()) for i, x in enumerate(self.input_nodes)]
        if len({dtype for _, dtype in dtype_infos}) != 1:
            dtype_desc = ", ".join(f"{name}={dtype}" for name, dtype in dtype_infos)
            raise NotImplementedError(f"Mixed dtype Cat is not implemented yet ({dtype_desc})")
        precision_bytes = mlir_common.get_dtype_nbytes(y.get_dtype())
        num_inputs = len(self.input_nodes)
        output_sizes = [sz for d, sz in enumerate(y.get_size()) if d != self.dim]

        if not output_sizes:
            return [[1]]

        max_tile_total = kernel.spad_info["spad_size"] // (
            kernel.vector_lane * precision_bytes * 2 * num_inputs
        )

        dim_tile_candidates = []
        for dim_size in output_sizes:
            max_tile = min(dim_size, max_tile_total)
            candidates = set()
            for mult in range(1, max_tile // kernel.vector_lane + 1):
                t = mult * kernel.vector_lane
                if t <= dim_size and dim_size % t == 0:
                    candidates.add(t)
            if max_tile > 0:
                for exp in range(int(math.log2(max_tile)) + 1):
                    t = 2 ** exp
                    if t <= dim_size and dim_size % t == 0:
                        candidates.add(t)
            candidates.add(dim_size)  # dim_size always divides itself
            dim_tile_candidates.append(sorted(candidates)[:5])

        tile_candidates = [
            list(combo)
            for combo in itertools.product(*dim_tile_candidates)
            if math.prod(combo) * (num_inputs + 1) * precision_bytes
               <= kernel.spad_info["spad_size"] * kernel.vector_lane
        ]

        if not tile_candidates:
            tile_candidates = [[1] * len(output_sizes)]

        tile_candidates.sort(key=lambda x: -math.prod(x))
        return tile_candidates[:4]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_excluded_dims(self, tile_sizes: list) -> list:
        """Return non-tiled dimension indices when rank exceeds the 4-dim limit."""
        max_tiled = 3
        if len(tile_sizes) <= max_tiled:
            return []
        sorted_dims = sorted(enumerate(tile_sizes), key=lambda x: x[1], reverse=True)
        excluded = [idx for idx, _ in sorted_dims[max_tiled:]]
        for idx in excluded:
            tile_sizes[idx] = 1
        return excluded

    def _calculate_input_tile_sizes(self, kernel, input_sizes, tile_sizes, num_inputs, rank, precision_bytes):
        """Calculate tile sizes along the concat dimension for each input."""
        non_dim_tile_elements = math.prod(tile_sizes) if tile_sizes else 1
        max_spad_per_input = kernel.spad_info["spad_size"] * kernel.vector_lane // 2
        extra_concat = math.ceil(max_spad_per_input / (non_dim_tile_elements * precision_bytes)) - num_inputs

        input_tile_sizes_dim = []
        for i in range(num_inputs):
            if extra_concat > 0 and non_dim_tile_elements > 0:
                tile_dim = min(input_sizes[i][self.dim], extra_concat)
                extra_concat -= tile_dim
            else:
                tile_dim = 1
            input_tile_sizes_dim.append(tile_dim)
        return input_tile_sizes_dim

    def _build_buffer_mapping(self, input_nodes):
        """Map actual buffer names to short template names (X0, X1, ...)."""
        name_map = {}
        template_names = []
        for x in input_nodes:
            actual = x.get_name()
            template = name_map.setdefault(actual, f"X{len(name_map)}")
            template_names.append(template)
        return name_map, template_names

    def _build_tile_descriptors(
        self, kernel, input_nodes, input_sizes, input_tile_sizes_dim, tile_sizes, rank,
        input_buffer_names, output_node, excluded_dims=None
    ):
        """Build tile descriptors for every input (and its paired output)."""
        if excluded_dims is None:
            excluded_dims = set()

        def make_tile_desc(tile_sz, vector_lane, name, offset):
            desc = mlir_common.MLIRMultiDimTile(
                tile_sz, vector_lane,
                vlane_split_axis=len(tile_sz) - 1,
                vlane_stride=1
            )
            desc.set_tile_size(tile_sz)
            desc.set_name(name)
            desc.offset = offset
            return desc

        output_offset = output_node.get_layout().offset
        input_tile_descs, output_tile_descs, unique_tile_descs = [], [], {}

        for i, x in enumerate(input_nodes):
            # Collect tile sizes for tiled dimensions only (skip excluded non-concat dims)
            tile_sz = []
            tile_idx = 0
            for d in range(rank):
                if d != self.dim:
                    if tile_idx not in excluded_dims:
                        tile_sz.append(tile_sizes[tile_idx])
                    tile_idx += 1
                else:
                    tile_sz.append(input_tile_sizes_dim[i])

            sram_name = f"{input_buffer_names[i].lower()}_cat_tile"
            input_tile_descs.append(make_tile_desc(tile_sz, kernel.vector_lane, sram_name, x.get_layout().offset))
            output_tile_descs.append(make_tile_desc(tile_sz, kernel.vector_lane, sram_name, output_offset))

            actual_name = x.get_name()
            if actual_name not in unique_tile_descs:
                unique_tile_descs[actual_name] = input_tile_descs[-1]

        return input_tile_descs, output_tile_descs, unique_tile_descs

    def _build_dma_info(
        self, input_nodes, input_sizes, output_strides,
        input_tile_descs, output_tile_descs,
        rank, num_inputs, excluded_dims=None
    ):
        """Build per-input DRAM offset affine maps and tile strides.

        Three stride concepts are maintained:

        * layout_strides (internal) - raw DRAM buffer strides for every rank
          dimension, used to compute the flat base-address affine map.
          These reflect how the tensor is physically laid out in DRAM.
        * dram_strides (returned,  ``def_dma_op dram_stride=``) - stride in
          DRAM per *tiled* dimension (excluded dims removed). The DMA engine
          uses these to walk DRAM when loading/storing a tile.
        * sram_strides (inside ``def_dma_op``, from tile_desc) - stride in
          SRAM per tiled dimension. The DMA engine uses these to place data
          into the SRAM tile buffer.

        Returns:
            input_offset_maps, input_offset_var_strs, input_dram_strides,
            output_offset_maps, output_offset_var_strs, output_dram_strides,
            cumulative_offsets
        """
        if excluded_dims is None:
            excluded_dims = set()

        def make_affine_map(idx_syms, strides, layout_offset):
            terms = []
            for j, s in enumerate(strides):
                s = int(s)
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
            dim_str = ", ".join(f"d{j}" for j in range(len(idx_syms)))
            return f"affine_map<({dim_str}) -> ({' + '.join(terms) if terms else '0'})>"

        cumulative_offsets = [0]
        for i in range(num_inputs - 1):
            cumulative_offsets.append(cumulative_offsets[-1] + input_sizes[i][self.dim])

        input_offset_maps, input_offset_var_strs, input_dram_strides = [], [], []
        output_offset_maps, output_offset_var_strs, output_dram_strides = [], [], []

        for i, x in enumerate(input_nodes):
            x_stride = x.get_layout().stride
            in_syms, in_layout_strides, in_dram_strides = [], [], []
            out_syms, out_layout_strides, out_dram_strides = [], [], []
            tile_idx = 0

            for d in range(rank):
                if d != self.dim:
                    in_syms.append(sympy.Symbol(f"index{d}"))
                    in_layout_strides.append(int(x_stride[d]))
                    out_syms.append(sympy.Symbol(f"index{d}"))
                    out_layout_strides.append(int(output_strides[d]))
                    if tile_idx not in excluded_dims:
                        in_dram_strides.append(int(x_stride[d]))
                        out_dram_strides.append(int(output_strides[d]))
                    tile_idx += 1
                else:
                    in_syms.append(sympy.Symbol(f"index_local{self.dim}_{i}"))
                    in_layout_strides.append(int(x_stride[d]))
                    out_syms.append(sympy.Symbol(f"index{self.dim}_{i}"))
                    out_layout_strides.append(int(output_strides[d]))
                    in_dram_strides.append(int(x_stride[d]))
                    out_dram_strides.append(int(output_strides[d]))

            input_offset_maps.append(make_affine_map(in_syms, in_layout_strides, input_tile_descs[i].offset))
            input_offset_var_strs.append(", ".join(f"%{s}" for s in in_syms))
            input_dram_strides.append(in_dram_strides)

            output_offset_maps.append(make_affine_map(out_syms, out_layout_strides, output_tile_descs[i].offset))
            output_offset_var_strs.append(", ".join(f"%{s}" for s in out_syms))
            output_dram_strides.append(out_dram_strides)

        return (input_offset_maps, input_offset_var_strs, input_dram_strides,
                output_offset_maps, output_offset_var_strs, output_dram_strides,
                cumulative_offsets)
