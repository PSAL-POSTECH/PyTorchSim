import math # sqrt
import sympy

from typing import List, Optional

import torch
from torch import empty_strided
from torch._inductor.ir import IRNode, TensorBox, FixedLayout
from torch._inductor.virtualized import V
from torch._inductor.select_algorithm import realize_inputs
from torch.backends.cuda import flash_sdp_enabled, mem_efficient_sdp_enabled

from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel


def _make_offset_map_with_sym(strides, sym_dim, sym_stride, offset=0):
    """Like _make_offset_map but injects a block symbol ``s`` into dimension ``sym_dim``.

    The effective index for that dimension becomes ``d{sym_dim} + sym_stride * s``.
    Use this to keep ``affine.for`` bounds static and encode the block contribution
    directly inside the ``affine.apply`` call that computes the DRAM offset.

    Args:
        strides:    per-dimension DRAM strides.
        sym_dim:    which dimension carries the block symbol.
        sym_stride: multiplier for the symbol (1 for abs-position loops like FLASH
                    ``%blk``; ``BlkS`` for block-index loops like PARTIAL ``%blk``).
        offset:     constant layout offset.

    Returns:
        MLIR affine_map string with one symbol, e.g.
        ``affine_map<(d0, d1, d2)[s] -> (d0 * 8192 + (d1 + 128 * s) * 64 + d2)>``
    """
    n = len(strides)
    terms = []
    for j, sv in enumerate(strides):
        sv = int(sv)
        if sv == 0:
            continue
        if j == sym_dim:
            inner = f"d{j} + s" if sym_stride == 1 else f"d{j} + {sym_stride} * s"
            terms.append(f"({inner})" if sv == 1 else f"({inner}) * {sv}")
        else:
            terms.append(f"d{j}" if sv == 1 else f"d{j} * {sv}")
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    if off:
        terms.append(str(off))
    dim_str = ", ".join(f"d{j}" for j in range(n))
    expr = " + ".join(terms) if terms else "0"
    return f"affine_map<({dim_str})[s] -> ({expr})>"


def _make_offset_map(strides, offset=0):
    """Generate an MLIR affine_map string for a flat DRAM base-address.

    Args:
        strides: list of integer per-dimension strides.
                 A stride of 0 means the dimension does not contribute.
        offset:  constant layout offset (e.g. from IRNode.get_layout().offset).

    Returns:
        MLIR affine_map string, e.g. ``affine_map<(d0, d1) -> (d0 * 128 + d1)>``
    """
    n = len(strides)
    terms = []
    for j, s in enumerate(strides):
        s = int(s)
        if s == 1:
            terms.append(f"d{j}")
        elif s != 0:
            terms.append(f"d{j} * {s}")
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    if off:
        terms.append(str(off))
    dim_str = ", ".join(f"d{j}" for j in range(n))
    expr = " + ".join(terms) if terms else "0"
    return f"affine_map<({dim_str}) -> ({expr})>"


def flash_sdpa_args(
        query : TensorBox,
        key   : TensorBox,
        value : TensorBox) -> list:
    """
    Arg processing for flash SDPA.
    Its logic is based on:
    mm_args() which is in torch._inductor.kernel.mm_common.py (142 line).
    """

    # Materialize input buffers for the codegen backend.
    query, key, value = realize_inputs(query, key, value)

    # query : (n, hq, l, e)
    # key   : (n, h, s, e)
    # value : (n, h, s, ev)
    # out   : (n, hq, l, ev)
    # n: Batch size
    # hq: query's head counts, h: key and value's head counts.
    # l: target sequence lenght and s: source sequence length.
    # e: embeding dimension of the query and key and ev: embeding dimension of the value.
    nq, hq, l, eq  = query.get_size()
    nk, hk, sk, ek = key.get_size()
    nk, hv, sv, ev = value.get_size()

    n = V.graph.sizevars.guard_equals(nq, nk)
    n = V.graph.sizevars.guard_equals(nq, nk)

    h = V.graph.sizevars.guard_equals(hk, hv)
    s = V.graph.sizevars.guard_equals(sk, sv)
    e = V.graph.sizevars.guard_equals(eq, ek)

    # While there are no theoretical requirements for e == ev,
    # this implementation currently enforces e == ev for simplicity.
    if e != ev:
        raise NotImplementedError(
            "Flash SDPA currently requires matching head dimensions between query and value (e == ev)."
        )

    # Minimal GQA support (single-batch only for now).
    # We map each query head to a KV head by grouping: hq = g * h.
    if hq != h:
        if n != 1:
            raise NotImplementedError("Flash SDPA GQA is currently supported only for n == 1.")
        if (hq % h) != 0:
            raise NotImplementedError(f"Flash SDPA GQA requires hq % h == 0 (hq: {hq}, h: {h}).")

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [n, hq, l, ev]
    )

    return [n, hq, h, l, s, e, ev, layout, query, key, value]

def calculate_scale(query: torch.Tensor, scale: float) -> float:
    """
    Calculate the scaling factor based on the head dimension if scale is None
    Otherwise, use the provided scale.
    """
    if scale is None:
        return 1.0 / math.sqrt(query.layout.size[-1])
    else:
        return scale


FLASH_SDPA_TEMPLATE = r"""
// SDPA kernel
// b = {{ b }}
// l = {{ l }}
// s = {{ s }}
// e = {{ e }}
// tile_l = {{ tile_l }}
// tile_s = {{ tile_s }}
// tile_e = {{ tile_e }}
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[query, key, value], outputs=[out], names_str="query, key, value, out", input_reorder=input_reorder)}} {
  // Inputs
  {{ kernel.def_sram_buffer("query", q_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("key", k_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("value", v_tile_desc, indent_size=2) }}

  // Output
  {{ kernel.def_sram_buffer("out", out_tile_desc, indent_size=2) }}

  // Intermediate buffers
  {{ kernel.def_sram_buffer("mul", mul_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("max", max_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("sum", sum_desc, indent_size=2) }}

  // Constants
  %c0 = arith.constant 0.0 : {{ data_stype }}
  %c1 = arith.constant 1.0 : {{ data_stype }}
  %c_scale = arith.constant {{ scale }} : {{ data_stype }}
  %c_neg_inf = arith.constant -1.0e+30 : {{ data_stype }}

  %v0_c = arith.constant dense<0.0> : vector<{{ chunk_size }}x{{ data_stype }}>
  %v0_l = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(tile_l, tile_e) }}x{{ data_stype }}>
  %v0_s = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(tile_s, tile_l) }}x{{ data_stype }}>
  %v0_2x = arith.constant dense<0.0> : vector<2x{{ data_stype }}>

  %v_neg_inf_c = arith.constant dense<-1.0e+30> : vector<{{ chunk_size }}x{{ data_stype }}>
  %v_neg_inf_2x = arith.constant dense<-1.0e+30> : vector<2x{{ data_stype }}>

  %v_scale = vector.broadcast %c_scale : {{ data_stype }} to vector<{{ tile_s }}x{{ data_stype }}>

  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %index0 = 0 to {{ b }} {
    affine.for %index3 = 0 to 1 step 1 {
      affine.for %index1 = 0 to {{ l }} step {{ tile_l }} {
        %q_dram_offset = affine.apply {{ q_offset_map }}(%index0, %index1, %index3)
        {{ kernel.def_dma_op("MVIN", "query", [], q_tile_desc, indent_size=8, dram_stride=q_dram_stride, dram_offset="q_dram_offset") }}

        affine.vector_store %v0_l, %out_buffer[0, 0, 0] : {{ out_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ kernel.get_spad_size_per_lane(tile_l, tile_e) }}x{{ data_stype }}>
        affine.vector_store %v_neg_inf_2x, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>
        affine.vector_store %v0_2x, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>

        %qt_buffer2D = memref.reinterpret_cast %q_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_l }}], strides: [{{ tile_l }}, 1] : {{ q_tile_desc.get_mlir_shape(data_stype) }} to memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>
        %ot_buffer2D = memref.reinterpret_cast %out_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_l }}], strides: [{{ tile_l }}, 1] : {{ out_tile_desc.get_mlir_shape(data_stype) }} to memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>

        affine.for %index2 = 0 to {{ s }} step {{ tile_s }} {
          %k_dram_offset = affine.apply {{ k_offset_map }}(%index0, %index2, %index3)
          {{ kernel.def_dma_op("MVIN", "key", [], k_tile_desc, indent_size=10, dram_stride=k_dram_stride, dram_offset="k_dram_offset") }}
          %v_dram_offset = affine.apply {{ v_offset_map }}(%index0, %index2, %index3)
          {{ kernel.def_dma_op("MVIN", "value", [], v_tile_desc, indent_size=10, dram_stride=v_dram_stride, dram_offset="v_dram_offset") }}

          affine.vector_store %v0_s, %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ kernel.get_spad_size_per_lane(tile_s, tile_l) }}x{{ data_stype }}>

          %k_buffer2D = memref.reinterpret_cast %k_buffer to offset: [0], sizes: [{{ tile_s }}, {{ tile_e }}], strides: [{{ tile_e }}, 1] : {{ k_tile_desc.get_mlir_shape(data_stype) }} to memref<{{ tile_s }}x{{ tile_e }}x{{ data_stype }}, 1>
          %vt_buffer2D = memref.reinterpret_cast %v_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_s }}], strides: [{{ tile_s }}, 1] : {{ v_tile_desc.get_mlir_shape(data_stype) }} to memref<{{ tile_e }}x{{ tile_s }}x{{ data_stype }}, 1>


          // key @ query.t and scaling.
          linalg.matmul
            { idx_map = array<i32: 1, 0, -1> }
            ins(%k_buffer2D, %qt_buffer2D : memref<{{ tile_s }}x{{ tile_e }}x{{ data_stype }}, 1>, memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>)
            outs(%mul_buffer : {{ mul_tile_desc.get_mlir_shape(data_stype) }})

          %raw_mul_vec = affine.vector_load %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ tile_s }}x{{ data_stype }}>
          %scaled_mul_vec = arith.mulf %raw_mul_vec, %v_scale :  vector<{{ tile_s }}x{{ data_stype }}>
          affine.vector_store %scaled_mul_vec, %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ tile_s }}x{{ data_stype }}>


          // Find new max.
          %old_max = affine.vector_load %max_buffer[0,0] : {{ max_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>

          %chunk_max_res = affine.for %index5 = 0 to {{ tile_s }} step {{ chunk_size }} iter_args(%iter_max=%v_neg_inf_c) -> (vector<{{ chunk_size }}x{{ data_stype }}>) {
            %chunk_val = affine.vector_load %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ chunk_size }}x{{ data_stype }}>
            %local_max = arith.maximumf %chunk_val, %iter_max : vector<{{ chunk_size }}x{{ data_stype }}>
            affine.yield %local_max : vector<{{ chunk_size }}x{{ data_stype }}>
          } { accumulation_loop=true }

          %max_cast = vector.shape_cast %chunk_max_res : vector<{{ chunk_size }}x{{ data_stype }}> to vector<{{ chunk_size // 2 }}x2x{{ data_stype }}>
          %max_reduced_1 = vector.multi_reduction <maximumf>, %max_cast, %v_neg_inf_2x [0] : vector<8x2x{{ data_stype }}> to vector<2x{{ data_stype }}>
          %max_shuffled = vector.shuffle %max_reduced_1, %max_reduced_1 [1, 0] : vector<2x{{ data_stype }}>, vector<2x{{ data_stype }}>
          %max_reduced_2 = arith.maximumf %max_reduced_1, %max_shuffled : vector<2x{{ data_stype }}>

          %new_max = arith.maximumf %max_reduced_2, %old_max : vector<2x{{ data_stype }}>
          affine.vector_store %new_max, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>


          // Compute rescale factors: exp(old_max - new_max)
          %max_diff = arith.subf %old_max, %new_max : vector<2x{{ data_stype }}>
          %max_diff_scalar = vector.extract %max_diff[0] : {{ data_stype }} from vector<2x{{ data_stype }}>

          %rescale_bcast_e = vector.broadcast %max_diff_scalar : {{ data_stype }} to vector<{{ tile_e }}x{{ data_stype }}>
          %exp_rescale_e = math.exp %rescale_bcast_e : vector<{{ tile_e }}x{{ data_stype }}>

          %rescale_bcast_2 = vector.broadcast %max_diff_scalar : {{ data_stype }} to vector<2x{{ data_stype }}>
          %exp_rescale_2 = math.exp %rescale_bcast_2 : vector<2x{{ data_stype }}>


          // Rescale previous out and sum accumulators
          %old_out = affine.vector_load %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>, vector<{{ tile_e }}x{{ data_stype }}>
          %rescaled_out = arith.mulf %exp_rescale_e, %old_out : vector<{{ tile_e }}x{{ data_stype }}>
          affine.vector_store %rescaled_out, %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>, vector<{{ tile_e }}x{{ data_stype }}>

          %old_sum = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>
          %rescaled_sum = arith.mulf %old_sum, %exp_rescale_2 : vector<2x{{ data_stype }}>


          // Shift scores and apply exp: exp(x - new_max)
          %scaled_scores_reload = affine.vector_load %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ tile_s }}x{{ data_stype }}>
          %new_max_scalar = vector.extract %new_max[0] : {{ data_stype }} from vector<2x{{ data_stype }}>
          %new_max_bcast = vector.broadcast %new_max_scalar : {{ data_stype }} to vector<{{ tile_s }}x{{ data_stype }}>

          %shifted_scores = arith.subf %scaled_scores_reload, %new_max_bcast : vector<{{ tile_s }}x{{ data_stype }}>
          %exp_scores = math.exp %shifted_scores :  vector<{{ tile_s }}x{{ data_stype }}>
          affine.vector_store %exp_scores, %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ tile_s }}x{{ data_stype }}>


          // accumulate current sum
          %chunk_sum_res = affine.for %index5 = 0 to {{ tile_s }} step {{ chunk_size }} iter_args(%iter_sum=%v0_c) -> (vector<{{ chunk_size }}x{{ data_stype }}>) {
            %chunk_exp = affine.vector_load %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ chunk_size }}x{{ data_stype }}>
            %local_sum = arith.addf %chunk_exp, %iter_sum : vector<{{ chunk_size }}x{{ data_stype }}>
            affine.yield %local_sum : vector<{{ chunk_size }}x{{ data_stype }}>
          } { accumulation_loop=true }

          %zero_2x = vector.broadcast %c0 : {{ data_stype }} to vector<2x{{ data_stype }}>
          %sum_cast = vector.shape_cast %chunk_sum_res : vector<{{ chunk_size }}x{{ data_stype }}> to vector<{{ chunk_size // 2 }}x2x{{ data_stype }}>
          %sum_reduced_1 = vector.multi_reduction <add>, %sum_cast, %zero_2x [0] : vector<8x2x{{ data_stype }}> to vector<2x{{ data_stype }}>
          %sum_shuffled = vector.shuffle %sum_reduced_1, %sum_reduced_1 [1, 0] : vector<2x{{ data_stype }}>, vector<2x{{ data_stype }}>
          %sum_reduced_2 = arith.addf %sum_reduced_1, %sum_shuffled : vector<2x{{ data_stype }}>

          %new_sum = arith.addf %sum_reduced_2, %rescaled_sum :  vector<2x{{ data_stype }}>
          affine.vector_store %new_sum, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>


          // value.t @ mul
          linalg.matmul
            { idx_map = array<i32: 2, 1, -1> }
            ins(%vt_buffer2D, %mul_buffer : memref<{{ tile_e }}x{{ tile_s }}x{{ data_stype }}, 1>, {{ mul_tile_desc.get_mlir_shape(data_stype) }})
            outs(%ot_buffer2D : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>)
        } { accumulation_loop=true }

        // out @ row_sum^(-1)
        %final_row_sum = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>
        %one_2x = vector.broadcast %c1 : {{ data_stype }} to vector<2x{{ data_stype }}>

        %reciprocal_row_sum_2x = arith.divf %one_2x, %final_row_sum : vector<2x{{ data_stype }}>
        %reciprocal_scalar = vector.extract %reciprocal_row_sum_2x[0] : {{ data_stype }} from vector<2x{{ data_stype }}>
        %reciprocal_bcast_e = vector.broadcast %reciprocal_scalar : {{ data_stype }} to vector<{{ tile_e }}x{{ data_stype }}>

        %accumulated_out = affine.vector_load %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>, vector<{{ tile_e }}x{{ data_stype }}>
        %stable_final_out = arith.mulf %accumulated_out, %reciprocal_bcast_e : vector<{{ tile_e }}x{{ data_stype }}>
        affine.vector_store %stable_final_out, %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>, vector<{{ tile_e }}x{{ data_stype }}>

        %out_dram_offset = affine.apply {{ out_offset_map }}(%index0, %index1, %index3)
        {{ kernel.def_dma_op("MVOUT", "out", [], out_tile_desc, indent_size=8, dram_stride=out_dram_stride, dram_offset="out_dram_offset") }}
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

class MLIRFlashSDPATemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, scale, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.scale = scale

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):

        # Except for kernel, other arguments are usually None.
        query, key, value, out, q_tensor, k_tensor, v_tensor, out_tensor, b, l, s, e, ev, n_extra_node, n_prologue_node = self.extract_info(template_buffer_node, epilogue_nodes, prologue_nodes)

        if tile_info is None:
            tile_l, tile_s, tile_e, subtile_l, subtile_s, subtile_e = self.select_tile(kernel, l, s, e, n_extra_node, 0, n_prologue_node)[0]
        else:
            tile_l, tile_s, tile_e, subtile_l, subtile_s, subtile_e = tile_info

        TOG_latency = l if tile_l > l else tile_l
        kernel.loop_size = [TOG_latency, tile_s, tile_e]

        # Select template code
        # Other templates will be added according to situations.
        nr_reduction_nodes = [node for node in epilogue_nodes if node.is_reduction()] if epilogue_nodes is not None else []
        if nr_reduction_nodes:
            raise NotImplementedError("FLASH_SDPA_REDUCTION_TEMPLATE is not implemented yet.")
        elif prologue_nodes:
            raise NotImplementedError("FLASH_SDPA_PROLOGUE_TEMPLATE is not implemented yet.")
        else:
            template = FLASH_SDPA_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index1", "index2": "index2", "index3": "index3"}
            nr_rdim = 0

        # Prepare tile descriptors for input and output tensors.
        # Intermediate buffers (transient data) do not require DRAM settings(dram stride and dram indices)
        # as they are not synchronized with external DRAM.
        # DRAM and SRAM tile shapes must match.
        vlane_stride = 1

        # (n, l, s, e, ev)
        loop_dim = [sympy.Symbol("index0"), sympy.Symbol("index1"), sympy.Symbol("index2"), sympy.Symbol("index3")]


        # Hardware constraint: The tile split axis is restricted.
        # To accommodate this, we compute (key @ query.t) instead of (query @ key.t).
        # SRAM settings
        vlane_split_axis = 1
        q_tile_size = [1, tile_l, tile_e]
        q_tile_stride = [0, tile_e, 1]
        q_tile_desc = mlir_common.MLIRMultiDimTile(q_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        q_tile_desc.set_tile_size_stride(q_tile_size, q_tile_stride)
        q_tile_desc.set_name("q_buffer")
        q_tile_desc.offset = query.get_layout().offset
        # DRAM settings
        q_stride = q_tensor.stride()

        # Since we use a weight-stationary approach in the Systolic Array (SA),
        # the split axis of the first operand differs from a standard linear algebra matmul.
        # The first operand (key) must be split along the column axis.
        # This logic aligns with the relationship between the dot product's summation direction and the hardware's accumulation direction in the SA.
        # SRAM settings
        vlane_split_axis = 2
        k_tile_size = [1, tile_s, tile_e]
        k_tile_stride = [0, 1, tile_s]
        k_tile_desc = mlir_common.MLIRMultiDimTile(k_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        k_tile_desc.set_tile_size_stride(k_tile_size, k_tile_stride)
        k_tile_desc.set_name("k_buffer")
        k_tile_desc.offset = key.get_layout().offset
        # DRAM settings
        k_stride = k_tensor.stride()

        # Since we compute mul = key @ query.t, we perform out.t = (value.t @ Softmax(mul).t).t,
        # which simplifies to (value.t @ Softmax(mul))
        # SRAM settings
        vlane_split_axis = 1
        v_tile_size = [1, tile_s, tile_e]
        v_tile_stride = [0, tile_e, 1]
        v_tile_desc = mlir_common.MLIRMultiDimTile(v_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        v_tile_desc.set_tile_size_stride(v_tile_size, v_tile_stride)
        v_tile_desc.set_name("v_buffer")
        v_tile_desc.offset = value.get_layout().offset
        # DRAM settings
        v_stride = v_tensor.stride()

        # Output is also stored in transposed format to match the value.t @ Softmax(mul) operation.
        # SRAM settings
        vlane_split_axis = 1
        out_tile_size = [1, tile_l, tile_e]
        out_tile_stride=[0, tile_e, 1]
        out_tile_desc = mlir_common.MLIRMultiDimTile(out_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        out_tile_desc.set_tile_size_stride(out_tile_size, out_tile_stride)
        out_tile_desc.set_name("out_buffer")
        # DRAM settings
        out_stride = out.get_layout().stride[1:]

        # Intermediate buffers

        # For mul = key @ query.t
        vlane_split_axis = 1
        mul_tile_size = [tile_s, tile_l]
        mul_tile_stride = [tile_l, 1]
        mul_tile_desc = mlir_common.MLIRMultiDimTile(mul_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        mul_tile_desc.set_tile_size_stride(mul_tile_size, mul_tile_stride)
        mul_tile_desc.set_name("mul_buffer")
        #FIXME. What is the offset? -> It doesn't matter at this time.

        # For storing maximum values per row
        vlane_split_axis = 0
        max_size = [tile_l, 2]
        max_stride = [2, 1]
        max_desc = mlir_common.MLIRMultiDimTile(max_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        max_desc.set_tile_size_stride(max_size, max_stride)
        max_desc.set_name("max_buffer")

        # For storing summation per row
        vlane_split_axis = 0
        sum_size = [tile_l, 2]
        sum_stride = [2, 1]
        sum_desc = mlir_common.MLIRMultiDimTile(sum_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        sum_desc.set_tile_size_stride(sum_size, sum_stride)
        sum_desc.set_name("sum_buffer")

        # For reduction
        chunk_size = 16

        # DMA strides and offset affine maps (dram_stride + dram_offset style)
        q_dram_stride  = [int(q_stride[0]), int(q_stride[1]), int(q_stride[2])]
        k_dram_stride  = [int(k_stride[0]), int(k_stride[1]), int(k_stride[2])]
        v_dram_stride  = [int(v_stride[0]), int(v_stride[1]), int(v_stride[2])]
        out_dram_stride = [int(out_stride[0]), int(out_stride[1]), int(out_stride[2])]

        q_offset_map   = _make_offset_map(q_dram_stride,   q_tile_desc.offset)
        k_offset_map   = _make_offset_map(k_dram_stride,   k_tile_desc.offset)
        v_offset_map   = _make_offset_map(v_dram_stride,   v_tile_desc.offset)
        out_offset_map = _make_offset_map(out_dram_stride, 0)

        # Keep out_idx only for epilogue_info (not in render_options)
        out_idx = [loop_dim[0]*out_stride[0], loop_dim[1]*out_stride[1], loop_dim[3]*out_stride[2]]

        kernel.render_options = dict(
            KERNEL_NAME = self.name,
            kernel = kernel,
            b = b,
            l = l,
            s = s,
            e = e,                             # Input sizes (dram)
            tile_l = tile_l,
            tile_s = tile_s,
            tile_e = tile_e,                   # Tile sizes (sram)
            data_stype="f32",
            query = query,
            key = key,
            value = value,
            out = out,                         # Inputs and output (dram)
            q_dram_stride  = q_dram_stride,
            k_dram_stride  = k_dram_stride,
            v_dram_stride  = v_dram_stride,
            out_dram_stride = out_dram_stride, # Per-dim DRAM strides
            q_offset_map   = q_offset_map,
            k_offset_map   = k_offset_map,
            v_offset_map   = v_offset_map,
            out_offset_map = out_offset_map,   # Affine maps for base address
            q_tile_desc = q_tile_desc,
            k_tile_desc = k_tile_desc,
            v_tile_desc = v_tile_desc,
            mul_tile_desc = mul_tile_desc,
            out_tile_desc = out_tile_desc,     # Tile descriptions (sram)
            max_desc = max_desc,
            sum_desc = sum_desc,               # Intermediate buffer descriptions (sram)
            scale = self.scale,
            chunk_size = chunk_size,
            input_reorder = self.input_reorder # ETC
        )

        code = self._template_from_string(template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["l"], kernel.render_options["s"], kernel.render_options["e"]], [kernel.render_options["tile_l"], kernel.render_options["tile_s"], kernel.render_options["tile_e"]])
        return code

    def extract_info(self, template_buffer_node, epilogue_nodes, prologue_nodes):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node

        query = self.input_nodes[0]
        key = self.input_nodes[1]
        value = self.input_nodes[2]
        out = self.output_node

        q_tensor = empty_strided(query.layout.size, query.layout.stride)
        k_tensor = empty_strided(key.layout.size, key.layout.stride)
        v_tensor = empty_strided(value.layout.size, value.layout.stride)
        out_tensor = empty_strided(out.layout.size, out.layout.stride)

        # Flatten batch and head dimensions (n, h) into a single dimension (b = n*h)
        q_tensor = q_tensor.view([-1, q_tensor.shape[-2], q_tensor.shape[-1]])
        k_tensor = k_tensor.view([-1, k_tensor.shape[-2], k_tensor.shape[-1]])
        v_tensor = v_tensor.view([-1, v_tensor.shape[-2], v_tensor.shape[-1]])
        out_tensor = out_tensor.view([-1, out_tensor.shape[-2], out_tensor.shape[-1]])

        b, l, s, e, ev = q_tensor.size(0), q_tensor.size(1), k_tensor.size(1), k_tensor.size(2), v_tensor.size(2)

        n_extra_node = len(epilogue_nodes) if epilogue_nodes is not None else 0
        n_prologue_node = len(prologue_nodes) if prologue_nodes is not None else 0

        return query, key, value, out, q_tensor, k_tensor, v_tensor, out_tensor, b, l, s, e, ev, n_extra_node, n_prologue_node

    # Reuse the existing function in MLIRBMMTemplate.
    def select_tile(self, kernel, l, s, e, n_extra_node, n_extra_read, n_prologue_node):

        # FIXME: Update the method for getting tile candidates once TestDmaFineGrained oass works correctly with Flash Attention.
        # tile_candidates = kernel.flash_sdpa_mapping(l, s, e, n_extra_node=n_extra_node)
        tile_candidates = [[kernel.vector_lane, kernel.vector_lane, e]]

        for idx, (tile_l, tile_s, tile_e) in enumerate(tile_candidates):
            subtile_l = tile_l if (tile_l < kernel.vector_lane) or n_prologue_node else kernel.vector_lane
            subtile_s = tile_s # if (tile_s < kernel.vector_lane) or prologue_nodes else kernel.vector_lane
            subtile_e = tile_e # if (tile_e < kernel.vector_lane) or prologue_nodes else kernel.vector_lane

            tile_candidates[idx] = tile_l,tile_s,tile_e,subtile_l,subtile_s,subtile_e

        return tile_candidates

