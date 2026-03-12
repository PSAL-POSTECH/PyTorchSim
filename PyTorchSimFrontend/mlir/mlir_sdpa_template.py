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

    # Support head dimensions larger than vector lanes by tiling e/ev.
    # For now, require multiples of vector lanes (covers 64/128 with vlanes=16).
    vector_lane = extension_config.vpu_num_lanes
    if (e % vector_lane) != 0:
        raise NotImplementedError(
            f"Flash SDPA currently requires e to be a multiple of vlanes (e: {e}, vlanes: {vector_lane})."
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
// subtile_l = {{ subtile_l }}
// subtile_s = {{ subtile_s }}
// subtile_e = {{ subtile_e }}
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
        {{ kernel.def_dma_op("MVIN", "query", q_idx, q_tile_desc, subtile_size=[1, subtile_l, subtile_e], indent_size=8) }}  
        
        affine.vector_store %v0_l, %out_buffer[0, 0, 0] : {{ out_tile_desc.get_mlir_shape(data_stype) }}, vector<{{ kernel.get_spad_size_per_lane(tile_l, tile_e) }}x{{ data_stype }}>
        affine.vector_store %v_neg_inf_2x, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}> 
        affine.vector_store %v0_2x, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>
              
        %qt_buffer2D = memref.reinterpret_cast %q_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_l }}], strides: [{{ tile_l }}, 1] : {{ q_tile_desc.get_mlir_shape(data_stype) }} to memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>
        %ot_buffer2D = memref.reinterpret_cast %out_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_l }}], strides: [{{ tile_l }}, 1] : {{ out_tile_desc.get_mlir_shape(data_stype) }} to memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>

        affine.for %index2 = 0 to {{ s }} step {{ tile_s }} {
          {{ kernel.def_dma_op("MVIN", "key", k_idx, k_tile_desc, subtile_size=[1, subtile_s, subtile_e], indent_size=10) }} 
          {{ kernel.def_dma_op("MVIN", "value", v_idx, v_tile_desc, subtile_size=[1, subtile_s, subtile_e], indent_size=10) }}

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
          }

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
          }
          
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
        }

        // out @ row_sum^(-1)
        %final_row_sum = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(data_stype) }}, vector<2x{{ data_stype }}>
        %one_2x = vector.broadcast %c1 : {{ data_stype }} to vector<2x{{ data_stype }}>
        
        %reciprocal_row_sum_2x = arith.divf %one_2x, %final_row_sum : vector<2x{{ data_stype }}>
        %reciprocal_scalar = vector.extract %reciprocal_row_sum_2x[0] : {{ data_stype }} from vector<2x{{ data_stype }}>
        %reciprocal_bcast_e = vector.broadcast %reciprocal_scalar : {{ data_stype }} to vector<{{ tile_e }}x{{ data_stype }}>
        
        %accumulated_out = affine.vector_load %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>, vector<{{ tile_e }}x{{ data_stype }}>
        %stable_final_out = arith.mulf %accumulated_out, %reciprocal_bcast_e : vector<{{ tile_e }}x{{ data_stype }}>
        affine.vector_store %stable_final_out, %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ data_stype }}, 1>, vector<{{ tile_e }}x{{ data_stype }}>

        {{ kernel.store_output(indent_size=8) }}
      } { accumulation_loop=true } 
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
        q_idx = [loop_dim[0]*q_stride[0], loop_dim[1]*q_stride[1], loop_dim[3]*q_stride[2]] # To keep index arguemnt order, we used index_list

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
        k_idx = [loop_dim[0]*k_stride[0], loop_dim[2]*k_stride[1], loop_dim[3]*k_stride[2]]

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
        v_idx = [loop_dim[0]*v_stride[0], loop_dim[2]*v_stride[1], loop_dim[3]*v_stride[2]] # To keep index arguemnt order, we used index_list

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
        out_idx = [loop_dim[0]*out_stride[0], loop_dim[1]*out_stride[1], loop_dim[3]*out_stride[2]]

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
            subtile_l = subtile_l, 
            subtile_s = subtile_s, 
            subtile_e = subtile_e,             # Subtile sizes (sram)  
            data_stype="f32",
            query = query, 
            key = key,
            value = value, 
            out = out,                         # Inputs and output (dram)
            q_idx = q_idx,
            k_idx = k_idx,
            v_idx = v_idx,
            out_idx = out_idx,                 # Strides (dram)       
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

        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            sram_var = "out_buffer",
            dram_var = "out",
            dram_idx = out_idx,
            dram_tile_desc = out_tile_desc,
            nr_rdim = nr_rdim,
            r_dim_size = 0,
            dim_aliasing = epilogue_dim_aliasing
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


# ---------------------------
# Decode-only GQA SDPA (Lq == 1)
# ---------------------------

DECODE_GQA_SDPA_TEMPLATE = r"""
// Decode GQA SDPA kernel (Lq == 1)
// B = {{ B }}
// Hq = {{ Hq }}
// H = {{ H }}
// g = {{ g }}
// S = {{ S }}
// Dh = {{ Dh }}
// BlkS = {{ BlkS }}
// tile_s = {{ tile_s }}
// tile_e = {{ tile_e }}
// dh_tiles = {{ dh_tiles }}
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[query, key, value], outputs=[out], names_str="query, key, value, out", input_reorder=input_reorder)}} {
  // IO buffers follow input dtype (fp16/bf16/f32)
  {{ kernel.def_sram_buffer("query", q_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("key", k_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("value", v_tile_desc, indent_size=2) }}
  // Softmax output used for SV matmul (io dtype)
  {{ kernel.def_sram_buffer("mul", mul_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("score", score_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("prob", prob_desc, indent_size=2) }}
  // Accumulator in fp32 (stable)
  {{ kernel.def_sram_buffer("out_acc", out_acc_tile_desc, indent_size=2) }}
  // Temp output in io dtype for SV matmul result
  {{ kernel.def_sram_buffer("out_io", out_io_tile_desc, indent_size=2) }}
  // Softmax running stats in fp32
  {{ kernel.def_sram_buffer("max", max_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("sum", sum_desc, indent_size=2) }}

  %c0 = arith.constant 0.0 : {{ acc_stype }}
  %c1 = arith.constant 1.0 : {{ acc_stype }}
  %c_scale = arith.constant {{ scale }} : {{ acc_stype }}
  %c_neg_inf = arith.constant -1.0e+30 : {{ acc_stype }}

  %v0_e_acc = arith.constant dense<0.0> : vector<{{ tile_e }}x{{ acc_stype }}>
  %v0_e_io = arith.constant dense<0.0> : vector<{{ tile_e }}x{{ io_stype }}>
  %v0_2x = arith.constant dense<0.0> : vector<2x{{ acc_stype }}>
  %v_neg_inf_2x = arith.constant dense<-1.0e+30> : vector<2x{{ acc_stype }}>
  %v0_s_acc = arith.constant dense<0.0> : vector<{{ tile_s }}x{{ acc_stype }}>

  %v_scale = vector.broadcast %c_scale : {{ acc_stype }} to vector<{{ tile_s }}x{{ acc_stype }}>

  {{ kernel.def_local_vars(indent_size=2) }}

  // kv_head parallelism is the natural unit for GQA reuse
  affine.for %kv = 0 to {{ H }} {
    // Process S in blocks (BlkS). Sequential inside a core.
    affine.for %blk = 0 to {{ S }} step {{ BlkS }} {
      // Initialize per-qsub accumulators for this (kv, blk)
      affine.for %qsub = 0 to {{ g }} {
        affine.vector_store %v_neg_inf_2x, %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
        affine.vector_store %v0_2x, %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
        affine.for %dht = 0 to {{ dh_tiles }} {
          affine.vector_store %v0_e_acc, %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_e }}x{{ acc_stype }}>
        }
      }

      affine.for %s0 = %blk to (%blk + {{ BlkS }}) step {{ tile_s }} {
        // Accumulate score per qsub so K tiles can be shared across qsub.
        affine.for %qsub = 0 to {{ g }} {
          affine.vector_store %v0_s_acc, %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_s }}x{{ acc_stype }}>
        }

        affine.for %k0 = 0 to {{ Dh }} step {{ tile_e }} {
          // Load K slice once for all qsub.
          {{ kernel.def_dma_op("MVIN", "key", kk_idx, k_tile_desc, subtile_size=[1, tile_s, tile_e], indent_size=10, padding=1) }}
          %k2D = memref.reinterpret_cast %k_buffer to offset: [0], sizes: [{{ tile_s }}, {{ tile_e }}], strides: [{{ tile_e }}, 1] : {{ k_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_s }}x{{ tile_e }}x{{ io_stype }}, 1>

          affine.for %qsub = 0 to {{ g }} {
            {{ kernel.def_dma_op("MVIN", "query", qk_idx, q_tile_desc, subtile_size=[1, 1, tile_e], indent_size=12) }}
            %q2D = memref.reinterpret_cast %q_buffer to offset: [0], sizes: [{{ tile_e }}, 1], strides: [1, 1] : {{ q_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x1x{{ io_stype }}, 1>

            // mul = k @ q  -> (tile_s x 1) in io dtype, then upcast and accumulate.
            linalg.matmul
              { idx_map = array<i32: 1, 0, -1> }
              ins(%k2D, %q2D : memref<{{ tile_s }}x{{ tile_e }}x{{ io_stype }}, 1>, memref<{{ tile_e }}x1x{{ io_stype }}, 1>)
              outs(%mul_buffer : {{ mul_tile_desc.get_mlir_shape(io_stype) }})

            %raw_mul_io = affine.vector_load %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>
            %raw_mul = arith.extf %raw_mul_io : vector<{{ tile_s }}x{{ io_stype }}> to vector<{{ tile_s }}x{{ acc_stype }}>
            %old_score = affine.vector_load %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_s }}x{{ acc_stype }}>
            %new_score = arith.addf %old_score, %raw_mul : vector<{{ tile_s }}x{{ acc_stype }}>
            affine.vector_store %new_score, %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_s }}x{{ acc_stype }}>
          } { accumulation_loop=true }
        } { accumulation_loop=true }

        affine.for %qsub = 0 to {{ g }} {
          %score_acc = affine.vector_load %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_s }}x{{ acc_stype }}>
          // scale after full Dh reduction
          %scaled_mul_vec = arith.mulf %score_acc, %v_scale : vector<{{ tile_s }}x{{ acc_stype }}>

            // Online softmax update (max/sum/out) identical to FLASH_SDPA_TEMPLATE but specialized to Lq==1.
            %old_max = affine.vector_load %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
            // Reduce max over tile_s
            %max_init = vector.broadcast %c_neg_inf : {{ acc_stype }} to vector<{{ tile_s }}x{{ acc_stype }}>
            %local_max_vec = arith.maximumf %scaled_mul_vec, %max_init : vector<{{ tile_s }}x{{ acc_stype }}>
            %max_cast = vector.shape_cast %local_max_vec : vector<{{ tile_s }}x{{ acc_stype }}> to vector<{{ tile_s // 2 }}x2x{{ acc_stype }}>
            %max_red1 = vector.multi_reduction <maximumf>, %max_cast, %v_neg_inf_2x [0] : vector<{{ tile_s // 2 }}x2x{{ acc_stype }}> to vector<2x{{ acc_stype }}>
            %max_shuf = vector.shuffle %max_red1, %max_red1 [1, 0] : vector<2x{{ acc_stype }}>, vector<2x{{ acc_stype }}>
            %max_red2 = arith.maximumf %max_red1, %max_shuf : vector<2x{{ acc_stype }}>
            %new_max = arith.maximumf %max_red2, %old_max : vector<2x{{ acc_stype }}>
            affine.vector_store %new_max, %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>

            // rescale = exp(old_max - new_max)
            %max_diff = arith.subf %old_max, %new_max : vector<2x{{ acc_stype }}>
            %max_diff_scalar = vector.extract %max_diff[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>
            %rescale_e = vector.broadcast %max_diff_scalar : {{ acc_stype }} to vector<{{ tile_e }}x{{ acc_stype }}>
            %exp_rescale_e = math.exp %rescale_e : vector<{{ tile_e }}x{{ acc_stype }}>
            %rescale_2 = vector.broadcast %max_diff_scalar : {{ acc_stype }} to vector<2x{{ acc_stype }}>
            %exp_rescale_2 = math.exp %rescale_2 : vector<2x{{ acc_stype }}>

            // out *= rescale
            %old_out = affine.vector_load %out_acc_buffer[%qsub, 0, 0] : {{ out_acc_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_e }}x{{ acc_stype }}>
            %rescaled_out = arith.mulf %exp_rescale_e, %old_out : vector<{{ tile_e }}x{{ acc_stype }}>
            affine.vector_store %rescaled_out, %out_acc_buffer[%qsub, 0, 0] : {{ out_acc_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_e }}x{{ acc_stype }}>

            // sum *= rescale
            %old_sum = affine.vector_load %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
            %rescaled_sum = arith.mulf %old_sum, %exp_rescale_2 : vector<2x{{ acc_stype }}>

            // exp(score - new_max)
            %new_max_scalar = vector.extract %new_max[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>
            %new_max_bcast = vector.broadcast %new_max_scalar : {{ acc_stype }} to vector<{{ tile_s }}x{{ acc_stype }}>
            %shifted = arith.subf %scaled_mul_vec, %new_max_bcast : vector<{{ tile_s }}x{{ acc_stype }}>
            %exp_scores = math.exp %shifted : vector<{{ tile_s }}x{{ acc_stype }}>
            // For SV matmul: downcast softmax output to io dtype (common in practice)
            %exp_scores_io = arith.truncf %exp_scores : vector<{{ tile_s }}x{{ acc_stype }}> to vector<{{ tile_s }}x{{ io_stype }}>
            affine.vector_store %exp_scores_io, %prob_buffer[%qsub, 0] : {{ prob_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>

            // sum += reduce(exp_scores)
            %sum_cast = vector.shape_cast %exp_scores : vector<{{ tile_s }}x{{ acc_stype }}> to vector<{{ tile_s // 2 }}x2x{{ acc_stype }}>
            %zero_2x = vector.broadcast %c0 : {{ acc_stype }} to vector<2x{{ acc_stype }}>
            %sum_red1 = vector.multi_reduction <add>, %sum_cast, %zero_2x [0] : vector<{{ tile_s // 2 }}x2x{{ acc_stype }}> to vector<2x{{ acc_stype }}>
            %sum_shuf = vector.shuffle %sum_red1, %sum_red1 [1, 0] : vector<2x{{ acc_stype }}>, vector<2x{{ acc_stype }}>
            %sum_red2 = arith.addf %sum_red1, %sum_shuf : vector<2x{{ acc_stype }}>
            %new_sum = arith.addf %sum_red2, %rescaled_sum : vector<2x{{ acc_stype }}>
            affine.vector_store %new_sum, %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>

        } { accumulation_loop=true }

        // 2) SV accumulation: for each output dh tile, load V once and share across qsub.
        affine.for %dht = 0 to {{ dh_tiles }} {
          %dh0 = affine.apply affine_map<(d0) -> (d0 * {{ tile_e }})>(%dht)
          {{ kernel.def_dma_op("MVIN", "value", v_idx, v_tile_desc, subtile_size=[1, tile_s, tile_e], indent_size=10, padding=0) }}
          %v2D = memref.reinterpret_cast %v_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_s }}], strides: [{{ tile_s }}, 1] : {{ v_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x{{ tile_s }}x{{ io_stype }}, 1>

          affine.for %qsub = 0 to {{ g }} {
            %prob_vec = affine.vector_load %prob_buffer[%qsub, 0] : {{ prob_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>
            affine.vector_store %prob_vec, %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>
            affine.vector_store %v0_e_io, %out_io_buffer[0, 0, 0] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_e }}x{{ io_stype }}>
            %out_io_2D = memref.reinterpret_cast %out_io_buffer to offset: [0], sizes: [{{ tile_e }}, 1], strides: [1, 1] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x1x{{ io_stype }}, 1>
            linalg.matmul
              { idx_map = array<i32: 2, 1, -1> }
              ins(%v2D, %mul_buffer : memref<{{ tile_e }}x{{ tile_s }}x{{ io_stype }}, 1>, {{ mul_tile_desc.get_mlir_shape(io_stype) }})
              outs(%out_io_2D : memref<{{ tile_e }}x1x{{ io_stype }}, 1>)

            %out_io_vec = affine.vector_load %out_io_buffer[0, 0, 0] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_e }}x{{ io_stype }}>
            %out_io_f32 = arith.extf %out_io_vec : vector<{{ tile_e }}x{{ io_stype }}> to vector<{{ tile_e }}x{{ acc_stype }}>
            %out_acc_vec = affine.vector_load %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_e }}x{{ acc_stype }}>
            %out_acc_new = arith.addf %out_acc_vec, %out_io_f32 : vector<{{ tile_e }}x{{ acc_stype }}>
            affine.vector_store %out_acc_new, %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_e }}x{{ acc_stype }}>
          } { accumulation_loop=true }
        } { accumulation_loop=true }
      } { accumulation_loop=true }

      // finalize per-qsub for this (kv, blk) and store out for all dh tiles
      affine.for %qsub = 0 to {{ g }} {
        %final_sum = affine.vector_load %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
        %one_2x = vector.broadcast %c1 : {{ acc_stype }} to vector<2x{{ acc_stype }}>
        %inv_sum_2x = arith.divf %one_2x, %final_sum : vector<2x{{ acc_stype }}>
        %inv_sum = vector.extract %inv_sum_2x[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>
        %inv_bcast = vector.broadcast %inv_sum : {{ acc_stype }} to vector<{{ tile_e }}x{{ acc_stype }}>

        affine.for %dht = 0 to {{ dh_tiles }} {
          %dh0 = affine.apply affine_map<(d0) -> (d0 * {{ tile_e }})>(%dht)
          %acc_out = affine.vector_load %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ tile_e }}x{{ acc_stype }}>
          %final_out_acc = arith.mulf %acc_out, %inv_bcast : vector<{{ tile_e }}x{{ acc_stype }}>
          %final_out_io = arith.truncf %final_out_acc : vector<{{ tile_e }}x{{ acc_stype }}> to vector<{{ tile_e }}x{{ io_stype }}>
          affine.vector_store %final_out_io, %out_io_buffer[0, 0, 0] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_e }}x{{ io_stype }}>
          {{ kernel.store_output(indent_size=10) }}
        }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }

  return
}
"""


class MLIRDecodeGQASDPATemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, scale, BlkS: int = 1024, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.scale = scale
        self.BlkS = BlkS

    def render(self, kernel: MLIRTemplateKernel, template_buffer_node=None, epilogue_nodes=None, prologue_nodes=None, tile_info=None, **kwargs):
        # Decode-only: q is (B,Hq,1,Dh)
        query, key, value, out = self.input_nodes[0], self.input_nodes[1], self.input_nodes[2], self.output_node

        # Materialize tensors for stride metadata
        q_tensor4 = empty_strided(query.layout.size, query.layout.stride)
        k_tensor4 = empty_strided(key.layout.size, key.layout.stride)
        v_tensor4 = empty_strided(value.layout.size, value.layout.stride)

        B, Hq, Lq, Dh = q_tensor4.shape
        Bk, H, S, Dhk = k_tensor4.shape
        assert B == 1, "Decode GQA template currently supports B==1"
        assert Lq == 1, "Decode GQA template requires Lq==1"
        assert Dh == Dhk
        g = Hq // H
        BlkS = min(int(self.BlkS), int(S))

        # Use 3D views to match the existing SDPA indexing scheme
        # q: (Hq, 1, Dh), k/v: (H, S, Dh), out: (Hq, 1, Dh)
        q_tensor = q_tensor4.view(Hq, 1, Dh)
        k_tensor = k_tensor4.view(H, S, Dh)
        v_tensor = v_tensor4.view(H, S, Dh)

        tile_s = kernel.vector_lane
        tile_e = kernel.vector_lane
        dh_tiles = int(Dh) // int(tile_e)

        io_stype = mlir_common.DTYPE_TO_MLIR[query.get_dtype()]
        acc_stype = "f32"

        # SRAM tiles: q(1x1xtile_e), k/v(1xtile_sxtile_e), mul(tile_sx1) in io dtype.
        # out_acc in f32; out_io temp in io dtype.
        vlane_stride = 1
        q_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_e], kernel.vector_lane, 1, vlane_stride)
        q_tile_desc.set_tile_size_stride([1, 1, tile_e], [0, tile_e, 1])
        q_tile_desc.set_name("q_buffer")
        q_tile_desc.offset = query.get_layout().offset

        k_tile_desc = mlir_common.MLIRMultiDimTile([1, tile_s, tile_e], kernel.vector_lane, 2, vlane_stride)
        k_tile_desc.set_tile_size_stride([1, tile_s, tile_e], [0, 1, tile_s])
        k_tile_desc.set_name("k_buffer")
        k_tile_desc.offset = key.get_layout().offset

        v_tile_desc = mlir_common.MLIRMultiDimTile([1, tile_s, tile_e], kernel.vector_lane, 1, vlane_stride)
        v_tile_desc.set_tile_size_stride([1, tile_s, tile_e], [0, tile_e, 1])
        v_tile_desc.set_name("v_buffer")
        v_tile_desc.offset = value.get_layout().offset

        mul_tile_desc = mlir_common.MLIRMultiDimTile([tile_s, 1], kernel.vector_lane, 1, vlane_stride)
        mul_tile_desc.set_tile_size_stride([tile_s, 1], [1, 1])
        mul_tile_desc.set_name("mul_buffer")

        score_desc = mlir_common.MLIRMultiDimTile([g, tile_s], kernel.vector_lane, 1, vlane_stride)
        score_desc.set_tile_size_stride([g, tile_s], [tile_s, 1])
        score_desc.set_name("score_buffer")

        prob_desc = mlir_common.MLIRMultiDimTile([g, tile_s], kernel.vector_lane, 1, vlane_stride)
        prob_desc.set_tile_size_stride([g, tile_s], [tile_s, 1])
        prob_desc.set_name("prob_buffer")

        # Per-qsub accumulators so KV tiles can be shared across qsub
        out_acc_tile_desc = mlir_common.MLIRMultiDimTile([g, dh_tiles, tile_e], kernel.vector_lane, 2, vlane_stride)
        out_acc_tile_desc.set_tile_size_stride([g, dh_tiles, tile_e], [dh_tiles * tile_e, tile_e, 1])
        out_acc_tile_desc.set_name("out_acc_buffer")

        out_io_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_e], kernel.vector_lane, 1, vlane_stride)
        out_io_tile_desc.set_tile_size_stride([1, 1, tile_e], [0, tile_e, 1])
        out_io_tile_desc.set_name("out_io_buffer")

        max_desc = mlir_common.MLIRMultiDimTile([g, 2], kernel.vector_lane, 0, vlane_stride)
        max_desc.set_tile_size_stride([g, 2], [2, 1])
        max_desc.set_name("max_buffer")

        sum_desc = mlir_common.MLIRMultiDimTile([g, 2], kernel.vector_lane, 0, vlane_stride)
        sum_desc.set_tile_size_stride([g, 2], [2, 1])
        sum_desc.set_name("sum_buffer")

        # Indices
        kv = sympy.Symbol("kv")
        qsub = sympy.Symbol("qsub")
        dh0 = sympy.Symbol("dh0")
        k0 = sympy.Symbol("k0")
        s0 = sympy.Symbol("s0")
        q_head = kv * g + qsub

        q_stride = q_tensor.stride()
        k_stride = k_tensor.stride()
        v_stride = v_tensor.stride()
        # out is (B,Hq,1,Dh) but we address it as (Hq,1,Dh)
        out_tensor = empty_strided(out.get_layout().size, out.get_layout().stride).view(Hq, 1, Dh)
        out_stride = out_tensor.stride()

        # QK indices use k0 reduction over Dh
        qk_idx = [q_head * q_stride[0], sympy.Integer(0), k0 * q_stride[2]]
        kk_idx = [kv * k_stride[0], s0 * k_stride[1], k0 * k_stride[2]]
        # V and output use dh0 tile offset
        v_idx = [kv * v_stride[0], s0 * v_stride[1], dh0 * v_stride[2]]
        out_idx = [q_head * out_stride[0], sympy.Integer(0), dh0 * out_stride[2]]

        kernel.loop_size = [tile_s, tile_e, 1]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            B=B,
            Hq=Hq,
            H=H,
            g=g,
            S=S,
            Dh=Dh,
            dh_tiles=dh_tiles,
            BlkS=BlkS,
            tile_s=tile_s,
            tile_e=tile_e,
            io_stype=io_stype,
            acc_stype=acc_stype,
            scale=self.scale,
            query=query,
            key=key,
            value=value,
            out=out,
            q_tile_desc=q_tile_desc,
            k_tile_desc=k_tile_desc,
            v_tile_desc=v_tile_desc,
            out_acc_tile_desc=out_acc_tile_desc,
            out_io_tile_desc=out_io_tile_desc,
            mul_tile_desc=mul_tile_desc,
            score_desc=score_desc,
            prob_desc=prob_desc,
            max_desc=max_desc,
            sum_desc=sum_desc,
            qk_idx=qk_idx,
            kk_idx=kk_idx,
            v_idx=v_idx,
            out_idx=out_idx,
            input_reorder=self.input_reorder,
        )

        kernel.epilogue_info = dict(
            output_node=self.output_node.name,
            sram_var="out_io_buffer",
            dram_var="out",
            dram_idx=out_idx,
            dram_tile_desc=out_io_tile_desc,
            nr_rdim=0,
            r_dim_size=0,
            dim_aliasing={"kv": "kv", "qsub": "qsub", "dh0": "dh0", "s0": "s0"},
        )

        return self._template_from_string(DECODE_GQA_SDPA_TEMPLATE).render(**kernel.render_options)


# ---------------------------
# Decode-only GQA SDPA: 2-kernel pipeline (partial blocks + reduce)
# ---------------------------

DECODE_GQA_SDPA_PARTIAL_TEMPLATE = r"""
// Decode GQA SDPA partial kernel (per sequence block)
// Produces partials per (kv,qsub,dh_tile,blk):
// - first half lanes: o_j (tile_e)
// - second half lanes: [m_j, l_j, 0, 0, ...] (tile_e)
// QK/softmax is computed once per (kv,qsub,s0) over full Dh using k0 reduction.
// SV then reuses those probabilities across all dh tiles.
// H = {{ H }}, g = {{ g }}, Dh = {{ Dh }}, dh_tiles = {{ dh_tiles }}, S = {{ S }}, BlkS = {{ BlkS }}, nblk = {{ nblk }}
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[query, key, value], outputs=[partial], names_str="query, key, value, partial", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("query", q_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("key", k_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("value", v_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("mul", mul_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("score", score_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("prob", prob_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("out_io", out_io_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("max", max_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("sum", sum_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("out_acc", out_acc_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("partial", partial_tile_desc, indent_size=2) }}

  %c0 = arith.constant 0.0 : f32
  %c_scale = arith.constant {{ scale }} : f32
  %c_neg_inf = arith.constant -1.0e+30 : f32

  %v0_e = arith.constant dense<0.0> : vector<{{ tile_e }}xf32>
  %v0_e_io = arith.constant dense<0.0> : vector<{{ tile_e }}x{{ io_stype }}>
  %v0_s = arith.constant dense<0.0> : vector<{{ tile_s }}xf32>
  %v0_2x = arith.constant dense<0.0> : vector<2xf32>
  %v_neg_inf_2x = arith.constant dense<-1.0e+30> : vector<2xf32>
  %v_scale = vector.broadcast %c_scale : f32 to vector<{{ tile_s }}xf32>

  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %kv = 0 to {{ H }} {
    affine.for %blk = 0 to {{ nblk }} step 1 {
      // Reset per-block accumulators for all qsub/dh tiles.
      affine.for %qsub = 0 to {{ g }} {
        affine.vector_store %v_neg_inf_2x, %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>
        affine.vector_store %v0_2x, %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
        affine.for %dht = 0 to {{ dh_tiles }} {
          affine.vector_store %v0_e, %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
        }
      }

      affine.for %s0 = ({{ BlkS }} * %blk) to ({{ BlkS }} * (%blk + 1)) step {{ tile_s }} {
        // Accumulate score per qsub so K tiles can be shared across qsub.
        affine.for %qsub = 0 to {{ g }} {
          affine.vector_store %v0_s, %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape("f32") }}, vector<{{ tile_s }}xf32>
        }

        affine.for %k0 = 0 to {{ Dh }} step {{ tile_e }} {
          {{ kernel.def_dma_op("MVIN", "key", kk_idx, k_tile_desc, subtile_size=[1, tile_s, tile_e], indent_size=10, padding=1) }}
          %k2D = memref.reinterpret_cast %k_buffer to offset: [0], sizes: [{{ tile_s }}, {{ tile_e }}], strides: [{{ tile_e }}, 1] : {{ k_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_s }}x{{ tile_e }}x{{ io_stype }}, 1>

          affine.for %qsub = 0 to {{ g }} {
            {{ kernel.def_dma_op("MVIN", "query", qk_idx, q_tile_desc, subtile_size=[1, 1, tile_e], indent_size=12) }}
            %q2D = memref.reinterpret_cast %q_buffer to offset: [0], sizes: [{{ tile_e }}, 1], strides: [1, 1] : {{ q_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x1x{{ io_stype }}, 1>
            linalg.matmul
              { idx_map = array<i32: 1, 0, -1> }
              ins(%k2D, %q2D : memref<{{ tile_s }}x{{ tile_e }}x{{ io_stype }}, 1>, memref<{{ tile_e }}x1x{{ io_stype }}, 1>)
              outs(%mul_buffer : {{ mul_tile_desc.get_mlir_shape(io_stype) }})
            %raw_mul_io = affine.vector_load %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>
            %raw_mul = arith.extf %raw_mul_io : vector<{{ tile_s }}x{{ io_stype }}> to vector<{{ tile_s }}xf32>
            %old_score = affine.vector_load %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape("f32") }}, vector<{{ tile_s }}xf32>
            %new_score = arith.addf %old_score, %raw_mul : vector<{{ tile_s }}xf32>
            affine.vector_store %new_score, %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape("f32") }}, vector<{{ tile_s }}xf32>
          } { accumulation_loop=true }
        } { accumulation_loop=true }

        // Softmax once per qsub; persist probabilities in SRAM for all SV dh tiles.
        affine.for %qsub = 0 to {{ g }} {
          %score = affine.vector_load %score_buffer[%qsub, 0] : {{ score_desc.get_mlir_shape("f32") }}, vector<{{ tile_s }}xf32>
          %scaled = arith.mulf %score, %v_scale : vector<{{ tile_s }}xf32>

          %old_max = affine.vector_load %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>
          %max_init = vector.broadcast %c_neg_inf : f32 to vector<{{ tile_s }}xf32>
          %local_max_vec = arith.maximumf %scaled, %max_init : vector<{{ tile_s }}xf32>
          %max_cast = vector.shape_cast %local_max_vec : vector<{{ tile_s }}xf32> to vector<{{ tile_s // 2 }}x2xf32>
          %max_red1 = vector.multi_reduction <maximumf>, %max_cast, %v_neg_inf_2x [0] : vector<{{ tile_s // 2 }}x2xf32> to vector<2xf32>
          %max_shuf = vector.shuffle %max_red1, %max_red1 [1, 0] : vector<2xf32>, vector<2xf32>
          %max_red2 = arith.maximumf %max_red1, %max_shuf : vector<2xf32>
          %new_max = arith.maximumf %max_red2, %old_max : vector<2xf32>
          affine.vector_store %new_max, %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>

          %max_diff = arith.subf %old_max, %new_max : vector<2xf32>
          %max_diff_scalar = vector.extract %max_diff[0] : f32 from vector<2xf32>
          %rescale_e = vector.broadcast %max_diff_scalar : f32 to vector<{{ tile_e }}xf32>
          %exp_rescale_e = math.exp %rescale_e : vector<{{ tile_e }}xf32>
          %rescale_2 = vector.broadcast %max_diff_scalar : f32 to vector<2xf32>
          %exp_rescale_2 = math.exp %rescale_2 : vector<2xf32>

          %old_sum = affine.vector_load %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
          %rescaled_sum = arith.mulf %old_sum, %exp_rescale_2 : vector<2xf32>

          affine.for %dht = 0 to {{ dh_tiles }} {
            %old_out = affine.vector_load %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
            %rescaled_out = arith.mulf %exp_rescale_e, %old_out : vector<{{ tile_e }}xf32>
            affine.vector_store %rescaled_out, %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
          }

          %new_max_scalar = vector.extract %new_max[0] : f32 from vector<2xf32>
          %new_max_bcast = vector.broadcast %new_max_scalar : f32 to vector<{{ tile_s }}xf32>
          %shifted = arith.subf %scaled, %new_max_bcast : vector<{{ tile_s }}xf32>
          %exp_scores = math.exp %shifted : vector<{{ tile_s }}xf32>
          %exp_scores_io = arith.truncf %exp_scores : vector<{{ tile_s }}xf32> to vector<{{ tile_s }}x{{ io_stype }}>
          affine.vector_store %exp_scores_io, %prob_buffer[%qsub, 0] : {{ prob_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>

          %sum_cast = vector.shape_cast %exp_scores : vector<{{ tile_s }}xf32> to vector<{{ tile_s // 2 }}x2xf32>
          %zero_2x = vector.broadcast %c0 : f32 to vector<2xf32>
          %sum_red1 = vector.multi_reduction <add>, %sum_cast, %zero_2x [0] : vector<{{ tile_s // 2 }}x2xf32> to vector<2xf32>
          %sum_shuf = vector.shuffle %sum_red1, %sum_red1 [1, 0] : vector<2xf32>, vector<2xf32>
          %sum_red2 = arith.addf %sum_red1, %sum_shuf : vector<2xf32>
          %new_sum = arith.addf %sum_red2, %rescaled_sum : vector<2xf32>
          affine.vector_store %new_sum, %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
        } { accumulation_loop=true }

        // For each output dh tile, load V once and share it across qsub.
        affine.for %dht = 0 to {{ dh_tiles }} {
          %dh0 = affine.apply affine_map<(d0) -> (d0 * {{ tile_e }})>(%dht)
          {{ kernel.def_dma_op("MVIN", "value", v_idx, v_tile_desc, subtile_size=[1, tile_s, tile_e], indent_size=10, padding=0) }}
          %v2D = memref.reinterpret_cast %v_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_s }}], strides: [{{ tile_s }}, 1] : {{ v_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x{{ tile_s }}x{{ io_stype }}, 1>

          affine.for %qsub = 0 to {{ g }} {
            %prob_vec = affine.vector_load %prob_buffer[%qsub, 0] : {{ prob_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>
            affine.vector_store %prob_vec, %mul_buffer[0, 0] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_s }}x{{ io_stype }}>
            affine.vector_store %v0_e_io, %out_io_buffer[0, 0, 0] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_e }}x{{ io_stype }}>
            %out_io_2D = memref.reinterpret_cast %out_io_buffer to offset: [0], sizes: [{{ tile_e }}, 1], strides: [1, 1] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x1x{{ io_stype }}, 1>
            linalg.matmul
              { idx_map = array<i32: 2, 1, -1> }
              ins(%v2D, %mul_buffer : memref<{{ tile_e }}x{{ tile_s }}x{{ io_stype }}, 1>, {{ mul_tile_desc.get_mlir_shape(io_stype) }})
              outs(%out_io_2D : memref<{{ tile_e }}x1x{{ io_stype }}, 1>)

            %out_io_vec = affine.vector_load %out_io_buffer[0, 0, 0] : {{ out_io_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_e }}x{{ io_stype }}>
            %out_io_f32 = arith.extf %out_io_vec : vector<{{ tile_e }}x{{ io_stype }}> to vector<{{ tile_e }}xf32>
            %out_acc_vec = affine.vector_load %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
            %out_acc_new = arith.addf %out_acc_vec, %out_io_f32 : vector<{{ tile_e }}xf32>
            affine.vector_store %out_acc_new, %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
          } { accumulation_loop=true }
        } { accumulation_loop=true }
      } { accumulation_loop=true }

      // Store packed partials for all qsub/dh tiles.
      affine.for %qsub = 0 to {{ g }} {
        %final_max = affine.vector_load %max_buffer[%qsub, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>
        %m_scalar = vector.extract %final_max[0] : f32 from vector<2xf32>
        %final_sum = affine.vector_load %sum_buffer[%qsub, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
        %l_scalar = vector.extract %final_sum[0] : f32 from vector<2xf32>
        %ml_vec = vector.broadcast %c0 : f32 to vector<{{ tile_e }}xf32>
        %ml0 = vector.insert %m_scalar, %ml_vec[0] : f32 into vector<{{ tile_e }}xf32>
        %ml1 = vector.insert %l_scalar, %ml0[1] : f32 into vector<{{ tile_e }}xf32>

        affine.for %dht = 0 to {{ dh_tiles }} {
          %out_vec = affine.vector_load %out_acc_buffer[%qsub, %dht, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
          %packed = vector.concat %out_vec, %ml1 : vector<{{ tile_pack }}xf32>
          affine.vector_store %packed, %partial_buffer[0, 0, 0] : {{ partial_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_pack }}xf32>
          {{ kernel.store_output(indent_size=10) }}
        }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""


DECODE_GQA_SDPA_REDUCE_TEMPLATE = r"""
// Decode GQA SDPA reduce kernel: merge partials across blocks
// Input partial shape: (HgDhTiles, nblk, tile_pack)
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[partial], outputs=[out], names_str="partial, out", input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("partial", partial_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("out_acc", out_acc_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("max", max_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("sum", sum_desc, indent_size=2) }}

  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %c_neg_inf = arith.constant -1.0e+30 : f32
  %v0_e = arith.constant dense<0.0> : vector<{{ tile_e }}xf32>
  %v0_2x = arith.constant dense<0.0> : vector<2xf32>
  %v_neg_inf_2x = arith.constant dense<-1.0e+30> : vector<2xf32>

  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %gh = 0 to {{ HgDhTiles }} {
    // reset merged accumulators
    affine.vector_store %v0_e, %out_acc_buffer[0, 0, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
    affine.vector_store %v_neg_inf_2x, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>
    affine.vector_store %v0_2x, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>

    affine.for %blk = 0 to {{ nblk }} {
      {{ kernel.def_dma_op("MVIN", "partial", partial_idx, partial_tile_desc, subtile_size=[1, 1, tile_pack], indent_size=8) }}
      %p = affine.vector_load %partial_buffer[0, 0, 0] : {{ partial_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_pack }}xf32>
      %p2 = vector.shape_cast %p : vector<{{ tile_pack }}xf32> to vector<2x{{ tile_e }}xf32>
      %o_j = vector.extract %p2[0] : vector<{{ tile_e }}xf32> from vector<2x{{ tile_e }}xf32>
      %ml_j = vector.extract %p2[1] : vector<{{ tile_e }}xf32> from vector<2x{{ tile_e }}xf32>
      %m_j = vector.extract %ml_j[0] : f32 from vector<{{ tile_e }}xf32>
      %l_j = vector.extract %ml_j[1] : f32 from vector<{{ tile_e }}xf32>

      %old_max = affine.vector_load %max_buffer[0, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>
      %m_old = vector.extract %old_max[0] : f32 from vector<2xf32>
      %m_new = arith.maximumf %m_old, %m_j : f32
      %m_new2 = vector.broadcast %m_new : f32 to vector<2xf32>
      affine.vector_store %m_new2, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape("f32") }}, vector<2xf32>

      %diff_old = arith.subf %m_old, %m_new : f32
      %diff_j = arith.subf %m_j, %m_new : f32
      %scale_old = math.exp %diff_old : f32
      %scale_j = math.exp %diff_j : f32
      %scale_old_e = vector.broadcast %scale_old : f32 to vector<{{ tile_e }}xf32>
      %scale_j_e = vector.broadcast %scale_j : f32 to vector<{{ tile_e }}xf32>

      %o_old = affine.vector_load %out_acc_buffer[0, 0, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
      %o_old_rs = arith.mulf %o_old, %scale_old_e : vector<{{ tile_e }}xf32>
      %o_j_rs = arith.mulf %o_j, %scale_j_e : vector<{{ tile_e }}xf32>
      %o_new = arith.addf %o_old_rs, %o_j_rs : vector<{{ tile_e }}xf32>
      affine.vector_store %o_new, %out_acc_buffer[0, 0, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>

      %old_sum = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
      %l_old = vector.extract %old_sum[0] : f32 from vector<2xf32>
      %l_new = arith.addf (arith.mulf %l_old, %scale_old : f32), (arith.mulf %l_j, %scale_j : f32) : f32
      %l_new2 = vector.broadcast %l_new : f32 to vector<2xf32>
      affine.vector_store %l_new2, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
    } { accumulation_loop=true }

    // finalize: out = o / l
    %sum2 = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape("f32") }}, vector<2xf32>
    %l = vector.extract %sum2[0] : f32 from vector<2xf32>
    %inv = arith.divf %c1, %l : f32
    %inv_e = vector.broadcast %inv : f32 to vector<{{ tile_e }}xf32>
    %o = affine.vector_load %out_acc_buffer[0, 0, 0] : {{ out_acc_tile_desc.get_mlir_shape("f32") }}, vector<{{ tile_e }}xf32>
    %out_f32 = arith.mulf %o, %inv_e : vector<{{ tile_e }}xf32>
    %out_io = arith.truncf %out_f32 : vector<{{ tile_e }}xf32> to vector<{{ tile_e }}x{{ io_stype }}>
    affine.vector_store %out_io, %out_buffer[0, 0, 0] : {{ out_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ tile_e }}x{{ io_stype }}>
    {{ kernel.store_output(indent_size=4) }}
  } { outer_loop=true }
  return
}
"""


class MLIRDecodeGQASDPAPartialTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, scale, BlkS: int = 1024, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.scale = scale
        self.BlkS = BlkS

    def render(self, kernel: MLIRTemplateKernel, template_buffer_node=None, epilogue_nodes=None, prologue_nodes=None, tile_info=None, **kwargs):
        query, key, value = self.input_nodes[0], self.input_nodes[1], self.input_nodes[2]
        partial = self.output_node

        q_tensor4 = empty_strided(query.layout.size, query.layout.stride)
        k_tensor4 = empty_strided(key.layout.size, key.layout.stride)
        v_tensor4 = empty_strided(value.layout.size, value.layout.stride)
        B, Hq, Lq, Dh = q_tensor4.shape
        _, H, S, _ = k_tensor4.shape
        assert B == 1 and Lq == 1
        g = Hq // H
        BlkS = min(int(self.BlkS), int(S))
        nblk = (int(S) + int(BlkS) - 1) // int(BlkS)

        io_stype = mlir_common.DTYPE_TO_MLIR[query.get_dtype()]
        tile_s = kernel.vector_lane
        tile_e = kernel.vector_lane
        tile_pack = tile_e * 2

        # Use 3D views for indices
        q_tensor = q_tensor4.view(Hq, 1, Dh)
        k_tensor = k_tensor4.view(H, S, Dh)
        v_tensor = v_tensor4.view(H, S, Dh)

        # Flatten (kv,qsub,dh_tile) into GH = H*g*(Dh/tile_e)
        dh_tiles = int(Dh) // int(tile_e)
        HgDhTiles = int(H) * int(g) * int(dh_tiles)

        # tile descs
        vlane_stride = 1
        q_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_e], kernel.vector_lane, 1, vlane_stride)
        q_tile_desc.set_tile_size_stride([1, 1, tile_e], [0, tile_e, 1])
        q_tile_desc.set_name("q_buffer")
        q_tile_desc.offset = query.get_layout().offset

        k_tile_desc = mlir_common.MLIRMultiDimTile([1, tile_s, tile_e], kernel.vector_lane, 2, vlane_stride)
        k_tile_desc.set_tile_size_stride([1, tile_s, tile_e], [0, 1, tile_s])
        k_tile_desc.set_name("k_buffer")
        k_tile_desc.offset = key.get_layout().offset

        v_tile_desc = mlir_common.MLIRMultiDimTile([1, tile_s, tile_e], kernel.vector_lane, 1, vlane_stride)
        v_tile_desc.set_tile_size_stride([1, tile_s, tile_e], [0, tile_e, 1])
        v_tile_desc.set_name("v_buffer")
        v_tile_desc.offset = value.get_layout().offset

        mul_tile_desc = mlir_common.MLIRMultiDimTile([tile_s, 1], kernel.vector_lane, 1, vlane_stride)
        mul_tile_desc.set_tile_size_stride([tile_s, 1], [1, 1])
        mul_tile_desc.set_name("mul_buffer")

        score_desc = mlir_common.MLIRMultiDimTile([g, tile_s], kernel.vector_lane, 1, vlane_stride)
        score_desc.set_tile_size_stride([g, tile_s], [tile_s, 1])
        score_desc.set_name("score_buffer")

        prob_desc = mlir_common.MLIRMultiDimTile([g, tile_s], kernel.vector_lane, 1, vlane_stride)
        prob_desc.set_tile_size_stride([g, tile_s], [tile_s, 1])
        prob_desc.set_name("prob_buffer")

        # Per-qsub, per-dh-tile accumulators so QK is computed once and SV expands across dh tiles.
        out_acc_tile_desc = mlir_common.MLIRMultiDimTile([g, dh_tiles, tile_e], kernel.vector_lane, 2, vlane_stride)
        out_acc_tile_desc.set_tile_size_stride([g, dh_tiles, tile_e], [dh_tiles * tile_e, tile_e, 1])
        out_acc_tile_desc.set_name("out_acc_buffer")

        max_desc = mlir_common.MLIRMultiDimTile([g, 2], kernel.vector_lane, 0, vlane_stride)
        max_desc.set_tile_size_stride([g, 2], [2, 1])
        max_desc.set_name("max_buffer")

        sum_desc = mlir_common.MLIRMultiDimTile([g, 2], kernel.vector_lane, 0, vlane_stride)
        sum_desc.set_tile_size_stride([g, 2], [2, 1])
        sum_desc.set_name("sum_buffer")

        out_io_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_e], kernel.vector_lane, 1, vlane_stride)
        out_io_tile_desc.set_tile_size_stride([1, 1, tile_e], [0, tile_e, 1])
        out_io_tile_desc.set_name("out_io_buffer")

        partial_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_pack], kernel.vector_lane, 1, vlane_stride)
        partial_tile_desc.set_tile_size_stride([1, 1, tile_pack], [0, tile_pack, 1])
        partial_tile_desc.set_name("partial_buffer")

        # Indices
        kv = sympy.Symbol("kv")
        qsub = sympy.Symbol("qsub")
        dht = sympy.Symbol("dht")
        dh0 = sympy.Symbol("dh0")
        k0 = sympy.Symbol("k0")
        blk = sympy.Symbol("blk")
        s0 = sympy.Symbol("s0")
        q_head = kv * g + qsub

        q_stride = q_tensor.stride()
        k_stride = k_tensor.stride()
        v_stride = v_tensor.stride()

        qk_idx = [q_head * q_stride[0], sympy.Integer(0), k0 * q_stride[2]]
        kk_idx = [kv * k_stride[0], s0 * k_stride[1], k0 * k_stride[2]]
        v_idx = [kv * v_stride[0], s0 * v_stride[1], dh0 * v_stride[2]]

        # partial tensor is view(HgDhTiles, nblk, tile_pack) contiguous
        p_tensor = empty_strided(partial.get_layout().size, partial.get_layout().stride).view(HgDhTiles, nblk, tile_pack)
        p_stride = p_tensor.stride()
        # group head index: ((kv*g + qsub)*dh_tiles + dht)
        gh = (kv * g + qsub) * dh_tiles + dht
        partial_idx = [gh * p_stride[0], blk * p_stride[1], sympy.Integer(0)]

        kernel.loop_size = [tile_s, tile_e, tile_pack]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            H=H,
            g=g,
            Dh=Dh,
            S=S,
            BlkS=BlkS,
            nblk=nblk,
            tile_s=tile_s,
            tile_e=tile_e,
            dh_tiles=dh_tiles,
            tile_pack=tile_pack,
            io_stype=io_stype,
            scale=self.scale,
            query=query,
            key=key,
            value=value,
            partial=partial,
            q_tile_desc=q_tile_desc,
            k_tile_desc=k_tile_desc,
            v_tile_desc=v_tile_desc,
            mul_tile_desc=mul_tile_desc,
            score_desc=score_desc,
            prob_desc=prob_desc,
            out_io_tile_desc=out_io_tile_desc,
            out_acc_tile_desc=out_acc_tile_desc,
            max_desc=max_desc,
            sum_desc=sum_desc,
            partial_tile_desc=partial_tile_desc,
            qk_idx=qk_idx,
            kk_idx=kk_idx,
            v_idx=v_idx,
            partial_idx=partial_idx,
            input_reorder=self.input_reorder,
        )

        kernel.epilogue_info = dict(
            output_node=self.output_node.name,
            sram_var="partial_buffer",
            dram_var="partial",
            dram_idx=partial_idx,
            dram_tile_desc=partial_tile_desc,
            nr_rdim=0,
            r_dim_size=0,
            dim_aliasing={"kv": "kv", "qsub": "qsub", "dht": "dht", "dh0": "dh0", "k0": "k0", "blk": "blk", "s0": "s0"},
        )
        return self._template_from_string(DECODE_GQA_SDPA_PARTIAL_TEMPLATE).render(**kernel.render_options)


class MLIRDecodeGQASDPAReduceTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, BlkS: int = 1024, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.BlkS = BlkS

    def render(self, kernel: MLIRTemplateKernel, template_buffer_node=None, epilogue_nodes=None, prologue_nodes=None, tile_info=None, **kwargs):
        partial = self.input_nodes[0]
        out = self.output_node

        tile_e = kernel.vector_lane
        tile_pack = tile_e * 2

        # Infer sizes from partial layout: (HgDhTiles, nblk, tile_pack)
        HgDhTiles, nblk, _ = partial.get_size()
        io_stype = mlir_common.DTYPE_TO_MLIR[out.get_dtype()]

        vlane_stride = 1
        partial_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_pack], kernel.vector_lane, 1, vlane_stride)
        partial_tile_desc.set_tile_size_stride([1, 1, tile_pack], [0, tile_pack, 1])
        partial_tile_desc.set_name("partial_buffer")
        partial_tile_desc.offset = partial.get_layout().offset

        out_acc_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_e], kernel.vector_lane, 1, vlane_stride)
        out_acc_tile_desc.set_tile_size_stride([1, 1, tile_e], [0, tile_e, 1])
        out_acc_tile_desc.set_name("out_acc_buffer")

        max_desc = mlir_common.MLIRMultiDimTile([1, 2], kernel.vector_lane, 0, vlane_stride)
        max_desc.set_tile_size_stride([1, 2], [2, 1])
        max_desc.set_name("max_buffer")

        sum_desc = mlir_common.MLIRMultiDimTile([1, 2], kernel.vector_lane, 0, vlane_stride)
        sum_desc.set_tile_size_stride([1, 2], [2, 1])
        sum_desc.set_name("sum_buffer")

        out_tile_desc = mlir_common.MLIRMultiDimTile([1, 1, tile_e], kernel.vector_lane, 1, vlane_stride)
        out_tile_desc.set_tile_size_stride([1, 1, tile_e], [0, tile_e, 1])
        out_tile_desc.set_name("out_buffer")

        # Indexing: partial is already 3D; out is (Hq,1,Dh) but view as (Hq*Dh/tile_e, 1, tile_e)
        p_tensor = empty_strided(partial.get_layout().size, partial.get_layout().stride)
        p_stride = p_tensor.stride()
        gh = sympy.Symbol("gh")
        blk = sympy.Symbol("blk")
        partial_idx = [gh * p_stride[0], blk * p_stride[1], sympy.Integer(0)]

        # out view
        out_tensor4 = empty_strided(out.get_layout().size, out.get_layout().stride)
        B, Hq, Lq, Dh = out_tensor4.shape
        assert B == 1 and Lq == 1
        dh_tiles = int(Dh) // int(tile_e)
        out_tensor = out_tensor4.view(Hq * dh_tiles, 1, tile_e)
        o_stride = out_tensor.stride()
        out_idx = [gh * o_stride[0], sympy.Integer(0), sympy.Integer(0)]

        kernel.loop_size = [tile_pack, tile_e, 1]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            HgDhTiles=HgDhTiles,
            nblk=nblk,
            tile_e=tile_e,
            tile_pack=tile_pack,
            io_stype=io_stype,
            partial=partial,
            out=out,
            partial_tile_desc=partial_tile_desc,
            out_acc_tile_desc=out_acc_tile_desc,
            max_desc=max_desc,
            sum_desc=sum_desc,
            out_tile_desc=out_tile_desc,
            partial_idx=partial_idx,
            out_idx=out_idx,
            input_reorder=self.input_reorder,
        )

        kernel.epilogue_info = dict(
            output_node=self.output_node.name,
            sram_var="out_buffer",
            dram_var="out",
            dram_idx=out_idx,
            dram_tile_desc=out_tile_desc,
            nr_rdim=0,
            r_dim_size=0,
            dim_aliasing={"gh": "gh", "blk": "blk"},
        )
        return self._template_from_string(DECODE_GQA_SDPA_REDUCE_TEMPLATE).render(**kernel.render_options)
