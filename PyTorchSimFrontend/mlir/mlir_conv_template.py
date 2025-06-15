import os
import math
from sympy import divisors, Range
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import PyTorchSimFrontend.extension_codecache as extension_codecache
from torch._inductor.codecache import get_hash
from PyTorchSimFrontend import extension_config

CONV_TEMPLATE = r"""
// Conv2D kernel
// BATCH = {{ BATCH }}
// I_C = {{ I_C }}
// I_H = {{ I_H }}
// I_W = {{ I_W }}
// O_C = {{ O_C }}
// K_H = {{ K_H }}
// K_W = {{ K_W }}
// O_H = {{ O_H }}
// O_W = {{ O_W }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// TILE_I_H={{ TILE_I_H }},
// TILE_I_W={{ TILE_I_W }},
// TILE_O_H={{ TILE_O_H }},
// TILE_O_W={{ TILE_O_W }},
// TILE_K_H={{ TILE_K_H }},
// TILE_K_W={{ TILE_K_W }},
// SUB_TILE_M={{ SUB_TILE_M }},
// SUB_TILE_N={{ SUB_TILE_N }},
// SUB_TILE_I_W={{ SUB_TILE_I_W }},
// SUB_TILE_K_H={{ SUB_TILE_K_H }},
// SUB_TILE_K_W={{ SUB_TILE_K_W }},
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DILATION_H = {{ DILATION_H }}
// DILATION_W = {{ DILATION_W }}
// DATA_STYPE = {{ DATA_STYPE }}
// DATA_SIZE = {{ DATA_SIZE }}

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ O_W * BATCH * O_C }} + d1 * {{ BATCH * O_C }} + d2 * {{ O_C }} + d3)> // output (O_H, O_W, BATCH, O_C)
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ (I_W + 2 * PADDING_W) * BATCH * I_C }} + d1 * {{ BATCH * I_C }} + d2 * {{ I_C }} + d3)> // input (I_H, I_W, BATCH, I_C)
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ K_W * I_C * O_C }} + d1 * {{ I_C * O_C }} + d2 * {{ O_C }} + d3)> // weight (K_H, K_W, I_C, O_C)
#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#map_I_W = affine_map<(d0, d1) -> (d0 * {{ STRIDE_W }} + d1)>
#offset_w_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_K_W * TILE_K, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_K, TILE_N) }})>
#offset_x_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_I_W * TILE_M, TILE_K) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_K) }})>
#offset_y_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_O_W * TILE_M, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }})>

memref.global @X_spad : memref<{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_conv_kernel(inputs=[X, W, BIAS], outputs=[Y], names_str="X, W, Bias, Y", padded_input_size=PADDED_INPUT_SIZE, input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %input_axis = arith.constant 3 : index
  %weight_axis = arith.constant 2 : index
  %input_buffer = memref.get_global @X_spad : memref<{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %weight_buffer = memref.get_global @W_spad : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %output_buffer = memref.get_global @Y_spad : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>
  %tag3 = memref.alloc() : memref<1xi32>
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_O_W * TILE_M, TILE_N) }}xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  {{- kernel.def_local_vars() }}

  affine.for %o_h = 0 to {{ O_H }} step {{ TILE_O_H }} {
    affine.for %o_w = 0 to {{ O_W }} step {{ TILE_O_W }} {
      affine.for %tile_m = 0 to {{ BATCH }} step {{ TILE_M }} {
        affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
          %index0 = affine.apply #map0(%o_h, %o_w, %tile_m, %tile_n)
          // Initialize output
          {%- if BIAS %}
          memref.dma_start %Bias[%tile_n], %output_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag0[%c0], %c0, %vstride
              : memref<{{ O_C }}xf32>, memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ TILE_O_H }}, {{ TILE_O_W }}, {{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_O_W * TILE_M * TILE_N }}, {{ TILE_M * TILE_N }}, 1, {{ TILE_M }}]}
          {%- else %}
          affine.vector_store %v0, %output_buffer[%c0, %c0, %c0, %c0] : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_O_W * TILE_M, TILE_N) }}xf32>
          {%- endif %}
          affine.for %k_h = 0 to {{ K_H }} step {{ TILE_K_H }} {
            affine.for %k_w = 0 to {{ K_W }} step {{ TILE_K_W }} {
              affine.for %tile_k = 0 to {{ I_C }} step {{ TILE_K }} {
                %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
                %index_i_w = affine.apply #map_I_W(%o_w, %k_w)
                %index1 = affine.apply #map1(%index_i_h, %index_i_w, %tile_m, %tile_k) // input index
                %index2 = affine.apply #map2(%k_h, %k_w, %tile_k, %tile_n) // weight index
                // Load input matrix
                memref.dma_start %X[%index1], %input_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag1[%c0], %input_axis, %vstride
                    : memref<{{ BATCH * I_C * (I_H + 2 * PADDING_H) * (I_W + 2 * PADDING_W) }}xf32>, memref<{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_I_H }}, {{ SUB_TILE_I_W }}, {{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[{{ TILE_I_W * TILE_M * TILE_K }}, {{ TILE_M * TILE_K }}, 1, {{ TILE_M }}]}
                // Load kernel matrix
                memref.dma_start %W[%index2], %weight_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag2[%c0], %input_axis, %vstride
                    : memref<{{ O_C * I_C * K_H * K_W }}xf32>, memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K_H }}, {{ SUB_TILE_K_W }}, {{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_K_W * TILE_K * TILE_N }}, {{ TILE_K * TILE_N }}, 1, {{ TILE_K }}]}
                affine.for %tile_k_h = 0 to {{ TILE_K_H }} { // loop order should be fixed for timing simulation. Do not change this order.
                  affine.for %tile_k_w = 0 to {{ TILE_K_W }} {
                    %offset_w = affine.apply #offset_w_map(%tile_k_h, %tile_k_w)
                    %W_buffer = memref.reinterpret_cast %weight_buffer to offset: [%offset_w], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                    affine.for %tile_o_h = 0 to {{ TILE_O_H }} {
                      affine.for %tile_o_w = 0 to {{ TILE_O_W }} {
                        %tile_i_h = affine.apply #map_I_H(%tile_o_h, %tile_k_h)
                        %tile_i_w = affine.apply #map_I_W(%tile_o_w, %tile_k_w)
                        %offset_x = affine.apply #offset_x_map(%tile_i_h, %tile_i_w)
                        %offset_y = affine.apply #offset_y_map(%tile_o_h, %tile_o_w)
                        %X_buffer = memref.reinterpret_cast %input_buffer to offset: [%offset_x], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>
                        %Y_buffer = memref.reinterpret_cast %output_buffer to offset: [%offset_y], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                        linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                              outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                      } { inner_loop=true }
                    } { inner_loop=true }
                  } { inner_loop=true }
                } { inner_loop=true }
              } { accumulation_loop=true }
            } { accumulation_loop=true }
          } { accumulation_loop=true }
          // Store output matrix
          {{kernel.store_output(indent_size=10)}}
        } { outer_loop=true }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

MULTI_TILE_CONV_TEMPLATE = r"""
// Multi Channel Tile Conv2D kernel
// BATCH = {{ BATCH }}
// I_C = {{ I_C }}
// I_H = {{ I_H }}
// I_W = {{ I_W }}
// O_C = {{ O_C }}
// K_H = {{ K_H }}
// K_W = {{ K_W }}
// O_H = {{ O_H }}
// O_W = {{ O_W }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// TILE_I_H={{ TILE_I_H }},
// TILE_I_W={{ TILE_I_W }},
// TILE_O_H={{ TILE_O_H }},
// TILE_O_W={{ TILE_O_W }},
// TILE_K_H={{ TILE_K_H }},
// TILE_K_W={{ TILE_K_W }},
// SUB_TILE_M={{ SUB_TILE_M }},
// SUB_TILE_N={{ SUB_TILE_N }},
// SUB_TILE_I_W={{ SUB_TILE_I_W }},
// SUB_TILE_K_H={{ SUB_TILE_K_H }},
// SUB_TILE_K_W={{ SUB_TILE_K_W }},
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DILATION_H = {{ DILATION_H }}
// DILATION_W = {{ DILATION_W }}
// DATA_STYPE = {{ DATA_STYPE }}
// DATA_SIZE = {{ DATA_SIZE }}

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ O_W * BATCH * O_C }} + d1 * {{ BATCH * O_C }} + d2 * {{ O_C }} + d3)> // output (O_H, O_W, BATCH, O_C)
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ (I_W + 2 * PADDING_W) * BATCH * I_C }} + d1 * {{ I_C * STRIDE_W }} + d2 * {{ I_C * (I_W + 2 * PADDING_W) }} + d3)> // input (I_H, BATCH, I_W, I_C)
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ K_W * I_C * O_C }} + d1 * {{ I_C * O_C }} + d2 * {{ O_C }} + d3)> // weight (K_H, K_W, I_C, O_C)
#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#offset_w_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(1 * TILE_K, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_K, TILE_N) }})>
#offset_x_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_O_W * TILE_M, TILE_K) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_K) }})>
#offset_y_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_O_W * TILE_M, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }})>

memref.global @X_spad : memref<{{ TILE_I_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K_H }}x{{ 1 }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_conv_kernel(inputs=[X, W, BIAS], outputs=[Y], names_str="X, W, Bias, Y", padded_input_size=PADDED_INPUT_SIZE, input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %input_axis = arith.constant 3 : index
  %weight_axis = arith.constant 2 : index
  %input_buffer = memref.get_global @X_spad : memref<{{ TILE_I_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %weight_buffer = memref.get_global @W_spad : memref<{{ TILE_K_H }}x{{ 1 }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %output_buffer = memref.get_global @Y_spad : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>
  %tag3 = memref.alloc() : memref<1xi32>
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_O_W * TILE_M, TILE_N) }}xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  {{- kernel.def_local_vars() }}

  affine.for %o_h = 0 to {{ O_H }} step {{ TILE_O_H }} {
    affine.for %o_w = 0 to {{ O_W }} step {{ TILE_O_W }} {
      affine.for %tile_m = 0 to {{ BATCH }} step {{ TILE_M }} {
        affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
          %index0 = affine.apply #map0(%o_h, %o_w, %tile_m, %tile_n)
          // Initialize output
          {%- if BIAS %}
          memref.dma_start %Bias[%tile_n], %output_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag0[%c0], %c0, %vstride
              : memref<{{ O_C }}xf32>, memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ TILE_O_H }}, {{ TILE_O_W }}, {{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_O_W * TILE_M * TILE_N }}, {{ TILE_M * TILE_N }}, 1, {{ TILE_M }}]}
          {%- else %}
          affine.vector_store %v0, %output_buffer[%c0, %c0, %c0, %c0] : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_O_W * TILE_M, TILE_N) }}xf32>
          {%- endif %}
          affine.for %k_h = 0 to {{ K_H }} step {{ TILE_K_H }} {
            affine.for %tile_k = 0 to {{ I_C * K_W }} step {{ TILE_K }} {
              %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
              %index1 = affine.apply #map1(%index_i_h, %o_w, %tile_m, %tile_k) // input index
              %index2 = affine.apply #map2(%k_h, %c0, %tile_k, %tile_n) // weight index
              // Load input matrix
              memref.dma_start %X[%index1], %input_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag1[%c0], %input_axis, %vstride
                  : memref<{{ BATCH * I_C * (I_H + 2 * PADDING_H) * (I_W + 2 * PADDING_W) }}xf32>, memref<{{ TILE_I_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_I_H }}, {{ SUB_TILE_I_W }}, {{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[{{ TILE_O_W * TILE_M * TILE_K }}, {{ TILE_M * TILE_K }}, 1, {{ TILE_M }}]}
              // Load kernel matrix
              memref.dma_start %W[%index2], %weight_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag2[%c0], %input_axis, %vstride
                  : memref<{{ O_C * I_C * K_H * K_W }}xf32>, memref<{{ TILE_K_H }}x{{ 1 }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K_H }}, {{ SUB_TILE_K_W }}, {{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_K_W * TILE_K * TILE_N }}, {{ TILE_K * TILE_N }}, 1, {{ TILE_K }}]}
              affine.for %tile_k_h = 0 to {{ TILE_K_H }} { // loop order should be fixed for timing simulation. Do not change this order.
                affine.for %tile_k_w = 0 to 1 {
                  %offset_w = affine.apply #offset_w_map(%tile_k_h, %tile_k_w)
                  %W_buffer = memref.reinterpret_cast %weight_buffer to offset: [%offset_w], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ TILE_K_H }}x{{ 1 }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                  affine.for %tile_o_h = 0 to {{ TILE_O_H }} {
                    affine.for %tile_o_w = 0 to {{ TILE_O_W }} {
                      %tile_i_h = affine.apply #map_I_H(%tile_o_h, %tile_k_h)
                      %offset_x = affine.apply #offset_x_map(%tile_i_h, %tile_o_w)
                      %offset_y = affine.apply #offset_y_map(%tile_o_h, %tile_o_w)
                      %X_buffer = memref.reinterpret_cast %input_buffer to offset: [%offset_x], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<{{ TILE_I_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>
                      %Y_buffer = memref.reinterpret_cast %output_buffer to offset: [%offset_y], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ TILE_O_H }}x{{ TILE_O_W }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                      linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                            outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                    } { inner_loop=true }
                  } { inner_loop=true }
                } { inner_loop=true }
              } { inner_loop=true }
            } { accumulation_loop=true }
          } { accumulation_loop=true }
          // Store output matrix
          {{kernel.store_output(indent_size=10)}}
        } { outer_loop=true }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

SINGLE_BATCH_CONV_TEMPLATE = r"""
// Single Batch Conv2D kernel
// BATCH = {{ BATCH }}
// I_C = {{ I_C }}
// I_H = {{ I_H }}
// I_W = {{ I_W }}
// O_C = {{ O_C }}
// K_H = {{ K_H }}
// K_W = {{ K_W }}
// O_H = {{ O_H }}
// O_W = {{ O_W }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// TILE_I_H={{ TILE_I_H }},
// TILE_I_W={{ TILE_I_W }},
// TILE_O_H={{ TILE_O_H }},
// TILE_O_W={{ TILE_O_W }},
// TILE_K_H={{ TILE_K_H }},
// TILE_K_W={{ TILE_K_W }},
// SUB_TILE_M={{ SUB_TILE_M }},
// SUB_TILE_N={{ SUB_TILE_N }},
// SUB_TILE_I_W={{ SUB_TILE_I_W }},
// SUB_TILE_K_H={{ SUB_TILE_K_H }},
// SUB_TILE_K_W={{ SUB_TILE_K_W }},
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DILATION_H = {{ DILATION_H }}
// DILATION_W = {{ DILATION_W }}
// DATA_STYPE = {{ DATA_STYPE }}
// DATA_SIZE = {{ DATA_SIZE }}

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ O_W * O_H * O_C }} + d1 * {{ O_W * O_C }} + d2 * {{ O_C }} + d3)> // output (BATCH, O_H, O_W, O_C)
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ (I_W + 2 * PADDING_W) * (I_H + 2 * PADDING_W) * I_C }} + d1 * {{ (I_W + 2 * PADDING_W) * I_C }} + d2 * {{ I_C }} + d3)> // input (BATCH, I_H, I_W, I_C) Stride should be changed if kernel stride > 1
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ K_W * I_C * O_C }} + d1 * {{ I_C * O_C }} + d2 * {{ O_C }} + d3)> // weight (K_H, K_W, I_C, O_C)
#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#map_I_W = affine_map<(d0, d1) -> (d0 * {{ STRIDE_W }} + d1)>
#offset_w_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_K_W * TILE_K, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_K, TILE_N) }})>
#offset_x_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_I_W, TILE_K) }} + d1)>
#offset_y_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }})>
memref.global @X_spad : memref<{{ 1 }}x{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_conv_kernel(inputs=[X, W, BIAS], outputs=[Y], names_str="X, W, Bias, Y", padded_input_size=PADDED_INPUT_SIZE, input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %input_axis = arith.constant 3 : index
  %weight_axis = arith.constant 2 : index
  %input_buffer = memref.get_global @X_spad : memref<{{ 1 }}x{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_K }}xf32, 1>
  %weight_buffer = memref.get_global @W_spad : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %output_buffer = memref.get_global @Y_spad : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>
  %tag3 = memref.alloc() : memref<1xi32>
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  {{- kernel.def_local_vars() }}
  affine.for %o_w = 0 to {{ O_W }} step {{ TILE_O_W }} {
  affine.for %o_h = 0 to {{ O_H }} step {{ TILE_O_H }} {
    affine.for %tile_m = 0 to {{ O_W }} step {{ TILE_M }} {
      affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
        %index0 = affine.apply #map0(%c0, %o_h, %tile_m, %tile_n)
        // Initialize output
        {%- if BIAS %}
        memref.dma_start %Bias[%tile_n], %output_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag0[%c0], %c0, %vstride // not implemented yet
            : memref<{{ O_C }}xf32>, memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ 1 }}, {{ TILE_O_H }}, {{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_O_H * TILE_M * TILE_N }}, {{ TILE_M * TILE_N }}, 1, {{ TILE_M }}]}
        {%- else %}
        affine.vector_store %v0, %output_buffer[%c0, %c0, %c0, %c0] : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}xf32>
        {%- endif %}
        affine.for %k_h = 0 to {{ K_H }} step {{ TILE_K_H }} {
          affine.for %k_w = 0 to {{ K_W }} step {{ TILE_K_W }} {
            affine.for %tile_k = 0 to {{ I_C }} step {{ TILE_K }} {
              %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
              %index_i_w = affine.apply #map_I_W(%o_w, %k_w)
              %index1 = affine.apply #map1(%c0, %index_i_h, %index_i_w, %tile_k) // input index
              %index2 = affine.apply #map2(%k_h, %k_w, %tile_k, %tile_n) // weight index
              // Load input matrix
              memref.dma_start %X[%index1], %input_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag1[%c0], %input_axis, %vstride
                  : memref<{{ BATCH * I_C * (I_H + 2 * PADDING_H) * (I_W + 2 * PADDING_W) }}xf32>, memref<{{ 1 }}x{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ 1 }}, {{ SUB_TILE_I_H }}, {{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[{{ TILE_I_H * TILE_I_W * TILE_K }}, {{ TILE_I_W * TILE_K }}, 1, {{ TILE_I_W }}]}
              // Load kernel matrix
              memref.dma_start %W[%index2], %weight_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag2[%c0], %input_axis, %vstride
                  : memref<{{ O_C * I_C * K_H * K_W }}xf32>, memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K_H }}, {{ SUB_TILE_K_W }}, {{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_K_W * TILE_K * TILE_N }}, {{ TILE_K * TILE_N }}, 1, {{ TILE_K }}]}
              affine.for %tile_k_h = 0 to {{ TILE_K_H }} { // loop order should be fixed for timing simulation. Do not change this order.
                affine.for %tile_k_w = 0 to {{ TILE_K_W }} {
                  %offset_w = affine.apply #offset_w_map(%tile_k_h, %tile_k_w)
                  %W_buffer = memref.reinterpret_cast %weight_buffer to offset: [%offset_w], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                  affine.for %tile_o_h = 0 to {{ TILE_O_H }} {
                    affine.for %tile_o_w = 0 to {{ 1 }} { // TILE_O_W
                      %tile_i_h = affine.apply #map_I_H(%tile_o_h, %tile_k_h)
                      %offset_x = affine.apply #offset_x_map(%tile_i_h, %tile_k_w)
                      %offset_y = affine.apply #offset_y_map(%tile_o_h, %tile_o_w)
                      %X_buffer = memref.reinterpret_cast %input_buffer to offset: [%offset_x], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<{{ 1 }}x{{ TILE_I_H }}x{{ TILE_I_W }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>
                      %Y_buffer = memref.reinterpret_cast %output_buffer to offset: [%offset_y], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                      linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                            outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                    } { inner_loop=true }
                  } { inner_loop=true }
                } { inner_loop=true }
              } { inner_loop=true }
            } { accumulation_loop=true }
          } { accumulation_loop=true }
        } { accumulation_loop=true }
        // Store output matrix
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

SINGLE_BATCH_CONV_STRIDE_TEMPLATE = r"""
// Single Batch Conv2D (Stride != 1) kernel
// BATCH = {{ BATCH }}
// I_C = {{ I_C }}
// I_H = {{ I_H }}
// I_W = {{ I_W }}
// O_C = {{ O_C }}
// K_H = {{ K_H }}
// K_W = {{ K_W }}
// O_H = {{ O_H }}
// O_W = {{ O_W }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// TILE_I_H={{ TILE_I_H }},
// TILE_I_W={{ TILE_I_W }},
// TILE_O_H={{ TILE_O_H }},
// TILE_O_W={{ TILE_O_W }},
// TILE_K_H={{ TILE_K_H }},
// TILE_K_W={{ TILE_K_W }},
// SUB_TILE_M={{ SUB_TILE_M }},
// SUB_TILE_N={{ SUB_TILE_N }},
// SUB_TILE_I_W={{ SUB_TILE_I_W }},
// SUB_TILE_K_H={{ SUB_TILE_K_H }},
// SUB_TILE_K_W={{ SUB_TILE_K_W }},
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DILATION_H = {{ DILATION_H }}
// DILATION_W = {{ DILATION_W }}
// DATA_STYPE = {{ DATA_STYPE }}
// DATA_SIZE = {{ DATA_SIZE }}

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ O_W * O_H * O_C }} + d1 * {{ O_W * O_C }} + d2 * {{ O_C }} + d3)> // output (BATCH, O_H, O_W, O_C)
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ (I_W + 2 * PADDING_W) * I_C }} + d1 * {{ I_C }} + d2 * {{ I_C * STRIDE_W }} + d3)> // input (I_H, (k_w), I_W, I_C) // duplicate for k_w
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ K_W * I_C * O_C }} + d1 * {{ I_C * O_C }} + d2 * {{ O_C }} + d3)> // weight (K_H, K_W, I_C, O_C)
#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#offset_w_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_K_W * TILE_K, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_K, TILE_N) }})>
#offset_x_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_M * TILE_K_W, TILE_K) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_K) }})>
#offset_y_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }})>

memref.global @X_spad : memref<{{ TILE_I_H }}x{{ TILE_K_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_conv_kernel(inputs=[X, W, BIAS], outputs=[Y], names_str="X, W, Bias, Y", padded_input_size=PADDED_INPUT_SIZE, input_reorder=input_reorder)}} {
  %c_mvin = arith.constant 2 : index
  %c_mvin2 = arith.constant 1 : index
  %c_mvin3 = arith.constant 14 : index
  %c_mvout = arith.constant 3 : index
  %vstride = arith.constant 1 : index
  %input_axis = arith.constant 3 : index
  %weight_axis = arith.constant 2 : index
  %input_buffer = memref.get_global @X_spad : memref<{{ TILE_I_H }}x{{ TILE_K_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %weight_buffer = memref.get_global @W_spad : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %output_buffer = memref.get_global @Y_spad : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %tag0 = memref.alloc() : memref<1xi32>
  %tag1 = memref.alloc() : memref<1xi32>
  %tag2 = memref.alloc() : memref<1xi32>
  %tag3 = memref.alloc() : memref<1xi32>
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  {{- kernel.def_local_vars() }}

  affine.for %o_h = 0 to {{ O_H }} step {{ TILE_O_H }} {
    affine.for %tile_m = 0 to {{ O_W }} step {{ TILE_M }} {
      affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
        %index0 = affine.apply #map0(%c0, %o_h, %tile_m, %tile_n)
        // Initialize output
        {%- if BIAS %}
        memref.dma_start %Bias[%tile_n], %output_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag0[%c0], %c0, %vstride // not implemented yet
            : memref<{{ O_C }}xf32>, memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ 1 }}, {{ TILE_O_H }}, {{ SUB_TILE_M }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_O_H * TILE_M * TILE_N }}, {{ TILE_M * TILE_N }}, 1, {{ TILE_M }}]}
        {%- else %}
        affine.vector_store %v0, %output_buffer[%c0, %c0, %c0, %c0] : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}xf32>
        {%- endif %}
        affine.for %k_h = 0 to {{ K_H }} step {{ TILE_K_H }} {
          affine.for %k_w = 0 to {{ K_W }} step {{ TILE_K_W }} {
            affine.for %tile_k = 0 to {{ I_C }} step {{ TILE_K }} {
              %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
              %index1 = affine.apply #map1(%index_i_h, %k_w, %tile_m, %tile_k) // input index
              %index2 = affine.apply #map2(%k_h, %k_w, %tile_k, %tile_n) // weight index
              // Load input matrix
              memref.dma_start %X[%index1], %input_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag1[%c0], %input_axis, %vstride
                  : memref<{{ BATCH * I_C * (I_H + 2 * PADDING_H) * (I_W + 2 * PADDING_W) }}xf32>, memref<{{ TILE_I_H }}x{{ TILE_K_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_I_H }}, {{ SUB_TILE_K_W }}, {{ SUB_TILE_M }}, {{ SUB_TILE_K }}], async=1, sram_stride=[{{ TILE_K_W * TILE_M * TILE_K }}, {{ TILE_M * TILE_K }}, 1, {{ TILE_M }}]}
              // Load kernel matrix
              memref.dma_start %W[%index2], %weight_buffer[%c0, %c0, %c0, %c0], %c_mvin, %tag2[%c0], %input_axis, %vstride
                  : memref<{{ O_C * I_C * K_H * K_W }}xf32>, memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32> { subtile_size=[{{ SUB_TILE_K_H }}, {{ SUB_TILE_K_W }}, {{ SUB_TILE_K }}, {{ SUB_TILE_N }}], async=1, sram_stride=[{{ TILE_K_W * TILE_K * TILE_N }}, {{ TILE_K * TILE_N }}, 1, {{ TILE_K }}]}
              affine.for %tile_k_h = 0 to {{ TILE_K_H }} { // loop order should be fixed for timing simulation. Do not change this order.
                affine.for %tile_k_w = 0 to {{ TILE_K_W }} {
                  %offset_w = affine.apply #offset_w_map(%tile_k_h, %tile_k_w)
                  %W_buffer = memref.reinterpret_cast %weight_buffer to offset: [%offset_w], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ TILE_K_H }}x{{ TILE_K_W }}x{{ TILE_K }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                  affine.for %tile_o_h = 0 to {{ TILE_O_H }} {
                    affine.for %tile_o_w = 0 to {{ 1 }} { // TILE_O_W
                      %tile_i_h = affine.apply #map_I_H(%tile_o_h, %tile_k_h)
                      %offset_x = affine.apply #offset_x_map(%tile_i_h, %tile_k_w)
                      %offset_y = affine.apply #offset_y_map(%tile_o_h, %tile_o_w)
                      %X_buffer = memref.reinterpret_cast %input_buffer to offset: [%offset_x], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : memref<{{ TILE_I_H }}x{{ TILE_K_W }}x{{ TILE_M }}x{{ TILE_K }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>
                      %Y_buffer = memref.reinterpret_cast %output_buffer to offset: [%offset_y], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : memref<{{ 1 }}x{{ TILE_O_H }}x{{ TILE_M }}x{{ TILE_N }}xf32, 1> to memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                      linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                            outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                    } { inner_loop=true }
                  } { inner_loop=true }
                } { inner_loop=true }
              } { inner_loop=true }
            } { accumulation_loop=true }
          } { accumulation_loop=true }
        } { accumulation_loop=true }
        // Store output matrix
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

WRAPPER_TEMPLATE = r"""
def {{ FUNC_NAME }}{{kernel.def_wrapper()}}:
    # Padding input
    padded_shape = list(X.shape)
    padded_shape[2] += 2 * {{ PADDING_H }}
    padded_shape[3] += 2 * {{ PADDING_W }}
    X_padding = torch.zeros(padded_shape, device=X.device)
    X_padding[:, :, {{ PADDING_H }}:X.shape[2] + {{ PADDING_H }}, {{ PADDING_W }}:X.shape[3] + {{ PADDING_W }}] = X

    # Holding original output tensor
    {%- for buf, name in kernel.get_conv_outputs().items() %}
    {{ name }}_t = {{ name }}
    {%- endfor %}

    # Tanspose inputs
    {%- for buf, name in kernel.get_conv_inputs().items() %}
      {%- if name == "X" %}
        {%- if MULTI_TILE %}
    {{ name }} = {{ name }}_padding.permute(2, 0, 3, 1).contiguous() # (BATCH, I_C, I_H, I_W) -> (I_H, BATCH, I_W, I_C)
        {%- elif SINGLE_BATCH %}
    {{ name }} = {{ name }}_padding.permute(0, 2, 3, 1).contiguous() # (BATCH, I_C, I_H, I_W) -> (BATCH, I_H, I_W, I_C)
        {%- else %}
    {{ name }} = {{ name }}_padding.permute(2, 3, 0, 1).contiguous() # (BATCH, I_C, I_H, I_W) -> (I_H, I_W, BATCH, I_C)
        {%- endif %}
      {%- elif name == "W" %}
    {{ name }} = {{ name }}.permute(2, 3, 1, 0).contiguous() # (O_C, I_C, K_H, K_W) -> (K_H, K_W, I_C, O_C)
      {%- elif name == "Bias" %}
    {{ name }} = {{ name }}
      {%- else %}
        {%- if SINGLE_BATCH %}
    {{ name }} = {{ name }}.permute(0, 2, 3, 1).contiguous()  if {{ name }}.dim() == 4 else {{ name }} # (BATCH, O_C, O_H, O_W) -> (BATCH, O_H, O_W, O_C)
        {%- else %}
    {{ name }} = {{ name }}.permute(2, 3, 0, 1).contiguous()  if {{ name }}.dim() == 4 else {{ name }} # (BATCH, O_C, O_H, O_W) -> (O_H, O_W, BATCH, O_C)
        {%- endif %}
      {%- endif %}
    {%- endfor %}

    # Transpose outputs
    {%- for buf, name in kernel.get_conv_outputs().items() %}
      {%- if SINGLE_BATCH %}
    {{ name }} = {{ name }}.permute(0, 2, 3, 1).contiguous() # (BATCH, O_C, O_H, O_W) -> (BATCH, O_H, O_W, O_C)
      {%- else %}
    {{ name }} = {{ name }}.permute(2, 3, 0, 1).contiguous() # (BATCH, O_C, O_H, O_W) ->  (O_H, O_W, BATCH, O_C)
      {%- endif %}
    {%- endfor %}

    # Launch kernel
    {{ KERNEL_NAME }}<DEF_CONV_WRAPPER>
    {%- if BACKENDSIM_EAGER_MODE %}
    yield ({{KERNEL_NAME}}, <DEF_CONV_WRAPPER>)
    {%- endif %}

    # Transpose back outputs
    {%- for buf, name in kernel.get_conv_outputs().items() %}
      {%- if SINGLE_BATCH %}
    {{ name }}_t.copy_({{ name }}.permute(0, 3, 1, 2).contiguous()) # (BATCH, O_H, O_W, O_C) -> (BATCH, O_C, O_H, O_W)
      {%- else %}
    {{ name }}_t.copy_({{ name }}.permute(2, 3, 0, 1).contiguous()) # (O_H, O_W, BATCH, O_C) -> (BATCH, O_C, O_H, O_W)
      {%- endif %}
    {%- endfor %}
"""

class MLIRConvTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None, **kwargs):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]
        self.dilation = kwargs["dilation"]
        self.weight_shape = [str(i) for i in input_nodes[1].layout.size]
        self.input_shape = [i for i in input_nodes[0].layout.size]
        self.function_name = "Conv2D_" + "_".join(self.weight_shape)+ "_" \
            + "_".join([str(i) for i in self.stride]) \
            + "_" + "_".join([str(i) for i in self.padding]) \
            + "_" + "_".join([str(i) for i in self.dilation])
        self.kernel_args = ['X', 'W', 'Bias', 'Y']

    def is_transposed(self, node):
        if isinstance(node, ReinterpretView):
            if node.layout.stride != node.data.layout.stride:
                if node.layout.stride[-2] == node.data.layout.stride[-1] and node.layout.stride[-1] == node.data.layout.stride[-2]:
                    return True
                else:
                  raise NotImplementedError("If the stride is not equal to the original stride, it should have been transposed.")
        return False

    def is_multi_tile(self, I_C):
        return I_C < (self.kernel.vector_lane // 8) # 8 is hard-coded for now. This should be changed to a better heuristic.

    def is_single_batch(self, BATCH):
        return BATCH == 1

    def get_padded_input_size(self, X):
        input_padded = list(X.layout.size)
        input_padded[2] += 2 * self.padding[0]
        input_padded[3] += 2 * self.padding[1]
        return math.prod(input_padded)

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        self.kernel = kernel
        self.epilogue_nodes = epilogue_nodes

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        if epilogue_nodes is not None:
          extra_node_rw = {
            item.name for epilogue_node in epilogue_nodes
            for item in epilogue_node.read_writes.reads | epilogue_node.read_writes.writes
            if item.name != Y.name
          }
        n_extra_node = len(extra_node_rw) if epilogue_nodes is not None else 0

        BATCH = X.layout.size[0]
        I_C = X.layout.size[1]
        O_C = W.layout.size[0]
        K_H = W.layout.size[2]
        K_W = W.layout.size[3]
        O_H = Y.layout.size[2] if template_buffer_node is None else template_buffer_node.layout.size[2]
        O_W = Y.layout.size[3] if template_buffer_node is None else template_buffer_node.layout.size[3]

        TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K = kernel.conv_combination_mapping(BATCH, O_C, I_C, K_H, K_W, O_H, O_W, self.stride, self.dilation, n_extra_node)
        SUB_TILE_M = TILE_M if TILE_M < kernel.vector_lane else kernel.vector_lane
        SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
        SUB_TILE_K = TILE_K
        TILE_I_H = 1 + (TILE_O_H - 1) * self.stride[0] + (TILE_K_H - 1) * self.dilation[0]
        TILE_I_W = 1 + (TILE_O_W - 1) * self.stride[1] + (TILE_K_W - 1) * self.dilation[1]
        SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W = 1, 1, 1, 1
        x_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_I_W * TILE_I_H * TILE_M, TILE_K)
        w_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_K_W * TILE_K_H * TILE_K, TILE_N)
        y_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_O_H * TILE_O_W * TILE_M, TILE_N)
        x_spad_size = TILE_I_W * TILE_I_H * TILE_M * TILE_K
        w_spad_size = TILE_K_W * TILE_K_H * TILE_K * TILE_N
        y_spad_size = TILE_O_H * TILE_O_W * TILE_M * TILE_N
        conv_template = CONV_TEMPLATE
        TOG_latency = BATCH if TILE_M > BATCH else TILE_M
        if self.is_single_batch(BATCH) and self.stride[0] != 1:
          conv_template = SINGLE_BATCH_CONV_STRIDE_TEMPLATE
          TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K = kernel.conv_single_batch_mapping(BATCH, O_C, I_C, K_H, K_W, O_H, O_W, self.stride, self.dilation, n_extra_node) # TODO: implement K_W
          TILE_I_H = 1 + (TILE_O_H - 1) * self.stride[0] + (TILE_K_H - 1) * self.dilation[0]
          x_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_K_W * TILE_I_H * TILE_M, TILE_K)
          w_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_K_W * TILE_K_H * TILE_K, TILE_N)
          y_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N)
          x_spad_size = TILE_K_W * TILE_I_H * TILE_M * TILE_K
          w_spad_size = TILE_K_W * TILE_K_H * TILE_K * TILE_N
          y_spad_size = TILE_O_H * TILE_M * TILE_N
          SUB_TILE_M = TILE_M if TILE_M < kernel.vector_lane else kernel.vector_lane
          SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
          SUB_TILE_K = TILE_K
          TOG_latency = O_W if TILE_M > O_W else TILE_M
        elif self.is_single_batch(BATCH) and self.stride[0] == 1:
          conv_template = SINGLE_BATCH_CONV_TEMPLATE
          TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K = kernel.conv_single_batch_mapping(BATCH, O_C, I_C, K_H, 1, O_H, O_W, self.stride, self.dilation, n_extra_node) # TODO: implement K_W
          TILE_I_H = 1 + (TILE_O_H - 1) * self.stride[0] + (TILE_K_H - 1) * self.dilation[0]
          TILE_I_W = 1 + (TILE_O_W - 1) * self.stride[1] + (TILE_K_W - 1) * self.dilation[1]
          SUB_TILE_M = TILE_I_W if TILE_I_W < kernel.vector_lane else kernel.vector_lane
          SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
          SUB_TILE_K = TILE_K
          x_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_I_W * TILE_I_H, TILE_K)
          w_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_K_W * TILE_K_H * TILE_K, TILE_N)
          y_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_O_H  * TILE_M, TILE_N)
          x_spad_size = TILE_I_W * TILE_I_H * TILE_K
          w_spad_size = TILE_K_W * TILE_K_H * TILE_K * TILE_N
          y_spad_size = TILE_O_H * TILE_M * TILE_N
          TOG_latency = O_W if TILE_M > O_W else TILE_M
        elif self.is_multi_tile(I_C):
          conv_template = MULTI_TILE_CONV_TEMPLATE
          TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K = kernel.conv_multi_tile_mapping(BATCH, O_C, I_C, K_H, K_W, O_H, O_W, self.stride, self.dilation, n_extra_node)
          TILE_I_W = 1 + (TILE_O_W - 1) * self.stride[1]
          TILE_I_H = 1 + (TILE_O_H - 1) * self.stride[0] + (TILE_K_H - 1) * self.dilation[0]
          SUB_TILE_K = TILE_K
          x_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_I_W * TILE_I_H * TILE_M, TILE_K)
          w_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_K_H * TILE_K, TILE_N)
          y_spad_size_per_lane = kernel.get_spad_size_per_lane(TILE_O_H * TILE_O_W * TILE_M, TILE_N)
          x_spad_size = TILE_I_W * TILE_I_H * TILE_M * TILE_K
          w_spad_size = TILE_K_H * TILE_K * TILE_N
          y_spad_size = TILE_O_H * TILE_O_W * TILE_M * TILE_N
        SUB_TILE_N = TILE_N if TILE_N > 512 else SUB_TILE_N
        TOG_latency = 8 if TOG_latency < 8 else TOG_latency
        kernel.loop_size = [TOG_latency, TILE_N, TILE_K]

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            X=X,
            W=W,
            BIAS=Bias,
            Y=Y,
            PADDED_INPUT_SIZE=self.get_padded_input_size(X),
            BATCH=X.layout.size[0],
            I_C=X.layout.size[1],
            I_H=X.layout.size[2],
            I_W=X.layout.size[3],
            O_C=O_C,
            K_H=K_H,
            K_W=K_W,
            O_H=O_H,
            O_W=O_W,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            TILE_I_H=TILE_I_H,
            TILE_I_W=TILE_I_W,
            TILE_O_H=TILE_O_H,
            TILE_O_W=TILE_O_W,
            TILE_K_H=TILE_K_H,
            TILE_K_W=TILE_K_W,
            SUB_TILE_M=SUB_TILE_M,
            SUB_TILE_N=SUB_TILE_N,
            SUB_TILE_K=SUB_TILE_K,
            SUB_TILE_I_H=SUB_TILE_I_H,
            SUB_TILE_I_W=SUB_TILE_I_W,
            SUB_TILE_K_H=SUB_TILE_K_H,
            SUB_TILE_K_W=SUB_TILE_K_W,
            PADDING_H=self.padding[0],
            PADDING_W=self.padding[1],
            STRIDE_H=self.stride[0],
            STRIDE_W=self.stride[1],
            DILATION_H=self.dilation[0],
            DILATION_W=self.dilation[1],
            DATA_STYPE="f32",
            DATA_SIZE=4,
            input_reorder=self.input_reorder
        )

        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            dependent_buf = [],
            sram_var = "output_buffer",
            dram_var = "Y",
            index_var = "index0",
            tag_var = "tag",
            vlane_split_axis = 3,
            vlane_stride = 1,
            mlir_dtype = kernel.render_options['DATA_STYPE'],
            dram_shape = f"memref<{BATCH * O_C * O_H * O_W}x{kernel.render_options['DATA_STYPE']}>",
            tile_size = (TILE_O_H, TILE_O_W, TILE_M, TILE_N) if conv_template in (CONV_TEMPLATE, MULTI_TILE_CONV_TEMPLATE) else (1, TILE_O_H, TILE_M, TILE_N),
            tile_stride = [TILE_O_W * TILE_M * TILE_N, TILE_M * TILE_N, 1, TILE_M]
        )
        code = self._template_from_string(conv_template).render(**kernel.render_options)
        self.header = f"float X_spad[{x_spad_size_per_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float W_spad[{w_spad_size_per_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{y_spad_size_per_lane}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header = f"float X_spad[{x_spad_size}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float W_spad[{w_spad_size}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float Y_spad[{y_spad_size}] __attribute__ ((section(\".spad\")));\n"

        kernel.add_loop_info([kernel.render_options["K_H"], kernel.render_options["K_W"], kernel.render_options["O_H"], kernel.render_options["O_W"], kernel.render_options["BATCH"], kernel.render_options["O_C"], kernel.render_options["I_C"]], [kernel.render_options["TILE_M"], kernel.render_options["TILE_N"], kernel.render_options["TILE_K"]])

        return code

    def outer_func_render(self, kernel_name, input_args):
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        eager_mode = int(os.environ.get('BACKENDSIM_EAGER_MODE', default=False))
        options = dict(
            kernel=self.kernel,
            KERNEL_NAME=kernel_name,
            FUNC_NAME=self.function_name + f"_{len(input_args)}",
            INPUT=X,
            WEIGHT=W,
            BIAS=Bias,
            OUTPUT=Y,
            PADDING_H=self.padding[0],
            PADDING_W=self.padding[1],
            MULTI_TILE=self.is_multi_tile(self.input_shape[1]),
            SINGLE_BATCH=self.is_single_batch(self.input_shape[0]),
            VALIDATION_MODE=extension_config.CONFIG_TORCHSIM_VALIDATION_MODE,
            BACKENDSIM_EAGER_MODE=eager_mode,
            input_reorder=self.input_reorder
        )
        code = self._template_from_string(WRAPPER_TEMPLATE).render(**options)
        return code, self.function_name + f"_{len(input_args)}"

    def get_arg_attributes(self):
        arg_attributes = []

        X = self.input_nodes[0]
        X_shape = [X.get_size()[i] for i in (2, 3, 0, 1)]
        X_shape[0] += 2 * self.padding[0]
        X_shape[1] += 2 * self.padding[1]

        def compute_stride(shape):
            stride = [1] * len(shape)
            for i in range(len(shape)-2, -1, -1):
                stride[i] = stride[i+1] * shape[i+1]
            return stride

        X_stride = compute_stride(X_shape)
        arg_attributes.append([X.data.data.name, [MLIRKernelArgs.MLIR_ARGS_IN, X.layout.dtype, math.prod(X_shape), X_shape, X_stride]])

        return arg_attributes

    def codegen_header(self, code, extra_headers):
        write_path = extension_codecache.get_write_path(code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        spike_write_path = os.path.join(write_path, "global_var.h")
        gem5_write_path = os.path.join(write_path, "gem5_global_var.h")
        if not os.path.exists(spike_write_path):
            write_atomic(spike_write_path, self.header+extra_headers[0])
        if not os.path.exists(gem5_write_path):
            write_atomic(gem5_write_path, self.gem5_header+extra_headers[1])
        self.hash_value = get_hash(code.strip())