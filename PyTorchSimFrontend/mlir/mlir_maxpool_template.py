import os
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import PyTorchSimFrontend.extension_codecache as extension_codecache

# This template only represents the DMA operations
TEMPLATE = r"""#map0 = affine_map<(d0, d1) -> (d0 * {{ W }} + d1)>
memref.global @X_spad : memref<{{ in_tile }}x{{ in_tile }}xf32, 1>
memref.global @Y_spad : memref<{{ out_tile }}x{{ out_tile }}xf32, 1>

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y")}} {
  %c_mvin = arith.constant 2 : index
  %c_mvout = arith.constant 3 : index
  %axis = arith.constant 1 : index
  %vstride = arith.constant 1 : index
  %X_buffer = memref.get_global @X_spad : memref<{{ in_tile }}x{{ in_tile }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ out_tile }}x{{ out_tile }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %c0 = arith.constant 0 : index
  affine.for %i = 0 to {{ BCH }} step {{ out_tile }} {
    affine.for %j = 0 to {{ W }} step {{ out_tile }} {
      %index0 = affine.apply #map0(%i, %j)
      memref.dma_start %X[%index0], %X_buffer[%c0, %c0], %c_mvin, %tag[%c0], %axis, %vstride : memref<{{ IN }}xf32>, memref<{{ in_tile }}x{{ in_tile }}xf32, 1>, memref<1xi32>
      memref.dma_start %Y_buffer[%c0, %c0], %Y[%index0], %c_mvout, %tag[%c0], %axis, %vstride : memref<{{ out_tile }}x{{ out_tile }}xf32, 1>, memref<{{ OUT }}xf32>, memref<1xi32>
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

class MLIRMaxPoolTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, kernel_size, stride, padding, dilation, ceil_mode):
        super().__init__("kernel", input_nodes, layout)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])
        X = self.input_nodes[0]
        Y = self.output_node
        out_tile = kernel.vector_lane
        in_tile = self.stride[0] * (out_tile - 1) + self.dilation[0] * (self.kernel_size[0] - 1) + 1 # padding should be considered? - 2 * self.padding
        B = Y.get_size()[0]
        C = Y.get_size()[1]
        H = Y.get_size()[2]
        W = Y.get_size()[3]
        BCH = B * C * H
        kernel.loop_size = None
        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            IN=X.get_numel(),
            OUT=Y.get_numel(),
            X=X,
            Y=Y,
            BCH=BCH,
            W=W,
            in_tile=in_tile,
            out_tile=out_tile,
            DATA_STYPE="f32",
        )
        kernel.epilogue_info = dict(
            output_node = self.output_node.name,
            dependent_buf = [],
            sram_var = "Y_buffer",
            dram_var = "Y",
            index_var = "index0",
            tag_var = "tag",
            vlane_split_axis = 1,
            vlane_stride = 1,
            mlir_dtype = kernel.render_options['DATA_STYPE'],
            tile_nr_dim = 2,
            dram_shape = f"memref<{kernel.render_options['OUT']}x{kernel.render_options['DATA_STYPE']}>",
            tile_shape = f"memref<{out_tile}x{out_tile}x{kernel.render_options['DATA_STYPE']}, 1>",
            tile_size = (out_tile, out_tile),
            tile_stride = [1, out_tile]
        )
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        self.header = f"float X_spad[{in_tile * in_tile // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.header += f"float Y_spad[{out_tile * out_tile // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header = f"float X_spad[{in_tile * in_tile // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"
        self.gem5_header += f"float Y_spad[{out_tile * out_tile // kernel.vector_lane}] __attribute__ ((section(\".spad\")));\n"

        kernel.add_loop_info([kernel.render_options["IN"]], [kernel.vector_lane, kernel.vector_lane])
        return code

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