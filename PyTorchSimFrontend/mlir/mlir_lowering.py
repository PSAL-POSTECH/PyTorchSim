from typing import List, Optional, Sequence

import torch
from torch._inductor.lowering import lowerings
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor import ir
from torch._inductor.virtualized import V
from torch._inductor.ir import TensorBox
from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate
from PyTorchSimFrontend.mlir.mlir_conv_template import MLIRConvTemplate
from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate

aten = torch.ops.aten

def tuned_mm(mat1, mat2, * ,layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    mlir_template = MLIRGemmTemplate([mat1, mat2], layout)

    return mlir_template.generate(input_nodes=[mat1, mat2], layout=layout).output_node()

def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    mlir_template = MLIRGemmTemplate([mat1, mat2, inp_expanded], layout)

    return mlir_template.generate().output_node()

def tuned_bmm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    mlir_template = MLIRBMMTemplate([mat1, mat2], layout)

    return mlir_template.generate().output_node()

def conv_layout(
    x: TensorBox,
    weight: TensorBox,
    bias: Optional[TensorBox],
    stride: Sequence[int],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
) -> ir.Layout:
    """Determine output layout for a convolution"""
    with V.graph.fake_mode:
        output = torch.ops.aten.convolution(
            ir.ir_node_to_tensor(x, guard_shape=True),
            ir.ir_node_to_tensor(weight, guard_shape=True),
            ir.ir_node_to_tensor(bias, guard_shape=True),
            stride,
            tuple(V.graph.sizevars.size_hint(p) for p in padding),
            dilation,
            transposed,
            tuple(V.graph.sizevars.size_hint(p) for p in output_padding),
            groups,
        )
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())

    return ir.FixedLayout(
        x.get_device(),
        x.get_dtype(),
        sizes,
        stride,
    )

def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
):
    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)

    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }

    layout = conv_layout(x, weight, None, **kwargs)
    mlir_template = MLIRConvTemplate([x, weight, bias], layout, **kwargs)
    return mlir_template.generate().output_node()

def maxpool_layout(
    x: TensorBox,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
) -> ir.Layout:
    """Determine output layout for a maxpool"""
    with V.graph.fake_mode:
        output, _ = torch.ops.aten.max_pool2d_with_indices(
            ir.ir_node_to_tensor(x, guard_shape=True),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())

    return ir.FixedLayout(
        x.get_device(),
        x.get_dtype(),
        sizes,
        stride,
    )

def custom_maxpool(
    x: TensorBox,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1],
    ceil_mode: bool = False
):
    kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
    }
    layout = maxpool_layout(x, kernel_size, stride, padding, dilation, ceil_mode)
    mlir_template = MLIRMaxPoolTemplate([x], layout, **kwargs)
    return mlir_template.generate().output_node(), x # FIXME: x is dummy IRNode, indices are not used in our case

lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})
lowerings.update({getattr(aten.addmm, overload): tuned_addmm for overload in aten.addmm.overloads()})
lowerings.update({getattr(aten.convolution, overload): convolution for overload in aten.convolution.overloads()})
lowerings.update({getattr(aten.bmm, overload): tuned_bmm for overload in aten.bmm.overloads()})