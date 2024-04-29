from typing import List, Optional, Sequence

import torch
from torch._inductor.lowering import lowerings
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor import ir
from torch._inductor.virtualized import V
from torch._inductor.ir import TensorBox
from PyTorchSimFrontend.llvm_gemm_template import LLVMGemmTemplate
from PyTorchSimFrontend.llvm_conv_template import LLVMConvTemplate

aten = torch.ops.aten

def tuned_mm(mat1, mat2, * ,layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    llvm_template = LLVMGemmTemplate([mat1, mat2], layout)

    return llvm_template.generate().output_node()

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
    llvm_template = LLVMConvTemplate([x, weight, bias], layout, **kwargs)
    return llvm_template.generate(kernel_caller_function="Conv2d").output_node()

lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})
lowerings.update({getattr(aten.convolution, overload): convolution for overload in aten.convolution.overloads()})