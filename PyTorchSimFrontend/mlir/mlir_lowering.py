import math
from typing import List, Optional, Sequence

import torch
from torch._inductor.lowering import lowerings, index_impl
from torch._inductor.kernel.mm_common import mm_args
# from torch._inductor.select_algorithm import ExternKernelChoice
from torch._inductor import ir
from torch._inductor.virtualized import V
from torch._inductor.ir import TensorBox
from PyTorchSimFrontend.extension_op import MLIRExternKernelChoice
from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate
from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate
from PyTorchSimFrontend.mlir.mlir_conv_template import MLIRConvTemplate
from PyTorchSimFrontend.mlir.mlir_conv_mt_template import MLIRConvMultiTileTemplate
from PyTorchSimFrontend.mlir.mlir_conv_sb_template import MLIRConvSingleBatchTemplate
from PyTorchSimFrontend.mlir.mlir_conv_sbs_template import MLIRConvSingleBatchStridedTemplate
from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate
from PyTorchSimFrontend.mlir.mlir_cat_template import MLIRCatTemplate
from PyTorchSimFrontend.mlir.mlir_sort_template import MLIRSortTemplate, MLIRStableSortTemplate
from PyTorchSimFrontend import extension_config

aten = torch.ops.aten
aten_spmm = MLIRExternKernelChoice(torch.sparse.mm, "custom_op::sparse_addmm")
_orig_sort_values_stable_lowering = lowerings.get(aten.sort.values_stable)

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

    x.realize()
    weight.realize()
    x = ir.ExternKernel.require_channels_last(x)
    BATCH = x.layout.size[0]
    I_C = x.layout.size[1]
    weight = ir.ExternKernel.require_channels_last(weight)
    layout = conv_layout(x, weight, None, **kwargs)

    # Select conv kernel
    if BATCH == 1 and stride[0] == 1 and extension_config.CONFIG_SINGLE_BATCH_CONV:
        mlir_template = MLIRConvSingleBatchTemplate([x, weight, bias], layout, **kwargs)
    elif BATCH == 1 and stride[0] != 1 and extension_config.CONFIG_SINGLE_BATCH_CONV:
        mlir_template = MLIRConvSingleBatchStridedTemplate([x, weight, bias], layout, **kwargs)
    elif I_C < extension_config.vpu_num_lanes // 8 and extension_config.CONFIG_MULTI_TILE_CONV: # 8 is hard-coded for now. This should be changed to a better heuristic.
        mlir_template = MLIRConvMultiTileTemplate([x, weight, bias], layout, **kwargs)
    else:
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
    x.realize()
    template_node = mlir_template.generate().output_node()
    return template_node, x # FIXME: x is dummy IRNode, indices are not used in our case

def sparse_addmm(*args, **kwargs):
    _, sp_mat1, sp_mat2 = args
    mat1_layout = sp_mat1.layout
    out_range = args[0].data.data.data.ranges
    size = [out_range[i] for i in args[0].data.dims]
    layout = ir.FlexibleLayout(
            device=mat1_layout.device, dtype=mat1_layout.dtype, size=size  # FIXME: Example code for aten op overwrite by externkernel call
        )
    return aten_spmm.bind((sp_mat1, sp_mat2), layout).output_node()

def custom_unsafe_index(x, indices):
    # We can't fuse indirect access + indexed_expression + computation
    if isinstance(x, TensorBox):
        x.realize()
    return index_impl(x, indices, check=False)


def _cat_layout(tensors: Sequence[TensorBox], dim: int) -> ir.Layout:
    with V.graph.fake_mode:
        output = torch.ops.aten.cat(
            [ir.ir_node_to_tensor(t, guard_shape=True) for t in tensors],
            dim,
        )
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())
    return ir.FixedLayout(
        tensors[0].get_device(),
        tensors[0].get_dtype(),
        sizes,
        stride,
    )

def custom_cat_default(tensors: Sequence[TensorBox], dim: int = 0):
    if tensors and dim < 0:
        dim += len(tensors[0].get_size())
    copy_default_lowering = lowerings.get(aten.copy_.default)
    empty_strided_lowering = lowerings.get(aten.empty_strided.default)
    new_tensors = []
    for t in tensors:
        t.realize()
        # If the tensor is backed by a view (ReinterpretView, PermuteView, etc.),
        # materialise it into a fresh contiguous FixedLayout buffer so the cat
        # kernel always receives plain, dense strides.
        if isinstance(t.data, ir.BaseView):
            sizes = list(t.get_size())
            strides = [math.prod(sizes[i + 1:]) for i in range(len(sizes))]
            new_buf = empty_strided_lowering(
                sizes, strides, dtype=t.get_dtype(), device=t.get_device()
            )
            tt = copy_default_lowering(new_buf, t)
        else:
            tt = t
        new_tensors.append(tt)

    layout = _cat_layout(new_tensors, dim)
    mlir_template = MLIRCatTemplate(list(new_tensors), layout, dim=dim)
    return mlir_template.generate().output_node()

def custom_sort_default(
    value: TensorBox,
    dim: int = -1,
    descending: bool = False,
    stable: Optional[bool] = None,
):
    if dim < 0:
        dim += len(value.get_size())

    value.realize()

    value_layout, index_layout = _sort_layouts(value, dim, descending)
    empty_strided_lowering = lowerings.get(aten.empty_strided.default)
    indices = empty_strided_lowering(
        value.get_size(),
        index_layout.stride,
        dtype=torch.int64,
        device=value.get_device(),
    )
    stable_required = True if stable is None else stable
    sort_template_cls = MLIRStableSortTemplate if stable_required else MLIRSortTemplate
    mlir_template = sort_template_cls(
        [value, indices],
        value_layout,
        dim=dim,
        descending=descending,
        stable=stable_required,
    )
    sorted_values = mlir_template.generate(template_buffer_node=value).output_node()
    return sorted_values, indices


def _sort_layouts(x: TensorBox, dim: int, descending: bool):
    with V.graph.fake_mode:
        v, i = torch.ops.aten.sort(
            ir.ir_node_to_tensor(x, guard_shape=True),
            dim,
            descending,
        )
        v_sizes = ir.convert_shape_to_inductor(v.size())
        v_stride = ir.convert_shape_to_inductor(v.stride())
        i_sizes = ir.convert_shape_to_inductor(i.size())
        i_stride = ir.convert_shape_to_inductor(i.stride())

    value_layout = ir.FixedLayout(x.get_device(), x.get_dtype(), v_sizes, v_stride)
    index_layout = ir.FixedLayout(x.get_device(), torch.int64, i_sizes, i_stride)
    return value_layout, index_layout

lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})
lowerings.update({getattr(aten.addmm, overload): tuned_addmm for overload in aten.addmm.overloads()})
lowerings.update({getattr(aten.convolution, overload): convolution for overload in aten.convolution.overloads()})
lowerings.update({getattr(aten.bmm, overload): tuned_bmm for overload in aten.bmm.overloads()})
lowerings.update({getattr(aten._sparse_addmm, overload): sparse_addmm for overload in aten._sparse_addmm.overloads()})
lowerings.update({getattr(aten._unsafe_index, overload): custom_unsafe_index for overload in aten._unsafe_index.overloads()})
lowerings.update({getattr(aten.cat, overload): custom_cat_default for overload in aten.cat.overloads()})
lowerings.update({getattr(aten.sort, overload): custom_sort_default for overload in aten.sort.overloads()})
    
if extension_config.CONFIG_USE_TIMING_POOLING:
    lowerings.update({getattr(aten.max_pool2d_with_indices, overload): custom_maxpool for overload in aten.max_pool2d_with_indices.overloads()}) # FIXME: maxpool should be implemented as a template
