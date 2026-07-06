import math
from typing import Any, Callable, List, Optional, Sequence

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
from PyTorchSimFrontend.mlir.mlir_sdpa_template import (
    MLIRFlashSDPATemplate,
    flash_sdpa_args,
    calculate_scale,
)
from PyTorchSimFrontend import extension_config

aten = torch.ops.aten
aten_spmm = MLIRExternKernelChoice(torch.sparse.mm, "custom_op::sparse_addmm")
_orig_sort_values_stable_lowering = lowerings.get(aten.sort.values_stable)


def _device_is_npu(device: Optional[torch.device]) -> bool:
    return device is not None and device.type == "npu"


def _tensor_args_all_npu(*roots, optional=()) -> bool:
    """True only if every tensor-like IR node under roots/optional is on an NPU device."""
    stack: list = list(roots) + list(optional)
    while stack:
        n = stack.pop()
        if n is None:
            continue
        if isinstance(n, (list, tuple)):
            stack.extend(n)
            continue
        get_dev = getattr(n, "get_device", None)
        if get_dev is None:
            continue
        if not _device_is_npu(get_dev()):
            return False
    return True


def _override_lowerings_npu(
    aten_op: Any,
    mlir_impl: Callable[..., Any],
    npu_ok: Callable[..., bool],
) -> None:
    """Register mlir_impl for each overload; fall back to the prior lowering if npu_ok is false."""
    for overload in aten_op.overloads():
        op = getattr(aten_op, overload)
        orig = lowerings.get(op)

        def wrapped(*args, _orig=orig, **kwargs):
            if not npu_ok(*args, **kwargs):
                return _orig(*args, **kwargs)
            return mlir_impl(*args, **kwargs)

        lowerings[op] = wrapped


def _mlir_tuned_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    mlir_template = MLIRGemmTemplate([mat1, mat2], layout)

    return mlir_template.generate(input_nodes=[mat1, mat2], layout=layout).output_node()

def _mlir_tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    mlir_template = MLIRGemmTemplate([mat1, mat2, inp_expanded], layout)

    return mlir_template.generate().output_node()

def _mlir_tuned_bmm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    mlir_template = MLIRBMMTemplate([mat1, mat2], layout)

    return mlir_template.generate().output_node()


def _flash_decode_pad_query(query: TensorBox, tile_l: int, l_real: int) -> TensorBox:
    """Zero-pad the query length L up to tile_l so the flash kernel runs its
    verified prefill path (query rows fill the vector lanes). query (n, hq, L, e) ->
    (n, hq, tile_l, e); the extra rows are dropped again by _flash_decode_slice_output.
    """
    pad_amt = tile_l - l_real
    # constant_pad_nd pad spec is [e_lo, e_hi, l_lo, l_hi, ...]:
    # pad the l dim at the high end by pad_amt.
    return lowerings[aten.constant_pad_nd](query, [0, 0, 0, pad_amt], 0.0)


def _flash_decode_slice_output(out: TensorBox, l_real: int) -> TensorBox:
    """Slice the padded output (n, hq, tile_l, ev) back to the real L rows
    (n, hq, L, ev) along the query dim. Lowers to a free reinterpret/view."""

    # slice spec is [target, dim_idx, start_idx, len]
    return lowerings[aten.slice.Tensor](out, len(out.get_size()) - 2, 0, l_real)


def _mlir_tuned_flash_sdpa(
        query             : TensorBox,
        key               : TensorBox,
        value             : TensorBox,
        attn_bias         : Optional[TensorBox] = None,
        dropout_p         : float = 0.0,
        is_causal         : bool = False,
        return_debug_mask : bool = False,
        scale             : Optional[float] = None,
        enable_gqa        : bool = False) -> tuple:
    # _fused_sdp_choice in C++ already guarantees:
    # dropout_p == 0.0
    # before routing here via SDPBackend::overrideable.
    # Non-matching shapes fall back to SDPBackend::math in C++ and decompose
    # into primitive ops (matmul/softmax) before reaching this lowering.
    scale = calculate_scale(query, scale)

    tile_l = extension_config.vpu_num_lanes
    l_real = V.graph.sizevars.size_hint(query.get_size()[-2])
    hq = V.graph.sizevars.size_hint(query.get_size()[-3])
    h = V.graph.sizevars.size_hint(key.get_size()[-3])
    kv_ratio = 1

    enable_decode = l_real < tile_l
    enable_gqa    = (hq != h) and (hq % h == 0)

    if enable_decode:
        query = _flash_decode_pad_query(query, tile_l, l_real)
    if enable_gqa:
        kv_ratio = int(hq / h)   

    N, Hq, H, L, S, E, Ev, layout, query, key, value = flash_sdpa_args(query, key, value)
    mlir_template = MLIRFlashSDPATemplate([query, key, value], layout, scale, kv_ratio, is_causal)
    out = mlir_template.generate().output_node()

    if enable_decode:
        out = _flash_decode_slice_output(out, l_real)
    if enable_gqa:
        pass 

    return (out, None, None, None, None, None, None, None, None)


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

def _mlir_convolution(
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

def _mlir_custom_maxpool(
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

def _mlir_sparse_addmm(*args, **kwargs):
    _, sp_mat1, sp_mat2 = args
    mat1_layout = sp_mat1.layout
    out_range = args[0].data.data.data.ranges
    size = [out_range[i] for i in args[0].data.dims]
    layout = ir.FlexibleLayout(
            device=mat1_layout.device, dtype=mat1_layout.dtype, size=size  # FIXME: Example code for aten op overwrite by externkernel call
        )
    return aten_spmm.bind((sp_mat1, sp_mat2), layout).output_node()

def _mlir_custom_unsafe_index(x, indices):
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

def _mlir_custom_cat_default(tensors: Sequence[TensorBox], dim: int = 0):
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

def _mlir_custom_sort_default(
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

_override_lowerings_npu(
    aten.mm,
    _mlir_tuned_mm,
    lambda mat1, mat2, **_: _tensor_args_all_npu(mat1, mat2),
)
_override_lowerings_npu(
    aten.addmm,
    _mlir_tuned_addmm,
    lambda inp, mat1, mat2, **_: _tensor_args_all_npu(inp, mat1, mat2),
)
_override_lowerings_npu(
    aten.convolution,
    _mlir_convolution,
    lambda *a, **_: len(a) >= 2
    and _tensor_args_all_npu(a[0], a[1], optional=(a[2] if len(a) > 2 else None,)),
)
_override_lowerings_npu(
    aten.bmm,
    _mlir_tuned_bmm,
    lambda mat1, mat2, **_: _tensor_args_all_npu(mat1, mat2),
)
_override_lowerings_npu(
    aten._sparse_addmm,
    _mlir_sparse_addmm,
    lambda *a, **_: len(a) >= 3 and _tensor_args_all_npu(a[1], a[2]),
)
_override_lowerings_npu(
    aten._unsafe_index,
    _mlir_custom_unsafe_index,
    lambda x, indices, **_: _tensor_args_all_npu(x, indices),
)
_override_lowerings_npu(
    aten.cat,
    _mlir_custom_cat_default,
    lambda *a, **_k: a and _tensor_args_all_npu(a[0]),
)
_override_lowerings_npu(
    aten.sort,
    _mlir_custom_sort_default,
    lambda *a, **_k: a and _tensor_args_all_npu(a[0]),
)

if extension_config.CONFIG_USE_TIMING_POOLING:
    _override_lowerings_npu(
        aten.max_pool2d_with_indices,
        _mlir_custom_maxpool,
        lambda *a, **_: bool(a) and _tensor_args_all_npu(a[0]),
    )

_override_lowerings_npu(
    aten._scaled_dot_product_fused_attention_overrideable,
    _mlir_tuned_flash_sdpa,
    lambda *a, **k: len(a) >= 3
    and _tensor_args_all_npu(
        a[0],
        a[1],
        a[2],
        optional=(a[3] if len(a) > 3 else k.get("attn_bias"),),
    ),
)
