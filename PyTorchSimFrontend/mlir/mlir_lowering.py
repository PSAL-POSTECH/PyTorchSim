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
from PyTorchSimFrontend.mlir.mlir_sort_template import MLIRSortTemplate
from PyTorchSimFrontend import extension_config

aten = torch.ops.aten
aten_spmm = MLIRExternKernelChoice(torch.sparse.mm, "custom_op::sparse_addmm")
_orig_cat_default_lowering = lowerings.get(aten.cat.default)
_orig_cat_out_lowering = lowerings.get(aten.cat.out)
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


def _can_use_cat_template(tensors: Sequence[TensorBox], dim: int) -> bool:
    # Current template specialization: 2 inputs, rank-2, dim in {0, 1}.
    if len(tensors) != 2:
        return False
    if not all(hasattr(t, "get_size") and hasattr(t, "get_dtype") and hasattr(t, "realize") for t in tensors):
        return False
    if tensors[0].get_dtype() != tensors[1].get_dtype():
        return False
    rank0 = len(tensors[0].get_size())
    rank1 = len(tensors[1].get_size())
    if rank0 != 2 or rank1 != 2:
        return False
    if dim < 0:
        dim += rank0
    if dim not in (0, 1):
        return False

    if dim == 0:
        cols0 = tensors[0].get_size()[1]
        cols1 = tensors[1].get_size()[1]
        return V.graph.sizevars.statically_known_equals(cols0, cols1)

    rows0 = tensors[0].get_size()[0]
    rows1 = tensors[1].get_size()[0]
    return V.graph.sizevars.statically_known_equals(rows0, rows1)


def _cat_fallback(reason: str, tensors: Sequence[TensorBox], dim: int):
    # Non-template cases delegate to the original lowering path.
    return _orig_cat_default_lowering(tensors, dim)


def _custom_cat_impl(tensors: Sequence[TensorBox], dim: int = 0):
    if _orig_cat_default_lowering is None:
        raise RuntimeError("Original aten.cat.default lowering is missing")
    if len(tensors) > 0:
        rank = len(tensors[0].get_size())
        if dim < 0:
            dim += rank
    if not _can_use_cat_template(tensors, dim):
        return _cat_fallback("default-path", tensors, dim)

    for t in tensors:
        t.realize()
    layout = _cat_layout(tensors, dim)
    mlir_template = MLIRCatTemplate(list(tensors), layout, dim=dim)
    return mlir_template.generate().output_node()


def custom_cat_default(tensors: Sequence[TensorBox], dim: int = 0):
    return _custom_cat_impl(tensors, dim)


def custom_cat_out(tensors: Sequence[TensorBox], dim: int = 0, out: Optional[TensorBox] = None):
    if _orig_cat_out_lowering is None:
        raise RuntimeError("Original aten.cat.out lowering is missing")
    if out is None:
        return _orig_cat_out_lowering(tensors, dim, out)

    copy_default_lowering = lowerings.get(aten.copy_.default)
    slice_tensor_lowering = lowerings.get(aten.slice.Tensor)
    if copy_default_lowering is None or slice_tensor_lowering is None:
        raise RuntimeError("cat.out lowering requires aten.copy_.default and aten.slice.Tensor lowerings")

    # Lower cat.out as a sequence of slice+copy ops so each piece still runs
    # through the existing compiled/simulated kernel path.
    if len(tensors) == 0:
        raise RuntimeError("cat.out requires at least one input tensor")
    if not all(hasattr(t, "get_size") and hasattr(t, "get_dtype") and hasattr(t, "realize") for t in tensors):
        raise RuntimeError("cat.out inputs must be tensor-like values")
    rank = len(tensors[0].get_size())
    if rank == 0:
        raise RuntimeError("cat.out does not support scalar inputs")
    if dim < 0:
        dim = dim + rank
    if dim < 0 or dim >= rank:
        raise RuntimeError(f"cat.out dim out of range: dim={dim}, rank={rank}")
    if any(len(t.get_size()) != rank for t in tensors):
        raise RuntimeError("cat.out inputs must have the same rank")
    if any(t.get_dtype() != tensors[0].get_dtype() for t in tensors):
        raise RuntimeError("cat.out inputs must have the same dtype")
    # cat semantics: all non-cat dimensions must be equal.
    for i in range(rank):
        if i == dim:
            continue
        base = tensors[0].get_size()[i]
        if any(not V.graph.sizevars.statically_known_equals(base, t.get_size()[i]) for t in tensors[1:]):
            raise RuntimeError(f"cat.out non-concatenated dimension mismatch at dim={i}")

    # Output shape must match concatenated shape.
    if not hasattr(out, "get_size"):
        raise RuntimeError("cat.out output must be tensor-like")
    out_sizes = list(out.get_size())
    if len(out_sizes) != rank:
        raise RuntimeError("cat.out output rank mismatch")
    for i in range(rank):
        if i == dim:
            continue
        if not V.graph.sizevars.statically_known_equals(out_sizes[i], tensors[0].get_size()[i]):
            raise RuntimeError(f"cat.out output shape mismatch at dim={i}")
    expected_cat = sum(t.get_size()[dim] for t in tensors)
    if not V.graph.sizevars.statically_known_equals(out_sizes[dim], expected_cat):
        raise RuntimeError(f"cat.out output concatenated dimension mismatch at dim={dim}")

    if isinstance(out, TensorBox):
        out.realize()

    offset = 0
    for src in tensors:
        src.realize()
        end = offset + src.get_size()[dim]
        dst_view = slice_tensor_lowering(out, dim, offset, end, 1)
        copy_default_lowering(dst_view, src)
        offset = end
    return out


def _custom_sort_values_impl(
    self: TensorBox,
    dim: int = -1,
    descending: bool = False,
    values: Optional[TensorBox] = None,
    indices: Optional[TensorBox] = None,
    stable: Optional[bool] = None,
):
    if values is None or indices is None:
        raise RuntimeError("sort.values* lowering requires both out tensors: values, indices")

    def _normalize_dim(rank: int, d: int) -> int:
        return d + rank if d < 0 else d

    if not hasattr(self, "get_size"):
        raise RuntimeError("sort.values* lowering requires TensorBox input")

    rank = len(self.get_size())
    norm_dim = _normalize_dim(rank, dim)
    if norm_dim < 0 or norm_dim >= rank:
        raise RuntimeError(f"sort.values* dim out of range: dim={dim}, rank={rank}")
    if rank != 2:
        raise RuntimeError(f"sort.values* lowering currently supports rank-2 only, got rank={rank}")
    if norm_dim not in (0, 1):
        raise RuntimeError(f"sort.values* lowering currently supports dim in {{0,1}} only, got dim={norm_dim}")

    self.realize()
    if isinstance(values, TensorBox):
        values.realize()
    if isinstance(indices, TensorBox):
        indices.realize()

    value_layout, _ = _sort_layouts(self, norm_dim, descending)
    mlir_template = MLIRSortTemplate(
        [self],
        value_layout,
        dim=norm_dim,
        descending=descending,
        stable=True if stable is None else stable,
        indices_node=indices,
    )
    sorted_values = mlir_template.generate(template_buffer_node=values, epilogue_nodes=[indices]).output_node()
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


def custom_sort_stable(
    self: TensorBox,
    *,
    stable: Optional[bool] = None,
    dim: int = -1,
    descending: bool = False,
):
    empty_strided_lowering = lowerings.get(aten.empty_strided.default)
    if empty_strided_lowering is None:
        if _orig_sort_values_stable_lowering is None:
            raise RuntimeError("sort.stable lowering requires aten.empty_strided.default")
        return _orig_sort_values_stable_lowering(self, dim=dim, descending=descending, stable=True)

    rank = len(self.get_size()) if hasattr(self, "get_size") else 0
    norm_dim = dim + rank if dim < 0 else dim
    if rank > 0 and (norm_dim < 0 or norm_dim >= rank):
        raise RuntimeError(f"sort.stable dim out of range: dim={dim}, rank={rank}")

    # Template specialization supports rank-2 and dim in {0,1}.
    if rank == 2 and norm_dim not in (0, 1):
        if _orig_sort_values_stable_lowering is None:
            raise RuntimeError("Original aten.sort.values_stable lowering is missing")
        return _orig_sort_values_stable_lowering(self, dim=dim, descending=descending, stable=True)

    try:
        value_layout, index_layout = _sort_layouts(self, norm_dim, descending)
        values = empty_strided_lowering(
            list(value_layout.size),
            list(value_layout.stride),
            dtype=value_layout.dtype,
            device=self.get_device(),
        )
        indices = empty_strided_lowering(
            list(index_layout.size),
            list(index_layout.stride),
            dtype=index_layout.dtype,
            device=self.get_device(),
        )
        return _custom_sort_values_impl(
            self=self,
            dim=dim,
            descending=descending,
            values=values,
            indices=indices,
            stable=True if stable is None else stable,
        )
    except Exception:
        if _orig_sort_values_stable_lowering is None:
            raise
        return _orig_sort_values_stable_lowering(self, dim=dim, descending=descending, stable=stable)


def custom_sort_values_stable(
    self: TensorBox,
    *,
    stable: Optional[bool] = None,
    dim: int = -1,
    descending: bool = False,
    values: Optional[TensorBox] = None,
    indices: Optional[TensorBox] = None,
):
    return _custom_sort_values_impl(
        self=self,
        dim=dim,
        descending=descending,
        values=values,
        indices=indices,
        stable=stable,
    )


lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})
lowerings.update({getattr(aten.addmm, overload): tuned_addmm for overload in aten.addmm.overloads()})
lowerings.update({getattr(aten.convolution, overload): convolution for overload in aten.convolution.overloads()})
lowerings.update({getattr(aten.bmm, overload): tuned_bmm for overload in aten.bmm.overloads()})
lowerings.update({getattr(aten._sparse_addmm, overload): sparse_addmm for overload in aten._sparse_addmm.overloads()})
lowerings.update({getattr(aten._unsafe_index, overload): custom_unsafe_index for overload in aten._unsafe_index.overloads()})

lowerings.update({aten.cat.default: custom_cat_default})
lowerings.update({aten.cat.out: custom_cat_out})

lowerings.update({aten.sort.stable: custom_sort_stable})
lowerings.update({aten.sort.values_stable: custom_sort_values_stable})
    
if extension_config.CONFIG_USE_TIMING_POOLING:
    lowerings.update({getattr(aten.max_pool2d_with_indices, overload): custom_maxpool for overload in aten.max_pool2d_with_indices.overloads()}) # FIXME: maxpool should be implemented as a template
