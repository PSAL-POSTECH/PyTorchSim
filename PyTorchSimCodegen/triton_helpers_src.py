"""The triton_helpers Inductor calls, vendored as SOURCE for the torch-free venv.

Pure triton -- @triton.jit over tl.* and nothing else -- which is what makes a
copy possible and is also the test for whether a helper belongs here. The NaN
handling is load-bearing: `mask |= a != a` is what tl.maximum does not do.
"""
VENDORED = {"promote_to_tensor", "is_floating",
            "minimum", "maximum", "min2", "max2", "any",
            "welford_reduce", "welford_combine", "welford",
            "sort_with_index", "select_one",
            "minimum_with_index", "maximum_with_index",
            "min_with_index", "max_with_index",
            "div_floor_integer", "remainder_integer"}


SRC = '''
import types as _types
import triton
import triton.language as tl

@triton.jit
def _torchsim_promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)

@triton.jit
def _torchsim_is_floating(x):
    return _torchsim_promote_to_tensor(x).dtype.is_floating()

@triton.jit
def _torchsim_minimum(a, b):
    mask = a < b
    if _torchsim_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)

@triton.jit
def _torchsim_maximum(a, b):
    mask = a > b
    if _torchsim_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)

@triton.jit
def _torchsim_min2(a, dim):
    return tl.reduce(a, dim, _torchsim_minimum)

@triton.jit
def _torchsim_max2(a, dim):
    return tl.reduce(a, dim, _torchsim_maximum)

@triton.jit
def _torchsim_any_combine(a, b):
    return a | b

@triton.jit
def _torchsim_any(a, dim):
    return tl.reduce(a, dim, _torchsim_any_combine)

@triton.jit
def _torchsim_welford_reduce(value, mean, m2, weight, first_iteration):
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2, new_weight

@triton.jit
def _torchsim_welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )

@triton.jit
def _torchsim_welford(mean, m2, weight, dim):
    return tl.reduce((mean, m2, weight), dim, _torchsim_welford_combine)

from triton.language.standard import _log2 as _torchsim_log2

@triton.jit
def _torchsim_compare_and_swap_with_index(
    x, idxs, rnumel, flip,
    i: tl.constexpr, n_dims: tl.constexpr,
    stable: tl.constexpr, descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    y = tl.reshape(x, shape)
    iy = y.to(idtype, bitcast=True)
    right_mask = tl.arange(0, 2)[None, :, None].to(idtype)
    left_mask = (1 - right_mask).to(idtype)
    ileft = tl.broadcast_to(tl.sum(iy * left_mask, 1).to(idtype)[:, None, :], shape)
    iright = tl.broadcast_to(tl.sum(iy * right_mask, 1).to(idtype)[:, None, :], shape)
    ileft = tl.reshape(ileft, x.shape)
    iright = tl.reshape(iright, x.shape)
    left = ileft.to(x.dtype, bitcast=True)
    right = iright.to(x.dtype, bitcast=True)

    y_idx = tl.reshape(idxs, shape)
    left_idx = tl.broadcast_to(
        tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    right_idx = tl.broadcast_to(
        tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    if rnumel is None:
        left_valid_mask = tl.full(x.shape, True, tl.int1)
        right_valid_mask = tl.full(x.shape, True, tl.int1)
    else:
        left_valid_mask = left_idx < rnumel
        right_valid_mask = right_idx < rnumel

    ix = x.to(idtype, bitcast=True)

    left_isnan = left != left
    right_isnan = right != right

    if descending:
        cond = left < right
        if _torchsim_is_floating(left):
            if not stable:
                cond = cond | right_isnan
            else:
                cond = cond | (right_isnan & (~left_isnan))
    else:
        cond = left > right
        if _torchsim_is_floating(left):
            if not stable:
                cond = cond | left_isnan
            else:
                cond = cond | (left_isnan & (~right_isnan))

    if stable:
        eq = left == right
        if _torchsim_is_floating(left):
            eq = eq | (left_isnan & right_isnan)
        cond = cond | (eq & (left_idx > right_idx))

    cond = (right_valid_mask > left_valid_mask) | (
        (right_valid_mask == left_valid_mask) & cond
    )
    cond = (cond ^ flip).to(tl.int1)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))

    return ret.to(x.dtype, bitcast=True), new_idxs

@triton.jit
def _torchsim_bitonic_merge_with_index(
    x, idxs, rnumel,
    stage: tl.constexpr, alternating: tl.constexpr, n_dims: tl.constexpr,
    stable: tl.constexpr, descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if alternating:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = False
    for i in tl.static_range(stage):
        x, idxs = _torchsim_compare_and_swap_with_index(
            x, idxs, rnumel, flip, i + (n_dims - stage), n_dims, stable, descending
        )
    return x, idxs

@triton.jit
def _torchsim_sort_with_index(
    x, idxs, rnumel,
    dim: tl.constexpr = None,
    stable: tl.constexpr = tl.constexpr(False),
    descending: tl.constexpr = tl.constexpr(False),
):
    x, idxs = tl.broadcast(x, idxs)
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    n_dims: tl.constexpr = _torchsim_log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, idxs = _torchsim_bitonic_merge_with_index(
            x, idxs, rnumel, i,
            alternating=i < n_dims, n_dims=n_dims,
            stable=stable, descending=descending,
        )
    return x, idxs

@triton.jit
def _torchsim_div_floor_integer(a, b):
    quot = a // b
    remainder = a % b
    fixed = tl.where(remainder != 0, quot - 1, quot)
    return tl.where((a < 0) != (b < 0), fixed, quot)

@triton.jit
def _torchsim_remainder_integer(a, b):
    remainder = a % b
    return tl.where((remainder != 0) & ((a < 0) != (b < 0)),
                    remainder + b, remainder)

@triton.jit
def _torchsim_maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if _torchsim_is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)

@triton.jit
def _torchsim_minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if _torchsim_is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)

@triton.jit
def _torchsim_max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, _torchsim_maximum_with_index)

@triton.jit
def _torchsim_min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, _torchsim_minimum_with_index)

@triton.jit
def _torchsim_select_one(x, mask, dim, keep_dims=False):
    idtype = tl.core.get_int_dtype(x.dtype.primitive_bitwidth, signed=False)
    ix = x.to(idtype, bitcast=True)
    iy = tl.sum(ix * mask, dim, keep_dims=keep_dims)
    return iy.to(x.dtype, bitcast=True)

triton_helpers = _types.ModuleType("triton_helpers")
triton_helpers.any = _torchsim_any
triton_helpers.welford_reduce = _torchsim_welford_reduce
triton_helpers.welford_combine = _torchsim_welford_combine
triton_helpers.welford = _torchsim_welford
triton_helpers.promote_to_tensor = _torchsim_promote_to_tensor
triton_helpers.is_floating = _torchsim_is_floating
triton_helpers.minimum = _torchsim_minimum
triton_helpers.maximum = _torchsim_maximum
triton_helpers.min2 = _torchsim_min2
triton_helpers.max2 = _torchsim_max2
triton_helpers.sort_with_index = _torchsim_sort_with_index
triton_helpers.select_one = _torchsim_select_one
triton_helpers.maximum_with_index = _torchsim_maximum_with_index
triton_helpers.minimum_with_index = _torchsim_minimum_with_index
triton_helpers.max_with_index = _torchsim_max_with_index
triton_helpers.min_with_index = _torchsim_min_with_index
triton_helpers.div_floor_integer = _torchsim_div_floor_integer
triton_helpers.remainder_integer = _torchsim_remainder_integer
'''
