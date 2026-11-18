"""The matmul family: a row times a matrix as vector work, matmuls with nothing to compute, and int8 matmuls
widened to int32."""

import torch
from torch import fx
from torch._inductor.pattern_matcher import Arg, CallFunction, CallFunctionVarArgs

from pytorchsim.frontend.inductor.postgrad import fake_value, register_rewrite

aten = torch.ops.aten
prims = torch.ops.prims


# mm_row, addmm_row: a 1 x K row times a matrix

def _is_row(match):
    """Real floating operands, the left a 1 x K row: an mm or a plain addmm (no beta or alpha)."""
    a, b = match.args[-2:]
    av, bv = fake_value(a), fake_value(b)
    return (av is not None and bv is not None and av.dim() == 2 and bv.dim() == 2
            and av.shape[0] == 1 and av.is_floating_point() and bv.is_floating_point())


def _row_times(a, b, acc=None):
    """The row stored first (a no-op when it already is), broadcast down the matrix, a multiply and a sum over
    K in `acc` when given (both operands widened to it): every block reads the stored row."""
    from torch._inductor import inductor_prims
    row = inductor_prims.force_stride_order(a, [a.shape[1], 1]).reshape(-1, 1)
    if acc is not None:
        row, b = row.to(acc), b.to(acc)
    return (row * b).sum(0, keepdim=True, dtype=acc)


@register_rewrite(CallFunction(aten.mm.default, Arg(), Arg()), _is_row)
def mm_row(node, a, b):
    """A 1 x K row times a K x N matrix as _row_times: vector work, where the mm template would run it on
    the array."""
    return (lambda a, b: _row_times(a, b)), [a, b]


@register_rewrite(CallFunction(aten.addmm.default, Arg(), Arg(), Arg()), _is_row)
def addmm_row(node, bias, a, b):
    """mm_row plus the bias."""
    return (lambda bias, a, b: _row_times(a, b) + bias), [bias, a, b]


# gemm_zero_length: matmuls with nothing to compute

def _is_zero_length(bias):
    """An mm-family node whose output is empty, or whose K is zero with no bias to add."""
    def check(match):
        node = match.output_node()
        out, a = fake_value(node), fake_value(node.args[bias])
        if out is None or a is None or not all(isinstance(s, int) for s in (*out.shape, *a.shape)):
            return False
        return out.numel() == 0 or (not bias and a.shape[-1] == 0)
    return check


def gemm_zero_length(node, *args, **kwargs):
    """A matmul with a zero-length M or N, or K and no bias, as zeros of its output: nothing to compute,
    where the mm template would be asked for an empty tile."""
    out = fake_value(node)
    shape, dtype = list(out.shape), out.dtype
    first = next(a for a in args if isinstance(a, fx.Node))
    return (lambda t: t.new_zeros(shape, dtype=dtype)), [first]


for _op, _bias in ((aten.mm.default, 0), (aten.bmm.default, 0), (aten.addmm.default, 1),
                   (aten.baddbmm.default, 1)):
    register_rewrite(CallFunctionVarArgs(_op), _is_zero_length(_bias))(gemm_zero_length)


# int_mm: an int8 matmul widened to int32 before an mm, as the int8 matmul it is

def _widened():
    """A fresh pattern for one operand converted to int32 (the matcher binds a pattern to one node)."""
    return CallFunction(prims.convert_element_type.default, Arg(), torch.int32)


def _is_int8_widened(match):
    """Both operands of the int32 mm are 2-D int8 tensors widened to int32."""
    vals = [fake_value(a) for a in match.args]
    return all(v is not None and v.dim() == 2 and v.dtype == torch.int8 for v in vals)


@register_rewrite(CallFunction(aten.mm.default, _widened(), _widened()), _is_int8_widened)
def int_mm(node, a, b):
    """mm(a.to(int32), b.to(int32)) of int8 a and b as aten._int_mm(a, b) on the mm template, a 1 x K row as
    _row_times summed in int32 (vector work, as mm_row): equal bit for bit to the int32 mm."""
    if fake_value(a).shape[0] == 1:
        return (lambda a, b: _row_times(a, b, torch.int32)), [a, b]
    return (lambda a, b: aten._int_mm(a, b)), [a, b]


# mm_widened_weight: an mm's right operand widened from a narrower dtype, stored row-major

def _is_widened_strided(match):
    """The right operand of a 2-D mm is a widening convert (int8 to bf16, say) whose result is not row-major,
    as a transposed weight's is: its strides follow the narrow source."""
    a, conv = match.args[0], match.output_node().args[1]
    src, wide = fake_value(conv.args[0]), fake_value(conv)
    return (src is not None and wide is not None and fake_value(a) is not None and wide.dim() == 2
            and src.dtype.itemsize < wide.dtype.itemsize and not wide.is_contiguous())


@register_rewrite(CallFunction(aten.mm.default, Arg(), CallFunction(prims.convert_element_type.default, Arg(),
                                                                   Arg())), _is_widened_strided)
def mm_widened_weight(node, a, w, dtype):
    """The widened right operand stored row-major (force_stride_order): the layout the mm template reads, so
    it can load the narrow source and widen it in place, where a transposed one conflicts and has no fallback."""
    from torch._inductor import inductor_prims
    n = fake_value(node).shape[1]
    return (lambda a, w: torch.mm(a, inductor_prims.force_stride_order(w.to(dtype), [n, 1]))), [a, w]
