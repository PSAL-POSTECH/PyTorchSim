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


def _row_times(a, b):
    """The row stored first (a no-op when it already is), broadcast down the matrix, a multiply and a sum over
    K: every block reads the stored row, where a row computed in the kernel is spread over the lanes per block."""
    from torch._inductor import inductor_prims
    return (inductor_prims.force_stride_order(a, [a.shape[1], 1]).reshape(-1, 1) * b).sum(0, keepdim=True)


@register_rewrite(CallFunction(aten.mm.default, Arg(), Arg()), _is_row)
def mm_row(node, a, b):
    """A 1 x K row times a K x N matrix as _row_times: vector work, where the mm template would run it on
    the array."""
    return _row_times, [a, b]


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
    """mm(a.to(int32), b.to(int32)) of int8 a and b as aten._int_mm(a, b), equal bit for bit: Inductor has no
    int32 mm template and runs it as an extern mm on the CPU, where _int_mm takes the mm template."""
    return (lambda a, b: aten._int_mm(a, b)), [a, b]
