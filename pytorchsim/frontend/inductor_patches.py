"""Our patches to Inductor so its Triton codegen serves npu, and install(), which applies them all.

Here: the lowering patches with no Inductor hook (a transposed conv lowered as its direct
equivalent, a grouped conv's grid per group, a zero-length GEMM as zeros) and npu's conv
decompositions. npu as a Triton GPU is triton_gpu.py, its choices choices.py, how a template's
config is picked selection.py, the ops with a Triton kernel of their own triton_ops/.
"""

import torch

from pytorchsim import config
from pytorchsim.frontend.choices import conv_groups
from pytorchsim.frontend.triton_gpu import is_npu, lowering_npu

_installed = False


def install():
    """Apply every patch once: npu as a Triton GPU, its choices, the lowering patches here, its
    template selection, the ops with a Triton kernel of their own, and Inductor's config."""
    global _installed
    if _installed:
        return
    from pytorchsim.frontend import choices, selection, triton_gpu, triton_ops

    triton_gpu.install()
    choices.install()
    _wrap_convolution_lowering()
    _size_grouped_conv_grid_per_group()
    _short_circuit_degenerate_gemms()
    _decompose_convolution_for_npu()
    selection.install()
    triton_ops.install()
    _configure_inductor()
    _installed = True


def _configure_inductor():
    """mm/conv template autotune on, at compile time off. In timing mode a template with an epilogue
    is picked by its fused cycles, Inductor timing the top configs one at a time in this process
    (TritonNPUScheduling.benchmark_codegened_module); select records picks, so its cache stays off."""
    from torch._inductor import config as inductor_config

    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    inductor_config.max_autotune_conv_backends = "ATEN,TRITON"
    inductor_config.triton.autotune_at_compile_time = False
    timed = bool(config.pytorchsim_timing_mode)
    inductor_config.benchmark_epilogue_fusion = timed
    inductor_config.max_epilogue_benchmarked_choices = config.codegen_autotune_template_topk
    inductor_config.autotune_local_cache = False
    if timed:
        inductor_config.compile_threads = 1


def _wrap_convolution_lowering():
    """Wrap aten.convolution's lowering: record its groups, and lower a transposed one as direct.

    The groups narrow the conv heuristic's BLOCK_N to one group's channels (choices.groups_now).
    """
    import functools
    import inspect

    from torch._inductor import ir
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.lowering import lowerings as L
    from torch._inductor.virtualized import V

    aten = torch.ops.aten
    prims = torch.ops.prims
    sig = inspect.signature(conv_kernel.convolution)

    def _per_dim(seq, n):
        seq = list(seq)
        return seq * n if len(seq) == 1 else seq

    def _spread(t, lead, extents, factor):
        """Put factor-1 zeros between neighbouring elements of each dim."""
        if all(f == 1 for f in factor):
            return t
        n = len(extents)
        paired = list(lead)
        for extent in extents:
            paired += [extent, 1]
        y = L[aten.view](t, paired)
        pad = []
        for i in reversed(range(n)):
            pad += [0, factor[i] - 1, 0, 0]
        y = L[aten.constant_pad_nd](y, pad, 0.0)
        y = L[aten.view](y, list(lead) +
                         [extents[i] * factor[i] for i in range(n)])
        for i in range(n):
            if factor[i] != 1:
                y = L[aten.slice](y, len(lead) + i, 0,
                                  extents[i] * factor[i] - (factor[i] - 1))
        return y

    def _rewrite(x, weight, stride, padding, dilation, output_padding, groups):
        """Return the (x, weight, *params) of the equivalent direct conv."""
        ints = V.graph.sizevars.guard_int_seq
        batch, in_chan, *spatial = ints(x.get_size())
        _, out_per_group, *kernel = ints(weight.get_size())
        n_sp = len(kernel)
        out_chan = out_per_group * groups
        stride = _per_dim(stride, n_sp)
        padding = _per_dim(padding, n_sp)
        dilation = _per_dim(dilation, n_sp)
        output_padding = _per_dim(output_padding, n_sp)

        y = _spread(x, [batch, in_chan], spatial, stride)

        pad = []
        for i in reversed(range(n_sp)):
            lo = dilation[i] * (kernel[i] - 1) - padding[i]
            pad += [lo, lo + output_padding[i]]
        if any(p != 0 for p in pad):
            y = L[aten.constant_pad_nd](y, pad, 0.0)

        w = L[aten.view](weight, [groups, in_chan // groups,
                                  out_chan // groups] + kernel)
        w = L[aten.permute](w, [0, 2, 1] + [3 + i for i in range(n_sp)])
        w = L[aten.view](w, [out_chan, in_chan // groups] + kernel)
        if any(k != 1 for k in kernel):
            w = L[prims.rev.default](w, [2 + i for i in range(n_sp)])

        w = _spread(w, [out_chan, in_chan // groups], kernel, dilation)

        return y, w, [1] * n_sp, [0] * n_sp, [1] * n_sp, [0] * n_sp

    def lower(inner, a, args, kwargs):
        """The lowering itself: a transposed npu conv as its direct equivalent, else inner."""
        if a is None:
            return inner(*args, **kwargs)
        x, weight = a["x"], a["weight"]
        if not a["transposed"] or ir.get_device_type(x) != "npu":
            return inner(*args, **kwargs)

        batchless = len(x.get_size()) == len(weight.get_size()) - 1
        if batchless:
            x = L[aten.expand](x, [1, *x.get_size()])

        y, w, stride, padding, dilation, output_padding = _rewrite(
            x, weight, a["stride"], a["padding"], a["dilation"],
            a["output_padding"], a["groups"])
        out = inner(y, w, a["bias"], stride, padding, dilation, False,
                    output_padding, a["groups"])
        return L[aten.squeeze](out, dim=0) if batchless else out

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
                groups = a.get("groups", 1)
            except TypeError:
                a, groups = None, 1
            prev = getattr(conv_groups, "value", 1)
            conv_groups.value = groups if isinstance(groups, int) else 1
            try:
                return lower(inner, a, args, kwargs)
            finally:
                conv_groups.value = prev

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _size_grouped_conv_grid_per_group():
    """Grid a grouped convolution over one group's channels, not all of them."""
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.select_algorithm import SymbolicGridFn

    orig2d = conv_kernel.conv2d_grid
    orig3d = conv_kernel.conv3d_grid

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        if not lowering_npu():
            return orig2d(n, c, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
        if not lowering_npu():
            return orig3d(n, c, d, h, w, meta)
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * d * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    conv_kernel.conv2d_grid = conv2d_grid
    conv_kernel.conv3d_grid = conv3d_grid
    for tmpl, fn in ((getattr(conv_kernel, "conv2d_template", None), conv2d_grid),
                     (getattr(conv_kernel, "conv3d_template", None), conv3d_grid)):
        if tmpl is not None:
            tmpl.grid = fn


def _short_circuit_degenerate_gemms():
    """Lower an npu mm/bmm/addmm/baddbmm with a zero-length M or N (or K, unbiased) as zeros."""
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.lowering import full, lowerings
    from torch._inductor.virtualized import V

    def wrap(op, bias):
        def wrapped(*args, _orig=lowerings[op], **kwargs):
            try:
                m, n, k, layout = mm_args(*args[bias:bias + 2],
                                          layout=kwargs.get("layout"))[:4]
                m, n, k = (int(V.graph.sizevars.optimization_hint(s)) for s in (m, n, k))
            except Exception:
                return _orig(*args, **kwargs)
            if not is_npu(layout):
                return _orig(*args, **kwargs)
            if m == 0 or n == 0 or (k == 0 and not bias):
                return full(layout.size, 0, dtype=layout.dtype,
                            device=layout.device)
            return _orig(*args, **kwargs)

        return wrapped

    aten = torch.ops.aten
    for op, bias in ((aten.mm, 0), (aten.bmm, 0),
                     (aten.addmm, 1), (aten.baddbmm, 1)):
        for name in op.overloads():
            o = getattr(op, name)
            if o in lowerings:
                lowerings[o] = wrap(o, bias)


def _decompose_convolution_for_npu():
    """Register npu's aten.convolution decompositions: a Conv1d reaches the 2-D template through
    torch's conv1d_to_conv2d, and a depthwise 2-D conv is a stencil on the vector unit. Declines
    non-npu inputs, transposed convs and output_padding; anything else keeps the template."""
    from torch._inductor.decomposition import conv1d_to_conv2d, register_decomposition

    aten = torch.ops.aten

    @register_decomposition([aten.convolution])
    def _npu_convolution(input, weight, bias, stride, padding, dilation,
                         transposed, output_padding, groups):
        if input.device.type != "npu" or transposed:
            return NotImplemented
        if any(p != 0 for p in output_padding):
            return NotImplemented
        if len(stride) == 1:
            return conv1d_to_conv2d(input, weight, bias, stride, padding, dilation, groups)
        if len(stride) == 2 and _is_depthwise(input, weight, groups):
            return _depthwise_stencil(input, weight, bias, stride, padding, dilation)
        return NotImplemented


def _is_depthwise(input, weight, groups):
    """One channel a group, in and out: the conv is a per-channel stencil, not a matmul."""
    return (groups > 1 and input.shape[1] == groups and weight.shape[0] == groups
            and weight.shape[1] == 1)


def _depthwise_stencil(x, w, bias, stride, padding, dilation):
    """A depthwise 2-D conv as taps: each a strided slice of the padded input times its weight, summed."""
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    g, _, kh, kw = w.shape
    x = torch.ops.aten.constant_pad_nd(x, [pw, pw, ph, ph])
    h_out = (x.shape[2] - dh * (kh - 1) - 1) // sh + 1
    w_out = (x.shape[3] - dw * (kw - 1) - 1) // sw + 1
    out = None
    for i in range(kh):
        for j in range(kw):
            tap = x[:, :, i * dh:i * dh + sh * (h_out - 1) + 1:sh,
                    j * dw:j * dw + sw * (w_out - 1) + 1:sw]
            term = tap * w[:, 0, i, j].reshape(1, g, 1, 1)
            out = term if out is None else out + term
    return out if bias is None else out + bias.reshape(1, g, 1, 1)
