"""One decoder layer of a real-size model test alone on the device, at one point of the TPU v6e grid,
``--seq S`` a prefill of S tokens, ``--decode S`` one step at position S - 1 that reads a cache of S already
filled and writes nothing to it, as the TPU decode row does (`_reads_given`); an MLA layer's attention is handed
the step's K and V whole, as the TPU harness does, see `_given_kv`. The layer's inputs and cache come from the model's
float32 reference (result_check.f32_reference) on the CPU, rounded to the dtypes the model's own forward
gives them; the device runs the model's layer on those, the reference the same values in float32.

Imported the way result_check is (tests/ on sys.path): ``import one_layer``.
"""

import contextlib
import copy

import torch

from result_check import f32, f32_reference


class _Stop(Exception):
    """Raised by a hook to end a forward at the first layer."""


def add_grid_args(parser, decode=True, seq=128):
    """--seq S (prefill, default `seq`) and, for a model that takes a KV cache, --decode S."""
    point = parser.add_mutually_exclusive_group()
    point.add_argument("--seq", type=int, default=seq, help="prefill: S tokens")
    if decode:
        point.add_argument("--decode", type=int, default=None, metavar="S",
                           help="one decode step at position S - 1 after a KV cache of S - 1 (--part layer)")


def _like(x, proto, device=None, swap=(None, None)):
    """`x` with every floating tensor cast to the dtype of its counterpart in `proto` (the same structure, or
    one torch.dtype for all; a missing counterpart keeps the tensor's own), moved to `device`, and the object
    swap[0] replaced by swap[1]."""
    if swap[0] is not None and x is swap[0]:
        return swap[1]
    if torch.is_tensor(x):
        dtype = proto if isinstance(proto, torch.dtype) else getattr(proto, "dtype", x.dtype)
        return x.to(device or x.device, dtype if x.is_floating_point() else x.dtype)
    if isinstance(x, (tuple, list)):
        return type(x)(_like(v, _part(proto, i), device, swap) for i, v in enumerate(x))
    if isinstance(x, dict):
        return {k: _like(v, _part(proto, k), device, swap) for k, v in x.items()}
    return x


def _part(proto, key):
    """`proto`'s entry `key` (None when it has none), or `proto` itself when it is a torch.dtype."""
    if isinstance(proto, torch.dtype):
        return proto
    if isinstance(proto, dict):
        return proto.get(key)
    if isinstance(proto, (tuple, list)) and key < len(proto):
        return proto[key]
    return None


def _first_tensor(out):
    """The layer's hidden-state output, whether it returns a tensor or a tuple starting with one."""
    return out if torch.is_tensor(out) else out[0]


def _run_until(body, layer, after, **inputs):
    """(args, kwargs) `body`'s forward of `inputs` hands `layer`; the forward ends before `layer` runs, or
    right after it when `after`."""
    seen = {}

    def before_hook(module, args, kwargs):
        seen.update(args=args, kwargs=kwargs)
        if not after:
            raise _Stop

    def after_hook(module, args, output):
        raise _Stop

    handles = [layer.register_forward_pre_hook(before_hook, with_kwargs=True)]
    if after:
        handles.append(layer.register_forward_hook(after_hook))
    try:
        body(**inputs)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    if not seen:
        raise RuntimeError(f"{type(body).__name__}'s forward never reached {type(layer).__name__}")
    return seen["args"], seen["kwargs"]


def _cache_like(cache, proto, device):
    """A copy of `cache` on `device`, every tensor of every cache layer cast to the dtype of the same tensor
    in `proto` (a cache the device-dtype model filled) or to `proto` when it is a torch.dtype."""
    out = copy.deepcopy(cache)
    for i, layer in enumerate(out.layers):
        ref = proto if isinstance(proto, torch.dtype) else vars(proto.layers[i])
        for name, value in vars(layer).items():
            setattr(layer, name, _like(value, ref if isinstance(ref, torch.dtype) else ref[name], device))
        if hasattr(layer, "device"):
            layer.device = device
        if getattr(layer, "dtype", None) is not None:
            layer.dtype = proto if isinstance(proto, torch.dtype) else proto.layers[i].dtype
    return out


def _mla_attention(layer):
    """`layer`'s MLA attention (DeepSeek-V3's: kv_b_proj up-projects a latent cache every step), or None."""
    return next((m for m in layer.modules() if hasattr(m, "kv_b_proj") and hasattr(m, "kv_lora_rank")), None)


def _given_kv_of(step, filled):
    """Per cache layer, the K and V its update would hand the step (`step`'s S - 1, or its sliding window, then
    the step's own token, the last of `filled`), built once off the device; None where a layer holds none."""
    out = []
    for a, b in zip(step.layers, filled.layers):
        keys, values = getattr(a, "keys", None), getattr(a, "values", None)
        if keys is None or getattr(b, "keys", None) is None:
            out.append(None)
            continue
        out.append((torch.cat([keys, b.keys[:, :, -1:]], 2), torch.cat([values, b.values[:, :, -1:]], 2)))
    return out


def _reads_given(cache, given, device, dtype=None):
    """`cache` whose update hands each layer its `given` K and V (on `device`, in `dtype` or as they are) and
    writes nothing: the TPU decode row reads its full cache in place, with no copy of it per step."""
    given = [g if g is None else tuple(t.to(device, dtype or t.dtype).contiguous() for t in g) for g in given]
    base = type(cache)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if given[layer_idx] is None:
            return base.update(self, key_states, value_states, layer_idx, *args, **kwargs)
        return given[layer_idx]

    cache.__class__ = type(f"Reads{base.__name__}", (base,), {"update": update})
    return cache


def _swap(x, old, new):
    """`x` (args, kwargs) with the object `old` replaced by `new` where it is a direct argument."""
    args, kwargs = x
    return (tuple(new if a is old else a for a in args),
            {k: new if v is old else v for k, v in kwargs.items()})


@contextlib.contextmanager
def _given_kv(impl, seen):
    """transformers' attention `impl` wrapped: a module carrying `_given_kv` attends those K and V, not its own;
    seen[module] is the key and value each module attended."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as fns
    inner, local = fns[impl], fns._local_mapping.get(impl)

    def attend(module, query, key, value, *args, **kwargs):
        key, value = getattr(module, "_given_kv", (key, value))
        seen[module] = (key, value)
        return inner(module, query, key, value, *args, **kwargs)

    fns[impl] = attend
    try:
        yield
    finally:
        if local is None:
            del fns[impl]
        else:
            fns[impl] = local


class _Point:
    """The `body` inputs of one grid point: `tokens(a, b)` is its tokens a to b, with their positions and the
    same cut of every token-indexed `extra` input."""

    def __init__(self, ids, extra):
        self.ids, self.extra = ids, extra
        self.positions = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)

    def tokens(self, a, b, **more):
        cut = {k: (v[:, a:b] if torch.is_tensor(v) else v) for k, v in self.extra.items()}
        return dict(input_ids=self.ids[:, a:b], position_ids=self.positions[:, a:b], **cut, **more)


@torch.no_grad()
def run(body, layer, config, ids, device, decode=False, compile_model=True, **extra):
    """(float32 reference output, device output) of `layer`, one of `body`'s layers (those before it run on
    the CPU), at the grid point of `ids` [B, S]; `extra` holds token-indexed `body` inputs. A two-token run of
    `body` itself gives the dtypes; `layer` then moves to `device`, so its weights are never copied."""
    from transformers import DynamicCache

    ref_body = f32_reference(body)
    ref_layer = ref_body.get_submodule(next(n for n, m in body.named_modules() if m is layer))
    point, s, cpu = _Point(ids, extra), ids.shape[1], torch.device("cpu")
    ref_cache = device_cache = None
    if decode:
        probe_cache, cache = DynamicCache(config=config), DynamicCache(config=config)
        _run_until(body, layer, True, **point.tokens(0, 1, past_key_values=probe_cache, use_cache=True))
        probe = _run_until(body, layer, False,
                           **point.tokens(1, 2, past_key_values=probe_cache, use_cache=True))
        _run_until(ref_body, ref_layer, True, **point.tokens(0, s - 1, past_key_values=cache, use_cache=True))
        rounded_cache = _cache_like(cache, probe_cache, cpu)
        ref_cache = _cache_like(rounded_cache, torch.float32, cpu)
        device_cache = _cache_like(rounded_cache, rounded_cache, device)
        args, kwargs = _run_until(ref_body, ref_layer, False,
                                  **point.tokens(s - 1, s, past_key_values=ref_cache, use_cache=True))
        if _mla_attention(layer) is None:
            filled = DynamicCache(config=config)
            _run_until(ref_body, ref_layer, True, **point.tokens(0, s, past_key_values=filled, use_cache=True))
            given = _given_kv_of(rounded_cache, _cache_like(filled, probe_cache, cpu))
            step_cache = ref_cache
            ref_cache = _reads_given(_cache_like(rounded_cache, torch.float32, cpu), given, cpu, torch.float32)
            device_cache = _reads_given(_cache_like(rounded_cache, rounded_cache, device), given, device)
            args, kwargs = _swap((args, kwargs), step_cache, ref_cache)
    else:
        probe = _run_until(body, layer, False, **point.tokens(0, 2, use_cache=False))
        args, kwargs = _run_until(ref_body, ref_layer, False, **point.tokens(0, s, use_cache=False))
    rounded = _like((args, kwargs), probe)

    ref_args, ref_kwargs = f32(rounded)
    mla, ref_mla = _mla_attention(layer), _mla_attention(ref_layer)
    if decode and mla is not None:
        return _run_mla_decode(layer, ref_layer, mla, ref_mla, rounded, ref_args, ref_kwargs, ref_cache,
                               device, compile_model)
    ref_out = _first_tensor(ref_layer(*ref_args, **ref_kwargs))
    layer.to(device)
    device_layer = torch.compile(layer, dynamic=False) if compile_model else layer
    device_args, device_kwargs = _like(rounded, rounded, device, (ref_cache, device_cache))
    return ref_out, _first_tensor(device_layer(*device_args, **device_kwargs))


def _run_mla_decode(layer, ref_layer, mla, ref_mla, rounded, ref_args, ref_kwargs, ref_cache, device,
                    compile_model):
    """The TPU harness's MLA decode: no cache, the new token alone projected, its attention handed the K and V
    the reference step attended ([B, H, S, 192] and [B, H, S, 128]), contiguous in the layer's dtype."""
    seen, dtype = {}, next(layer.parameters()).dtype
    with _given_kv(ref_mla.config._attn_implementation, seen):
        ref_out = _first_tensor(ref_layer(*ref_args, **ref_kwargs))
    layer.to(device)
    mla._given_kv = tuple(t.to(device, dtype, memory_format=torch.contiguous_format) for t in seen[ref_mla])
    device_layer = torch.compile(layer, dynamic=False) if compile_model else layer
    device_args, device_kwargs = _like(rounded, rounded, device, (ref_cache, None))
    if "use_cache" in device_kwargs:
        device_kwargs["use_cache"] = False
    try:
        with _given_kv(mla.config._attn_implementation, {}):
            return ref_out, _first_tensor(device_layer(*device_args, **device_kwargs))
    finally:
        del mla._given_kv
