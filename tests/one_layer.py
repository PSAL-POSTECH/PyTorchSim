"""One decoder layer of a real-size model on the device at a TPU v6e grid point: ``--seq S`` a prefill,
``--decode S`` a step at S - 1 writing its K and V into slot S - 1 in place, as the TPU decode row does;
inputs from the float32 reference, rounded to the forward's dtypes. ``import one_layer`` (tests/ on path)."""

import contextlib
import copy

import torch

from result_check import check_result, f32, f32_reference


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


def _writes_in_place(cache, device, dtype=None):
    """`cache` (CPU) on `device` in `dtype`, each layer's K and V grown by one empty slot that update fills in
    place with the step's own K and V, at a Python-int slot so the store address is a compile-time constant
    (tokenwise into the page tensor itself); the layer gets the whole buffer."""
    from pytorchsim import config
    slots, pages = [], []
    for layer in cache.layers:
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None:
            slots.append(None)
            pages.append(None)
            continue
        kv = [torch.cat([t, t.new_zeros(*t.shape[:2], 1, t.shape[3])], 2).to(dtype or t.dtype)
              for t in (keys, values)]
        tokenwise = config.CONFIG_KVCACHE_LAYOUT == "tokenwise" and kv[0].shape == kv[1].shape
        page = _get_pages(*kv).to(device) if tokenwise else None
        layer.keys, layer.values = _get_views(page) if tokenwise else [t.to(device).contiguous() for t in kv]
        pages.append(page)
        slots.append(int(keys.shape[2]))
    base = type(cache)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        slot, page = self._slots[layer_idx], self._pages[layer_idx]
        if slot is None:
            return base.update(self, key_states, value_states, layer_idx, *args, **kwargs)
        if page is not None:
            new = torch.stack([key_states, value_states], 2).permute(0, 3, 1, 2, 4).flatten(2, 3)
            page[:, slot:slot + 1].copy_(new)
            return _get_views(page)
        layer = self.layers[layer_idx]
        layer.keys[:, :, slot:slot + 1].copy_(key_states)
        layer.values[:, :, slot:slot + 1].copy_(value_states)
        return layer.keys, layer.values

    cache.__class__ = type(f"InPlace{base.__name__}", (base,), {"update": update})
    cache._slots, cache._pages = slots, pages
    return cache


def _check_written(ref_cache, device_cache, before):
    """Each layer's written slot of `device_cache` against `ref_cache`'s, and every slot before it as placed
    (`before`); prints a line, not a verdict (run_grid counts one per point): a mismatch fails through
    check_result."""
    for i, slot in enumerate(device_cache._slots):
        if slot is None:
            continue
        for name in ("keys", "values"):
            got = getattr(device_cache.layers[i], name).cpu()
            want = getattr(ref_cache.layers[i], name)[:, :, -1:].float()
            if not torch.equal(got[:, :, :-1], before[i][name]):
                raise AssertionError(f"layer {i} {name}: a slot before the step's moved")
            what, tol = f"KV cache layer {i} {name} slot {got.shape[2] - 1}", 5e-2 * float(want.abs().max())
            if not torch.allclose(got[:, :, -1:].float(), want, rtol=5e-2, atol=tol):
                check_result(what, got[:, :, -1:], want, rtol=5e-2, atol=tol)
            print(f"{what} written: max diff {(got[:, :, -1:].float() - want).abs().max().item():.3g}")


def _paged(k, v):
    """K and V [B, H, S, D] as views of one tokenwise page tensor [B, S, 2H, D], each head's K then its V, as
    ragged paged attention's kv_pages hold them (pages in order: page_indices the identity, as the TPU row runs)."""
    if k.shape != v.shape:
        return k, v
    return _get_views(_get_pages(k, v))


def _get_pages(k, v):
    """The tokenwise page tensor [B, S, 2H, D] of K and V [B, H, S, D]: head h's K at 2h, its V at 2h + 1."""
    b, h, s, d = k.shape
    pages = torch.empty(b, s, 2 * h, d, dtype=k.dtype, device=k.device)
    pages[:, :, 0::2] = k.transpose(1, 2)
    pages[:, :, 1::2] = v.transpose(1, 2)
    return pages


def _get_views(pages):
    """K and V [B, H, S, D] as views of tokenwise `pages`."""
    return pages[:, :, 0::2].transpose(1, 2), pages[:, :, 1::2].transpose(1, 2)


def _swap(x, old, new):
    """`x` (args, kwargs) with the object `old` replaced by `new` where it is a direct argument."""
    args, kwargs = x
    return (tuple(new if a is old else a for a in args),
            {k: new if v is old else v for k, v in kwargs.items()})


def _drop_plain_mask(kwargs):
    """`kwargs` without an attention_mask saying only what no mask does (all keys at L 1, causal at L == S),
    as the TPU rows pass none: ragged paged attention and causal splash read the shared KV heads, unmasked."""
    m = kwargs.get("attention_mask")
    if not torch.is_tensor(m) or m.dim() != 4 or m.shape[-2] not in (1, m.shape[-1]):
        return kwargs
    keep = m if m.dtype == torch.bool else m == 0
    if m.dtype != torch.bool and not torch.all((m == 0) | (m <= torch.finfo(m.dtype).min / 2)):
        return kwargs
    q, k = m.shape[-2:]
    if not torch.equal(keep, torch.ones(q, k, dtype=torch.bool).tril(k - q).expand_as(keep)):
        return kwargs
    return {**kwargs, "attention_mask": None}


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
            step_cache = ref_cache
            ref_cache = _writes_in_place(_cache_like(rounded_cache, torch.float32, cpu), cpu, torch.float32)
            device_cache = _writes_in_place(_cache_like(rounded_cache, rounded_cache, cpu), device)
            before = [None if t is None else {n: getattr(l, n)[:, :, :-1].cpu() for n in ("keys", "values")}
                      for l, t in zip(device_cache.layers, device_cache._slots)]
            args, kwargs = _swap((args, kwargs), step_cache, ref_cache)
    else:
        probe = _run_until(body, layer, False, **point.tokens(0, 2, use_cache=False))
        args, kwargs = _run_until(ref_body, ref_layer, False, **point.tokens(0, s, use_cache=False))
    kwargs = _drop_plain_mask(kwargs)
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
    with torch.npu.TOGSimulator():
        device_out = _first_tensor(device_layer(*device_args, **device_kwargs))
    if decode:
        _check_written(ref_cache, device_cache, before)
    return ref_out, device_out


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
        with _given_kv(mla.config._attn_implementation, {}), torch.npu.TOGSimulator():
            return ref_out, _first_tensor(device_layer(*device_args, **device_kwargs))
    finally:
        del mla._given_kv
