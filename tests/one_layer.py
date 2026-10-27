"""One decoder layer of a real-size model test alone on the device, at one point of the TPU v6e grid,
``--seq S`` a prefill of S tokens, ``--decode S`` one step at position S - 1 after a DynamicCache of the S - 1
before it (attention reads S keys, as the TPU decode row). The layer's inputs are what the model's own CPU
forward hands that layer; the device cache is a copy moved by prefetch(), so only the layer is timed.

Imported the way result_check is (tests/ on sys.path): ``import one_layer``.
"""

import copy

import torch


class _Stop(Exception):
    """Raised by a hook to end a forward at the first layer."""


def add_grid_args(parser, decode=True, seq=128):
    """--seq S (prefill, default `seq`) and, for a model that takes a KV cache, --decode S."""
    point = parser.add_mutually_exclusive_group()
    point.add_argument("--seq", type=int, default=seq, help="prefill: S tokens")
    if decode:
        point.add_argument("--decode", type=int, default=None, metavar="S",
                           help="one decode step at position S - 1 after a KV cache of S - 1 (--part layer)")


def _to(x, device, cache=None, device_cache=None):
    """`x` with every tensor in it, through tuples, lists and dicts, moved to `device`, and `cache` swapped
    for `device_cache`."""
    if cache is not None and x is cache:
        return device_cache
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, device, cache, device_cache) for v in x)
    if isinstance(x, dict):
        return {k: _to(v, device, cache, device_cache) for k, v in x.items()}
    return x


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


def cache_on(cache, device):
    """A copy of `cache` on `device`, every cache layer moved by its own prefetch()."""
    moved = copy.deepcopy(cache)
    for layer in moved.layers:
        layer.device = device
        layer.prefetch()
    return moved


@torch.no_grad()
def run(body, layer, config, ids, device, decode=False, compile_model=True, **extra):
    """(CPU output, device output) of `layer`, one of `body`'s layers (those before it run on the CPU), at
    the grid point of `ids` [B, S]; `extra` goes to every `body` call. `layer` moves to `device` after its
    CPU run, so its weights are never copied."""
    from transformers import DynamicCache

    positions = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
    cache = device_cache = None
    if decode:
        cache = DynamicCache(config=config)
        _run_until(body, layer, True, input_ids=ids[:, :-1], position_ids=positions[:, :-1],
                   past_key_values=cache, use_cache=True, **extra)
        args, kwargs = _run_until(body, layer, False, input_ids=ids[:, -1:], position_ids=positions[:, -1:],
                                  past_key_values=cache, use_cache=True, **extra)
        device_cache = cache_on(cache, device)
    else:
        args, kwargs = _run_until(body, layer, False, input_ids=ids, position_ids=positions, use_cache=False,
                                  **extra)
    device_args, device_kwargs = _to((args, kwargs), device, cache, device_cache)

    cpu_out = _first_tensor(layer(*args, **kwargs))
    layer.to(device)
    device_layer = torch.compile(layer, dynamic=False) if compile_model else layer
    device_out = _first_tensor(device_layer(*device_args, **device_kwargs))
    return cpu_out, device_out
