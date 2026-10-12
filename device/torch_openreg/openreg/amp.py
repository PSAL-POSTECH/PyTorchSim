import torch

import torch_openreg._C  # type: ignore[misc]

from . import _lazy_init


__all__ = [
    "is_autocast_enabled",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
    "get_amp_supported_dtype",
]

def is_autocast_enabled():
    return torch_openreg._C.is_autocast_enabled()


def set_autocast_enabled(enabled: bool) -> None:
    torch_openreg._C.set_autocast_enabled(enabled)


def get_autocast_dtype():
    return torch_openreg._C.get_autocast_dtype()


def set_autocast_dtype(dtype) -> None:
    torch_openreg._C.set_autocast_dtype(dtype)


def get_amp_supported_dtype():
    return torch_openreg._C.get_amp_supported_dtype()