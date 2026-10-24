import torch

import torch_openreg._C  # type: ignore[misc]

from . import _lazy_init, current_device, device_count


__all__ = [
    "get_rng_state",
    "set_rng_state",
    "manual_seed",
    "manual_seed_all",
    "initial_seed",
    "_is_in_bad_fork",
]


def get_rng_state(device="npu"):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("npu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.get_state()


def set_rng_state(new_state, device="npu"):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("npu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    default_generator.set_state(new_state)


def initial_seed() -> int:
    _lazy_init()
    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.initial_seed()


def manual_seed(seed: int) -> None:
    seed = int(seed)

    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    default_generator.manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    seed = int(seed)

    for idx in range(device_count()):
        default_generator = torch_openreg._C._get_default_generator(idx)
        default_generator.manual_seed(seed)

def _is_in_bad_fork():
    """Never: the simulator has none of CUDA's fork issues."""
    return False