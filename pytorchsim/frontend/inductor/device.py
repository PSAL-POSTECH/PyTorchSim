"""Whether a thing is on npu, the device tests the npu patches share."""

import torch


def is_npu(obj):
    """Whether a device, device string, tensor, layout or node is on this device."""
    dev = getattr(obj, "device", obj)
    dev = getattr(dev, "device", dev)
    if isinstance(dev, str):
        dev = torch.device(dev)
    return getattr(dev, "type", None) == "npu"


def is_lowering_npu():
    """Whether the graph being lowered right now targets this device."""
    from torch._inductor.virtualized import V

    try:
        return V.graph.get_current_device_or_throw().type == "npu"
    except (AttributeError, RuntimeError):
        return False
