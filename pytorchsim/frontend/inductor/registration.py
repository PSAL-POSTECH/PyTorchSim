"""npu as a Triton GPU to Inductor: in GPU_TYPES, which Inductor has no hook for, so it patches
its module state; is_npu and lowering_npu are the device tests the other patches share.
"""


def install():
    """Make Inductor take npu as a Triton GPU."""
    _register_npu_as_gpu()


def _register_npu_as_gpu():
    """Add npu to Inductor's GPU_TYPES, so Triton codegen is chosen for it."""
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


def lowering_npu():
    """Whether the graph being lowered right now targets this device."""
    from torch._inductor.virtualized import V

    try:
        return V.graph.get_current_device_or_throw().type == "npu"
    except Exception:  # noqa: BLE001
        return False


def is_npu(obj):
    """Whether a device, device string, tensor, layout or node is on this device."""
    dev = getattr(obj, "device", obj)
    dev = getattr(dev, "device", dev)
    if isinstance(dev, str):
        return dev.split(":")[0] == "npu"
    return getattr(dev, "type", None) == "npu"
