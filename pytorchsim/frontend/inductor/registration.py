"""npu as an Inductor backend: its DeviceInterface (device_interface.py), a Triton GPU (GPU_TYPES,
which has no hook, so it patches module state), its codegen and post-grad pass and its device op
overrides. is_npu and lowering_npu are the device tests the patches share."""

def install():
    """Make Inductor take npu as a Triton GPU and register npu's device interface, codegen,
    post-grad pass and op overrides."""
    _register_device_interface()
    _register_npu_as_gpu()
    _register_backend()
    _register_device_op_overrides()


def _register_backend():
    """npu's codegen route: Inductor's Triton codegen through TritonNPUScheduling and its wrapper, with
    pytorchsim_post_grad_pass run on every npu graph."""
    from torch._inductor.codegen.common import register_backend_for_device

    from pytorchsim.frontend.inductor.codegen import TritonNPUScheduling, TritonNPUWrapperCodegen
    from pytorchsim.frontend.inductor.post_grad_pass import pytorchsim_post_grad_pass

    register_backend_for_device("npu", TritonNPUScheduling, TritonNPUWrapperCodegen,
                                device_custom_pass=pytorchsim_post_grad_pass)


def _register_device_interface():
    """What Inductor asks of the device: its properties, Triton and bf16 (device_interface.py)."""
    from pytorchsim.frontend.inductor import device_interface

    device_interface.install()


def _register_device_op_overrides():
    """The device code Inductor writes into an npu wrapper (device_op_overrides.py)."""
    from pytorchsim.frontend.inductor import device_op_overrides

    device_op_overrides.install()


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
