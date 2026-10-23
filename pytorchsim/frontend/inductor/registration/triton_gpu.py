"""npu as a Triton GPU to Inductor, by the one patch Inductor has no hook for: npu in GPU_TYPES (so
Triton codegen is chosen for it). has_triton and the backend hash are upstream's."""


def install():
    """The patch."""
    _add_gpu_type()


def _add_gpu_type():
    """Add npu to Inductor's GPU_TYPES, so Triton codegen is chosen for it."""
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")
