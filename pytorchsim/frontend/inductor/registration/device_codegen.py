"""npu's codegen route in Inductor's device_codegens (register_backend_for_device): Triton codegen
through TritonNPUScheduling and its wrapper, with pytorchsim_post_grad_pass on every npu graph."""


def install():
    """Register npu's scheduling, wrapper codegen and device_custom_pass."""
    from torch._inductor.codegen.common import register_backend_for_device

    from pytorchsim.frontend.inductor.codegen import TritonNPUScheduling, TritonNPUWrapperCodegen
    from pytorchsim.frontend.inductor.post_grad_pass import pytorchsim_post_grad_pass

    register_backend_for_device("npu", TritonNPUScheduling, TritonNPUWrapperCodegen,
                                device_custom_pass=pytorchsim_post_grad_pass)
