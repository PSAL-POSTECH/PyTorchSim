import sys
import os
import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import torch_openreg._C  # type: ignore[misc]
import torch_openreg.openreg

torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module("npu", torch_openreg.openreg)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
import pytorchsim.config  # noqa: F401
from pytorchsim.frontend import inductor_patches as _inductor_patches
from pytorchsim.frontend import rewrite_fx_graph as _rewrite_fx_graph

_inductor_patches.install()

# The `npu` codegen route: Inductor's own Triton codegen, lowered by the
# compiler passes. Registered here because Inductor registers a backend per
# device, once.
from pytorchsim.codegen import TritonNPUScheduling, TritonNPUWrapperCodegen
torch._inductor.codegen.common.register_backend_for_device(
    "npu",
    TritonNPUScheduling,
    TritonNPUWrapperCodegen,
    device_custom_pass=_rewrite_fx_graph.npu_post_grad_pass,
)

torch_openreg.openreg.init()
torch_openreg.openreg.register_eager_to_compile(
    torch_openreg.openreg.DEFAULT_EAGER_TO_COMPILE)
sys.modules['torch.npu'] = torch_openreg.openreg

def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass