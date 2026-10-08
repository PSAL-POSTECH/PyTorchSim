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
import PyTorchSimFrontend.extension_config  # noqa: F401
# What an op falls back TO is the CPU, and Inductor refuses to generate code for
# complex operators, so neither of these is about codegen.
import PyTorchSimFrontend.extension_decomposition  # noqa: F401
from PyTorchSimFrontend import extension_grouped_mm as _grouped_mm
from PyTorchSimFrontend import extension_topk as _topk

_grouped_mm.install()
_topk.install()
import PyTorchSimFrontend.extension_complex_to_real  # noqa: F401

# The `npu` codegen route: Inductor's own Triton codegen, lowered by the
# compiler passes. Registered here because Inductor registers a backend per
# device, once. See PyTorchSimFrontend/triton_backend/README.md.
from PyTorchSimFrontend.triton_backend import (
    TritonNPUScheduling, TritonNPUWrapperCodegen)
torch._inductor.codegen.common.register_backend_for_device(
    "npu",
    lambda scheduling: TritonNPUScheduling(scheduling),
    TritonNPUWrapperCodegen
)

torch_openreg.openreg.init()
torch_openreg.openreg.register_eager_to_compile(
    torch_openreg.openreg.DEFAULT_EAGER_TO_COMPILE)
sys.modules['torch.npu'] = torch_openreg.openreg

def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass