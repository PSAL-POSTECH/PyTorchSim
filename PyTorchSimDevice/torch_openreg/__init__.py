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
from PyTorchSimFrontend import extension_config as _extension_config
from PyTorchSimFrontend.mlir.mlir_codegen_backend import ExtensionWrapperCodegen

# Two mutually exclusive codegen routes for `npu`, chosen here because Inductor
# registers a backend per device, once.
#   MLIR   (default)  hand-written MLIR emission, PyTorchSimFrontend/mlir
#   Triton (opt-in)   Inductor's own Triton codegen + the triton-npu passes,
#                     TORCHSIM_TRITON_CODEGEN=1. WIP; see
#                     PyTorchSimFrontend/triton_backend/README.md
if _extension_config.CONFIG_TRITON_CODEGEN:
    from PyTorchSimFrontend.triton_backend import (
        TritonNPUScheduling, TritonNPUWrapperCodegen)
    torch._inductor.codegen.common.register_backend_for_device(
        "npu",
        lambda scheduling: TritonNPUScheduling(scheduling),
        TritonNPUWrapperCodegen
    )
else:
    from PyTorchSimFrontend.mlir.mlir_scheduling import MLIRScheduling
    torch._inductor.codegen.common.register_backend_for_device(
        "npu",
        lambda scheduling: MLIRScheduling(scheduling),
        ExtensionWrapperCodegen
    )

torch_openreg.openreg.init()
sys.modules['torch.npu'] = torch_openreg.openreg

def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass