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
from pytorchsim import frontend as _frontend

_frontend.install()

torch_openreg.openreg.init()
torch_openreg.openreg.register_eager_to_compile(
    torch_openreg.openreg.DEFAULT_EAGER_TO_COMPILE)
sys.modules['torch.npu'] = torch_openreg.openreg

def _autoload():
    """A placeholder registered as torch's device-backend entry point."""
    pass