"""The `npu` codegen route: Inductor's Triton backend + the compiler's passes.

    Inductor -> TritonNPUScheduling.define_kernel   (scheduling.py)
             -> torchsim_compile                  (codecache.py, kernel_spec.py)
             -> pytorchsim-triton-compiler, in a subprocess         (compiler_bridge.py)
             -> Spike (functional.py) / TOGSim (timing.py)

Registered for `npu` at device import; see PyTorchSimDevice/torch_openreg.
"""

from . import _triton_compat, inductor_patches

_triton_compat.install()
inductor_patches.install()

from .scheduling import TritonNPUScheduling
from .wrapper_codegen import TritonNPUWrapperCodegen
