"""The `npu` codegen route: Inductor's Triton backend + the compiler's passes.

    Inductor -> TritonNPUScheduling.define_kernel   (scheduling.py)
             -> torchsim_compile                  (driver/codecache.py, kernel_spec.py)
             -> pytorchsim-triton-compiler, in a subprocess         (driver/compiler_bridge.py)
             -> Spike (simulator/functional.py) / TOGSim (simulator/timing.py)

Registered for `npu` at device import; see device/torch_openreg.
"""

from pytorchsim.frontend import inductor_patches

inductor_patches.install()

from pytorchsim.codegen.scheduling import TritonNPUScheduling
from pytorchsim.codegen.wrapper_codegen import TritonNPUWrapperCodegen
