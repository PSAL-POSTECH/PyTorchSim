"""The `npu` codegen route: Inductor's Triton backend + the compiler's passes.

    Inductor -> TritonNPUScheduling.define_kernel   (inductor_hooks.py)
             -> torchsim_compile                  (driver/codecache.py, driver/compiler_input.py)
             -> pytorchsim-triton-compiler, in a subprocess         (driver/compiler_bridge.py)
             -> Spike (simulator/functional.py) / TOGSim (simulator/timing.py)

Registered for `npu` at device import; see device/torch_openreg.
"""

from pytorchsim.codegen.inductor_hooks import TritonNPUScheduling, TritonNPUWrapperCodegen
