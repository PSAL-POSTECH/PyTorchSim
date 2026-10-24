"""The `npu` codegen route: Inductor's Triton backend + the compiler's passes.

    Inductor -> TritonNPUScheduling.define_kernel   (scheduling.py)
             -> torchsim_compile                  (backend/codecache.py, backend/compiler_input.py)
             -> pytorchsim-triton-compiler, in a subprocess         (backend/compiler_bridge.py)
             -> Spike (simulator/functional.py) / TOGSim (simulator/timing.py)

Registered for `npu` by frontend/inductor/registration.py.
"""

from pytorchsim.frontend.inductor.codegen.scheduling import TritonNPUScheduling, TritonNPUWrapperCodegen
