"""The `npu` codegen route: Inductor's Triton backend + the compiler's passes.

    Inductor -> TritonNPUScheduling.define_kernel   (scheduling.py; the wrapper module: wrapper.py,
                                                     a model's own Triton kernels: user_kernel.py)
             -> torchsim_compile                  (backend/codecache.py, backend/compiler_input.py)
             -> pytorchsim-triton-compiler, in a subprocess         (backend/compiler_bridge.py)
             -> Spike (simulator/functional/__init__.py) / TOGSim (simulator/timing/__init__.py)

Registered for `npu` by frontend/inductor/registration.py.
"""

from pytorchsim.frontend.inductor.codegen.scheduling import TritonNPUScheduling
from pytorchsim.frontend.inductor.codegen.wrapper import TritonNPUWrapperCodegen
