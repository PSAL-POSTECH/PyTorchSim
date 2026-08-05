"""The Triton codegen route: Inductor's Triton backend + the tnpu lowering passes.

Inductor decides what to compute; triton-npu decides how it maps onto the NPU,
in place of the hand-emitted MLIR under `PyTorchSimFrontend/mlir/`.

    Inductor  ->  TritonNPUScheduling.define_kernel      (scheduling.py)
                     |  triton kernel SOURCE TEXT
                     v
              ->  TritonNPUCodeCache.load                (codecache.py)
                     |  a tnpu KernelSpec file           (kernel_spec.py)
                     v
              ->  triton-npu, in a subprocess            (tnpu_bridge.py)
                     ttir -> ttshared -> tnpu passes -> RISC-V ELF
                     v
              ->  Spike (functional) / TOGSim (timing)

The two routes are mutually exclusive and chosen at device-registration time by
`extension_config.CONFIG_TRITON_CODEGEN` (env `TORCHSIM_TRITON_CODEGEN=1`),
default off. README.md has the measured coverage and the gap list.
"""

from . import _triton_compat, inductor_templates

# Before anything imports Inductor's Triton codegen: it needs `triton` in THIS
# interpreter, and on a GPU-less box its backend hash cannot be computed.
_triton_compat.install()
inductor_templates.install()

from .scheduling import TritonNPUScheduling  # noqa: E402,F401
from .wrapper_codegen import TritonNPUWrapperCodegen  # noqa: E402,F401
