"""The Triton codegen route: Inductor's Triton backend + the tnpu lowering passes.

WHAT THIS REPLACES
------------------
The production path emits MLIR by hand (`PyTorchSimFrontend/mlir/`, ~5,500 lines
of string emission that also decides tiling, vectorization, DMA and scratchpad
placement). This route keeps Inductor's OWN Triton codegen for "what to compute"
and hands the resulting Triton kernel to triton-npu for "how to map it onto the
NPU":

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
`extension_config.CONFIG_TRITON_CODEGEN` (env `TORCHSIM_TRITON_CODEGEN=1`).
Default OFF -- nothing here is on the production path yet.

STATUS: scaffolding. The seams are wired and each stage names precisely what it
still owes; see README.md for the gap list. Expect failures, not results.
"""

from . import _triton_compat, inductor_templates

# Before anything imports Inductor's Triton codegen: it needs `triton` in THIS
# interpreter, and on a GPU-less box its backend hash cannot be computed.
_triton_compat.install()
# ... and before any lowering runs: mm/bmm/addmm/conv reach Inductor's own
# Triton templates only if `npu` is in GPU_TYPES when `use_triton_template` asks.
inductor_templates.install()

from .scheduling import TritonNPUScheduling  # noqa: E402,F401
from .wrapper_codegen import TritonNPUWrapperCodegen  # noqa: E402,F401
