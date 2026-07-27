"""Compile cache for the Triton route -- the counterpart of extension_codecache.

`triton_npu_compile` is what the generated wrapper calls, exactly where the MLIR
route calls `custom_async_compile.mlir(...)`. It compiles the Triton kernel via
tnpu and returns the callable the wrapper then invokes per launch.

    define_kernel   ->  triton_npu_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

Layout mirrors the MLIR route so the two are comparable: one directory per source
hash under the dump path, holding the generated tnpu kernel file and every tnpu
artifact (01-ttir.mlir ... 05-*.elf).
"""

import os

from filelock import FileLock
from torch._inductor.codecache import get_hash

from PyTorchSimFrontend import extension_config
from . import kernel_spec, tnpu_bridge

logger = extension_config.setup_logger()

LOCK_TIMEOUT = 600


def _write_path(src_code):
    return os.path.join(extension_config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip())[1:12])


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper.

    Holds the compile result; each call is one launch of the whole grid.
    """

    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta
        self.elf = os.path.join(workdir, f"05-{kernel_name}.elf")

    def __call__(self, *args):
        raise NotImplementedError(
            f"{self.kernel_name}: compiled to {self.elf}, but the launch is not "
            f"wired yet. Two pieces are missing and both are tracked in "
            f"triton_backend/README.md:\n"
            f"  1. functional -- marshal the caller's tensors into "
            f"{self.workdir}/runtime/*.raw, run Spike on the ELF, read the "
            f"outputs back into the caller's tensors;\n"
            f"  2. timing -- emit trace.so + trace_cycles.tsv from the tnpu IR "
            f"and hand them to TOGSim (needs the build_tog adapters).\n"
            f"Compilation itself succeeded, so the codegen half of this route "
            f"is exercised by getting this far.")


def triton_npu_compile(src_code, meta, kernel_name):
    """Compile one Inductor-generated Triton kernel through tnpu.

    Called from the generated wrapper at module import time (same point as
    `custom_async_compile.mlir`). Synchronous for now: the MLIR route's thread
    pool buys nothing until the pipeline itself is proven.
    """
    write_path = _write_path(src_code)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = os.path.join(write_path, f"05-{kernel_name}.elf")
        if not os.path.isfile(elf):
            kernel_spec.write_spec_file(src_code, meta, spec_path,
                                        tnpu_bridge.tnpu_dir())
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)      # the unmodified Inductor source, for diffing
            tnpu_bridge.run_pipeline(spec_path, write_path, to_stage="binary")
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
