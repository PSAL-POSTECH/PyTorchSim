"""Compile cache for the Triton route -- the counterpart of extension_codecache.

`triton_npu_compile` is what the generated wrapper calls, exactly where the MLIR
route calls `custom_async_compile.mlir(...)`. It compiles the Triton kernel via
tnpu and returns the callable the wrapper then invokes per launch.

    define_kernel   ->  triton_npu_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

Layout mirrors the MLIR route so the two are comparable: one directory per source
hash under the dump path, holding the generated tnpu kernel file and every tnpu
artifact (01-ttir.mlir ... *-<kernel>.elf).
"""

import os

from filelock import FileLock
from torch._inductor.codecache import get_hash

from PyTorchSimFrontend import extension_config
from . import functional, kernel_spec, timing, tnpu_bridge

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

    def __call__(self, *args):
        """One launch of the whole grid: run it on Spike, then time it.

        Spike runs first so the caller's output tensors hold real values even if
        TOGSim fails -- the two halves are independent.
        """
        if extension_config.pytorchsim_functional_mode:
            written = functional.run(self.workdir, self.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        # AND THE OTHER HALF IS SWITCHED TOO, which the paragraph above already
        # claims: "the two halves are independent". Only the functional one was
        # -- the timing half ran whatever the config said, so a graph being
        # checked for VALUES paid for a cycle simulation of every kernel it
        # touched. That is the whole cost of an e2e run: mobilenet_v2's
        # depthwise convolutions launch a grid of [144, 2, 49] each, and the
        # model took over two hours to reach kernel 16 of 57 with timing on and
        # minutes with it off. `pytorchsim_timing_mode` is the switch the MLIR
        # route already reads (extension_codecache.py), so this route reads the
        # same one rather than inventing a second name.
        if not extension_config.pytorchsim_timing_mode:
            # NOT "[TOGSim]". The sweep buckets a failure by matching its
            # output, and its togsim bucket is `TOGSim|trace\.so|SIGSEGV|...` --
            # so a line carrying that word puts every failing test in this mode
            # into the wrong bucket whatever actually went wrong. MEASURED:
            # tests/system/test_triton_codegen.py came back "[togsim]" for a
            # failure that had nothing to do with it.
            logger.warning(
                "[timing] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        if not os.path.isfile(os.path.join(self.workdir, timing.TRACE_SO)):
            timing.emit_trace(self.workdir, self.meta)
        result = timing.run_togsim(self.workdir, meta=self.meta, args=args)
        logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


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
        elf = tnpu_bridge.stage_artifact(write_path, f"{kernel_name}.elf")
        if elf is None:
            # Before write_spec_file, which rejects exactly the kernels whose
            # source is worth keeping.
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)      # the unmodified Inductor source
            kernel_spec.write_spec_file(src_code, meta, spec_path,
                                        tnpu_bridge.tnpu_dir())
            timing.store_meta(write_path, meta)   # lets the timing step run standalone
            tnpu_bridge.run_pipeline(spec_path, write_path, to_stage="binary")
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
