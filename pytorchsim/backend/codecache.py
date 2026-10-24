"""Compile cache for the codegen route: torchsim_compile(src, meta, kernel_name) turns one kernel into
a launcher the call site calls with the kernel's args; one directory per source hash holds the
compiler's kernel file and every artifact."""

import os

from filelock import FileLock
from torch._inductor.codecache import get_hash

from pytorchsim import config, contract
from pytorchsim.backend import retile, compiler_bridge, compiler_input, launch_hooks
from pytorchsim.backend.build_identity import cached_elf, store_identity
from pytorchsim.simulator import functional, timing, wallclock

logger = config.setup_logger()

LOCK_TIMEOUT = 600


def _write_path(src_code, meta):
    """The kernel's workdir, keyed by everything its ELF is built from: the source, and the
    constexprs, grid and arguments the spec bakes in. A user-defined Triton kernel's source is
    the same for every launch; its constexprs and grid are not."""
    baked = (meta.get("signature"), meta.get("constants"), meta.get("fixed_config"),
             meta.get("template_grid"),
             [(a["name"], a["role"], a["dtype"], a["numel"]) for a in meta.get("args", ())])
    return os.path.join(config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip() + repr(baked))[1:12])


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper: each call launches the whole
    grid, Spike first so the tensors hold real values even if TOGSim fails."""
    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta

    def __call__(self, *args):
        launch_hooks.notify_launch(self, args)
        if config.pytorchsim_functional_mode:
            with wallclock.span(wallclock.SPIKE, self.kernel_name):
                written = functional.run(self.workdir, self.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        if not config.pytorchsim_timing_mode:
            logger.warning(
                "[timing] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        result = timing.run(self.workdir, self.meta, args)
        if isinstance(result, int):
            logger.info("[TOGSim] %s queued as kernel %d; the stream's cycles "
                        "are reported when the simulator closes",
                        self.kernel_name, result)
        else:
            logger.info("[TOGSim] %s simulated: %d cycles", self.kernel_name, result["total_cycles"])
        return result


def torchsim_compile(src_code, meta, kernel_name):
    """Compile one Inductor-generated Triton kernel through the compiler, synchronously, when the
    generated wrapper is imported."""
    write_path = _write_path(src_code, meta)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = cached_elf(write_path, kernel_name)
        if (elf is not None and config.pytorchsim_timing_mode
                and contract.artifact(write_path, "tog_so") is None):
            logger.info("[torchsim-compile] %s: cached without --tog, rebuilding for timing",
                        kernel_name)
            elf = None
        if elf is None:
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            compiler_input.store_meta(write_path, meta)
            retries = retile.smaller_tiles(src_code, meta)
            while True:
                compiler_input.write(src_code, meta, spec_path)
                try:
                    with wallclock.span(wallclock.TORCHSIM_COMPILE, kernel_name):
                        compiler_bridge.run_pipeline(
                            spec_path, write_path, to_stage="torchsim-compile",
                            tog=bool(config.pytorchsim_timing_mode))
                    wallclock.ingest_compile(write_path, kernel_name)
                    break
                except compiler_bridge.CompilerError as exc:
                    why = retile.too_big(exc)
                    if why is None:
                        raise
                    retry = next(retries, None)
                    if retry is None:
                        logger.warning("[torchsim-compile] %s: %s, and no smaller tile is "
                                       "left to try", kernel_name, why)
                        raise
                    tile, src_code, tuned = retry
                    meta.update(tuned)
                    logger.info("[torchsim-compile] %s: %s, trying %s", kernel_name, why, tile)
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            compiler_input.store_meta(write_path, meta)
            store_identity(write_path)
        else:
            compiler_input.load_tuned_blocks(write_path, meta)
        logger.info("[torchsim-compile] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
