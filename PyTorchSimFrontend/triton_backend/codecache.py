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
import re

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
        # touched. `pytorchsim_timing_mode` is the switch the MLIR route already
        # reads (extension_codecache.py), so this route reads the same one
        # rather than inventing a second name.
        if not extension_config.pytorchsim_timing_mode:
            logger.warning(
                "[TOGSim] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        if not os.path.isfile(os.path.join(self.workdir, timing.TRACE_SO)):
            timing.emit_trace(self.workdir, self.meta)
        result = timing.run_togsim(self.workdir, meta=self.meta, args=args)
        logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


#: tnpu's machine-readable half of a scratchpad refusal (tnpu/spad.py,
#: SPAD_OVERFLOW_MARKER). The rest of that message is advice for a person.
_SPAD_OVERFLOW_RE = re.compile(
    r"tnpu-spad-overflow: usage=(\d+) budget=(\d+)")


def _spad_overflow(exc):
    """(usage, budget) if this failure was the scratchpad, else None.

    READ OFF `exc.output`, NOT `str(exc)`. TnpuError's message is a summary --
    it keeps the last few lines that look like a diagnostic (`error:`, an
    exception name, an assertion) so Inductor, which prints only `str(exc)`,
    shows something useful. The marker looks like none of those on purpose: it
    is addressed to this function, not to a reader, and widening that filter to
    let it through would put it in front of the reader instead. The raw stage
    output is where a contract belongs.
    """
    m = _SPAD_OVERFLOW_RE.search(getattr(exc, "output", None) or str(exc))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _shrink_reduction_blocks(meta, usage, budget):
    """Divide every reduction block by enough to fit, in place. False if stuck.

    WHY THE BLOCK SIZE IS THE FREE VARIABLE AND THE BUFFER COUNT IS NOT.
    `fixed_config_for` sizes R0_BLOCK against a budget divided by
    `_REDUCTION_LIVE_TILES`, a constant standing in for how many block-sized
    tiles the kernel will keep live -- and that count is a property of the
    LOWERING, which does not exist until the block size has been chosen. The
    constant's own comment says so ("the count is a property of the kernel and
    not a constant") and then guesses anyway, at 12. ViT's first LayerNorm,
    fused with a patch convolution, an addmm and a transpose, lowers to 41
    scratchpad globals:

        R0_BLOCK 512   77504 bytes/lane   over the 65536 budget
        R0_BLOCK 256   38680             fits
        R0_BLOCK 128   19356

    So the guess cannot be made right by picking a bigger number -- 41 would
    cost every ordinary reduction three quarters of its tile. It can only be
    CORRECTED, and the correction is one recompile: tnpu measures the real
    thing and says by how much.

    `usage / budget` rounded up to a power of two, so an overshoot of 1.18x
    halves once and an overshoot of 5x goes straight to an eighth rather than
    walking there. Only reduction blocks move: XBLOCK is the lane axis and
    shrinking it would leave lanes idle without freeing a byte per lane.
    """
    factor = 1
    while usage > budget * factor:
        factor *= 2
    blocks = {k: v for k, v in (meta.get("fixed_config") or {}).items()
              if k.startswith("R") and k.endswith("_BLOCK") and v and v > 1}
    if not blocks:
        return False
    for k, v in blocks.items():
        meta["fixed_config"][k] = max(1, v // factor)
    return True


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
            timing.store_meta(write_path, meta)   # lets the timing step run standalone
            while True:
                kernel_spec.write_spec_file(src_code, meta, spec_path,
                                            tnpu_bridge.tnpu_dir())
                try:
                    tnpu_bridge.run_pipeline(spec_path, write_path,
                                             to_stage="binary")
                    break
                except tnpu_bridge.TnpuError as exc:
                    over = _spad_overflow(exc)
                    if over is None or not _shrink_reduction_blocks(meta, *over):
                        raise
                    logger.info(
                        "[triton-npu] %s: %d bytes/lane over a budget of %d, "
                        "retrying with %s", kernel_name, over[0], over[1],
                        {k: v for k, v in meta["fixed_config"].items()
                         if k.endswith("_BLOCK")})
            # The spec now records the block sizes that actually compiled, and
            # timing.store_meta above wrote the ones that did not. Restate it so
            # a standalone timing run launches the grid the ELF was built for.
            timing.store_meta(write_path, meta)
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
