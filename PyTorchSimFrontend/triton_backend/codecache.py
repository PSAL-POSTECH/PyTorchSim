"""Compile cache for the codegen route: one kernel in, one launcher out.

    define_kernel   ->  triton_npu_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

One directory per source hash, holding the compiler kernel file and every artifact.
"""

import itertools
import os
import re

from filelock import FileLock
from torch._inductor.codecache import get_hash

from PyTorchSimFrontend import extension_config

from . import breakdown, functional, kernel_spec, provenance, timing, compiler_bridge

logger = extension_config.setup_logger()

LOCK_TIMEOUT = 600

#: THE MARKER IS A TEXT CONTRACT ACROSS A PROCESS BOUNDARY. The compiler runs
#: as a subprocess, so this string cannot be imported from it -- it is spelled
#: `spad.SPAD_OVERFLOW_MARKER` there and pinned by tile_spad_over_budget_marker.
_SPAD_OVERFLOW_RE = re.compile(r"torchsim-spad-overflow: usage=(\d+) budget=(\d+)")


def _write_path(src_code):
    return os.path.join(extension_config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip())[1:12])


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper.

    Each call is one launch of the whole grid, Spike first so the tensors hold
    real values even if TOGSim fails. Both halves switch on the config keys.
    """
    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta

    def __call__(self, *args):
        if extension_config.pytorchsim_functional_mode:
            with breakdown.span(breakdown.SPIKE, self.kernel_name):
                written = functional.run(self.workdir, self.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        if not extension_config.pytorchsim_timing_mode:
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
            logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


def _spad_overflow(exc):
    """(usage, budget) if this failure was the scratchpad, else None.

    Read off `exc.output`, NOT `str(exc)`: CompilerError's message keeps only lines
    that look like a diagnostic, and this marker is addressed to this function.
    """
    m = _SPAD_OVERFLOW_RE.search(getattr(exc, "output", None) or str(exc))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _get_tile_candidates(meta):
    """Every tile the search may try, LARGEST FIRST: each block the kernel takes as
    an argument halves down to 2, never 1. Equal sizes go to the larger reduction
    block; the first is the tile fixed_config_for pinned."""
    cfg = meta.get("fixed_config") or {}
    signature = meta.get("signature") or {}
    movable = [k for k, v in cfg.items() if k in signature and v and v >= 2]
    ranges = []
    for k in movable:
        vs, v = [], cfg[k]
        while v >= 2:
            vs.append(v)
            v //= 2
        ranges.append(vs)

    def _product(tile, keep):
        n = 1
        for k, v in tile.items():
            if keep(k):
                n *= v
        return n
    tiles = [dict(zip(movable, vs)) for vs in itertools.product(*ranges)]
    tiles.sort(key=lambda t: (-_product(t, lambda k: True),
                              -_product(t, lambda k: k.startswith("R")),
                              [-t[k] for k in movable]))
    return tiles


def triton_npu_compile(src_code, meta, kernel_name):
    """Compile one Inductor-generated Triton kernel through the compiler.

    Called from the generated wrapper at module import time. Synchronous, on
    purpose: a thread pool buys nothing until the pipeline itself is proven.
    """
    write_path = _write_path(src_code)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = compiler_bridge.artifact(write_path, "elf")
        if elf is not None and not provenance.matches(write_path):
            logger.info(
                "[torchsim-compile] %s: cached artifacts carry a different toolchain "
                "or machine identity, rebuilding", kernel_name)
            provenance.clear_stale(write_path)
            elf = None
        if elf is None:
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            timing.store_meta(write_path, meta)
            tiles = iter(_get_tile_candidates(meta))
            next(tiles)
            while True:
                kernel_spec.write_spec_file(src_code, meta, spec_path,
                                            compiler_bridge.tnpu_dir())
                try:
                    with breakdown.span(breakdown.TORCHSIM_COMPILE, kernel_name):
                        compiler_bridge.run_pipeline(
                            spec_path, write_path, to_stage="torchsim-compile",
                            tog=bool(extension_config.pytorchsim_timing_mode))
                    breakdown.ingest_compile(write_path, kernel_name)
                    break
                except compiler_bridge.CompilerError as exc:
                    over = _spad_overflow(exc)
                    if over is None:
                        raise
                    tile = next(tiles, None)
                    if tile is None:
                        logger.warning(
                            "[torchsim-compile] %s: %d bytes/lane over a budget of %d, and "
                            "no tile with every block >= 2 is left to try",
                            kernel_name, over[0], over[1])
                        raise
                    meta["fixed_config"].update(tile)
                    logger.info(
                        "[torchsim-compile] %s: %d bytes/lane over a budget of %d, trying "
                        "%s", kernel_name, over[0], over[1],
                        {k: v for k, v in meta["fixed_config"].items()
                         if k.endswith("BLOCK")})
            timing.store_meta(write_path, meta)
            provenance.store(write_path)
        logger.info("[torchsim-compile] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
