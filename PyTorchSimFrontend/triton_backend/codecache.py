"""Compile cache for the codegen route: one kernel in, one launcher out.

    define_kernel   ->  triton_npu_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

One directory per source hash, holding the compiler kernel file and every artifact.
"""

import copy
import itertools
import json
import os
import re
import time

from filelock import FileLock
from torch._inductor.codecache import get_hash

from PyTorchSimFrontend import extension_config

from . import breakdown, functional, kernel_spec, provenance, session, timing, compiler_bridge

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


_TILE_RE = {k: re.compile(rf"^(\s*{k}\s*:\s*tl\.constexpr\s*=\s*)(\S+)$", re.M)
            for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "EVEN_K")}
_DIM_RE = {k: re.compile(rf"^\s*{k} = (\d+)$", re.M) for k in "MNK"}


def _mm_template(src_code, meta):
    """(M, N, K, input bytes, output bytes) of an mm template kernel -- one program per output tile, its
    blocks stated as constexprs -- else None."""
    if (meta.get("template_grid") is None or "bmm" in meta.get("kernel_name", "")
            or not all(r.search(src_code) for r in _TILE_RE.values())):
        return None
    dims = [_DIM_RE[k].search(src_code) for k in "MNK"]
    if not all(dims):
        return None
    import torch
    nbytes = {a["role"]: getattr(torch, a["dtype"]).itemsize for a in meta["args"]}
    return tuple(int(d.group(1)) for d in dims) + (nbytes["in"], nbytes.get("out", nbytes["in"]))


def _retile(src_code, meta, mnk, tile):
    """The same mm kernel at block `tile` (M, N, K): its constexprs and its grid."""
    (m, n, k), (bm, bn, bk) = mnk, tile
    vals = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "EVEN_K": k % bk == 0}
    for key, v in vals.items():
        src_code = _TILE_RE[key].sub(lambda mt, v=v: f"{mt.group(1)}{v}", src_code)
    meta = copy.deepcopy(meta)
    meta["template_grid"] = [-(-m // bm) * -(-n // bn), 1, 1]
    return src_code, meta


def _time_tile(src_code, meta, kernel_name, workdir, timeout):
    """TOGSim cycles of one candidate, simulated alone; inf if it does not compile or run --
    develop's autotune, where a scratchpad overflow is a tile that never finishes."""
    from Simulator.simulator import TOGSimulator
    os.makedirs(workdir, exist_ok=True)
    spec_path = os.path.join(workdir, f"{kernel_name}_spec.py")
    with open(os.path.join(workdir, "kernel.py"), "w") as f:
        f.write(src_code)
    timing.store_meta(workdir, meta)
    kernel_spec.write_spec_file(src_code, meta, spec_path, compiler_bridge.tnpu_dir())
    try:
        compiler_bridge.run_pipeline(spec_path, workdir, to_stage="torchsim-compile", tog=True)
        timing.emit_trace(workdir, meta)
        mine = session.link_shared(workdir, (timing.TRACE_SO, timing.CYCLE_TSV))
        timing.write_shape(workdir, meta)
        result = TOGSimulator.run_standalone(os.path.join(mine, "tile_graph.onnx"),
                                             os.path.join(mine, "attribute"),
                                             autotune_mode=True, timeout_sec=timeout)
        return TOGSimulator.get_result_from_file(result)[-1]
    except (compiler_bridge.CompilerError, RuntimeError, FileNotFoundError) as exc:
        why = "scratchpad overflow" if _spad_overflow(exc) else type(exc).__name__
        logger.info("[autotune] %s %s: %s", kernel_name, workdir, why)
        return float("inf")


def _autotune_template(src_code, meta, kernel_name, write_path):
    """develop's template autotune: the top-k mapped tiles, each compiled with what is fused
    into it and simulated alone; the fewest cycles wins. The first tile to finish sets the
    others' time limit (its wall time plus codegen_autotune_wall_slack_sec)."""
    shape = _mm_template(src_code, meta)
    if shape is None:
        return src_code, meta
    from .inductor_templates import _gemm_tiles
    m, n, k, size, out_size = shape
    strategy = extension_config.codegen_mapping_strategy
    known = _recorded_tile(m, n, k) if "external" in strategy else None
    if known is not None:
        logger.info("[autotune] %s BLOCK_M/N/K=%s: from %s", kernel_name, known,
                    extension_config.codegen_external_mapping_file)
        return _retile(src_code, meta, (m, n, k), known)
    if "autotune" not in strategy:
        return src_code, meta
    tiles = [(c.block_m, c.block_n, c.block_k) for c in _gemm_tiles(m, n, k, size, out_size)]
    tiles = tiles[:extension_config.codegen_autotune_template_topk]
    best, timeout = (float("inf"), src_code, meta, None), None
    for tile in tiles:
        s, mt = _retile(src_code, meta, (m, n, k), tile)
        t0 = time.perf_counter()
        cycles = _time_tile(s, mt, kernel_name,
                            os.path.join(write_path, "autotune", "x".join(map(str, tile))), timeout)
        if cycles != float("inf") and timeout is None:
            timeout = time.perf_counter() - t0 + extension_config.codegen_autotune_wall_slack_sec
        logger.info("[autotune] %s BLOCK_M/N/K=%s: %s cycles", kernel_name, tile, cycles)
        if cycles < best[0]:
            best = (cycles, s, mt, tile)
    if best[0] != float("inf"):
        _record_tile(m, n, k, best[3])
    return best[1], best[2]


def _recorded_tile(m, n, k):
    """develop's external mapping: `{"M_N_K": {"TILE_M", "TILE_N", "TILE_K"}}`, else None."""
    path = extension_config.codegen_external_mapping_file
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        t = json.load(f).get(f"{m}_{n}_{k}")
    return None if t is None else (t["TILE_M"], t["TILE_N"], t["TILE_K"])


def _record_tile(m, n, k, tile):
    """Add an autotuned tile to the external mapping file, so the next compile reads it."""
    path = extension_config.codegen_external_mapping_file
    if not path:
        return
    with FileLock(path + ".lock", timeout=LOCK_TIMEOUT):
        data = json.load(open(path)) if os.path.isfile(path) else {}
        data[f"{m}_{n}_{k}"] = dict(zip(("TILE_M", "TILE_N", "TILE_K"), map(int, tile)))
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)


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
        if (elf is not None and extension_config.pytorchsim_timing_mode
                and compiler_bridge.artifact(write_path, "trace_so") is None):
            logger.info("[torchsim-compile] %s: cached without --tog, rebuilding for timing",
                        kernel_name)
            elf = None
        if elf is None:
            src_code, tuned = _autotune_template(src_code, meta, kernel_name, write_path)
            meta.update(tuned)
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
        else:
            # a cached template may have been autotuned to another block: its grid is the stored one
            stored = os.path.join(write_path, timing.META_JSON)
            if meta.get("template_grid") is not None and os.path.isfile(stored):
                with open(stored) as f:
                    meta["template_grid"] = json.load(f)["template_grid"]
        logger.info("[torchsim-compile] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
