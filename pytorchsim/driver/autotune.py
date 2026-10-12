"""Which tile a template kernel is built at: autotuned on TOGSim, or retried smaller when the compiler refuses it.

Autotune simulates each candidate alone, so this is the one place a build runs the simulator; the
retry on a too-big tile runs in every mode. A tile settled on is recorded in the external mapping.
"""

import copy
import itertools
import json
import os
import re
import time

from filelock import FileLock

from pytorchsim import config
from pytorchsim.codegen import tiling
from pytorchsim import contract
from pytorchsim.driver import compiler_bridge, compiler_input
from pytorchsim.simulator import timing

logger = config.setup_logger()

MAPPING_LOCK_TIMEOUT = 600


#: THE MARKER IS A TEXT CONTRACT ACROSS A PROCESS BOUNDARY. The compiler runs
#: as a subprocess, so this string cannot be imported from it -- it is spelled
#: `spad.SPAD_OVERFLOW_MARKER` there and pinned by tile_spad_over_budget_marker.
_SPAD_OVERFLOW_RE = re.compile(r"torchsim-spad-overflow: usage=(\d+) budget=(\d+)")


def _spad_overflow(exc):
    """(usage, budget) if this failure was the scratchpad, else None.

    Read off `exc.output`, NOT `str(exc)`: CompilerError's message keeps only lines
    that look like a diagnostic, and this marker is addressed to this function.
    """
    m = _SPAD_OVERFLOW_RE.search(getattr(exc, "output", None) or str(exc))
    return (int(m.group(1)), int(m.group(2))) if m else None


def too_big(exc):
    """Why this failure means the tile is too big, else None: the scratchpad budget it broke, the
    register file (llc spilled, which the compiler's ELF refuses) or Triton's own block size."""
    over = _spad_overflow(exc)
    if over is not None:
        return "%d bytes/lane over a budget of %d" % over
    said = getattr(exc, "output", None) or str(exc)
    if "spill not allowed" in said:
        return "the vector register file spills"
    if "exceeds triton maximum tensor numel" in said:
        return "a block holds more elements than Triton accepts"
    return None


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


def _retile(src_code, meta, mnk, tile):
    """The same mm or bmm kernel at block `tile` (M, N, K): its constexprs and its grid, the batch
    axes of the grid kept."""
    (m, n, k), (bm, bn, bk) = mnk, tile
    vals = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "EVEN_K": k % bk == 0}
    for key, v in vals.items():
        src_code = tiling.TILE_RE[key].sub(lambda mt, v=v: f"{mt.group(1)}{v}", src_code)
    meta = copy.deepcopy(meta)
    meta["template_grid"] = [-(-m // bm) * -(-n // bn)] + list(meta["template_grid"][1:])
    return src_code, meta


def _time_tile(src_code, meta, kernel_name, workdir, timeout):
    """TOGSim cycles of one candidate, simulated alone; inf if it does not compile or run --
    develop's autotune, where a scratchpad overflow is a tile that never finishes."""
    from pytorchsim.simulator.togsim import TOGSimulator
    os.makedirs(workdir, exist_ok=True)
    spec_path = os.path.join(workdir, f"{kernel_name}_spec.py")
    with open(os.path.join(workdir, "kernel.py"), "w") as f:
        f.write(src_code)
    timing.store_meta(workdir, meta)
    compiler_input.write(src_code, meta, spec_path, compiler_bridge.compiler_dir())
    try:
        compiler_bridge.run_pipeline(spec_path, workdir, to_stage="torchsim-compile", tog=True)
        timing.emit_trace(workdir, meta)
        mine = contract.link_shared(workdir, (timing.TRACE_SO, timing.CYCLE_TSV))
        timing.write_shape(workdir, meta)
        timing.write_tensors(workdir)
        result = TOGSimulator.run_standalone(os.path.join(mine, "tile_graph.onnx"),
                                             os.path.join(mine, "attribute"),
                                             autotune_mode=True, timeout_sec=timeout)
        return TOGSimulator.get_result_from_file(result)[-1]
    except (compiler_bridge.CompilerError, RuntimeError, FileNotFoundError) as exc:
        why = "scratchpad overflow" if _spad_overflow(exc) else type(exc).__name__
        logger.info("[autotune] %s %s: %s", kernel_name, workdir, why)
        return float("inf")


def _template_tiles(src_code, meta):
    """The mapped tiles of an mm template kernel, best first: the ones codegen recorded, else worked out here."""
    tiles = meta.get("template_tiles")
    if tiles is None:
        tiles = tiling.template_tiles(src_code, meta)
    return [tuple(t) for t in tiles]


def smaller_tiles(src_code, meta):
    """(tile, source, meta) to retry with after a scratchpad overflow, in order: the other
    mapped tiles of an mm template, else each smaller fixed_config tile."""
    shape = tiling.mm_template_shape(src_code, meta)
    if shape is not None:
        mnk = shape[:3]
        now = tuple(int(tiling.TILE_RE[k].search(src_code).group(2))
                    for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        for tile in _template_tiles(src_code, meta):
            if tile != now:
                yield (dict(zip(("BLOCK_M", "BLOCK_N", "BLOCK_K"), tile)),
                       *_retile(src_code, meta, mnk, tile))
        return
    tiles = iter(_get_tile_candidates(meta))
    next(tiles, None)
    for tile in tiles:
        meta = copy.deepcopy(meta)
        meta["fixed_config"].update(tile)
        yield {k: v for k, v in tile.items() if k.endswith("BLOCK")}, src_code, meta


def autotune_template(src_code, meta, kernel_name, write_path):
    """develop's template autotune: the top-k mapped tiles, each compiled with what is fused
    into it and simulated alone; the fewest cycles wins. Only in timing mode, since cycles
    are what it ranks by; the first tile to finish limits the others' wall time."""
    shape = tiling.mm_template_shape(src_code, meta)
    if shape is None:
        return src_code, meta
    m, n, k = shape[:3]
    strategy = config.codegen_mapping_strategy
    known = _recorded_tile(m, n, k) if "external" in strategy else None
    if known is not None:
        logger.info("[autotune] %s BLOCK_M/N/K=%s: from %s", kernel_name, known,
                    config.codegen_external_mapping_file)
        return _retile(src_code, meta, (m, n, k), known)
    if "autotune" not in strategy or not config.pytorchsim_timing_mode:
        return src_code, meta
    tiles = _template_tiles(src_code, meta)[:config.codegen_autotune_template_topk]
    best, timeout = (float("inf"), src_code, meta, None), None
    for tile in tiles:
        s, mt = _retile(src_code, meta, (m, n, k), tile)
        t0 = time.perf_counter()
        cycles = _time_tile(s, mt, kernel_name,
                            os.path.join(write_path, "autotune", "x".join(map(str, tile))), timeout)
        if cycles != float("inf") and timeout is None:
            timeout = time.perf_counter() - t0 + config.codegen_autotune_wall_slack_sec
        logger.info("[autotune] %s BLOCK_M/N/K=%s: %s cycles", kernel_name, tile, cycles)
        if cycles < best[0]:
            best = (cycles, s, mt, tile)
    if best[0] != float("inf"):
        _record_tile(m, n, k, best[3])
    return best[1], best[2]


def _recorded_tile(m, n, k):
    """develop's external mapping: `{"M_N_K": {"TILE_M", "TILE_N", "TILE_K"}}`, else None."""
    path = config.codegen_external_mapping_file
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        t = json.load(f).get(f"{m}_{n}_{k}")
    return None if t is None else (t["TILE_M"], t["TILE_N"], t["TILE_K"])


def _record_tile(m, n, k, tile):
    """Add an autotuned tile to the external mapping file, so the next compile reads it."""
    path = config.codegen_external_mapping_file
    if not path:
        return
    with FileLock(path + ".lock", timeout=MAPPING_LOCK_TIMEOUT):
        data = json.load(open(path)) if os.path.isfile(path) else {}
        data[f"{m}_{n}_{k}"] = dict(zip(("TILE_M", "TILE_N", "TILE_K"), map(int, tile)))
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
