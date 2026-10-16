"""How a kernel is built: its candidates autotuned on TOGSim when Inductor's knobs ask for it, the
first otherwise, and retried smaller when the compiler refuses it.

A template's candidates are its mapped tiles; every other kernel's are its XBLOCK, from the largest
fixed_config_for pinned down to the lanes, and what the compiler is asked beside it. Autotune simulates each
candidate alone, so this is the one place a build runs the simulator; the retry on a too-big tile
runs in every mode. A template tile settled on is recorded in the external mapping.

What the search does not do: it times candidates on TOGSim only (none is run on Spike, so a candidate
fast because it is wrong is not caught here), never moves R0_BLOCK or YBLOCK, and asks a reduction's
trips only at its largest XBLOCK.
"""

import copy
import itertools
import json
import os
import re

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


def _time_tile(src_code, meta, kernel_name, workdir, timeout, seen=None):
    """TOGSim cycles of one candidate, simulated alone; inf if it does not compile or run --
    develop's autotune. TOGSim 1.3.0 refuses a tile over the spad at load, where an earlier one never
    finishes it; the refusal and the timeout both reach here as RuntimeError. A candidate the
    compiler built into the same kernel at the same grid as one in `seen` takes its cycles, not
    another simulation (a pipelining count the compiler does not apply)."""
    from pytorchsim.simulator.togsim import TOGSimulator
    os.makedirs(workdir, exist_ok=True)
    spec_path = os.path.join(workdir, f"{kernel_name}_spec.py")
    with open(os.path.join(workdir, "kernel.py"), "w") as f:
        f.write(src_code)
    timing.store_meta(workdir, meta)
    compiler_input.write(src_code, meta, spec_path, compiler_bridge.compiler_dir())
    try:
        compiler_bridge.run_pipeline(spec_path, workdir, to_stage="torchsim-compile", tog=True,
                                     env=compile_env(meta))
        key = _build_key(workdir, meta)
        if seen is not None and key is not None and key in seen:
            logger.info("[autotune] %s %s: the same build as an earlier candidate", kernel_name,
                        workdir)
            return seen[key]
        timing.write_cycle_tables(workdir, meta)
        mine = contract.link_shared(workdir, timing.SHARED)
        timing.write_shape(workdir, meta)
        timing.write_tensors(workdir)
        result = TOGSimulator.run_standalone(os.path.join(mine, "tile_graph.onnx"),
                                             os.path.join(mine, "attribute"),
                                             autotune_mode=True, timeout_sec=timeout)
        cycles = TOGSimulator.get_result_from_file(result)[-1]
        if seen is not None and key is not None:
            seen[key] = cycles
        return cycles
    except (compiler_bridge.CompilerError, RuntimeError, FileNotFoundError) as exc:
        why = "scratchpad overflow" if _spad_overflow(exc) else type(exc).__name__
        logger.info("[autotune] %s %s: %s", kernel_name, workdir, why)
        return float("inf")


#: The kernel object's artifacts a candidate's cycles depend on: the program Spike runs, the one gem5
#: times, and the TOG TOGSim replays. Read by the kind the manifest names them, not by file name.
BUILD_ARTIFACTS = ("elf", "cycle_elf", "tog_so")


def _build_key(workdir, meta):
    """What the compiler built in `workdir` and the grid it runs over, or None: two candidates
    with the same key time the same. The bytes of each of BUILD_ARTIFACTS, as the kernel object
    names them, and the launch grid."""
    import hashlib
    paths = [contract.artifact(workdir, kind) for kind in BUILD_ARTIFACTS]
    if any(p is None for p in paths):
        return None
    h = hashlib.sha256(json.dumps(contract.grid_of(meta)).encode())
    for path in paths:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


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


def autotune_on(template):
    """Whether Inductor's own knobs ask for autotune: torch.compile(mode="max-autotune") sets
    max_autotune for every kernel, max_autotune_pointwise asks it for the kernels that are not a
    template. max_autotune_gemm is not one of them: the front end sets it so mm lowers to the Triton
    template at all. Cycles are what candidates rank by, so only in timing mode."""
    from torch._inductor import config as inductor_config
    asked = inductor_config.max_autotune or (
        not template and inductor_config.max_autotune_pointwise)
    return bool(asked) and bool(config.pytorchsim_timing_mode)


def build_policy():
    """What decides which candidate a kernel is built at, besides its source: part of the cache key,
    so a build chosen under one policy is never reused under another (an as-made build by a
    max-autotune run, or the reverse)."""
    on = {"template": autotune_on(template=True), "kernel": autotune_on(template=False)}
    policy = {"autotune": on, "mapping": config.codegen_mapping_strategy}
    if "external" in policy["mapping"]:
        policy["mapping_file"] = config.codegen_external_mapping_file
    if on["template"]:
        policy["template_topk"] = config.codegen_autotune_template_topk
    if on["kernel"]:
        policy["kernel_topk"] = config.codegen_autotune_kernel_topk
    return policy


#: The meta keys tune() and the too-big retry choose. A cached build is launched with the stored
#: ones: the grid TOGSim replays is computed from fixed_config, or is the template's own.
TUNED_KEYS = ("fixed_config", "template_grid", "compile_env")

#: A hung simulator, not a slow candidate: every candidate gets this same TOGSim wall limit, the
#: bound a compile gets (compiler_bridge.run_pipeline).
CANDIDATE_TIMEOUT_SEC = 1800


def tune(src_code, meta, kernel_name, write_path):
    """The (source, meta) a kernel is built at: its candidates, each compiled and simulated alone,
    the fewest cycles winning; the first alone when there is one, or autotune is off. A template's
    candidates are its mapped tiles, every other kernel's its XBLOCK from the largest down and what
    the compiler is asked beside it. A template tile settled on is recorded in the external mapping.

    Every candidate gets the same wall limit, CANDIDATE_TIMEOUT_SEC: a limit taken from an earlier
    candidate's wall time would rank by host load and by order, and the first candidate (the
    largest block, the fewest programs) is the quickest to simulate."""
    shape = tiling.mm_template_shape(src_code, meta)
    if shape is not None:
        cands = _template_candidates(src_code, meta, shape[:3], kernel_name)
    else:
        cands = _kernel_candidates(src_code, meta)
    if len(cands) == 1:
        return cands[0][1], cands[0][2]
    best, seen = (float("inf"), cands[0]), {}
    for label, s, mt in cands:
        cycles = _time_tile(s, mt, kernel_name, os.path.join(write_path, "autotune", label),
                            CANDIDATE_TIMEOUT_SEC, seen)
        logger.info("[autotune] %s %s: %s cycles", kernel_name, label, cycles)
        if cycles < best[0]:
            best = (cycles, (label, s, mt))
    if shape is not None and best[0] != float("inf"):
        _record_tile(*shape[:3], tuple(int(v) for v in best[1][0].split("x")))
    return best[1][1], best[1][2]


def _template_candidates(src_code, meta, mnk, kernel_name):
    """(label, source, meta) per mapped tile of an mm template, best first: the external mapping's
    tile alone when it has this shape, the top-k under autotune, else the source as codegen made it."""
    known = _recorded_tile(*mnk) if "external" in config.codegen_mapping_strategy else None
    if known is not None:
        logger.info("[autotune] %s BLOCK_M/N/K=%s: from %s", kernel_name, known,
                    config.codegen_external_mapping_file)
        return [("x".join(map(str, known)), *_retile(src_code, meta, mnk, known))]
    if not autotune_on(template=True):
        return [("as-made", src_code, meta)]
    tiles = _template_tiles(src_code, meta)[:max(1, config.codegen_autotune_template_topk)]
    return [("x".join(map(str, t)), *_retile(src_code, meta, mnk, t)) for t in tiles]


#: What the compiler is asked, per candidate: into how many trips torchsim-pipeline-reductions cuts
#: a reduction's tile (1 = whole). Passed through the compiler's environment. Which counts the pass
#: applies is the compiler's to decide: one it does not apply builds the same kernel as 1, and
#: _time_tile takes that build's cycles without simulating it again.
PIPE_STAGES = (1, 2, 4, 8, 16, 32, 64, 128)
PIPE_ENV = "TORCHSIM_PIPE_STAGES"

#: Every compiler option a build may be asked through the environment, at the value a build that
#: does not ask gets. Each is set on every compile, so an exported one never reaches a candidate
#: or the final build unasked.
COMPILE_ENV_DEFAULTS = {PIPE_ENV: 1}


def compile_env(meta):
    """The compiler environment a build runs with: COMPILE_ENV_DEFAULTS, overridden by what autotune
    chose for it (meta["compile_env"]), as strings."""
    env = dict(COMPILE_ENV_DEFAULTS)
    env.update((meta or {}).get("compile_env") or {})
    return {k: str(v) for k, v in env.items()}


def _kernel_candidates(src_code, meta):
    """(label, source, meta) per build of a kernel that is not a template; the first is the build
    as made, its blocks what fixed_config_for pinned -- XBLOCK as large as a lane's scratchpad
    holds. Under autotune, XBLOCK halves down to the lanes: a smaller block is more programs, one's
    loads overlapping another's arithmetic, where one big program waits for its whole tile. Then a
    reduction is asked each of PIPE_STAGES trips at the largest XBLOCK. The first
    codegen_autotune_kernel_topk are kept, in that order, and the rest are logged as dropped."""
    if not autotune_on(template=False):
        return [("as-made", src_code, meta)]
    kernel_name = meta.get("kernel_name")
    cfg = meta.get("fixed_config") or {}
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    xb = cfg.get("XBLOCK") or 0
    builds = []
    while xb >= lanes or not builds:
        builds.append((xb, 1))
        xb //= 2
    if any(k.startswith("R") and v for k, v in cfg.items()):
        if any(n is None for p, n in (meta.get("numels") or {}).items() if p.startswith("r")):
            logger.info("[autotune] %s: a reduction numel is not static, so no trip count is "
                        "asked", kernel_name)
        else:
            builds += [(builds[0][0], n) for n in PIPE_STAGES[1:]]
    keep = max(1, config.codegen_autotune_kernel_topk)
    if len(builds) > keep:
        logger.info("[autotune] %s: codegen_autotune_kernel_topk=%d drops %s", kernel_name, keep,
                    " ".join(f"x{x}-s{n}" for x, n in builds[keep:]))
    out = []
    for x, n in builds[:keep]:
        mt = copy.deepcopy(meta)
        if x:
            mt["fixed_config"]["XBLOCK"] = x
        mt["compile_env"] = {PIPE_ENV: n}
        out.append((f"x{x}-s{n}", src_code, mt))
    return out


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
