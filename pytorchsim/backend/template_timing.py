"""A template config timed in TOGSim cycles, in timing mode: every config of a timed template at once
(time_choices, a slow one cut), and a module Inductor wrote to benchmark a fusion (time_module). What a
timed pick leaves for next time is template_record.py."""

import concurrent.futures
import os
import time

from pytorchsim import config, contract
from pytorchsim.frontend.inductor.codegen import kernel_metadata
from pytorchsim.backend import retile, compiler_bridge, compiler_input, template_record
from pytorchsim.simulator import timing

logger = config.setup_logger()


def _time_tile(src_code, meta, kernel_name, workdir, cut=None):
    """TOGSim cycles of one kernel, simulated alone; inf if it does not compile or run, or `cut`
    stops its simulation. TOGSim refuses a tile over the spad at load, and that refusal and a cut
    reach here as RuntimeError."""
    from pytorchsim.simulator.timing.togsim import Cut, TOGSimProcess
    os.makedirs(workdir, exist_ok=True)
    spec_path = os.path.join(workdir, f"{kernel_name}_spec.py")
    with open(os.path.join(workdir, "kernel.py"), "w") as f:
        f.write(src_code)
    compiler_input.store_meta(workdir, meta)
    compiler_input.write(src_code, meta, spec_path)
    try:
        compiler_bridge.run_pipeline(spec_path, workdir, to_stage="torchsim-compile", tog=True)
        timing.write_cycle_tables(workdir, meta)
        mine = contract.link_shared(workdir, timing.SHARED)
        timing.write_shape(workdir, meta)
        timing.write_tensors(workdir)
        result = TOGSimProcess.run_standalone(os.path.join(mine, "tile_graph.onnx"),
                                              os.path.join(mine, "attribute"),
                                              autotune_mode=True, cut=cut)
        return TOGSimProcess.read_result(result)["total_cycles"]
    except (compiler_bridge.CompilerError, RuntimeError, FileNotFoundError) as exc:
        why = retile.too_big(exc) or (f"cut, {exc}" if isinstance(exc, Cut) else type(exc).__name__)
        logger.info("[select] %s %s: %s", kernel_name, workdir, why)
        return float("inf")


def _workdir(name):
    return os.path.join(config.get_dump_path(), "select", name)


def times(choices):
    """Whether these choices are picked by their cycles: some are configs of a timed template."""
    return any(template_record.is_timed(c) for c in choices)


#: A config still simulating this many times the wall time the first config to finish took is cut.
#: The configs run at once, so a loaded host slows them alike and the ratio stands.
CUT_FACTOR = 3


def time_choices(choices):
    """Cycles per choice for AlgorithmSelectorCache: each config of a timed template as Inductor
    rendered it, all at once, every other choice inf. A config whose simulation runs past
    CUT_FACTOR times the wall time of the first one timed is cut, and is inf."""
    out = {c: float("inf") for c in choices}
    timed = [c for c in choices if template_record.is_timed(c)]
    jobs = [(c, *kernel_metadata.benchmark_request_meta(c.bmreq)) for c in timed]
    t0, first = time.monotonic(), []

    def cut():
        if first and time.monotonic() - t0 > CUT_FACTOR * first[0]:
            return f"past {CUT_FACTOR}x the {first[0]:.0f} s the first config took"
        return None

    def one(job):
        c, src, meta = job
        got = _time_tile(src, meta, c.bmreq.kernel_name, _workdir(c.bmreq.module_cache_key), cut)
        if got != float("inf") and not first:
            first.append(time.monotonic() - t0)
        return got

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
        cycles = pool.map(one, jobs)
        for (c, _, _), got in zip(jobs, cycles):
            logger.info("[select] %s %s: %s cycles", c.name, c.description, got)
            c.annotations["ktc"].annotations[template_record.SOURCE] = template_record.TIMED
            out[c] = got
    if timed and all(out[c] == float("inf") for c in timed):
        raise RuntimeError(
            f"no template config of {[c.name for c in timed]} compiled and ran on TOGSim; "
            f"the [select] lines above say why each one did not")
    return out


def time_module(src_code, meta, path):
    """TOGSim cycles of a kernel module Inductor wrote to benchmark a fusion, named by its path."""
    name = os.path.splitext(os.path.basename(path))[0]
    cycles = _time_tile(src_code, meta, meta["kernel_name"], _workdir(name))
    logger.info("[select] %s: %s cycles", path, cycles)
    return cycles
