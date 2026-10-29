"""A template config timed in TOGSim cycles, in timing mode: every config of a timed template at once
(time_choices, a slow one cut), and a module Inductor wrote to benchmark a fusion (time_module). What a
timed pick leaves for next time is template_record.py."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time

from pytorchsim import config, contract
from pytorchsim.frontend.inductor.autotune import graph_prefetch
from pytorchsim.frontend.inductor.codegen import kernel_metadata
from pytorchsim.backend import retile, compiler_bridge, compiler_input, template_record
from pytorchsim.simulator import timing

logger = config.setup_logger()


def _time_tile(src_code, meta, kernel_name, workdir, cut=None):
    """TOGSim cycles of one kernel, simulated alone; inf if it does not compile in the bridge's timeout
    or run, or `cut` stops its simulation. TOGSim's refusal of a tile over the spad and a cut reach here
    as RuntimeError."""
    from pytorchsim.simulator.timing.togsim import Cut
    os.makedirs(workdir, exist_ok=True)
    spec_path = os.path.join(workdir, f"{kernel_name}_spec.py")
    with open(os.path.join(workdir, "kernel.py"), "w") as f:
        f.write(src_code)
    compiler_input.store_meta(workdir, meta)
    compiler_input.write(src_code, meta, spec_path)
    try:
        compiler_bridge.run_pipeline(spec_path, workdir, to_stage="pytorchsim-compile", tog=True)
        timing.write_cycle_tables(workdir, meta)
        mine = contract.link_shared(workdir, timing.SHARED)
        timing.write_shape(workdir, meta)
        timing.write_tensors(workdir)
        return timing.simulate_standalone(workdir, mine, _records(workdir, meta),
                                          autotune_mode=True, cut=cut)["total_cycles"]
    except subprocess.TimeoutExpired as exc:
        logger.info("[select] %s %s: not compiled in %.0f s", kernel_name, workdir, exc.timeout)
        return float("inf")
    except (compiler_bridge.CompilerError, RuntimeError, FileNotFoundError) as exc:
        why = retile.too_big(exc) or (f"cut, {exc}" if isinstance(exc, Cut) else f"{type(exc).__name__}: {exc}")
        logger.info("[select] %s %s: %s", kernel_name, workdir, why)
        return float("inf")


def _records(workdir, meta):
    """The runtime attribute records the TOG of the kernel in `workdir` reads: "" when it declares none, else a
    Spike run's on zeros, the launch's own when no record reads an input (its sources); records that read one
    have no values to be timed by here, a RuntimeError naming those inputs."""
    import torch
    from pytorchsim.simulator import functional
    attributes = ((contract.kernel_object(workdir) or {}).get("runtime_attributes") or {}).get("attributes") or []
    if not attributes:
        return ""
    read = sorted({s for a in attributes for s in a.get("sources", ())})
    if read:
        raise RuntimeError(f"its runtime attributes read {read}, whose launch values a benchmark does not have")
    functional.run(workdir, meta, [torch.zeros(a["numel"], dtype=getattr(torch, a["dtype"])) for a in meta["args"]])
    return timing.attribute_dir(workdir)


def _workdir(name):
    return os.path.join(config.get_dump_path(), "select", name)


#: Cycles this process already timed, by _timing_key: a shape repeated across a graph's nodes is timed once.
_timed = {}


def _timing_key(src_code, meta):
    """What a kernel's cycles follow from: its source and its metadata, less the names that differ
    between graph nodes of one shape (the kernel's, each operand's buffer). A buffer is its index in
    order of first use, so operands that share one still do."""
    meta = dict(meta)
    meta.pop("kernel_name", None)
    order = {}
    args = []
    for a in meta.get("args") or []:
        a = dict(a)
        if a.get("buffer") is not None:
            a["buffer"] = order.setdefault(a["buffer"], len(order))
        args.append(a)
    meta["args"] = args
    said = json.dumps(meta, sort_keys=True, default=repr)
    return hashlib.sha256(src_code.encode()).hexdigest(), hashlib.sha256(said.encode()).hexdigest()


def times(choices):
    """Whether these choices are picked by their cycles: the strategy autotunes, and some are
    configs of a timed template that no record or external mapping fixed."""
    return template_record.autotunes() and any(template_record.is_timed(c) for c in choices)


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
    for c, _, meta in jobs:
        meta["graph_prefetch"] = graph_prefetch.choice_statement(c.annotations["ktc"], meta["args"])
    keys = {c: _timing_key(src, meta) for c, src, meta in jobs}
    for c, _, _ in jobs:
        if keys[c] in _timed:
            logger.info("[select] %s %s: %s cycles, timed before in this process",
                        c.name, c.description, _timed[keys[c]])
            c.annotations["ktc"].annotations[template_record.SOURCE] = template_record.TIMED
            out[c] = _timed[keys[c]]
    jobs = [j for j in jobs if keys[j[0]] not in _timed]
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
            out[c] = _timed[keys[c]] = got
    if timed and all(out[c] == float("inf") for c in timed):
        raise RuntimeError(
            f"no template config of {[c.name for c in timed]} compiled and ran on TOGSim; "
            f"the [select] lines above say why each one did not")
    return out


def time_module(src_code, meta, path):
    """TOGSim cycles of a kernel module Inductor wrote to benchmark a fusion, named by its path."""
    key = _timing_key(src_code, meta)
    if key in _timed:
        logger.info("[select] %s: %s cycles, timed before in this process", path, _timed[key])
        return _timed[key]
    name = os.path.splitext(os.path.basename(path))[0]
    cycles = _timed[key] = _time_tile(src_code, meta, meta["kernel_name"], _workdir(name))
    logger.info("[select] %s: %s cycles", path, cycles)
    return cycles
