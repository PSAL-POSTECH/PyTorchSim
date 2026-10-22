"""Which config a template kernel is built at: Inductor's own selection, timed on TOGSim.

Inductor offers each mm/bmm/addmm/baddbmm config as a choice and keeps the one with the least time;
in timing mode that time is TOGSim cycles. `time_choices` times the choices as Inductor rendered
them, and `time_module` a module Inductor's epilogue-fusion benchmark wrote, so a template with an
epilogue is picked by its fused cycles. Outside timing mode the offered order decides.

A pick is recorded in Inductor's lookup-table format, one file per machine under `RECORD_DIR`: a
recorded config is the only choice offered the next time, timed or not. What it records is the pick
without an epilogue, so a recorded template built with one fused is `RECORDED_UNFUSED`.
"""

import concurrent.futures
import hashlib
import json
import os
import time

from pytorchsim import config, contract
from pytorchsim.codegen import kernel_metadata
from pytorchsim.driver import autotune, compiler_bridge, compiler_input
from pytorchsim.simulator import timing

logger = config.setup_logger()

RECORD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "selection")
#: What a timed run appends to, under the dump path; scripts/selection_merge.py folds it into a record.
RECORD_LOG = "selection.jsonl"

#: The templates whose configs are timed. Another template (a convolution) keeps the offered order.
TIMED_TEMPLATES = ("triton::mm", "triton::bmm")

#: The KernelTemplateChoice annotation that says where a choice came from.
SOURCE = "tile_source"
RECORDED = "recorded"
RECORDED_UNFUSED = "recorded_unfused"
TIMED = "timed"
HEURISTIC = "heuristic"


def _time_tile(src_code, meta, kernel_name, workdir, cut=None):
    """TOGSim cycles of one kernel, simulated alone; inf if it does not compile or run, or `cut`
    stops its simulation. TOGSim refuses a tile over the spad at load, and that refusal and a cut
    reach here as RuntimeError."""
    from pytorchsim.simulator.togsim import Cut, TOGSimProcess
    os.makedirs(workdir, exist_ok=True)
    spec_path = os.path.join(workdir, f"{kernel_name}_spec.py")
    with open(os.path.join(workdir, "kernel.py"), "w") as f:
        f.write(src_code)
    timing.store_meta(workdir, meta)
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
        return TOGSimProcess.get_result_from_file(result)[-1]
    except (compiler_bridge.CompilerError, RuntimeError, FileNotFoundError) as exc:
        why = autotune.too_big(exc) or (f"cut, {exc}" if isinstance(exc, Cut) else type(exc).__name__)
        logger.info("[select] %s %s: %s", kernel_name, workdir, why)
        return float("inf")


def _workdir(name):
    return os.path.join(config.get_dump_path(), "select", name)


def _timed(choice):
    from torch._inductor.select_algorithm import TritonTemplateCaller
    ktc = choice.annotations.get("ktc")
    return (isinstance(choice, TritonTemplateCaller) and ktc is not None
            and ktc.template.uid in TIMED_TEMPLATES)


def times(choices):
    """Whether these choices are picked by their cycles: some are configs of a timed template."""
    return any(_timed(c) for c in choices)


#: A config still simulating this many times the wall time the first config to finish took is cut.
#: The configs run at once, so a loaded host slows them alike and the ratio stands.
CUT_FACTOR = 3


def time_choices(choices):
    """Cycles per choice for AlgorithmSelectorCache: each config of a timed template as Inductor
    rendered it, all at once, every other choice inf. A config whose simulation runs past
    CUT_FACTOR times the wall time of the first one timed is cut, and is inf."""
    out = {c: float("inf") for c in choices}
    timed = [c for c in choices if _timed(c)]
    # Rendered here: the meta reads Inductor's state, which the worker threads do not carry.
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
            c.annotations["ktc"].annotations[SOURCE] = TIMED
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


def keep_recorded(choices):
    """Inductor's preprocessing step: a recorded choice alone, so it is taken without a benchmark."""
    for c in choices:
        ktc = c.annotations.get("ktc")
        if ktc is not None and ktc.annotations.get(SOURCE) == RECORDED:
            return [c]
    return choices


def record(timings, name, input_nodes, choices, *_):
    """Inductor's feedback saver: append a timed pick to RECORD_LOG in lookup-table form, keyed
    the way the choices handler looks it up. Called before any fusion is benchmarked, so this is
    the pick without an epilogue."""
    from torch._inductor.virtualized import V

    if not config.pytorchsim_timing_mode:
        return
    timed = {c: t for c, t in timings.items() if _timed(c) and t != float("inf")}
    if not timed:
        return
    best = min(timed, key=timed.get)
    ktc = best.annotations.get("ktc")
    if ktc is None or ktc.annotations.get(SOURCE) == RECORDED:
        return
    key = V.choices.make_lookup_key(ktc.inputs, name, include_device=True)
    if key is None:
        return
    entry = {"template_id": ktc.template.uid, **ktc.params.to_kwargs()}
    src_hash = getattr(ktc.template, "src_hash", None)
    if src_hash is not None:
        entry["template_hash"] = src_hash
    line = json.dumps({"identity": identity(), "key": key, "config": entry}, sort_keys=True)
    with open(os.path.join(config.get_dump_path(), RECORD_LOG), "a") as f:
        f.write(line + "\n")


def device_key():
    """The device part of a lookup key: the machine's name."""
    return compiler_bridge.target().name


def record_path():
    return os.path.join(RECORD_DIR, f"{device_key()}.json")


#: The compiler this tree pins, and with it TOGSim.
PIN_FILE = os.path.join(config.CONFIG_TORCHSIM_DIR, "third_party", "pytorchsim-triton-backend.json")


def identity():
    """What a recorded pick was timed with: the compiler this tree pins, the machine and TOGSim's
    runtime ABI."""
    from pytorchsim.driver import codecache
    with open(PIN_FILE) as f:
        pin = json.load(f)["pytorchsim_triton_backend"]["ref"]
    whole = codecache.identity()
    return {"compiler": pin, "machine": whole["machine"], "tog_abi": whole["tog_abi"]}


_table = None


def lookup_table():
    """This machine's recorded picks, {lookup key: [config]}; empty when none are recorded or
    they were timed with another identity."""
    global _table
    if _table is None:
        try:
            with open(record_path()) as f:
                rec = json.load(f)
        except FileNotFoundError:
            rec = None
        if rec is None:
            _table = {}
        elif rec["identity"] != identity():
            logger.warning("[select] %s was timed with another compiler, machine or TOGSim; its "
                           "picks are not used", record_path())
            _table = {}
        else:
            _table = rec["table"]
    return _table


def record_digest():
    """A digest of this machine's record file, for Inductor's cache key: a graph compiled
    against other picks is not reused."""
    try:
        with open(record_path(), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except (FileNotFoundError, compiler_bridge.NoTarget):
        return "none"
