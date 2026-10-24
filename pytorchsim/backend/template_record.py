"""A template config's record: where a choice came from (SOURCE), the timed templates, and the picks a timed
run appends to RECORD_LOG, recorded per machine under RECORD_DIR in Inductor's lookup-table format and taken as
the only choice next time; its digest is part of Inductor's graph cache key."""

import hashlib
import json
import os

from pytorchsim import config
from pytorchsim.backend import compiler_bridge

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


def is_timed(choice):
    from torch._inductor.select_algorithm import TritonTemplateCaller
    ktc = choice.annotations.get("ktc")
    return (isinstance(choice, TritonTemplateCaller) and ktc is not None
            and ktc.template.uid in TIMED_TEMPLATES)


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
    timed = {c: t for c, t in timings.items() if is_timed(c) and t != float("inf")}
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
    from pytorchsim.backend import build_identity
    with open(PIN_FILE) as f:
        pin = json.load(f)["pytorchsim_triton_backend"]["ref"]
    whole = build_identity.identity()
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


def graph_digest():
    """A digest of what an npu graph is compiled against, for Inductor's cache key: the machine
    (its name, the keys the compiler reads, num_cores) and its recorded picks."""
    try:
        t = compiler_bridge.target()
        machine = {"name": t.name, "keys": t.as_dict(), "num_cores": config.CONFIG_NUM_CORES}
    except compiler_bridge.NoTarget:
        machine = None
    said = json.dumps({"machine": machine, "record": record_digest()}, sort_keys=True)
    return hashlib.sha256(said.encode()).hexdigest()[:16]
