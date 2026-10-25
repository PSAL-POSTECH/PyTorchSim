"""A template config's record: where a choice came from (SOURCE), the timed templates, and the picks a timed
run appends to RECORD_LOG, recorded per machine under RECORD_DIR in Inductor's lookup-table format and taken as
the only choice next time; its digest is part of Inductor's graph cache key. codegen_mapping_strategy says
whether tiles are timed (autotunes) and whether codegen_external_mapping_file's tiles come first."""

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
TIMED_TEMPLATES = ("triton::mm", "triton::bmm", "triton::mm_resident")


#: The KernelTemplateChoice annotation that says where a choice came from.
SOURCE = "tile_source"


RECORDED = "recorded"


RECORDED_UNFUSED = "recorded_unfused"


TIMED = "timed"


HEURISTIC = "heuristic"


#: The one config a rule of the compiler's contract gives (the resident mm, copy_node.plan): nothing to time.
PLANNED = "planned"


#: A tile codegen_external_mapping_file states for the shape (an external- strategy).
EXTERNAL = "external"


EXTERNAL_KEYS = ("TILE_M", "TILE_N", "TILE_K")


#: The templates an external tile is built at: their BLOCK_M/N/K are the tile.
EXTERNAL_TEMPLATES = ("triton::mm", "triton::bmm")


def is_timed(choice):
    from torch._inductor.select_algorithm import TritonTemplateCaller
    ktc = choice.annotations.get("ktc")
    return (isinstance(choice, TritonTemplateCaller) and ktc is not None
            and ktc.template.uid in TIMED_TEMPLATES
            and ktc.annotations.get(SOURCE) not in (RECORDED, EXTERNAL))


def keep_recorded(choices):
    """Inductor's preprocessing step: a recorded or external choice alone, taken without a benchmark."""
    for c in choices:
        ktc = c.annotations.get("ktc")
        if ktc is not None and ktc.annotations.get(SOURCE) in (RECORDED, EXTERNAL):
            return [c]
    return choices


def record(timings, name, input_nodes, choices, *_):
    """Inductor's feedback saver: append a timed pick to RECORD_LOG in lookup-table form, keyed
    the way the choices handler looks it up. Called before any fusion is benchmarked, so this is
    the pick without an epilogue."""
    from torch._inductor.virtualized import V

    if not config.pytorchsim_timing_mode or not autotunes():
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


def autotunes():
    """Whether codegen_mapping_strategy times the heuristic's top tiles (in timing mode)."""
    return config.codegen_mapping_strategy.endswith("autotune")


_warned_untimed = []


def warn_untimed_autotune():
    """Say once that an autotune strategy a user stated is not timed outside timing mode."""
    if autotunes() and not _warned_untimed and \
            config.run_setting_source("codegen_mapping_strategy") != "default":
        _warned_untimed.append(True)
        logger.warning("[select] codegen_mapping_strategy %s times tiles in timing mode only; "
                       "TORCHSIM_TIMING_MODE is 0, so the heuristic's first tile is taken",
                       config.codegen_mapping_strategy)


_external = {}


def external_table():
    """{"M_N_K": (TILE_M, TILE_N, TILE_K)} of codegen_external_mapping_file; {} unless the strategy
    is an external- one. A file missing, or an entry not exactly TILE_M, TILE_N, TILE_K, is refused."""
    strategy = config.codegen_mapping_strategy
    if not strategy.startswith("external-"):
        return {}
    path = config.codegen_external_mapping_file
    if not path:
        raise ValueError(f"codegen_mapping_strategy {strategy} names no codegen_external_mapping_file")
    try:
        mtime = os.path.getmtime(path)
    except OSError as e:
        raise ValueError(f"codegen_external_mapping_file {path}: {e.strerror}") from None
    if _external.get(path, (None,))[0] != mtime:
        with open(path) as f:
            raw = json.load(f)
        table = {}
        for shape, tile in raw.items():
            if not isinstance(tile, dict) or sorted(tile) != sorted(EXTERNAL_KEYS) or \
                    not all(isinstance(tile[k], int) and tile[k] > 0 for k in EXTERNAL_KEYS):
                raise ValueError(f"{path}: {shape} is {tile!r}; an entry is exactly "
                                 f"{', '.join(EXTERNAL_KEYS)}, positive integers")
            table[shape] = tuple(tile[k] for k in EXTERNAL_KEYS)
        _external[path] = (mtime, table)
    return _external[path][1]


def external_tile(m, n, k):
    """The tile the external mapping states for an [m, k] @ [k, n] (a bmm's per batch), or None."""
    return external_table().get(f"{m}_{n}_{k}")


def selection_digest():
    """What picks a template's tile besides the machine and the record, for Inductor's cache key:
    the strategy, the top-k it times and the external mapping file's contents."""
    strategy = config.codegen_mapping_strategy
    parts = [strategy, str(config.codegen_autotune_template_topk)]
    if strategy.startswith("external-"):
        external_table()
        with open(config.codegen_external_mapping_file, "rb") as f:
            parts.append(hashlib.sha256(f.read()).hexdigest()[:16])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


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
    (its name, the keys the compiler reads, num_cores), its recorded picks and selection_digest()."""
    try:
        t = compiler_bridge.target()
        machine = {"name": t.name, "keys": t.as_dict(), "num_cores": config.CONFIG_NUM_CORES}
    except compiler_bridge.NoTarget:
        machine = None
    said = json.dumps({"machine": machine, "record": record_digest(),
                       "selection": selection_digest()}, sort_keys=True)
    return hashlib.sha256(said.encode()).hexdigest()[:16]
