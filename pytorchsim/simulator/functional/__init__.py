"""The functional path: a launch's tensors -> runtime/*.raw -> Spike -> its tensors.

    run(workdir, meta, args)    write the inputs, run Spike (spike.py), read the outputs back
    write_inputs, read_outputs  tensors <-> .raw files; the binary is shape-specialised, so a launch
                                whose shapes differ from the compiled ones is refused
    replay                      a saved run's outputs, opt-in (TORCHSIM_TRITON_REPLAY=1)
"""

import os

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge
from pytorchsim.contract import storage_span
from pytorchsim.simulator.functional import spike

logger = config.setup_logger()

REPLAY_DIR = ".triton_replay"


class ShapeMismatch(RuntimeError):
    """The launch does not match the shapes the binary was compiled for."""


#: Dtypes numpy cannot name, and the unsigned int of the same width that carries their bytes
#: instead. The .raw file holds the ORIGINAL bits either way -- the view is a reinterpretation,
#: never a conversion. Same table as the compiler's contract/kernel_object.py NP_DTYPE.
_NP_CARRIER = {"float8_e5m2": "uint8", "float8_e4m3fn": "uint8",
               "bfloat16": "uint16"}


def run(workdir, meta, args):
    """Execute the kernel on the launch's tensors. Returns the names written.

    No lock: every file this writes is under `workdir.runtime_dir`, which is
    this process's own, so concurrent launches of one kernel never meet.
    Replay is OFF by default -- a result out of a file is not one the simulator
    produced today; it is for the inner loop, not for reporting.
    """
    runtime = write_inputs(workdir, meta, args)

    key = None
    if os.environ.get("TORCHSIM_TRITON_REPLAY", "0") == "1":
        key = _replay_key(workdir, meta, runtime)
        if key and _replay(workdir, meta, runtime, key):
            logger.info("[Spike] %s replayed %s (same ELF, same inputs)",
                        meta["kernel_name"], key)
            return read_outputs(workdir, meta, args)

    manifest = contract.kernel_object(workdir)
    if manifest is None:
        raise FileNotFoundError(
            f"{workdir} has no kernel.json -- compile the kernel first")
    raw_paths = {a["name"]: os.path.join(runtime, f"{a['name']}.raw")
                 for a in manifest["args"]}
    values = dict(zip(meta.get("call_params", ()), args))
    try:
        spike.launch(workdir, manifest, runtime, raw_paths, logger.debug, scalar_values=values)
    except spike.launcher().SpikeError as e:
        raise RuntimeError(f"[Spike] {meta['kernel_name']} failed:\n{e}") from None

    if key:
        _save_replay(workdir, meta, runtime, key)
    return read_outputs(workdir, meta, args)


def write_inputs(workdir, meta, args):
    """Write every arg as runtime/<name>.raw. Returns the runtime directory.

    Inputs are scattered to the addresses the kernel will compute; outputs are
    zeros, because the wrapper dumps by argv position and a gap shifts them.
    """
    import numpy as np
    import torch

    pairs = tensor_args(meta, args)
    _check(meta, pairs)

    runtime = contract.runtime_dir(workdir)
    for m, t in pairs:
        path = os.path.join(runtime, f"{m['name']}.raw")
        if m["role"] not in ("in", "inout"):
            np.zeros(m["numel"], dtype=_np_dtype(m["dtype"])).tofile(path)
            continue
        _as_carrier(flat_input(m, t)).numpy().tofile(path)
    return runtime


def flat_input(m, t):
    """The launch tensor `t` of arg `m` as the kernel sees it: a CPU tensor of m's numel elements, each
    at the address `t`'s strides put it, what its .raw file holds."""
    import torch

    cpu = t.detach().to("cpu")
    flat = torch.zeros(m["numel"], dtype=cpu.dtype, device="cpu")
    shape, strides = list(t.shape), list(t.stride())
    src = cpu
    for i in range(len(strides) - 1, -1, -1):
        if strides[i] == 0:
            src = src.select(i, 0)
            shape.pop(i)
            strides.pop(i)
    flat.as_strided(shape, strides).copy_(src)
    return flat


def read_outputs(workdir, meta, args):
    """Copy the .raw files Spike wrote back into the launch's output tensors.

    Gathered back the way they were scattered: the kernel wrote the addresses
    the strides name, so `view_as` would be the same defect as `.contiguous()`.
    """
    import numpy as np
    import torch

    runtime = contract.runtime_dir(workdir)
    written = []
    for m, t in tensor_args(meta, args):
        if m["role"] not in ("out", "inout"):
            continue
        path = os.path.join(runtime, f"{m['name']}.raw")
        flat = np.fromfile(path, dtype=_np_dtype(m["dtype"]))
        if flat.size != m["numel"]:
            raise RuntimeError(
                f"{path} holds {flat.size} element(s), expected {m['numel']} "
                f"-- Spike did not write the whole tensor")
        # view, not .to(): the carrier holds the output's own bits, and a
        # conversion here would read an fp8 byte as the integer 60.
        stored = torch.from_numpy(flat)
        if stored.dtype != t.dtype:
            stored = stored.view(t.dtype)
        t.copy_(stored.as_strided(t.shape, t.stride()))
        written.append(m["name"])
    return written


def tensor_args(meta, args):
    """[(arg_meta, tensor)] for the launch, paired by position.

    Inductor passes the tensors first and the numels after, in signature order,
    so `meta["args"]` lines up with the leading arguments.
    """
    import torch

    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    metas = meta["args"]
    if len(tensors) != len(metas):
        raise ShapeMismatch(
            f"{meta['kernel_name']}: launch passed {len(tensors)} tensor(s), "
            f"but the spec declares {len(metas)} ({[m['name'] for m in metas]})")
    return list(zip(metas, tensors))


def _check(meta, pairs):
    for m, t in pairs:
        if _storage_numel(t) != m["numel"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' spans "
                f"{_storage_numel(t)} element(s) of storage, but the binary was "
                f"compiled for {m['numel']}. "
                f"the compiler bakes the extents and the grid into the kernel, so a "
                f"dynamic-shape graph reuses an ELF that does not fit. The timing "
                f"path does handle this (it takes the grid at run time); set "
                f"TORCHSIM_FUNCTIONAL_MODE=0 to study cycles alone, or keep "
                f"shapes static to check values.")
        if str(t.dtype).removeprefix("torch.") != m["dtype"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' is {t.dtype}, but the "
                f"binary was compiled for {m['dtype']}")


def _storage_numel(t):
    """How many elements of storage `t` spans -- its last addressable one.

    NOT `t.numel()`: the .raw round trip is a flat file and has to be as long
    as the addresses the kernel computes. Same definition as the compile side.
    """
    return storage_span(t.shape, t.stride())


def _np_dtype(name):
    import numpy as np
    if name == "bool":
        return np.dtype("bool")
    return np.dtype(_NP_CARRIER.get(name, name))


def _as_carrier(t):
    """`t` retyped to whatever numpy can hold, bit for bit."""
    import torch

    carrier = _NP_CARRIER.get(str(t.dtype).removeprefix("torch."))
    return t.view(getattr(torch, carrier)) if carrier else t


def _replay(workdir, meta, runtime, key):
    """Put a saved run's outputs, and its runtime attributes, back in runtime/, or say they are not
    there: a saved run without the attributes a kernel declares is a miss."""
    import shutil

    saved = os.path.join(_replay_root(workdir), key)
    names = _outputs_of(meta)
    if not all(os.path.isfile(os.path.join(saved, f"{n}.raw")) for n in names):
        return False
    attributes = _attribute_dir_of(workdir)
    if attributes and not os.path.isdir(os.path.join(saved, attributes)):
        return False
    for n in names:
        shutil.copyfile(os.path.join(saved, f"{n}.raw"),
                        os.path.join(runtime, f"{n}.raw"))
    if attributes:
        shutil.rmtree(os.path.join(runtime, attributes), ignore_errors=True)
        shutil.copytree(os.path.join(saved, attributes), os.path.join(runtime, attributes))
    return True


def _save_replay(workdir, meta, runtime, key):
    import shutil

    saved = os.path.join(_replay_root(workdir), key)
    os.makedirs(saved, exist_ok=True)
    for n in _outputs_of(meta):
        shutil.copyfile(os.path.join(runtime, f"{n}.raw"),
                        os.path.join(saved, f"{n}.raw"))
    attributes = _attribute_dir_of(workdir)
    if attributes:
        shutil.rmtree(os.path.join(saved, attributes), ignore_errors=True)
        shutil.copytree(os.path.join(runtime, attributes), os.path.join(saved, attributes))


def _attribute_dir_of(workdir):
    """runtime/'s directory of runtime attribute records when the kernel declares any, else None."""
    manifest = contract.kernel_object(workdir)
    if manifest is None or not manifest.get("runtime_attributes"):
        return None
    return compiler_bridge.compiler_contract("kernel_object").RUNTIME_ATTRIBUTE_DIR


def _replay_key(workdir, meta, runtime):
    """What this launch's outputs are a function of.

    The ELF's own bytes, so a fix anywhere in the compiler misses, and the bytes of
    every input. The Triton source is already in the workdir path.
    """
    import hashlib

    h = hashlib.sha256()
    elf = [f for f in sorted(os.listdir(workdir)) if f.endswith(".elf")]
    if not elf:
        return None
    with open(os.path.join(workdir, elf[0]), "rb") as f:
        h.update(f.read())
    for m in meta["args"]:
        h.update(("%s|%s|%s|%s;" % (m["name"], m["role"], m["dtype"],
                                    m["numel"])).encode())
        if m["role"] in ("in", "inout"):
            with open(os.path.join(runtime, f"{m['name']}.raw"), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:32]


def _replay_root(workdir):
    """Beside the workdirs, not inside one.

    A compiler-side fix is picked up by DELETING `outputs/triton_*`, and a cache
    kept inside a workdir would go with it every time it was most wanted.
    """
    return os.path.join(os.path.dirname(os.path.abspath(workdir)), REPLAY_DIR)


def _outputs_of(meta):
    return [m["name"] for m in meta["args"] if m["role"] in ("out", "inout")]
