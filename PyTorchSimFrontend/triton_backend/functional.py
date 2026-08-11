"""The functional half of the Triton route: real tensors -> Spike -> real tensors.

The timing half (timing.py) tells you how long the kernel takes; this tells you
whether it computed the right thing. tnpu's stage 6 already runs the ELF under
Spike, but on inputs it generates itself. Here the launch's own tensors are
written as the `.raw` files stage 6 reads, and the outputs are copied back:

    run(workdir, meta, args)    args -> runtime/*.raw -> spike -> args

The binary is shape-specialised -- the spec bakes the grid, the scalar values and
the memref extents in -- so a launch whose shapes differ from the compiled ones
is rejected rather than silently run against the wrong bounds.
"""

import os
import subprocess

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()

RUNTIME_DIR = "runtime"


class ShapeMismatch(RuntimeError):
    """The launch does not match the shapes the binary was compiled for."""


def _np_dtype(name):
    import numpy as np
    return np.dtype("bool" if name == "bool" else name)


def tensor_args(meta, args):
    """[(arg_meta, tensor)] for the launch, paired by position.

    Inductor passes the tensors first and the numels after, in signature order,
    so `meta["args"]` (tensors only) lines up with the leading arguments.
    """
    import torch

    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    metas = meta["args"]
    if len(tensors) != len(metas):
        raise ShapeMismatch(
            f"{meta['kernel_name']}: launch passed {len(tensors)} tensor(s), "
            f"but the spec declares {len(metas)} ({[m['name'] for m in metas]})")
    return list(zip(metas, tensors))


def _storage_numel(t):
    """How many elements of storage `t` spans -- its last addressable one.

    NOT `t.numel()`. The kernel addresses with the tensor's STRIDES, and a
    tensor whose strides leave gaps reaches past its element count: attention's
    probabilities arrive as `empty_strided((1, 12, 197, 197), (465792, 38816,
    197, 1))`, seven elements of hole at the end of every head. The .raw round
    trip below is a flat file, so it has to be as long as the addresses the
    kernel computes -- read at 465708 the last head is 77 elements short, and
    every head after the first is read at the wrong offset.

    Matches kernel_spec._buffer_numel, which is the same definition on the
    compile side. The two must agree: one sizes the buffer the binary is
    handed, the other the file that fills it.

    IT REPLACES A PERMUTATION, and the measurement that bought that permutation
    still holds -- `as_strided` gives the same answer where it applied. The
    round trip used to `.permute(...).contiguous()` into memory order, because
    `.contiguous()` alone is LOGICAL order and a channels-last tensor is a
    different permutation of the same values:

        measured   resnet18's first conv, whose input the graph put in
                   channels-last. The output matched a torch reference taken
                   over the file read channels-last to 1e-5 and differed from
                   the real answer in 602115 of 802816 elements; PyTorchSim's
                   own per-kernel check said 602108 and the same 1.38288.

    A permutation can only reorder, though, so it could never place a stride
    that skips. Saying it once with the tensor's own strides covers both.
    """
    if t.dim() == 0:
        return 1
    return 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))


def _check(meta, pairs):
    for m, t in pairs:
        if _storage_numel(t) != m["numel"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' spans "
                f"{_storage_numel(t)} element(s) of storage, but the binary was "
                f"compiled for {m['numel']}. "
                f"tnpu bakes the extents, the grid and the scalar values into "
                f"the kernel, so a dynamic-shape graph reuses an ELF that does "
                f"not fit. The timing path does handle this (it takes the grid "
                f"at run time); set pytorchsim_functional_mode: False to study "
                f"cycles alone, or keep shapes static to check values.")
        if str(t.dtype).removeprefix("torch.") != m["dtype"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' is {t.dtype}, but the "
                f"binary was compiled for {m['dtype']}")


def write_inputs(workdir, meta, args):
    """Write every arg as runtime/<name>.raw. Returns the runtime directory.

    Outputs are written too, as zeros: the wrapper loads and dumps by argv
    position, so a missing file shifts every later one.
    """
    import numpy as np

    pairs = tensor_args(meta, args)
    _check(meta, pairs)

    runtime = os.path.join(workdir, RUNTIME_DIR)
    os.makedirs(runtime, exist_ok=True)
    for m, t in pairs:
        path = os.path.join(runtime, f"{m['name']}.raw")
        if m["role"] in ("in", "inout"):
            import torch
            cpu = t.detach().to("cpu")
            flat = torch.zeros(m["numel"], dtype=cpu.dtype)
            # SCATTERED TO THE ADDRESSES THE KERNEL WILL COMPUTE, in one
            # statement, because that is what `as_strided` means. It subsumes
            # the permute-to-memory-order this used to do -- a channels-last
            # tensor lands the same way -- and it also handles the case that one
            # could not express: strides with GAPS in them. Any hole stays zero;
            # nothing logically reads one.
            flat.as_strided(cpu.shape, cpu.stride()).copy_(cpu)
            flat.numpy().tofile(path)
        else:
            np.zeros(m["numel"], dtype=_np_dtype(m["dtype"])).tofile(path)
    return runtime


def read_outputs(workdir, meta, args):
    """Copy the .raw files Spike wrote back into the launch's output tensors."""
    import numpy as np
    import torch

    runtime = os.path.join(workdir, RUNTIME_DIR)
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
        # Gathered back the way it was scattered, by the same one statement:
        # the kernel wrote the addresses the strides name, so read them from
        # there. `view_as(t)` is logical order and is the same defect as
        # `.contiguous()` on the way in -- see _storage_numel.
        stored = torch.from_numpy(flat).as_strided(t.shape, t.stride())
        t.copy_(stored.to(t.dtype))
        written.append(m["name"])
    return written


REPLAY_DIR = ".triton_replay"


def _replay_root(workdir):
    """Beside the workdirs, not inside one.

    A tnpu-side fix is picked up by DELETING `outputs/triton_*`, which is the
    project's own instruction and is what forces the pipeline to run again. A
    cache kept inside a workdir would go with it every time it was most wanted,
    so it lives one level up and is keyed strictly enough not to need the
    deletion: the ELF's bytes are in the key, so the rebuilt kernel misses.
    """
    return os.path.join(os.path.dirname(os.path.abspath(workdir)), REPLAY_DIR)


def _replay_key(workdir, meta, runtime):
    """What this launch's outputs are a function of.

    A GRAPH IS RE-RUN TO SEE THE NEXT LAYER, not the ones already settled, and
    Spike is the whole cost -- resnet18 spends minutes there per conv and the
    first twenty are unchanged between runs. So a launch whose every input is
    the one it had last time can replay that answer instead of simulating it,
    when `TORCHSIM_TRITON_REPLAY=1` asks for it.

    THE KEY IS EVERYTHING THE ANSWER DEPENDS ON: the ELF's own bytes, so a fix anywhere in tnpu misses (the workdir
    is keyed by the TRITON source alone and would not), and the bytes of every
    input, so a different tensor misses. The Triton source is already in the
    workdir path. Nothing else reaches the kernel.
    """
    import hashlib

    h = hashlib.sha256()
    elf = [f for f in sorted(os.listdir(workdir)) if f.endswith(".elf")]
    if not elf:
        return None                       # nothing compiled yet; do not replay
    with open(os.path.join(workdir, elf[0]), "rb") as f:
        h.update(f.read())
    for m in meta["args"]:
        h.update(("%s|%s|%s|%s;" % (m["name"], m["role"], m["dtype"],
                                    m["numel"])).encode())
        if m["role"] in ("in", "inout"):
            with open(os.path.join(runtime, f"{m['name']}.raw"), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:32]


def _outputs_of(meta):
    return [m["name"] for m in meta["args"] if m["role"] in ("out", "inout")]


def _replay(workdir, meta, runtime, key):
    """Put a saved run's outputs back in runtime/, or say it is not there."""
    import shutil

    saved = os.path.join(_replay_root(workdir), key)
    names = _outputs_of(meta)
    if not all(os.path.isfile(os.path.join(saved, f"{n}.raw")) for n in names):
        return False
    for n in names:
        shutil.copyfile(os.path.join(saved, f"{n}.raw"),
                        os.path.join(runtime, f"{n}.raw"))
    return True


def _save_replay(workdir, meta, runtime, key):
    import shutil

    saved = os.path.join(_replay_root(workdir), key)
    os.makedirs(saved, exist_ok=True)
    for n in _outputs_of(meta):
        shutil.copyfile(os.path.join(runtime, f"{n}.raw"),
                        os.path.join(saved, f"{n}.raw"))


def run(workdir, meta, args, timeout_sec=None):
    """Execute the kernel on the launch's tensors. Returns the names written.

    Returns the same names whether Spike ran or a saved run was replayed; the
    caller is told which in the log, because "it passed" means something
    different when nothing was simulated.
    """
    from filelock import FileLock

    from . import tnpu_bridge

    spec = os.path.join(workdir, f"{meta['kernel_name']}_spec.py")
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"{spec} not found -- compile the kernel first")

    # ONE LAUNCH AT A TIME PER KERNEL, and the lock has to span all three steps.
    # `runtime/` is a FIXED name under the kernel's hash directory, and that
    # directory is shared by every process that compiled the same source -- two
    # test sessions in one TORCHSIM_DUMP_PATH, most obviously. The compile is
    # locked (codecache.py); the launch was not, so A's write_inputs, B's
    # write_inputs, A's spike, A's read_outputs interleaves freely and A reads
    # back the answer to B's tensors. It does not raise: the file is there and
    # the right size, it is just someone else's data.
    #
    # MEASURED: two sessions sharing TORCHSIM_DUMP_PATH on
    # tests/ops/elementwise/test_add.py -- "VectorAdd Test Failed", max abs diff
    # 1.25, on a kernel that passes alone. Wrong values reported as a compiler
    # failure is the most expensive bug this repo can produce.
    #
    # Serialising rather than giving each process its own runtime/ because the
    # directory name is tnpu's (tnpu/spike.py joins "runtime" itself) and that is
    # another repo. Contention is per (kernel, concurrent launch) and one spike
    # run is seconds; wrong answers are not a tradeoff against seconds.
    with FileLock(os.path.join(workdir, ".launch.lock"), timeout=1800):
        return _run_locked(workdir, meta, args, spec, timeout_sec, tnpu_bridge)


def _run_locked(workdir, meta, args, spec, timeout_sec, tnpu_bridge):
    """`run` with the per-kernel launch lock already held."""
    runtime = write_inputs(workdir, meta, args)

    # OFF BY DEFAULT. A full run is the thing being trusted, and a result that
    # came out of a file is not a result the simulator produced today -- the key
    # argues it would have been the same, but an argument is not a measurement.
    # Turn it on for the inner loop, where the same graph is re-run to reach the
    # kernel actually being worked on, and leave it off for anything reported.
    key = None
    if os.environ.get("TORCHSIM_TRITON_REPLAY", "0") == "1":
        key = _replay_key(workdir, meta, runtime)
        if key and _replay(workdir, meta, runtime, key):
            logger.info("[Spike] %s replayed %s (same ELF, same inputs)",
                        meta["kernel_name"], key)
            return read_outputs(workdir, meta, args)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # keep tnpu on its own MLIR bindings
    proc = subprocess.run(
        [extension_config.CONFIG_TNPU_PYTHON, "-m", "tnpu.spike", spec, workdir],
        capture_output=True, text=True, cwd=tnpu_bridge.tnpu_dir(), env=env,
        timeout=timeout_sec)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[Spike] {meta['kernel_name']} failed:\n"
            + (proc.stdout + proc.stderr)[-2000:])

    if key:
        _save_replay(workdir, meta, runtime, key)
    return read_outputs(workdir, meta, args)
