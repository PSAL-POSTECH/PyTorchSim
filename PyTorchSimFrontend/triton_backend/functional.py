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


def _check(meta, pairs):
    for m, t in pairs:
        if t.numel() != m["numel"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' has {t.numel()} "
                f"element(s), but the binary was compiled for {m['numel']}. "
                f"tnpu bakes the extents, the grid and the scalar values into "
                f"the kernel, so a dynamic-shape graph reuses an ELF that does "
                f"not fit. The timing path does handle this (it takes the grid "
                f"at run time); set pytorchsim_functional_mode: False to study "
                f"cycles alone, or keep shapes static to check values.")
        if str(t.dtype).removeprefix("torch.") != m["dtype"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' is {t.dtype}, but the "
                f"binary was compiled for {m['dtype']}")


def _memory_order(t):
    """`t`'s axes ordered the way its storage is, outermost first.

    THE KERNEL INDEXES WITH THE TENSOR'S STRIDES, not with its shape. Inductor
    reads those strides at compile time and writes them into the source as
    constants -- resnet18's first conv carries `stride_xc = 1`, because the
    graph put its input in channels-last -- so the bytes the kernel is handed
    have to be the storage in ADDRESS order. `.contiguous()` produces logical
    order instead, which for a channels-last tensor is a different permutation
    of the same values, and the kernel then reads every element from the wrong
    place while computing perfectly correctly on what it found.

    MEASURED on resnet18's first conv: the output matched a torch reference
    taken over the file read channels-last to 1e-5, and differed from the real
    answer in 602115 of 802816 elements. PyTorchSim's own per-kernel check said
    602108 and the same 1.38288 -- the same divergence from the other side.
    """
    return sorted(range(t.dim()), key=lambda i: -t.stride(i))


def _inverse(order):
    """The permutation that puts `order` back."""
    inv = [0] * len(order)
    for slot, axis in enumerate(order):
        inv[axis] = slot
    return inv


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
            cpu = t.detach().to("cpu")
            (cpu.permute(*_memory_order(cpu)).contiguous()
                .numpy().reshape(-1).tofile(path))
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
        # Scattered back the way it was gathered: the kernel wrote address
        # order, so the flat buffer is read in the tensor's storage order and
        # permuted home. `view_as(t)` is logical order and is the same defect
        # as `.contiguous()` on the way in -- see _memory_order.
        order = _memory_order(t)
        stored = torch.from_numpy(flat).view(*[t.shape[i] for i in order])
        t.copy_(stored.permute(*_inverse(order)).to(t.dtype))
        written.append(m["name"])
    return written


def run(workdir, meta, args, timeout_sec=None):
    """Execute the kernel on the launch's tensors. Returns the names written."""
    from . import tnpu_bridge

    spec = os.path.join(workdir, f"{meta['kernel_name']}_spec.py")
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"{spec} not found -- compile the kernel first")

    write_inputs(workdir, meta, args)

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

    return read_outputs(workdir, meta, args)
