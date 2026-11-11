"""Where a launch's tensors sit in simulated DRAM, the one file Spike and TOGSim both read: per kernel argument in
order, the address the device allocator gave its storage plus its offset (the compiler's TENSOR_ADDRS_FILE).
Spike's copy goes beside the .raw files in runtime/, TOGSim's into the launch directory beside tog.so."""

import os

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge

logger = config.setup_logger()


class ShapeMismatch(RuntimeError):
    """The launch does not match the shapes the binary was compiled for."""


def write_tensor_addrs(workdir, meta, args, launch_dir=None):
    """Write tensor_addrs_file() for Spike (runtime/) or, given `launch_dir`, for TOGSim (there). Refused past the
    machine's dram_size; no file (a stale one removed, the reason logged) when a tensor is not an npu one."""
    into = contract.runtime_dir(workdir) if launch_dir is None else launch_dir
    path = os.path.join(into, tensor_addrs_file())
    lines = _get_addr_lines(workdir, meta, args)
    if lines is None:
        if os.path.exists(path):
            os.remove(path)
        return None
    with open(path, "w") as f:
        f.write("".join(lines))
    return path


def _get_addr_lines(workdir, meta, args):
    """The file's lines, or None, logged, when some argument is on no npu storage (template timing's CPU zeros)."""
    import torch

    ko = compiler_bridge.compiler_contract("kernel_object")
    pairs = tensor_args(meta, args) if any(torch.is_tensor(a) for a in args) else []
    off = [m["name"] for m, t in pairs if t.device.type != "npu"]
    if off or not pairs:
        logger.info("%s: %s on no npu storage, so no %s and its arguments are laid out from 0",
                    meta["kernel_name"], ", ".join(off) or "no tensor", ko.TENSOR_ADDRS_FILE)
        return None
    by_name = {m["name"]: t for m, t in pairs}
    target = compiler_bridge.target()
    most = torch.npu.sim_address_alignment()
    lines = []
    for a in contract.kernel_object(workdir)["args"]:
        addr, size = torch.npu.sim_address(by_name[a["name"]]), ko.nbytes(a)
        if addr + size > target.dram_size:
            raise RuntimeError(
                f"{meta['kernel_name']}: argument {a['name']} sits at [{addr:#x}, {addr + size:#x}) of simulated "
                f"DRAM, past the {target.dram_size:#x} bytes of {target.file}: the npu storages alive now do not "
                f"fit that machine")
        lines.append(ko.tensor_addr_line(a["name"], addr, size, min(addr & -addr or most, most), a["role"]))
    return lines


def tensor_addrs_file():
    """The file a launch's tensor addresses are in, the compiler's kernel_object.TENSOR_ADDRS_FILE."""
    return compiler_bridge.compiler_contract("kernel_object").TENSOR_ADDRS_FILE


def tensor_args(meta, args):
    """[(arg_meta, tensor)] for the launch, paired by position: Inductor passes the tensors first,
    in signature order, so meta["args"] lines up with the leading arguments."""
    import torch

    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    metas = meta["args"]
    if len(tensors) != len(metas):
        raise ShapeMismatch(
            f"{meta['kernel_name']}: launch passed {len(tensors)} tensor(s), "
            f"but the spec declares {len(metas)} ({[m['name'] for m in metas]})")
    return list(zip(metas, tensors))
