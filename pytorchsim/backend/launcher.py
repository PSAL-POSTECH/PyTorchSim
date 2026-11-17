"""The runner PTB's Triton driver launches a compiled kernel through (driver.set_launcher): the ELF and
kernel.json it is handed become a workdir, and the launch runs on Spike and, in timing mode, TOGSim."""

import base64
import hashlib
import json
import math
import os

from filelock import FileLock

from pytorchsim import config
from pytorchsim.backend import compiler_input, config_pick, launch_hooks
from pytorchsim.simulator import functional, timing, wallclock

logger = config.setup_logger()

LOCK_TIMEOUT = 600
_AXES = ("x", "y", "z")


class Launch:
    """What launch_hooks observers see of a launch: the kernel, where its object is, and its meta."""

    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta


def install():
    """Register run_launch as the runner of PTB's Triton driver."""
    from triton.backends.pytorchsim.driver import set_launcher

    set_launcher(run_launch)


def run_launch(binary, kernel_object, grid, args, artifacts=None):
    """One launch over `grid`: Spike first, so the tensors hold real values even if TOGSim fails, then (timing
    mode) TOGSim; returns TOGSim's result, None otherwise or when the launch is queued on the open simulator."""
    manifest = json.loads(kernel_object)
    workdir = _get_workdir(binary, kernel_object, manifest, artifacts)
    meta = _get_meta(manifest, grid, args)
    launch = Launch(manifest["name"], workdir, meta)
    if config_pick.measuring():
        config_pick.report(_simulate_alone(workdir, meta, args))
        return None
    launch_hooks.notify_launch(launch, args)
    if config.pytorchsim_functional_mode:
        with wallclock.span(wallclock.SPIKE, launch.kernel_name):
            written = functional.run(workdir, meta, args)
        logger.info("[Spike] %s wrote %s", launch.kernel_name, written)
    else:
        logger.warning("[Spike] %s: functional mode is off, so the output tensors keep "
                       "whatever they held", launch.kernel_name)

    if not config.pytorchsim_timing_mode:
        logger.warning("[timing] %s: timing mode is off, so no cycles are reported", launch.kernel_name)
        return None
    result = timing.run(workdir, meta, args, launch_hooks.take_call_names(len(args)))
    if result is None:
        logger.info("[TOGSim] %s queued on the open TOGSimulator; the stream's cycles are reported "
                    "when the simulator closes", launch.kernel_name)
    else:
        logger.info("[TOGSim] %s simulated: %d cycles", launch.kernel_name, result["total_cycles"])
    return result


def _store_meta(workdir, meta, args, names):
    """workdir/meta.json, once: the launch's meta with each tensor's sizes and buffer, which npuwattch reads."""
    import torch

    path = os.path.join(workdir, compiler_input.META_JSON)
    if os.path.exists(path):
        return
    tensors = [(i, a) for i, a in enumerate(args) if torch.is_tensor(a)]
    stored = dict(meta, args=[dict(m, sizes=list(t.shape), buffer=names[i] if names else None)
                              for m, (i, t) in zip(meta["args"], tensors)])
    with open(path + ".tmp", "w") as f:
        json.dump(stored, f)
    os.replace(path + ".tmp", path)


def _simulate_alone(workdir, meta, args):
    """TOGSim cycles of this launch run alone (a configuration being timed): inf when it does not simulate."""
    from pytorchsim import contract
    from pytorchsim.backend import template_timing

    try:
        timing.write_cycle_tables(workdir, meta)
        mine = contract.link_shared(workdir, timing.SHARED)
        timing.write_shape(workdir, meta, args)
        timing.write_tensors(workdir)
        return timing.simulate_standalone(workdir, mine, template_timing._records(workdir, meta),
                                          autotune_mode=True)["total_cycles"]
    except (RuntimeError, FileNotFoundError) as exc:
        logger.info("[select] %s: %s", meta["kernel_name"], exc)
        return float("inf")


def _get_workdir(binary, kernel_object, manifest, artifacts):
    """The directory of this binary's kernel object, one per (ELF, kernel.json) under the dump path, written
    once, with the other files its object names beside it."""
    key = hashlib.sha256(bytes(binary) + kernel_object.encode()).hexdigest()[:11]
    workdir = os.path.join(config.get_dump_path(), f"triton_{key}")
    os.makedirs(workdir, exist_ok=True)
    arts = manifest["artifacts"]
    files = {arts["elf"]: bytes(binary), "kernel.json": kernel_object.encode()}
    for item in json.loads(artifacts).values() if artifacts else ():
        files[item["name"]] = base64.b64decode(item["data"])
    with FileLock(os.path.join(workdir, ".launch.lock"), timeout=LOCK_TIMEOUT):
        for name, data in files.items():
            path = os.path.join(workdir, name)
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".tmp", "wb") as f:
                    f.write(data)
                os.chmod(path + ".tmp", 0o755)
                os.replace(path + ".tmp", path)
    return workdir


def _get_meta(manifest, grid, args):
    """The launch's meta, as the functional and timing runners read it: the manifest's tensors by role,
    dtype and storage span, its scalars ({name, ctype, default}), the constexprs it was built with, each
    argument's parameter name in order, the launch's extents per axis."""
    import torch

    tensors = iter(manifest["args"])
    scalars = iter(manifest.get("scalars", ()))
    call_params = [next(tensors)["name"] if torch.is_tensor(a) else next(scalars)["name"] for a in args]
    axes = manifest["abi"]["parallel_axes"]
    return {
        "kernel_name": manifest["name"],
        "args": [{"name": a["name"], "role": a["role"], "dtype": a["dtype"],
                  "numel": math.prod(a["shape"]), "sizes": None} for a in manifest["args"]],
        "call_params": call_params,
        "numels": {},
        "template_grid": None,
        "launch_grid": tuple(int(g) for g in grid),
        "launch_extents": [int(grid[_AXES.index(a)]) for a in axes],
        "user_kernel": launch_hooks.user_kernel_of(manifest["name"]),
        "scalars": [dict(x) for x in manifest.get("scalars", ())],
        "constexprs": dict(manifest.get("constexprs", {})),
    }
