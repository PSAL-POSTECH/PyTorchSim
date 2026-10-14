"""The functional half of the Triton route: real tensors -> Spike -> real tensors.

    run(workdir, meta, args)    args -> runtime/*.raw -> spike -> args

The binary is shape-specialised, so a launch whose shapes differ from the
compiled ones is rejected rather than run against the wrong bounds.
"""

import concurrent.futures as cf
import os
import re
import shutil
import subprocess

from pytorchsim import config

from pytorchsim import contract
from pytorchsim.contract import storage_span
from pytorchsim.driver import compiler_bridge

logger = config.setup_logger()

REPLAY_DIR = ".triton_replay"


class ShapeMismatch(RuntimeError):
    """The launch does not match the shapes the binary was compiled for."""


#: Dtypes numpy cannot name, and the unsigned int of the same width that carries their bytes
#: instead. The .raw file holds the ORIGINAL bits either way -- the view is a reinterpretation,
#: never a conversion. Same table as the compiler's contract/kernel_object.py NP_DTYPE.
_NP_CARRIER = {"float8_e5m2": "uint8", "float8_e4m3fn": "uint8",
               "bfloat16": "uint16"}


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


def _storage_numel(t):
    """How many elements of storage `t` spans -- its last addressable one.

    NOT `t.numel()`: the .raw round trip is a flat file and has to be as long
    as the addresses the kernel computes. Same definition as the compile side.
    """
    return storage_span(t.shape, t.stride())


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
        _as_carrier(flat).numpy().tofile(path)
    return runtime


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


def _replay_root(workdir):
    """Beside the workdirs, not inside one.

    A compiler-side fix is picked up by DELETING `outputs/triton_*`, and a cache
    kept inside a workdir would go with it every time it was most wanted.
    """
    return os.path.join(os.path.dirname(os.path.abspath(workdir)), REPLAY_DIR)


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
    scalars = argv_scalars(manifest, dict(zip(meta.get("call_params", ()), args)))
    try:
        launch(workdir, manifest, runtime, raw_paths, logger.debug, scalars=scalars)
    except SpikeError as e:
        raise RuntimeError(f"[Spike] {meta['kernel_name']} failed:\n{e}") from None

    if key:
        _save_replay(workdir, meta, runtime, key)
    return read_outputs(workdir, meta, args)


_VCIX_ENV = os.environ.get("VCIX_ENV_ROOT", "/opt/vcix-env")


SPIKE = os.environ.get("TORCHSIM_SPIKE", os.environ.get(
    "TORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike")))


PK = os.environ.get("TORCHSIM_PK", os.environ.get(
    "TORCHSIM_COMPILE_PK", os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")))


#: Second copy of the compiler's contract/kernel_object.py ITEMSIZE. The two
#: describe the same .raw files, so an entry added to one belongs in both.
ITEMSIZE = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
            "float8_e5m2": 1, "float8_e4m3fn": 1,
            "int64": 8, "int32": 4, "int16": 2, "int8": 1,
            "uint64": 8, "uint32": 4, "uint16": 2, "uint8": 1, "bool": 1}


TRAP_HINT = {200: "INVALID_SPAD_ACCESS", 201: "STACK_OVERFLOW"}


class SpikeError(RuntimeError):
    pass


class _NotSplittable(Exception):
    """Which program ran last decides the answer, so the split cannot give it."""


def artifact(obj_dir, manifest, kind):
    """One compiler output, by the name the manifest gives it."""
    rel = manifest.get("artifacts", {}).get(kind)
    if not rel:
        have = ", ".join(sorted(manifest.get("artifacts", {}))) or "none"
        raise SpikeError(f"{obj_dir}: kernel object has no `{kind}`. Has: {have}")
    return os.path.join(obj_dir, rel)


def argv_order(manifest):
    """The .raw names in the order the wrapper's main() expects them."""
    return [a["name"] for a in manifest["args"]]


def argv_scalars(manifest, values):
    """The scalar values main() reads after the tensor paths, as strings in parameter order:
    the launch's by name, else the manifest's defaults."""
    out = []
    for s in manifest.get("scalars", []):
        v = values.get(s["name"], s["default"])
        if v is None:
            raise SpikeError(f"{manifest['name']}: no value for scalar '{s['name']}' in this "
                             f"launch, and the spec gave it no default")
        out.append(repr(float(v)) if s["ctype"] in ("float", "double", "_Float16")
                   else str(int(v)))
    return out


def grid3(manifest):
    """The launch grid padded to three axes."""
    g = list(manifest["grid"]) + [1, 1, 1]
    return g[0], g[1], g[2]


def writable(manifest):
    """The arguments the kernel writes -- out and inout."""
    return [a for a in manifest["args"] if a["role"] in ("out", "inout")]


def splittable(manifest):
    """Whether the grid may be cut into ranges across processes."""
    return not any(a["role"] == "inout" for a in manifest["args"])


def nbytes(arg):
    """Bytes one argument occupies as a .raw file."""
    n = 1
    for d in arg["shape"]:
        n *= d
    return n * ITEMSIZE[arg["dtype"]]


def _run(cmd, cwd, log):
    """Spawn one process, raising SpikeError with what it said."""
    log("[Spike] $ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        said = (proc.stdout + proc.stderr).strip().splitlines()
        hint = TRAP_HINT.get(proc.returncode)
        raise SpikeError(
            f"spike exit code {proc.returncode}"
            + (f" ({hint})" if hint else "")
            + ("\n" + "\n".join(said[-40:]) if said else ""))
    return proc


def launch(obj_dir, manifest, runtime, raw_paths, log, jobs=None, scalars=()):
    """Run the whole launch grid, leaving each written argument in its .raw."""
    elf = artifact(obj_dir, manifest, "elf")
    shutil.rmtree(os.path.join(runtime, "indirect_access"), ignore_errors=True)
    os.makedirs(os.path.join(runtime, "indirect_access"))
    os.makedirs(os.path.join(runtime, "dma_access"), exist_ok=True)

    argv_paths = [os.path.basename(raw_paths[n]) for n in argv_order(manifest)]
    lo, hi = manifest["abi"]["kernel_addr"]
    try:
        run_under = ["--isa", manifest["run"]["spike_isa"],
                     f"--extlib={compiler_bridge.model(manifest)}",
                     "-m" + contract.spike_memory(manifest),
                     f"--machine-config={contract.machine_config(obj_dir, manifest)}"]
    except (compiler_bridge.NoTarget, contract.KernelObjectMismatch) as e:
        raise SpikeError(str(e)) from None
    cmd = [
        SPIKE,
        *run_under,
        f"--kernel-addr={lo:x}:{hi:x}",
        f"--base-path={runtime}",
        PK, elf, *argv_paths, *scalars,
    ]

    total = 1
    for g in grid3(manifest):
        total *= g
    n = _jobs(manifest, total, log) if jobs is None else max(1, min(jobs, total))
    if n > 1:
        _split(manifest, cmd, runtime, raw_paths, total, n, log)
    else:
        _run(cmd, runtime, log)
    return runtime


def _jobs(manifest, total, log):
    """How many processes to spread the launch grid over."""
    if not splittable(manifest):
        log("[Spike] one process: an inout argument is read as well as written")
        return 1
    env = os.environ.get("TORCHSIM_SPIKE_JOBS",
                         os.environ.get("TORCHSIM_COMPILE_SPIKE_JOBS", "")).strip()
    if env:
        return max(1, min(int(env), total))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(8, cpus // 2, total))


def _split(manifest, cmd, runtime, raw_paths, total, jobs, log):
    """Run the grid as `jobs` processes over disjoint program ranges."""
    edges = [(total * w) // jobs for w in range(jobs + 1)]
    ranges = [(edges[w], edges[w + 1]) for w in range(jobs)
              if edges[w] < edges[w + 1]]
    log(f"[Spike] {total} programs over {len(ranges)} processes")

    workers, cmds = [], []
    for w, (lo, hi) in enumerate(ranges):
        d = os.path.join(runtime, f"part{w}")
        _prepare(manifest, raw_paths, d)
        workers.append(d)
        argv = [_worker_path(manifest, raw_paths, d, runtime, c) for c in cmd]
        argv[argv.index(f"--base-path={runtime}")] = f"--base-path={d}"
        cmds.append(argv + [str(lo), str(hi)])

    with cf.ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        for f in cf.as_completed([pool.submit(_run, c, runtime, log)
                                  for c in cmds]):
            f.result()

    try:
        _merge_outputs(manifest, raw_paths, workers)
    except _NotSplittable as why:
        log(f"[Spike] {why} -- running the grid in one process instead")
        for d in workers:
            shutil.rmtree(d, ignore_errors=True)
        _run(cmd, runtime, log)
        return

    _merge_traces(runtime, workers)
    for d in workers:
        shutil.rmtree(d, ignore_errors=True)


def _worker_path(manifest, raw_paths, worker, runtime, item):
    """Redirect a written .raw to the worker's copy; leave everything else."""
    for a in writable(manifest):
        if item == os.path.basename(raw_paths[a["name"]]):
            return os.path.join(os.path.relpath(worker, runtime), item)
    return item


def _prepare(manifest, raw_paths, worker):
    """Give a worker its own dump dirs and its own copy of every output."""
    shutil.rmtree(worker, ignore_errors=True)
    os.makedirs(os.path.join(worker, "indirect_access"))
    os.makedirs(os.path.join(worker, "dma_access"))
    for a in writable(manifest):
        src = raw_paths[a["name"]]
        shutil.copyfile(src, os.path.join(worker, os.path.basename(src)))


def _merge_outputs(manifest, raw_paths, workers):
    """Fold each worker's copy of a written buffer back into the real one.

    Raises _NotSplittable when two ranges disagree about a byte, since which of
    them ran last would then decide the answer.
    """
    import numpy as np

    for a in writable(manifest):
        dest = raw_paths[a["name"]]
        n = nbytes(a)
        base = np.zeros(n, dtype=np.uint8)
        merged, written = base.copy(), np.zeros(n, dtype=bool)
        for d in workers:
            got = np.fromfile(os.path.join(d, os.path.basename(dest)),
                              dtype=np.uint8)
            if got.size != n:
                raise SpikeError(f"{d} wrote {got.size} bytes for "
                                 f"{a['name']}, expected {n}")
            changed = got != base
            clash = changed & written & (got != merged)
            if clash.any():
                raise _NotSplittable(
                    f"two program ranges wrote {a['name']} byte "
                    f"{int(np.flatnonzero(clash)[0])} differently")
            merged[changed] = got[changed]
            written |= changed
        merged.tofile(dest)


def _merge_traces(runtime, workers):
    """Gather the workers' index dumps: keyed ones (indirect_index_<key>_<n>) by name,
    since the key names the program; unkeyed ones renumbered in program order."""
    dst = os.path.join(runtime, "indirect_access")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)

    n = 0
    for d in workers:
        src = os.path.join(d, "indirect_access")
        names = os.listdir(src)
        unkeyed = sorted((f for f in names if re.fullmatch(r"indirect_index\d+\.raw", f)),
                         key=lambda f: int(re.findall(r"\d+", f)[-1]))
        for name in names:
            if name not in unkeyed:
                os.replace(os.path.join(src, name), os.path.join(dst, name))
        for name in unkeyed:
            os.replace(os.path.join(src, name),
                       os.path.join(dst, f"indirect_index{n}.raw"))
            n += 1

    wrote = [d for d in workers if os.listdir(os.path.join(d, "dma_access"))]
    if len(wrote) > 1:
        raise SpikeError("more than one program range wrote dma_access/; it "
                         "has no per-range naming. Set TORCHSIM_SPIKE_JOBS=1.")
    for d in wrote:
        for name in os.listdir(os.path.join(d, "dma_access")):
            os.replace(os.path.join(d, "dma_access", name),
                       os.path.join(runtime, "dma_access", name))
