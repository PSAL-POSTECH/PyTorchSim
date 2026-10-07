"""Run a compiled kernel object on Spike, over tensors already on disk.

Reads kernel.json for the kernel's address range, the grid, the argument roles
and the machine.
Never opens the compiler's source or the kernel's Python spec.
"""

import concurrent.futures as cf
import os
import re
import shutil
import subprocess

_VCIX_ENV = os.environ.get("VCIX_ENV_ROOT", "/opt/vcix-env")
SPIKE = os.environ.get("TORCHSIM_SPIKE", os.environ.get(
    "TORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike")))
PK = os.environ.get("TORCHSIM_PK", os.environ.get(
    "TORCHSIM_COMPILE_PK", os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")))
SPIKE_ISA = os.environ.get("TORCHSIM_SPIKE_ISA",
                           os.environ.get("TORCHSIM_COMPILE_SPIKE_ISA", "rv64gcv_zfh_xvcixaccel"))

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


def _write_machine_yaml(t, path):
    """The machine in the spelling the vcix accelerator model reads (--machine-config)."""
    kb, rem = divmod(t["spad_size"], 1024)
    if rem:
        raise SpikeError(f"scratchpad {t['spad_size']} B is not whole kilobytes")
    with open(path, "w") as f:
        f.write(f"vpu_num_lanes: {t['vector_lanes']}\n"
                f"vpu_spad_size_kb_per_lane: {kb}\n"
                f"vpu_spad_base_vaddr: 0x{t['spad_vaddr']:x}\n"
                f"vpu_vector_length_bits: {t['vlen_bits']}\n")
    return path


def _machine(manifest):
    """The target fields this launch needs, as ints."""
    t = dict(manifest["target"])
    for k in ("spad_vaddr", "spad_paddr", "dram_base", "dram_size",
              "spad_size", "vector_lanes", "vlen_bits"):
        v = t[k]
        t[k] = int(v, 0) if isinstance(v, str) else int(v)
    return t


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


def launch(obj_dir, manifest, runtime, raw_paths, log, jobs=None):
    """Run the whole launch grid, leaving each written argument in its .raw."""
    elf = artifact(obj_dir, manifest, "elf")
    os.makedirs(os.path.join(runtime, "indirect_access"), exist_ok=True)
    os.makedirs(os.path.join(runtime, "dma_access"), exist_ok=True)

    argv_paths = [os.path.basename(raw_paths[n]) for n in argv_order(manifest)]
    t = _machine(manifest)
    spad_bytes = t["spad_size"] * t["vector_lanes"]
    lo, hi = manifest["abi"]["kernel_addr"]
    from PyTorchSimFrontend import extension_config
    yml = _write_machine_yaml(t, os.path.join(runtime, "machine.yml"))
    cmd = [
        SPIKE,
        "--isa", SPIKE_ISA,
        f"--extlib={extension_config.CONFIG_VCIX_MODEL}",
        "-m" + f"0x{t['dram_base']:x}:0x{t['dram_size']:x},"
               f"0x{t['spad_paddr']:x}:0x{spad_bytes:x}",
        f"--machine-config={yml}",
        f"--kernel-addr={lo:x}:{hi:x}",
        f"--base-path={runtime}",
        PK, elf, *argv_paths,
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
    """Reassemble spike's per-DMA index dumps in program order."""
    dst = os.path.join(runtime, "indirect_access")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)

    n = 0
    for d in workers:
        src = os.path.join(d, "indirect_access")
        for name in sorted(os.listdir(src),
                           key=lambda f: int(re.findall(r"\d+", f)[-1])):
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
