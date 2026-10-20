"""Compile cache for the codegen route: one kernel in, one launcher out.

    define_kernel   ->  torchsim_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

One directory per source hash, holding the compiler kernel file and every artifact.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess

from filelock import FileLock
from torch._inductor.codecache import get_hash

from pytorchsim import config

from pytorchsim import contract
from pytorchsim.driver import autotune, compiler_bridge, compiler_input
from pytorchsim.simulator import functional, timing, wallclock

logger = config.setup_logger()

LOCK_TIMEOUT = 600

_LAUNCH_OBSERVERS = {}
_EVERY_LAUNCH = []
_EVERY_EXTERN = []


def user_kernel_id(kernel):
    """A user-defined Triton kernel's name in its launch meta: its module and qualified name."""
    return f"{kernel.__module__}.{kernel.__qualname__}"


def observe_launches(kernel, callback):
    """Call `callback()` at every launch of the user-defined Triton kernel `kernel`."""
    _LAUNCH_OBSERVERS[user_kernel_id(kernel)] = callback


def observe_every_launch(callback):
    """Call `callback(launcher)` at every launch of any kernel."""
    _EVERY_LAUNCH.append(callback)


def observe_every_extern(callback):
    """Call `callback(call)` at every call Inductor leaves to aten (an extern kernel or a fallback):
    `call` is its op, e.g. "aten.sort.stable". It runs as aten does, not on the simulator."""
    _EVERY_EXTERN.append(callback)


def torchsim_extern(call):
    """The generated wrapper's note before an extern call; see observe_every_extern."""
    for callback in _EVERY_EXTERN:
        callback(call)


def _write_path(src_code, meta):
    """The kernel's workdir, keyed by everything its ELF is built from: the source, and the
    constexprs, grid and arguments the spec bakes in. A user-defined Triton kernel's source is
    the same for every launch; its constexprs and grid are not."""
    baked = (meta.get("signature"), meta.get("constants"), meta.get("fixed_config"),
             meta.get("template_grid"),
             [(a["name"], a["role"], a["dtype"], a["numel"]) for a in meta.get("args", ())])
    return os.path.join(config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip() + repr(baked))[1:12])


STAMP_NAME = "provenance.json"


_STALE_PATTERNS = ("[0-9][0-9]-*", "tog.so", "tog_cycles.tsv", "tog_units.tsv",
                   "trace.so", "trace_cycles.tsv", "trace_units.tsv", "timing.json")


_identity = None


def _run_tool(cmd, cwd):
    """(rc, stdout) of a command, with stderr folded in on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return proc.returncode, proc.stdout if proc.returncode == 0 else proc.stderr


def _compiler_git(compiler_dir):
    """The the compiler checkout's commit, with a digest of any uncommitted change."""
    rc, head = _run_tool(["git", "rev-parse", "HEAD"], compiler_dir)
    if rc != 0:
        return None
    rc, diff = _run_tool(["git", "diff", "HEAD"], compiler_dir)
    dirty = hashlib.sha256(diff.encode()).hexdigest()[:12] if diff.strip() else ""
    return {"commit": head.strip(), "dirty": dirty}


def _tool_paths(compiler_dir):
    """The tool paths the compiler itself would resolve, asked in its own interpreter."""
    code = (f"import json; from {compiler_bridge.COMPILER_PKG} import config as c; "
            "print(json.dumps({n: p for n, p, _ in c.CHECKS}))")
    proc = subprocess.run(
        [config.CONFIG_TORCHSIM_COMPILE_PYTHON, "-c", code],
        capture_output=True, text=True, cwd=compiler_dir, env=compiler_bridge.compiler_env())
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _stat_tools(paths):
    """Each tool as [path, size, mtime_ns], or None where it does not resolve."""
    out = {}
    for name, path in sorted(paths.items()):
        resolved = path if os.sep in path else shutil.which(path)
        try:
            st = os.stat(resolved)
            out[name] = [resolved, st.st_size, st.st_mtime_ns]
        except (OSError, TypeError):
            out[name] = None
    return out


def _tog_abi():
    """A digest of the runtime header a cached tog.so was compiled against. TOGSim refuses a TOG
    of another TOGSIM_ABI_VERSION when it loads it; with the header in the artifact's identity
    any change to it is a cache miss before that."""
    path = os.path.join(config.togsim_include_dir(), "togsim_runtime.h")
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError as exc:
        raise FileNotFoundError(
            f"{path} cannot be read ({exc.strerror}): TOGSIM_DIR must name a TOGSim checkout, "
            f"the one the compiler carries as third_party/TOGSim unless set") from exc


def identity():
    """This process's build identity, computed once and reused."""
    global _identity
    if _identity is None:
        compiler_dir = compiler_bridge.compiler_dir()
        paths = _tool_paths(compiler_dir)
        if paths is None:
            logger.warning(
                "[provenance] the compiler's toolchain could not be introspected; "
                "cached kernels are guarded by the compiler commit only")
        _identity = {
            "the compiler": _compiler_git(compiler_dir),
            "tools": _stat_tools(paths) if paths else None,
            "machine": compiler_bridge.target().as_dict(),
            "tog_abi": _tog_abi(),
        }
    return _identity


def identity_matches(workdir):
    """Whether `workdir`'s stamp names the identity that would build it today."""
    try:
        with open(os.path.join(workdir, STAMP_NAME)) as fh:
            return json.load(fh) == identity()
    except (OSError, ValueError):
        return False


def store_identity(workdir):
    """Write the current identity next to the artifacts it describes."""
    with open(os.path.join(workdir, STAMP_NAME), "w") as fh:
        json.dump(identity(), fh, indent=1, sort_keys=True)


def clear_stale(workdir):
    """Remove the manifest, stage artifacts and timing derivatives built by another identity. The
    trace* names are what a frontend from before TOGSim ABI 16 left in the workdir."""
    for pattern in (contract.KERNEL_MANIFEST,) + _STALE_PATTERNS:
        for path in glob.glob(os.path.join(workdir, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


def cached_elf(workdir, kernel_name):
    """The ELF an earlier compile left in `workdir` if this identity built it, else None with that
    build's leftovers removed. A kernel.json of another format is such a leftover unless this
    identity wrote it: then the compiler and this reader disagree, and that is raised."""
    foreign = False
    try:
        elf = contract.artifact(workdir, "elf")
    except contract.KernelObjectMismatch as exc:
        if identity_matches(workdir):
            raise contract.KernelObjectMismatch(
                f"{workdir}: the kernel object is not format {contract.KERNEL_FORMAT}, and its "
                f"{STAMP_NAME} says this toolchain wrote it, so rebuilding will write the same. "
                f"contract.KERNEL_FORMAT and the compiler pinned in "
                f"third_party/pytorchsim-triton-backend.json must be moved together.") from exc
        elf, foreign = None, True
    if foreign or (elf is not None and not identity_matches(workdir)):
        logger.info(
            "[torchsim-compile] %s: cached artifacts carry a different toolchain, machine "
            "or kernel-object identity, rebuilding", kernel_name)
        clear_stale(workdir)
        return None
    return elf


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper.

    Each call is one launch of the whole grid, Spike first so the tensors hold
    real values even if TOGSim fails. Both halves switch on the config keys.
    """
    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta

    def __call__(self, *args):
        observer = _LAUNCH_OBSERVERS.get(self.meta.get("user_kernel"))
        if observer is not None:
            observer()
        for callback in _EVERY_LAUNCH:
            callback(self)
        if config.pytorchsim_functional_mode:
            with wallclock.span(wallclock.SPIKE, self.kernel_name):
                written = functional.run(self.workdir, self.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        if not config.pytorchsim_timing_mode:
            logger.warning(
                "[timing] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        result = timing.run(self.workdir, self.meta, args)
        if isinstance(result, int):
            logger.info("[TOGSim] %s queued as kernel %d; the stream's cycles "
                        "are reported when the simulator closes",
                        self.kernel_name, result)
        else:
            logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


def torchsim_compile(src_code, meta, kernel_name):
    """Compile one Inductor-generated Triton kernel through the compiler.

    Called from the generated wrapper at module import time. Synchronous, on
    purpose: a thread pool buys nothing until the pipeline itself is proven.
    """
    write_path = _write_path(src_code, meta)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = cached_elf(write_path, kernel_name)
        if (elf is not None and config.pytorchsim_timing_mode
                and contract.artifact(write_path, "tog_so") is None):
            logger.info("[torchsim-compile] %s: cached without --tog, rebuilding for timing",
                        kernel_name)
            elf = None
        if elf is None:
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            timing.store_meta(write_path, meta)
            retries = autotune.smaller_tiles(src_code, meta)
            while True:
                compiler_input.write(src_code, meta, spec_path)
                try:
                    with wallclock.span(wallclock.TORCHSIM_COMPILE, kernel_name):
                        compiler_bridge.run_pipeline(
                            spec_path, write_path, to_stage="torchsim-compile",
                            tog=bool(config.pytorchsim_timing_mode))
                    wallclock.ingest_compile(write_path, kernel_name)
                    break
                except compiler_bridge.CompilerError as exc:
                    why = autotune.too_big(exc)
                    if why is None:
                        raise
                    retry = next(retries, None)
                    if retry is None:
                        logger.warning("[torchsim-compile] %s: %s, and no smaller tile is "
                                       "left to try", kernel_name, why)
                        raise
                    tile, src_code, tuned = retry
                    meta.update(tuned)
                    logger.info("[torchsim-compile] %s: %s, trying %s", kernel_name, why, tile)
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            timing.store_meta(write_path, meta)
            store_identity(write_path)
        logger.info("[torchsim-compile] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
