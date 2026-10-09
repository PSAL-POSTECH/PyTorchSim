"""Identity of the toolchain and machine that build a kernel's artifacts.

A cached ELF is reusable only when what would build it today is what built it;
`matches` answers that at cache admission and `store` records it after a build.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess

from PyTorchSimFrontend import config

from . import compiler_bridge

logger = config.setup_logger()

STAMP_NAME = "provenance.json"

_STALE_PATTERNS = ("[0-9][0-9]-*", "trace.so", "trace_cycles.tsv", "timing.json")

_current = None


def _run(cmd, cwd):
    """(rc, stdout) of a command, with stderr folded in on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return proc.returncode, proc.stdout if proc.returncode == 0 else proc.stderr


def _compiler_git(compiler_dir):
    """The the compiler checkout's commit, with a digest of any uncommitted change."""
    rc, head = _run(["git", "rev-parse", "HEAD"], compiler_dir)
    if rc != 0:
        return None
    rc, diff = _run(["git", "diff", "HEAD"], compiler_dir)
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


def _trace_abi():
    """A digest of the runtime header a cached trace.so was compiled against.

    TOGSim declares that it refuses a producer whose ABI does not match and does
    not check; making the header part of the artifact's identity gets the same
    protection as a cache miss rather than a load-time refusal.
    """
    from Simulator.trace_build import default_include_dir

    path = os.path.join(default_include_dir(), "togsim_runtime.h")
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return None


def current():
    """This process's build identity, computed once and reused."""
    global _current
    if _current is None:
        compiler_dir = compiler_bridge.compiler_dir()
        paths = _tool_paths(compiler_dir)
        if paths is None:
            logger.warning(
                "[provenance] the compiler's toolchain could not be introspected; "
                "cached kernels are guarded by the compiler commit only")
        _current = {
            "the compiler": _compiler_git(compiler_dir),
            "tools": _stat_tools(paths) if paths else None,
            "machine": compiler_bridge.machine(),
            "trace_abi": _trace_abi(),
        }
    return _current


def matches(workdir):
    """Whether `workdir`'s stamp names the identity that would build it today."""
    try:
        with open(os.path.join(workdir, STAMP_NAME)) as fh:
            return json.load(fh) == current()
    except (OSError, ValueError):
        return False


def store(workdir):
    """Write the current identity next to the artifacts it describes."""
    with open(os.path.join(workdir, STAMP_NAME), "w") as fh:
        json.dump(current(), fh, indent=1, sort_keys=True)


def clear_stale(workdir):
    """Remove stage artifacts and timing derivatives built by another identity."""
    for pattern in _STALE_PATTERNS:
        for path in glob.glob(os.path.join(workdir, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass
