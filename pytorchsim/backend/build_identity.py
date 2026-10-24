"""What a cached kernel was built with, so a workdir another toolchain built is not reused: the
compiler's commit, its tools, the machine and the TOGSim ABI, stamped beside the artifacts."""

import glob
import hashlib
import json
import os
import shutil
import subprocess

from pytorchsim import config, contract
from pytorchsim.backend import compiler_bridge

logger = config.setup_logger()

STAMP_NAME = "provenance.json"


_STALE_PATTERNS = ("[0-9][0-9]-*", "tog.so", "tog_cycles.tsv", "tog_units.tsv", "timing.json")


_identity = None


def _run_tool(cmd, cwd):
    """(rc, stdout) of a command, with stderr folded in on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return proc.returncode, proc.stdout if proc.returncode == 0 else proc.stderr


def _compiler_git(compiler_dir):
    """The compiler checkout's commit, with a digest of any uncommitted change."""
    rc, head = _run_tool(["git", "rev-parse", "HEAD"], compiler_dir)
    if rc != 0:
        return None
    rc, diff = _run_tool(["git", "diff", "HEAD"], compiler_dir)
    dirty = hashlib.sha256(diff.encode()).hexdigest()[:12] if diff.strip() else ""
    return {"commit": head.strip(), "dirty": dirty}


_PROBE = ("import json; from {pkg} import toolchain as t; "
          "print(json.dumps({{'tools': dict(t.TOOLS), 'togsim_api': t.togsim_version(), "
          "'togsim_include': t.TOGSIM_INCLUDE}}))")


def _compiler_facts(compiler_dir):
    """The tools, TOGSim API version and TOGSim header directory the compiler itself resolves, asked
    in its own interpreter. A probe that cannot answer raises with what the compiler said."""
    proc = subprocess.run(
        [config.CONFIG_TORCHSIM_COMPILE_PYTHON, "-c", _PROBE.format(pkg=compiler_bridge.COMPILER_PKG)],
        capture_output=True, text=True, cwd=compiler_dir, env=compiler_bridge.compiler_env())
    try:
        facts = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else None
    except (ValueError, IndexError):
        facts = None
    if not isinstance(facts, dict) or not facts.get("tools") or not facts.get("togsim_include"):
        said = (proc.stderr or proc.stdout).strip()[-600:]
        raise RuntimeError(
            f"the compiler in {compiler_dir} did not state its toolchain (exit {proc.returncode}), so "
            f"no cached kernel can be trusted to be its build: {said}")
    return facts


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


def _tog_abi(facts):
    """The TOGSim API a cached tog.so was compiled against, as the compiler resolves it: its stated
    version and a digest of every header in its TOGSim include directory."""
    include = facts["togsim_include"]
    names = sorted(n for n in os.listdir(include) if n.endswith((".h", ".hpp"))) \
        if os.path.isdir(include) else []
    if "togsim.hpp" not in names:
        raise FileNotFoundError(
            f"{include} holds no togsim.hpp: the compiler's TOGSim (TOGSIM_DIR, else its "
            f"third_party/TOGSim) is not a TOGSim checkout")
    digest = hashlib.sha256()
    for name in names:
        with open(os.path.join(include, name), "rb") as fh:
            digest.update(name.encode() + b"\0" + fh.read())
    return {"api": facts["togsim_api"], "headers": digest.hexdigest()[:16]}


def identity():
    """This process's build identity, computed once and reused."""
    global _identity
    if _identity is None:
        compiler_dir = compiler_bridge.compiler_dir()
        facts = _compiler_facts(compiler_dir)
        _identity = {
            "compiler": _compiler_git(compiler_dir),
            "tools": _stat_tools(facts["tools"]),
            "machine": compiler_bridge.target().as_dict(),
            "tog_abi": _tog_abi(facts),
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
    """Remove the manifest, stage artifacts and timing derivatives built by another identity; one another
    process removed first is gone already, and one that cannot be removed is raised."""
    for pattern in (contract.KERNEL_MANIFEST,) + _STALE_PATTERNS:
        for path in glob.glob(os.path.join(workdir, pattern)):
            try:
                os.remove(path)
            except FileNotFoundError:
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
