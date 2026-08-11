"""Run the triton-npu pipeline, out of process.

WHY A SUBPROCESS
----------------
tnpu's passes run on LLVM 23's MLIR python bindings; this process holds LLVM 20's
(TORCHSIM_LLVM_PATH). `mlir` ships without an `__init__.py`, so it is a NAMESPACE
package whose `__path__` is the union of every `mlir/` directory on sys.path --
two LLVMs in one interpreter silently merge and fail later with an AttributeError
from a generated dialect module (tnpu/config.py:activate_bindings documents the
exact failure). They cannot share an interpreter, so tnpu gets its own.

The seam between them is a FILE, which is measured to work: LLVM 23 prints the
IR, and LLVM 20's bindings parse it back without complaint (verified by feeding
tnpu's 04-custom.mlir to PyTorchSim's build_tog). That is what makes the split
viable rather than merely necessary.
"""

import os
import re
import subprocess

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()


class TnpuError(RuntimeError):
    """A tnpu stage failed. Inductor reports only str(exc), so the stage's own
    diagnostic has to travel in the message."""

    #: How a failing stage names itself: MLIR diagnostics and exception lines.
    _SIGNAL = re.compile(
        r"^(?!\s|Traceback|During handling|The above)"
        r"(.*\berror:\s.*|.*failed to legalize.*|"
        r"[\w.]*(?:Error|Exception)\b.*|.*Assertion.*)$", re.M)
    #: Frames and carets: context, not the diagnostic.
    _FRAME = re.compile(r'^\s|^\s*File "|^\s*\^')

    def __init__(self, message, cmd=None, output=None):
        self.cmd = cmd
        self.output = output
        if output:
            hits = [h.strip() for h in self._SIGNAL.findall(output)
                    if not self._FRAME.match(h)]
            if not hits:
                hits = [l for l in output.strip().splitlines()
                        if l.strip() and not self._FRAME.match(l)]
            message = message + "\n  " + "\n  ".join(l[:300] for l in hits[-3:])
        super().__init__(message)


def tnpu_dir():
    d = extension_config.CONFIG_TNPU_DIR
    if not os.path.isdir(d):
        raise TnpuError(
            f"triton-npu checkout not found at {d}. It is a separate repository "
            f"and is not vendored; clone it there or set TNPU_DIR.")
    return d


def machine():
    """The machine the kernel is being compiled for, FROM THE TOGSIM YAML.

    THE YAML IS THE HARDWARE DESCRIPTION AND THEREFORE THE AUTHORITY. It is what
    TOGSim simulates and what a user edits to study a different machine; tnpu's
    config.py holds DEFAULTS for the same quantities so it can run standalone.
    Two copies of a number with no direction between them drift, and these
    already have: the YAML says 128 KB of scratchpad per lane while tnpu's
    default is 64 KB, so a reduction block sized against one and linked against
    the other died at LINK time with an error naming the scratchpad rather than
    the block size.

    Reading the YAML here is only half of that. `tnpu_env` below hands the same
    numbers to every tnpu subprocess through the environment variables its
    config.py already reads, so changing the YAML changes what the compiler
    targets instead of only changing what this side believes. That is the half
    that makes them one number rather than two that happen to agree.
    """
    spad = extension_config.CONFIG_SPAD_INFO["spad_size"]
    return {"lanes": int(extension_config.vpu_num_lanes),
            "vlen_bits": int(extension_config.vpu_vector_length_bits),
            "spad_size": int(spad)}


def tnpu_env():
    """The environment for a tnpu subprocess: this machine, and no PYTHONPATH.

    PYTHONPATH goes because `mlir` is a namespace package -- a stale entry
    pointing at LLVM 20's mlir_core is picked up before tnpu's own
    activate_bindings() runs, and the two LLVMs merge silently.

    The three TNPU_* names are the ones tnpu/config.py reads for exactly these
    quantities, so this is telling it rather than overriding it. Anything the
    caller already set in the environment wins, which keeps the pinned-worktree
    workflow (TNPU_SPAD_SIZE=... for a one-off experiment) working.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    m = machine()
    env.setdefault("TNPU_VECTORLANE_SIZE", str(m["lanes"]))
    env.setdefault("TNPU_VLEN_BITS", str(m["vlen_bits"]))
    env.setdefault("TNPU_SPAD_SIZE", str(m["spad_size"]))
    return env


def doctor():
    """Return (ok, output) for tnpu's own toolchain check."""
    proc = subprocess.run(
        [extension_config.CONFIG_TNPU_PYTHON, os.path.join(tnpu_dir(), "run.py"), "doctor"],
        capture_output=True, text=True, cwd=tnpu_dir())
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_pipeline(spec_path, workdir, to_stage="binary", from_stage="ttir",
                 verbose=False, timeout=1800):
    """Drive tnpu's stages over `spec_path`, writing artifacts into `workdir`.

    Stops at `to_stage`. The default is `binary` (through the RISC-V ELF):
    stage 6 (spike) needs the caller's real tensors as .raw files and stage 7
    compares against a per-kernel torch reference, neither of which exists on
    the Inductor route -- correctness is a graph-level property there.

    Returns the workdir on success; raises TnpuError with tnpu's own stage
    report (which names the failing command and its stderr) otherwise.
    """
    cmd = [extension_config.CONFIG_TNPU_PYTHON,
           os.path.join(tnpu_dir(), "run.py"), spec_path,
           "--from", from_stage, "--to", to_stage, "--workdir", workdir]
    if verbose:
        cmd.append("-v")

    # tnpu deliberately does not read TORCHSIM_LLVM_PATH (it would drag the
    # backend back to LLVM 20 and break the textual seam). tnpu_env drops the
    # stale PYTHONPATH and hands over the machine this repo's YAML describes.
    env = tnpu_env()

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=tnpu_dir(), env=env, timeout=timeout)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        # run.py prints a stage table; the diagnostic itself only reaches
        # stage.log.
        log = os.path.join(workdir, "stage.log")
        if os.path.isfile(log):
            with open(log, errors="replace") as fh:
                output += "\n" + fh.read()
        raise TnpuError(f"tnpu pipeline failed (exit {proc.returncode})",
                        cmd=" ".join(cmd), output=output)
    logger.debug("[triton-npu] %s", output)
    return workdir

def stage_artifact(workdir, suffix):
    """The stage file ending in `suffix`, whatever number tnpu gave it.

    THE NUMBERS ARE NOT AN INTERFACE. tnpu renumbers its stages whenever one is
    added or split -- the post-vcix IR has been 04-custom.mlir and is now
    05-custom.mlir -- and every hardcoded number here turns that into a
    FileNotFoundError raised from the launcher, long after the pipeline
    succeeded. The suffix is the stable half of the name, so match on it and let
    the number be tnpu's business.

    Returns None when nothing matches, so callers can say what they wanted.
    """
    import glob
    hits = sorted(glob.glob(os.path.join(workdir, f"*-{suffix}")))
    return hits[-1] if hits else None
