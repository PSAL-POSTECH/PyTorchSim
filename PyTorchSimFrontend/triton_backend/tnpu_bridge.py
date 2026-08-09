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

    env = dict(os.environ)
    # tnpu deliberately does not read TORCHSIM_LLVM_PATH (it would drag the
    # backend back to LLVM 20 and break the textual seam), but a stale
    # PYTHONPATH pointing at LLVM 20's mlir_core would still be picked up by the
    # namespace package before tnpu's own activate_bindings() runs.
    env.pop("PYTHONPATH", None)

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
