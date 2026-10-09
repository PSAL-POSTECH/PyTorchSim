"""Run pytorchsim-triton-compiler in its own process and read the kernel object it leaves.

The seam is kernel.json in the kernel's workdir; nothing here imports the compiler.
"""

import json
import os
import re
import subprocess

from PyTorchSimFrontend import config

logger = config.setup_logger()


class CompilerError(RuntimeError):
    """A compiler stage failed. Inductor reports only str(exc), so the stage's own
    diagnostic has to travel in the message."""
    _SIGNAL = re.compile(
        r"^(?!\s|Traceback|During handling|The above)"
        r"(.*\berror:\s.*|.*failed to legalize.*|"
        r"[\w.]*(?:Error|Exception)\b.*|.*Assertion.*|"
        r"[A-Za-z_]\w*(?:\.\w+)+: .+|\[(?:transform|lower|binary)\] .*(?:refused|not allowed).*)$", re.M)
    _FRAME = re.compile(r'^\s|^\s*File "|^\s*\^')
    _STAGE_FAIL = re.compile(r"^\[\d+/\d+\] \S+\s+FAIL\s.*\n\n(  \S.*)$", re.M)

    def __init__(self, message, cmd=None, output=None):
        self.cmd = cmd
        self.output = output
        if output:
            hits = [h.strip() for h in self._STAGE_FAIL.findall(output)]
            if not hits:
                hits = [h.strip() for h in self._SIGNAL.findall(output)
                        if not self._FRAME.match(h)]
            if not hits:
                hits = [l for l in output.strip().splitlines()
                        if l.strip() and not self._FRAME.match(l)]
            message = message + "\n  " + "\n  ".join(l[:300] for l in hits[-3:])
        super().__init__(message)


#: The compiler's package. One name since pytorchsim-triton-compiler fe0ee08.
COMPILER_PKG = "pytorchsim_triton_compiler"


def compiler_dir():
    d = config.CONFIG_TORCHSIM_COMPILE_DIR
    if not os.path.isdir(d):
        raise CompilerError(
            f"pytorchsim-triton-compiler checkout not found at {d}. It is a separate repository "
            f"and is not vendored; clone it there or set TORCHSIM_COMPILE_DIR.")
    return d


def machine():
    """The machine the kernel is compiled for, from the TOGSim YAML.

    The YAML is the hardware description and therefore the authority.
    """
    return dict(config.CONFIG_MACHINE)


def target_path():
    """Write this machine as a target description and return its path.

    Named for the config it came from, so two configs in one dump path do not
    overwrite each other. Written whole and renamed into place, since several
    processes may reach this at once.
    """
    import json

    m = machine()
    name = os.path.splitext(os.path.basename(
        os.environ.get("TOGSIM_CONFIG", "togsim")))[0]
    out = os.path.join(config.get_dump_path(), f"target-{name}.json")
    doc = dict(m, name=name,
               provenance={"togsim_config": os.environ.get("TOGSIM_CONFIG", ""),
                           "address_map": "PyTorchSimFrontend/config.py"})
    body = json.dumps(doc, indent=2) + "\n"
    if not os.path.isfile(out) or open(out).read() != body:
        tmp = f"{out}.{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(body)
        os.replace(tmp, out)
    return out


def compiler_env():
    """The environment for a compiler subprocess: this machine, no PYTHONPATH, and
    no device backend autoload.

    TORCHSIM_COMPILE_TARGET NAMES THE WHOLE MACHINE, and used to name three of its seven
    fields while the compiler supplied the rest from a shipped default. One
    description, written from the YAML this process is running.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env.setdefault("TORCHSIM_COMPILE_TARGET", target_path())
    return env


def doctor():
    """Return (ok, output) for the compiler's own toolchain check."""
    proc = subprocess.run(
        [config.CONFIG_TORCHSIM_COMPILE_PYTHON,
         os.path.join(compiler_dir(), "pytorchsim-triton-compiler"), "doctor"],
        capture_output=True, text=True, cwd=compiler_dir())
    return proc.returncode == 0, proc.stdout + proc.stderr



def run_pipeline(spec_path, workdir, to_stage="torchsim-compile", tog=False, timeout=1800):
    """Drive the compiler's stages over `spec_path`, writing artifacts into `workdir`.

    Stops at `to_stage`, by default `torchsim-compile` -- the ELF: spike and verify
    want tensors and a per-kernel reference this route has no graph-level answer for.
    """
    cmd = [config.CONFIG_TORCHSIM_COMPILE_PYTHON,
           os.path.join(compiler_dir(), "pytorchsim-triton-compiler"), spec_path,
           "--from", "triton-compile", "--to", to_stage, "--workdir", workdir] + (["--tog"] if tog else [])

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=compiler_dir(), env=compiler_env(), timeout=timeout)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        log = os.path.join(workdir, "stage.log")
        if os.path.isfile(log):
            with open(log, errors="replace") as fh:
                output += "\n" + fh.read()
        raise CompilerError(f"the compiler pipeline failed (exit {proc.returncode})",
                        cmd=" ".join(cmd), output=output)
    logger.debug("[torchsim-compile] %s", output)
    return workdir


#: The compiler's manifest. Its schema is declared in the compiler
#: (pytorchsim_triton_compiler/contract/kernel_object.py); this is a reader, and the format field is what stops
#: the two from drifting silently.
KERNEL_MANIFEST = "kernel.json"
KERNEL_FORMAT = 1


def kernel_object(workdir):
    """The manifest the compiler left in `workdir`, or None if it did not get that far."""
    path = os.path.join(workdir, KERNEL_MANIFEST)
    try:
        with open(path) as fh:
            got = json.load(fh)
    except (OSError, ValueError):
        return None
    if got.get("format") != KERNEL_FORMAT:
        raise CompilerError(
            f"{path}: kernel object format {got.get('format')!r}, this reader "
            f"knows {KERNEL_FORMAT}. The compiler and this frontend disagree "
            f"about the contract; rebuild the kernel cache.")
    return got


def artifact(workdir, kind):
    """One compiler output by the name the manifest gives it, or None.

    THE STAGE NUMBERS ARE NOT AN INTERFACE and this is what replaced globbing
    for them: the compiler renumbers when a stage is added, and the post-vcix IR has
    already moved from 04- to 05- once.
    """
    got = kernel_object(workdir)
    if got is None:
        return None
    rel = got.get("artifacts", {}).get(kind)
    if not rel:
        return None
    full = os.path.join(workdir, rel)
    return full if os.path.exists(full) else None
