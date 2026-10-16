"""Run pytorchsim-triton-compiler in its own process, for the machine TORCHSIM_TARGET names. The
machine file and the model library are read with the compiler's own contract modules, imported from
that same checkout; what a compile leaves is read through contract.py."""

import importlib
import os
import re
import subprocess

from pytorchsim import config

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
    _STDERR_BLOCK = re.compile(r"^  stderr  :\n((?:    .*\n?)+)", re.M)

    def __init__(self, message, cmd=None, output=None):
        self.cmd = cmd
        self.output = output
        if output:
            hits = [h.strip() for h in self._STAGE_FAIL.findall(output)]
            if hits:
                said = "\n".join(l.strip() for b in self._STDERR_BLOCK.findall(output)
                                 for l in b.splitlines())
                hits += [h.strip() for h in self._SIGNAL.findall(said)
                         if not self._FRAME.match(h) and h.strip() not in hits[-1]][-2:]
            if not hits:
                hits = [h.strip() for h in self._SIGNAL.findall(output)
                        if not self._FRAME.match(h)]
            if not hits:
                hits = [l for l in output.strip().splitlines()
                        if l.strip() and not self._FRAME.match(l)]
            message = message + "\n  " + "\n  ".join(l[:300] for l in hits[-3:])
        super().__init__(message)


#: The compiler's package.
COMPILER_PKG = "pytorchsim_triton_compiler"


def compiler_dir():
    d = config.CONFIG_TORCHSIM_COMPILE_DIR
    if not os.path.isdir(d):
        raise CompilerError(
            f"pytorchsim-triton-compiler checkout not found at {d}. It is a separate repository "
            f"and is not vendored; clone it there or set TORCHSIM_COMPILE_DIR.")
    return d


class NoTarget(RuntimeError):
    """TORCHSIM_TARGET names no machine file the compiler takes."""


def compiler_contract(name):
    """The compiler's contract/<name> module, from the checkout TORCHSIM_COMPILE_DIR names."""
    try:
        mod = importlib.import_module(f"{COMPILER_PKG}.contract.{name}")
    except ImportError as e:
        raise CompilerError(f"{COMPILER_PKG} is not importable here ({e}): install the checkout "
                            f"at {compiler_dir()} (pip install -e)") from None
    here = os.path.realpath(compiler_dir()) + os.sep
    if not os.path.realpath(mod.__file__).startswith(here):
        raise CompilerError(f"{COMPILER_PKG} imports from {mod.__file__}, and the compiler this "
                            f"process runs is the checkout at {compiler_dir()}")
    return mod


_TARGETS = {}


def _bare_env():
    """The environment a compiler subprocess starts from: no PYTHONPATH, no device backend autoload."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return env


def target():
    """The machine TORCHSIM_TARGET names, read by the compiler's target module: a bare name is
    $TOGSIM_DIR/configs/<name>.yml, a relative path is the caller's. Once per (target, cwd)."""
    if os.environ.get("TOGSIM_CONFIG"):
        raise NoTarget(f"TOGSIM_CONFIG={os.environ['TOGSIM_CONFIG']} is retired: the machine is "
                       f"one file, which TORCHSIM_TARGET names and TOGSim times on. Unset it")
    said, here = config.CONFIG_TORCHSIM_TARGET, os.getcwd()
    if (said, here) not in _TARGETS:
        t = compiler_contract("target")
        try:
            _TARGETS[(said, here)] = t.load(said)
        except t.TargetError as e:
            raise NoTarget(f"TORCHSIM_TARGET={said}: {e}") from None
    return _TARGETS[(said, here)]


def run_options(obj_dir, manifest):
    """Spike's --isa and -m, the model library, its --machine-config and the gem5 script for the
    kernel object in `obj_dir`: the compiler's kernel_object.run_options() of the machine it was
    compiled for. TORCHSIM_GEM5_SCRIPT, when set, is taken for the script."""
    ko, v = compiler_contract("kernel_object"), compiler_contract("vcix_model")
    machine = compiled_for(manifest)
    try:
        return ko.run_options(machine, obj_dir, config.CONFIG_VCIX_BUILD, manifest,
                              gem5_script=os.environ.get("TORCHSIM_GEM5_SCRIPT"))
    except v.ModelError as e:
        raise NoTarget(f"{e}: {manifest['name']} was compiled for {machine.name}") from None
    except ko.ObjectError as e:
        raise NoTarget(str(e)) from None


def compiled_for(manifest):
    """The machine a kernel object was compiled for: the file its manifest names, refused when that
    file now states otherwise than the manifest recorded (the compiler's target.as_compiled)."""
    t = compiler_contract("target")
    try:
        return t.as_compiled(manifest.get("target"), f"{manifest['name']}: manifest target")
    except t.TargetError as e:
        raise NoTarget(str(e)) from None


def compiler_env():
    """The environment for a compiler subprocess: this machine and this TOGSim. TOGSIM_DIR is the
    checkout whose header this process digests and whose Simulator it starts."""
    env = _bare_env()
    env["TORCHSIM_COMPILE_TARGET"] = target().file
    env["TOGSIM_DIR"] = config.CONFIG_TOGSIM_DIR
    return env


def doctor():
    """Return (ok, output) for the compiler's own toolchain check."""
    proc = subprocess.run(
        [config.CONFIG_TORCHSIM_COMPILE_PYTHON,
         os.path.join(compiler_dir(), "pytorchsim-triton-compiler"), "doctor"],
        capture_output=True, text=True, cwd=compiler_dir())
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_pipeline(spec_path, workdir, to_stage="torchsim-compile", tog=False, timeout=1800,
                 env=None):
    """Drive the compiler's stages over `spec_path`, writing artifacts into `workdir`, with `env`
    (the options a build asks the compiler, autotune.compile_env) over compiler_env().

    Stops at `to_stage`, by default `torchsim-compile` -- the ELF: spike and verify
    want tensors and a per-kernel reference this route has no graph-level answer for.
    """
    cmd = [config.CONFIG_TORCHSIM_COMPILE_PYTHON,
           os.path.join(compiler_dir(), "pytorchsim-triton-compiler"), spec_path,
           "--from", "triton-compile", "--to", to_stage, "--workdir", workdir] + (["--tog"] if tog else [])

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=compiler_dir(), env=dict(compiler_env(), **(env or {})),
                          timeout=timeout)
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
