"""Run pytorchsim-triton-compiler in its own process, for the machine the TOGSim YAML describes.

Nothing here imports the compiler; what it leaves is read through contract.py.
"""

import json
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


#: The compiler's package. One name since pytorchsim-triton-compiler fe0ee08.
COMPILER_PKG = "pytorchsim_triton_compiler"


def compiler_dir():
    d = config.CONFIG_TORCHSIM_COMPILE_DIR
    if not os.path.isdir(d):
        raise CompilerError(
            f"pytorchsim-triton-compiler checkout not found at {d}. It is a separate repository "
            f"and is not vendored; clone it there or set TORCHSIM_COMPILE_DIR.")
    return d


def _bare_env():
    """The environment for the compiler's interpreter: no PYTHONPATH, no device backend autoload."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return env


_written = set()


def _written_by_compiler(name, *argv):
    """`name` in the dump path, for this TOGSim config, written once a process by the compiler's
    interpreter running `argv` plus the path to write; aside, then renamed in, as processes race."""
    out = os.path.join(config.get_dump_path(), name)
    key = (config.CONFIG_TOGSIM_CONFIG, out)
    if key in _written and os.path.isfile(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}"
    cmd = [config.CONFIG_TORCHSIM_COMPILE_PYTHON, *argv, tmp]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=compiler_dir(), env=_bare_env())
    if proc.returncode != 0:
        raise CompilerError(f"writing {out} failed (exit {proc.returncode}):\n{proc.stderr.strip()}",
                            cmd=" ".join(cmd))
    os.replace(tmp, out)
    _written.add(key)
    return out


def _stem():
    return os.path.splitext(os.path.basename(config.CONFIG_TOGSIM_CONFIG))[0]


def target_path():
    """The target description PTB's emit_target.py writes from this TOGSim config, the compiler's
    TORCHSIM_COMPILE_TARGET; named for the config, so two in one dump path do not collide."""
    return _written_by_compiler(f"target-{_stem()}.json",
                                os.path.join(compiler_dir(), "tools", "emit_target.py"),
                                config.CONFIG_TOGSIM_CONFIG, "-o")


def machine():
    """The machine the kernel is compiled for: that target description less its name and provenance,
    which say where it came from, so two configs of one machine are one machine to the kernel cache."""
    with open(target_path()) as fh:
        return {k: v for k, v in json.load(fh).items() if k not in ("name", "provenance")}


def model_yaml():
    """The same machine in the spelling the accelerator model reads, for Spike's and gem5's
    --machine-config, as PTB's Target writes it."""
    return _written_by_compiler(
        f"model-{_stem()}.yml", "-c",
        f"import sys; from {COMPILER_PKG}.contract import target; "
        f"target.load(sys.argv[1]).write_model_yaml(sys.argv[2])", target_path())


def vcix_model():
    """The accelerator model Spike and gem5 load: TORCHSIM_COMPILE_VCIX_MODEL, else the one PTB's
    setup builds for this machine's matmul_unit."""
    return os.environ.get("TORCHSIM_COMPILE_VCIX_MODEL") or os.path.join(
        os.environ.get("TORCHSIM_PREFIX", "/workspace"), "vcix-build",
        {"systolic": "libtpu.so", "tpuv6_mxu": "libtpuv6.so"}[machine()["matmul_unit"]])


def compiler_env():
    """The environment for a compiler subprocess: this machine and this TOGSim. The target is written
    from the TOGSim YAML this process runs, and TOGSIM_DIR is the checkout whose header this process
    digests and whose Simulator it starts."""
    env = _bare_env()
    env["TORCHSIM_COMPILE_TARGET"] = target_path()
    env["TOGSIM_DIR"] = config.CONFIG_TOGSIM_DIR
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
