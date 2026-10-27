"""Run pytorchsim-triton-compiler in its own process, for the machine TORCHSIM_TARGET names. The
machine file and the model library are read with the compiler's own contract modules, imported from
that same checkout; what a compile leaves is read through contract.py."""

import importlib
import os
import subprocess

from pytorchsim import config

logger = config.setup_logger()


class CompilerError(RuntimeError):
    """A compiler stage failed. Inductor reports only str(exc), so the stage's own
    diagnostic travels in the message; `failure` is what the compiler left in the
    workdir's failure.json (contract/failure), None when it stopped outside every stage."""

    def __init__(self, message, cmd=None, failure=None):
        self.cmd = cmd
        self.failure = failure
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
    return compiler_module(f"contract.{name}")


def toolchain():
    """The compiler's toolchain module: Spike, pk, gem5 and the Coprocessor models, where it put them."""
    return compiler_module("toolchain")


def compiler_module(name):
    """The compiler's module `name` (dotted, under its package), from the checkout
    TORCHSIM_COMPILE_DIR names."""
    try:
        mod = importlib.import_module(f"{COMPILER_PKG}.{name}")
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


def runtime_attribute_dir(workdir):
    """runtime/'s directory of the runtime attribute records a launch's functional run writes, when the
    kernel object declares any, else None; its name is the compiler's kernel_object contract's."""
    from pytorchsim import contract
    manifest = contract.kernel_object(workdir)
    if manifest is None or not manifest.get("runtime_attributes"):
        return None
    return os.path.join(contract.runtime_dir(workdir), compiler_contract("kernel_object").RUNTIME_ATTRIBUTE_DIR)


def run_options(obj_dir, manifest, keys=None):
    """Spike's --isa and -m, the model library, its --machine-config and the gem5 script (or only
    `keys` of them) for the kernel object in `obj_dir`: the compiler's kernel_object.run_options() of
    the machine it was compiled for."""
    ko, v = compiler_contract("kernel_object"), compiler_contract("vcix_model")
    machine = compiled_for(manifest)
    try:
        return ko.run_options(machine, obj_dir, toolchain().COPROCESSOR_BUILD, manifest, keys)
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
    env["PYTORCHSIM_COMPILE_TARGET"] = target().file
    env["TOGSIM_DIR"] = config.CONFIG_TOGSIM_DIR
    return env


def run_pipeline(spec_path, workdir, to_stage="pytorchsim-compile", tog=False, timeout=1800):
    """Drive the compiler's stages over `spec_path`, writing artifacts into `workdir`, up to
    `to_stage` (by default `pytorchsim-compile`, the ELF)."""
    cmd = [config.CONFIG_TORCHSIM_COMPILE_PYTHON,
           os.path.join(compiler_dir(), "pytorchsim-triton-compiler"), spec_path,
           "--from", "triton-compile", "--to", to_stage, "--workdir", workdir] + (["--tog"] if tog else [])

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=compiler_dir(), env=compiler_env(), timeout=timeout)
    if proc.returncode != 0:
        failure = compiler_contract("failure").read(workdir)
        said = (f"at {failure['stage']} ({failure['kind']}): {failure['message']}" if failure
                else f"outside every stage:\n{proc.stderr}")
        raise CompilerError(f"the compiler pipeline failed (exit {proc.returncode}) {said}",
                            cmd=" ".join(cmd), failure=failure)
    logger.debug("[torchsim-compile] %s", proc.stdout + proc.stderr)
    return workdir
