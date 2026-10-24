import os
import sys
import yaml
import logging

CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
#: The vcix-accelerator environment: the RISC-V toolchain, pk, Spike and gem5.
CONFIG_VCIX_ENV_ROOT = os.environ.get("VCIX_ENV_ROOT", "/opt/vcix-env")
CONFIG_GEM5_PATH = os.environ.get('GEM5_PATH', default=os.path.join(
    CONFIG_VCIX_ENV_ROOT, "gem5", "build", "RISCV", "gem5.opt"))
CONFIG_SPIKE_PATH = os.environ.get("TORCHSIM_SPIKE", os.environ.get(
    "TORCHSIM_COMPILE_SPIKE", os.path.join(CONFIG_VCIX_ENV_ROOT, "spike", "bin", "spike")))
CONFIG_PK_PATH = os.environ.get("TORCHSIM_PK", os.environ.get(
    "TORCHSIM_COMPILE_PK", os.path.join(CONFIG_VCIX_ENV_ROOT, "pk", "riscv64-unknown-elf", "bin", "pk")))
#: How many Spike processes a launch's grid is split across; empty lets the launcher choose.
CONFIG_SPIKE_JOBS = os.environ.get("TORCHSIM_SPIKE_JOBS", os.environ.get("TORCHSIM_COMPILE_SPIKE_JOBS", "")).strip()
#: The compiler checkout (stages 1-5, ttir to the RISC-V ELF), a separate repository, not vendored.
CONFIG_TORCHSIM_COMPILE_DIR = os.environ.get(
    "TORCHSIM_COMPILE_DIR", default=os.path.join(CONFIG_TORCHSIM_DIR, "pytorchsim-triton-compiler"))
#: Where the accelerator models are built; a kernel object's accelerator names the one it runs on.
CONFIG_VCIX_BUILD = os.path.join(os.environ.get("TORCHSIM_PREFIX", "/workspace"), "vcix-build")
#: The interpreter the compiler's own process runs under.
CONFIG_TORCHSIM_COMPILE_PYTHON = os.environ.get("TORCHSIM_COMPILE_PYTHON", default=sys.executable)


def _togsim_dir():
    """The TOGSim repository root: TOGSIM_DIR, else the checkout the compiler carries as third_party/TOGSim."""
    return os.environ.get("TOGSIM_DIR") or os.path.join(
        CONFIG_TORCHSIM_COMPILE_DIR, "third_party", "TOGSim")


CONFIG_TOGSIM_DIR = _togsim_dir()
CONFIG_TOGSIM_CONFIG_DIR = os.path.join(CONFIG_TOGSIM_DIR, "configs")


def get_dump_path():
    """Resolve TORCHSIM_DUMP_PATH and re-point Inductor's cache dir at it.

    Read on every call, so a TORCHSIM_DUMP_PATH changed mid-session moves both together.
    """
    dump_path = os.environ.get(
        "TORCHSIM_DUMP_PATH",
        default=os.path.join(CONFIG_TORCHSIM_DIR, "outputs"),
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(dump_path, ".torchinductor")
    return dump_path


#: PyTorchSim's run settings, not the machine's: name -> (environment variable, default, type).
RUN_SETTINGS = {
    "pytorchsim_functional_mode": ("TORCHSIM_FUNCTIONAL_MODE", 1, int),
    "pytorchsim_timing_mode": ("TORCHSIM_TIMING_MODE", 0, int),
    "pytorchsim_functional_verify_per_kernel": ("TORCHSIM_FUNCTIONAL_VERIFY_PER_KERNEL", 0, int),
    "codegen_autotune_template_topk": ("TORCHSIM_CODEGEN_AUTOTUNE_TEMPLATE_TOPK", 4, int),
}


def _run_setting(name):
    env, default, kind = RUN_SETTINGS[name]
    got = os.environ.get(env)
    return kind(default if got is None or got == "" else got)


def __getattr__(name):
    """The settings read on each access: TORCHSIM_TARGET names a TOGSim machine file
    ($TOGSIM_DIR/configs/<name>.yml) or a path to one, which the compiler compiles for and TOGSim times on."""
    if name == "CONFIG_TORCHSIM_TARGET":
        return os.environ.get("TORCHSIM_TARGET") or "tpuv6e"
    if name == "CONFIG_MACHINE":
        from pytorchsim.backend import compiler_bridge
        return compiler_bridge.target()
    if name == "CONFIG_MACHINE_FILE":
        from pytorchsim.backend import compiler_bridge
        return compiler_bridge.target().file
    if name == "CONFIG_NUM_CORES":
        from pytorchsim.backend import compiler_bridge
        with open(compiler_bridge.target().file) as f:
            return yaml.safe_load(f)["num_cores"]

    if name == "pytorchsim_functional_verify_per_kernel":
        return bool(_run_setting(name)) and bool(_run_setting("pytorchsim_functional_mode"))
    if name in RUN_SETTINGS:
        return _run_setting(name)

    if name == "CONFIG_TOGSIM_DEBUG_LEVEL":
        return os.environ.get("TOGSIM_DEBUG_LEVEL", "")
    if name == "CONFIG_TORCHSIM_LOG_PATH":
        return os.environ.get('TORCHSIM_LOG_PATH', default = os.path.join(CONFIG_TORCHSIM_DIR, "togsim_results"))

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

CONFIG_DEBUG_MODE = int(os.environ.get('TORCHSIM_DEBUG_MODE', default=0))


def togsim_simulator(togsim_dir=None):
    """The Simulator a TOGSim checkout builds in its own tree; CONFIG_TOGSIM_DIR's unless another is named."""
    return os.path.join(togsim_dir or CONFIG_TOGSIM_DIR, "build", "bin", "Simulator")


def setup_logger(name=None, level=None):
    """The logger `name` (the calling module's by default), formatted alike across modules, at `level`
    (DEBUG in debug mode, else INFO)."""
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'PyTorchSim')

    name = name.lower()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if level is None:
            level = logging.DEBUG if CONFIG_DEBUG_MODE else logging.INFO
        logger.setLevel(level)

    return logger