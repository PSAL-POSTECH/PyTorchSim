import os
import sys
import yaml
import logging

CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
CONFIG_GEM5_PATH = os.environ.get('GEM5_PATH', default=os.path.join(
    os.environ.get("VCIX_ENV_ROOT", "/opt/vcix-env"), "gem5", "build", "RISCV", "gem5.opt"))
# --- Triton codegen route ----------------------------------------------------
# The pytorchsim-triton-compiler checkout that owns stages 1-5 (ttir -> ttshared -> the compiler passes
# -> RISC-V ELF). It is a SEPARATE repository, deliberately not vendored.
CONFIG_TORCHSIM_COMPILE_DIR = os.environ.get(
    "TORCHSIM_COMPILE_DIR", default=os.path.join(CONFIG_TORCHSIM_DIR, "pytorchsim-triton-compiler"))
# Where the accelerator models are built (a kernel object's accelerator names which one it runs on).
CONFIG_VCIX_BUILD = os.path.join(os.environ.get("TORCHSIM_PREFIX", "/workspace"), "vcix-build")
# the compiler runs in its own process, under this interpreter unless overridden.
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


# PyTorchSim's run settings: (environment variable, default, type). They are not the machine's.
RUN_SETTINGS = {
    "pytorchsim_functional_mode": ("TORCHSIM_FUNCTIONAL_MODE", 1, int),
    "pytorchsim_timing_mode": ("TORCHSIM_TIMING_MODE", 0, int),
    "pytorchsim_functional_verify_per_kernel": ("TORCHSIM_FUNCTIONAL_VERIFY_PER_KERNEL", 0, int),
    "codegen_mapping_strategy": ("TORCHSIM_CODEGEN_MAPPING_STRATEGY", "autotune", str),
    "codegen_autotune_template_topk": ("TORCHSIM_CODEGEN_AUTOTUNE_TEMPLATE_TOPK", 4, int),
    "codegen_external_mapping_file": ("TORCHSIM_CODEGEN_EXTERNAL_MAPPING_FILE", "", str),
    "codegen_autotune_wall_slack_sec": ("TORCHSIM_CODEGEN_AUTOTUNE_WALL_SLACK_SEC", 15, float),
}


def _run_setting(name):
    env, default, kind = RUN_SETTINGS[name]
    got = os.environ.get(env)
    return kind(default if got is None or got == "" else got)


def __getattr__(name):
    # The machine: TORCHSIM_TARGET names a TOGSim machine file ($TOGSIM_DIR/configs/<name>.yml) or
    # gives a path to one. The compiler compiles for it and TOGSim times on it.
    if name == "CONFIG_TORCHSIM_TARGET":
        return os.environ.get("TORCHSIM_TARGET") or "tpuv6e"
    if name == "CONFIG_MACHINE":
        from pytorchsim.driver import compiler_bridge
        return compiler_bridge.target()
    if name == "CONFIG_MACHINE_FILE":
        from pytorchsim.driver import compiler_bridge
        return compiler_bridge.target().file
    if name == "CONFIG_NUM_CORES":
        from pytorchsim.driver import compiler_bridge
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


def togsim_include_dir():
    """Where togsim_runtime.h lives: the header a TOG is compiled against."""
    return os.path.join(CONFIG_TOGSIM_DIR, "include")


def togsim_simulator(togsim_dir=None):
    """The Simulator a TOGSim checkout builds in its own tree; CONFIG_TOGSIM_DIR's unless another is named."""
    return os.path.join(togsim_dir or CONFIG_TOGSIM_DIR, "build", "bin", "Simulator")


def setup_logger(name=None, level=None):
    """
    Setup a logger with consistent formatting across all modules.

    Args:
        name: Logger name (default: __name__ of calling module)
        level: Logging level (default: DEBUG if CONFIG_DEBUG_MODE else INFO)

    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        # Get the calling module's name
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'PyTorchSim')

    # Convert logger name to lowercase
    name = name.lower()
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set log level
        if level is None:
            level = logging.DEBUG if CONFIG_DEBUG_MODE else logging.INFO
        logger.setLevel(level)

    return logger