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
CONFIG_GEM5_SCRIPT = os.environ.get("TORCHSIM_GEM5_SCRIPT", default=os.path.join(
    CONFIG_TORCHSIM_COMPILE_DIR, "third_party", "vcix-accelerator", "examples", "tpu", "gem5",
    "script_systolic.py"))
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


def __getattr__(name):
    # TOGSim config
    config_path = os.environ.get('TOGSIM_CONFIG', default=os.path.join(
        CONFIG_TOGSIM_CONFIG_DIR, "systolic_ws_256x256_c1_simple_noc_tpuv6e_functional_only.yml"))
    if name == "CONFIG_TOGSIM_CONFIG":
        return config_path

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"{config_path} not found: TOGSIM_DIR must name a TOGSim checkout (its configs/ holds the "
            f"machine configs; now {CONFIG_TOGSIM_DIR}), or TOGSIM_CONFIG a machine config file")
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    if name == "CONFIG_NUM_CORES":
        return config_yaml["num_cores"]

    if name == "pytorchsim_functional_mode":
        return config_yaml['pytorchsim_functional_mode']
    if name == "pytorchsim_timing_mode":
        env = os.environ.get("TORCHSIM_TIMING_MODE")
        if env is not None:
            return int(env)
        return config_yaml.get('pytorchsim_timing_mode', 1)
    # Sub-option of functional mode: compare every realized Spike buffer against a CPU
    # golden to localize the first kernel whose value diverges. Auto-disabled when
    # functional mode is off (there are no Spike values to verify).
    if name == "pytorchsim_functional_verify_per_kernel":
        return bool(config_yaml.get('pytorchsim_functional_verify_per_kernel', False)) \
            and bool(config_yaml['pytorchsim_functional_mode'])

    # Tile mapping (develop's keys): "autotune" times the top-k template tiles and keeps
    # the fastest; a tile that does not compile counts as never finishing.
    if name == "codegen_mapping_strategy":
        return config_yaml.get("codegen_mapping_strategy", "heuristic")
    if name == "codegen_autotune_template_topk":
        return int(config_yaml.get("codegen_autotune_template_topk", 4))
    if name == "codegen_external_mapping_file":
        return config_yaml.get("codegen_external_mapping_file", "")
    if name == "codegen_autotune_wall_slack_sec":
        return float(config_yaml.get("codegen_autotune_wall_slack_sec", 15))

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