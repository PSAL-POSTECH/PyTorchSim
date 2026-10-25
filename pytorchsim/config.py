import os
import sys
import yaml
import logging

CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
#: How many Spike processes a launch's grid is split across; empty lets the launcher choose.
CONFIG_SPIKE_JOBS = os.environ.get("TORCHSIM_SPIKE_JOBS", "").strip()
#: The compiler checkout (stages 1-5, ttir to the RISC-V ELF), a separate repository, not vendored.
CONFIG_TORCHSIM_COMPILE_DIR = os.environ.get(
    "TORCHSIM_COMPILE_DIR", default=os.path.join(CONFIG_TORCHSIM_DIR, "pytorchsim-triton-compiler"))
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


RUN_SETTINGS = {
    "pytorchsim_functional_mode": ("TORCHSIM_FUNCTIONAL_MODE", 1, int),
    "pytorchsim_timing_mode": ("TORCHSIM_TIMING_MODE", 0, int),
    "pytorchsim_functional_verify_per_kernel": ("TORCHSIM_FUNCTIONAL_VERIFY_PER_KERNEL", 0, int),
    "codegen_autotune_template_topk": ("TORCHSIM_CODEGEN_AUTOTUNE_TEMPLATE_TOPK", 4, int),
    "codegen_mapping_strategy": ("TORCHSIM_CODEGEN_MAPPING_STRATEGY", "autotune", str),
    "codegen_external_mapping_file": ("TORCHSIM_CODEGEN_EXTERNAL_MAPPING_FILE", "", str),
}

MAPPING_STRATEGIES = ("heuristic", "autotune", "external-then-heuristic", "external-then-autotune")

_machine_settings_cache = {}


def machine_settings():
    """({run setting: value}, path) of the machine file TORCHSIM_TARGET names: a run setting is its
    environment variable, else this key, else its default. The compiler and TOGSim read no such key."""
    from pytorchsim.backend import compiler_bridge
    path = compiler_bridge.target().file
    mtime = os.path.getmtime(path)
    got = _machine_settings_cache.get(path)
    if got is None or got[0] != mtime:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        got = (mtime, {k: raw[k] for k in RUN_SETTINGS if raw.get(k) is not None})
        _machine_settings_cache[path] = got
    return got[1], path


def run_setting_source(name):
    """Where a run setting's value comes from: "env", "machine" or "default"."""
    env = RUN_SETTINGS[name][0]
    if os.environ.get(env, "") != "":
        return "env"
    return "machine" if name in machine_settings()[0] else "default"


def _run_setting(name):
    env, default, kind = RUN_SETTINGS[name]
    source = run_setting_source(name)
    if source == "env":
        return kind(os.environ[env])
    if source == "machine":
        return kind(machine_settings()[0][name])
    return default


def _mapping_strategy():
    got = _run_setting("codegen_mapping_strategy")
    if got not in MAPPING_STRATEGIES:
        raise ValueError(f"codegen_mapping_strategy {got!r} ({run_setting_source('codegen_mapping_strategy')}) "
                         f"is none of {', '.join(MAPPING_STRATEGIES)}")
    return got


def _external_mapping_file():
    """The external mapping file, absolute: a relative path the machine file states is its
    directory's, one TORCHSIM_CODEGEN_EXTERNAL_MAPPING_FILE states is the working directory's."""
    got = _run_setting("codegen_external_mapping_file")
    if not got:
        return ""
    if run_setting_source("codegen_external_mapping_file") == "machine" and not os.path.isabs(got):
        got = os.path.join(os.path.dirname(machine_settings()[1]), got)
    return os.path.abspath(got)


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
    if name == "codegen_mapping_strategy":
        return _mapping_strategy()
    if name == "codegen_external_mapping_file":
        return _external_mapping_file()
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