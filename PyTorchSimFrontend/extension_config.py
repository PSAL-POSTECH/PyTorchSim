import os
import sys
import importlib
import yaml
import logging

CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
CONFIG_GEM5_PATH = os.environ.get('GEM5_PATH', default="/workspace/gem5/build/RISCV/gem5.opt")
CONFIG_TORCHSIM_LLVM_PATH = os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")

CONFIG_TORCHSIM_DUMP_MLIR_IR = int(os.environ.get("TORCHSIM_DUMP_MLIR_IR", default=False))
CONFIG_TORCHSIM_DUMP_LLVM_IR = int(os.environ.get("TORCHSIM_DUMP_LLVM_IR", default=False))

def __getattr__(name):
    # TOGSim config
    config_path = os.environ.get('TOGSIM_CONFIG',
                default=f"{CONFIG_TORCHSIM_DIR}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
    if name == "CONFIG_TOGSIM_CONFIG":
        return config_path

    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    # Hardware info config
    if name == "vpu_num_lanes":
        return config_yaml["vpu_num_lanes"]
    if name == "CONFIG_SPAD_INFO":
        return {
          "spad_vaddr" : 0xD0000000,
          "spad_paddr" : 0x2000000000,
          "spad_size" : config_yaml["vpu_spad_size_kb_per_lane"] << 10 # Note: spad size per lane
        }

    if name == "CONFIG_NUM_CORES":
        return config_yaml["num_cores"]
    if name == "vpu_vector_length_bits":
        return config_yaml["vpu_vector_length_bits"]

    if name == "pytorchsim_functional_mode":
        return config_yaml['pytorchsim_functional_mode']
    if name == "pytorchsim_timing_mode":
        return config_yaml['pytorchsim_timing_mode']

    # Mapping strategy
    if name == "codegen_mapping_strategy":
        codegen_mapping_strategy = config_yaml["codegen_mapping_strategy"]
        assert(codegen_mapping_strategy in ["heuristic", "autotune", "external-then-heuristic", "external-then-autotune"]), "Invalid mapping strategy!"
        return codegen_mapping_strategy

    if name == "codegen_external_mapping_file":
        return config_yaml["codegen_external_mapping_file"]

    # Autotune config
    if name == "codegen_autotune_max_retry":
        return config_yaml["codegen_autotune_max_retry"]
    if name == "codegen_autotune_template_topk":
        return config_yaml["codegen_autotune_template_topk"]

    # Compiler Optimization
    if name == "codegen_compiler_optimization":
        opt_level = config_yaml["codegen_compiler_optimization"]
        valid_opts = {
            "fusion",
            "reduction_epilogue",
            "reduction_reduction",
            "prologue",
            "single_batch_conv",
            "multi_tile_conv",
            "subtile"
        }
        if opt_level == "all" or opt_level == "none":
            pass
        elif isinstance(opt_level, list):
            # Check if provided list contains only valid options
            invalids = set(opt_level) - valid_opts
            assert not invalids, f"Invalid optimization options found: {invalids}"
        else:
            assert False, "Invalid format: Must be 'all', none, or a list of options."
        return opt_level

    # Advanced fusion options
    is_opt_enabled = lambda key: (__getattr__("codegen_compiler_optimization") == "all") or \
                                 (isinstance(__getattr__("codegen_compiler_optimization"), list) and \
                                  key in __getattr__("codegen_compiler_optimization"))
    if name == "CONFIG_FUSION":
        return is_opt_enabled("fusion")
    if name == "CONFIG_FUSION_REDUCTION_EPILOGUE":
        return is_opt_enabled("reduction_epilogue") # Fixed typo here as well
    if name == "CONFIG_FUSION_REDUCTION_REDUCTION":
        return is_opt_enabled("reduction_reduction")
    if name == "CONFIG_FUSION_PROLOGUE":
        return is_opt_enabled("prologue")
    if name == "CONFIG_SINGLE_BATCH_CONV":
        return is_opt_enabled("single_batch_conv")
    if name == "CONFIG_MULTI_TILE_CONV":
        return is_opt_enabled("multi_tile_conv")
    if name == "CONFIG_SUBTILE":
        return is_opt_enabled("subtile")

    if name == "CONFIG_TOGSIM_DEBUG_LEVEL":
        return os.environ.get("TOGSIM_DEBUG_LEVEL", "")
    if name == "CONFIG_TORCHSIM_DUMP_PATH":
        return os.environ.get('TORCHSIM_DUMP_PATH', default = CONFIG_TORCHSIM_DIR)
    if name == "CONFIG_TORCHSIM_LOG_PATH":
        return os.environ.get('TORCHSIM_LOG_PATH', default = os.path.join(CONFIG_TORCHSIM_DIR, "togsim_results"))

# SRAM Buffer allocation plan
def load_plan_from_module(module_path):
    if module_path is None:
      return None

    try:
        spec = importlib.util.spec_from_file_location("plan_module", module_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'plan'):
            return module.plan
        return None
    except Exception as e:
        print(f"[Warning] Failed to load SRAM buffer plan from module: {e}")
        return None

CONFIG_SRAM_BUFFER_PLAN_PATH = os.environ.get("SRAM_BUFFER_PLAN_PATH", default=None)
CONFIG_SRAM_BUFFER_PLAN = load_plan_from_module(CONFIG_SRAM_BUFFER_PLAN_PATH)

# For ILS experiment
CONFIG_TLS_MODE = int(os.environ.get('TORCHSIM_TLS_MODE', default=1))

CONFIG_USE_TIMING_POOLING = int(os.environ.get('TORCHSIM_USE_TIMING_POOLING', default=0))

CONFIG_DEBUG_MODE = int(os.environ.get('TORCHSIM_DEBUG_MODE', default=1))


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