import os
import re
import shlex
import subprocess
import torch

from PyTorchSimFrontend import extension_config
from torch._inductor.codecache import get_hash, write, write_atomic
from torch._inductor.async_compile import AsyncCompile
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen
from Simulator.simulator import FunctionalSimulator, CycleSimulator, TOGSimulator

# Configure logger for extension_codecache module (WARNING level by default)
logger = extension_config.setup_logger()

LOCK_TIMEOUT = 600

def hash_prefix(hash_value):
    return hash_value[1:12]

def get_write_path(src_code):
    return os.path.join(extension_config.get_dump_path(), hash_prefix(get_hash(src_code.strip())))


_HEADER_BY_HASH = {}
def store_header(src_code, spike_header, gem5_header):
    _HEADER_BY_HASH[get_hash(src_code.strip())] = (spike_header, gem5_header)
def get_header(src_code):
    return _HEADER_BY_HASH.get(get_hash(src_code.strip()))


def get_lock_path(write_path):
    """Return lock file path for the given write_path (per-source_code lock)."""
    return os.path.join(write_path, ".compile.lock")

def dump_metadata(args, arg_attributes, path):
    meta_path = os.path.join(path, "meta.txt")
    if os.path.isfile(meta_path):
        return

    with open(meta_path, "a") as file:
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            file.write(f'{arg_name}=({arg_attribute[0]}, {arg.dtype}, {arg.shape})\n')
    return

def mlir_compile_command(filename, vectorlane_size, vlen=256):
    # The C++ -dma-fine-grained and -test-pytorchsim-to-vcix passes are ported to
    # Python (passes/dma_fine_grained.py, lower_to_vcix.py), run in-process between
    # loop-padding and the standard lowering. So mlir-opt now runs only loop-padding
    # (-> _padded.mlir); the Python fine-grained + vcix passes produce _custom.mlir.
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding --allow-unregistered-dialect \
            {'--mlir-print-ir-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_MLIR_IR else ''} \
            {filename}.mlir -o {filename}_padded.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {filename}_llvm.mlir -o {filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -filetype=obj \
                {'--print-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_LLVM_IR else ''} \
                -O2 {filename}.ll -o {filename}.o
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -O2 {filename}.ll -o {filename}.s
        """,
    ).strip()]

def mlir_gem5_compile_command(filename, sample_filename, vectorlane_size, vlen=256):
    # See mlir_compile_command: -dma-fine-grained and -test-pytorchsim-to-vcix are
    # Python passes run in-process; mlir-opt runs only loop-padding here.
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding='timing_mode=1' --allow-unregistered-dialect \
            {'--mlir-print-ir-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_MLIR_IR else ''} \
            {filename}.mlir -o {sample_filename}_padded.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {sample_filename}_llvm.mlir -o {sample_filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -filetype=obj \
                {'--print-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_LLVM_IR else ''} \
                -O2 {sample_filename}.ll -o {sample_filename}.o
        """,
    ).strip()]

class SpadOverflowError(Exception):
    def __init__(self, message="SPAD overflow occurred."):
        super().__init__(message)

class TileSizeError(Exception):
    def __init__(self, message="SPAD overflow occurred."):
        super().__init__(message)

class MLIRCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)   # Todo: Cache

    @staticmethod
    def _load_library(path):
        pass

    @classmethod
    def load(cls, source_code,
             validation_wrapper_name="validation_wrapper",
             validation_binary_name="validation_bin",
             cycle_wrapper_name="cycle_wrapper",
             cycle_binary_name="cycle_bin",
             arg_attributes=[], vectorlane_size=16,
             spad_info=None, origins=None, silent_mode=False, **kwargs):
        vlen = kwargs['vlen']
        vlenb = vlen // 8
        write_path = get_write_path(source_code)
        os.makedirs(write_path, exist_ok=True)
        global_var_header = kwargs.get("global_var_header")
        if global_var_header is not None:
            write_atomic(os.path.join(write_path, "global_var.h"), global_var_header)
        gem5_global_var_header = kwargs.get("gem5_global_var_header")
        if gem5_global_var_header is not None:
            write_atomic(os.path.join(write_path, "gem5_global_var.h"), gem5_global_var_header)
        # The compile rewrites the kernel .mlir in place and reads it back, and two
        # compiles of the same source share a write_path. Hold the per-path lock across
        # the build, and skip it when a prior build finished (its trace.so exists).
        from filelock import FileLock
        from PyTorchSimFrontend.mlir.passes import (
            run_python_passes, run_module_passes, POST_OPT_PASSES,
            run_standard_lowering, run_tog,
        )
        trace_so_path = os.path.join(write_path, "trace.so")
        lock = FileLock(get_lock_path(write_path), timeout=LOCK_TIMEOUT)
        with lock:
            key, input_path = write(source_code, "mlir", specified_dir=write_path)
            if os.path.isfile(trace_so_path):
                return key
            # Run the Python out-of-line MLIR passes (MLIR bindings) on the kernel
            # .mlir in place, before mlir-opt. Currently lowers torchsim.vlane_idx
            # (replaces the old C++ -global-idx pass); add more in passes/__init__.py.
            run_python_passes(input_path, vectorlane=vectorlane_size)
            new_input_path = os.path.splitext(input_path)[0]
            raw_tog_path = new_input_path + "_tog.py"
            sample_mlir_path = new_input_path + "_sample"
            validation_binary_path = os.path.join(write_path, validation_binary_name)
            gem5_cmds = mlir_gem5_compile_command(new_input_path, sample_mlir_path, vectorlane_size)

            if spad_info is not None:
                link_option = f"-Wl,--section-start=.spad=0x{spad_info['spad_vaddr']:x}"
            else:
                link_option = ""
            # Compile a validation binary and measure its .spad section to reject
            # over-spad tilings, even in timing-only mode -- else the tiling wedges
            # TOGSim. Spike *execution* stays gated on functional_mode (run_spike).
            new_link_option = link_option + " -Wl,--wrap=malloc -Wl,--wrap=free"
            cmds = mlir_compile_command(new_input_path, vectorlane_size, vlen=vlen)
            opt_pad_cmd = shlex.split(cmds[0])
            translate_cmd = shlex.split(cmds[1])
            llc_cmd = shlex.split(cmds[2])
            llc_asm_cmd = shlex.split(cmds[3])
            try:
                # loop-padding (mlir-opt) -> Python fine-grained + vcix (one parse/print)
                subprocess.check_call(opt_pad_cmd)
                run_module_passes(new_input_path + "_padded.mlir",
                                  new_input_path + "_custom.mlir",
                                  POST_OPT_PASSES, vectorlane=vectorlane_size, vlen=vlen)
                # Standard MLIR -> LLVM-dialect lowering (registered upstream
                # passes) runs in-process via the bindings PassManager, picking
                # up after the custom mlir-opt passes (memref-to-gemmini).
                run_standard_lowering(new_input_path + "_custom.mlir", new_input_path + "_llvm.mlir")
                subprocess.check_call(translate_cmd)
                subprocess.check_call(llc_cmd)
                subprocess.check_call(llc_asm_cmd)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                logger.error(f"Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
                assert(0)

            val_llvm_caller = MLIRKernelCallerCodeGen(extension_config.pytorchsim_functional_mode, arg_attributes)
            val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
            val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name,
                                               validation_binary_name, new_link_option)

            # Only the .spad section consumes the scratchpad; the stack frame lives in main memory (sp in the -m region, not the scratchpad vaddr) so it is not charged against the per-lane spad budget.
            spad_usage = val_llvm_caller.get_spad_size(validation_binary_path)
            # Budget per dispatch = half the spad: two work-items run concurrently
            # (double-buffer), so each must fit in spad/2 or they deadlock competing for
            # the shared spad. Matches the GEMM tiling gate (max_spad_size = spad/2).
            spad_budget = extension_config.CONFIG_SPAD_INFO["spad_size"] // 2
            if spad_budget < spad_usage:
                logger.debug(
                    f"Scratchpad size exceeded: required {spad_usage} bytes, but only "
                    f"{spad_budget} bytes (spad/2, double-buffer budget) available."
                )
                raise SpadOverflowError()

            # Launch tile graph generator
            gem5_pad_cmd = shlex.split(gem5_cmds[0])
            gem5_translate_cmd = shlex.split(gem5_cmds[1])
            gem5_llc_cmd = shlex.split(gem5_cmds[2])

            try:
                # mlir-opt now runs only loop-padding and writes the post-vcix IR; the
                # tile-operation-graph pass is ported to Python. run_tog reads that IR and
                # writes the TOG plus the mutated IR (step rewrite + compute markers).
                subprocess.check_call(gem5_pad_cmd)
                run_module_passes(sample_mlir_path + "_padded.mlir",
                                  sample_mlir_path + "_postvcix.mlir",
                                  POST_OPT_PASSES, vectorlane=vectorlane_size, vlen=vlen)
                run_tog(sample_mlir_path + "_postvcix.mlir", raw_tog_path,
                        sample_mlir_path + "_custom.mlir",
                        sample_mode=extension_config.CONFIG_TLS_MODE,
                        vectorlane=vectorlane_size)
                # Standard MLIR -> LLVM-dialect lowering in-process (see functional path).
                run_standard_lowering(sample_mlir_path + "_custom.mlir", sample_mlir_path + "_llvm.mlir", timing=True)
                subprocess.check_call(gem5_translate_cmd)
                subprocess.check_call(gem5_llc_cmd)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                logger.error(f"Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
                assert(0)

            if not extension_config.pytorchsim_timing_mode:
                return key

            # Generate MLIR kernel calller and binary for cycle calculation
            cycle_llvm_caller = MLIRKernelCallerCodeGen(False, arg_attributes, cycle_sim=True)
            cycle_llvm_caller.generate_wrapper_file(write_path, cycle_wrapper_name)
            cycle_llvm_caller.compile_wih_kernel(write_path, key + "_sample", cycle_wrapper_name, cycle_binary_name, link_option)

            # Run cyclesim
            cyclesim = CycleSimulator()
            cycle_list = cyclesim.compile_and_simulate(os.path.join(write_path, cycle_binary_name), vectorlane_size, silent_mode=silent_mode)
            cycle_list_for_trace = list(cycle_list)

            # Per-tile cycle offsets, shared with the trace cycle-table below.
            w_offset, x_offset = vectorlane_size, vectorlane_size
            if kwargs['loop_size'] is not None and kwargs['loop_size'][-3] < vectorlane_size:
                x_offset = kwargs['loop_size'][-3]
            if kwargs['loop_size'] is not None and kwargs['loop_size'][-1] < vectorlane_size:
                w_offset = kwargs['loop_size'][-1]
            w_offset = 0 # max(w_offset - x_offset, 0)

            # Trace pipeline (sole sim path): emit the compiled trace producer .so +
            # the cycle-table TSV from the post-vcix IR and gem5 cycle_list/offsets;
            # TOGSim builds its C++ TOG from this via trace_to_tilegraph.
            try:
                import mlir.ir as ir
                from PyTorchSimFrontend.mlir.passes import (
                    build_skeleton as _bs, cycle_table as _ct, lower_to_emitc as _l2e)
                pv = sample_mlir_path + "_postvcix.mlir"
                _ctx = ir.Context(); _ctx.allow_unregistered_dialects = True
                with _ctx:
                    _mod = ir.Module.parse(open(pv).read(), _ctx)
                    _bs.build_skeleton(_mod)
                    _ntiles = len(_ct._compute_types(_mod))
                    # align lengths: gem5 gives one numCycles per compute node;
                    # pad with the last value / truncate if it disagrees.
                    _cl = list(cycle_list_for_trace)
                    if _cl and len(_cl) != _ntiles:
                        _cl = (_cl + [_cl[-1]] * _ntiles)[:_ntiles]
                    _tbl = _ct.build_cycle_table(_mod, _cl, x_offset, w_offset)
                _ct.dump_cycle_table_tsv(_tbl, os.path.join(write_path, "trace_cycles.tsv"), origins=origins)
                _l2e.build_trace_so(pv, os.path.join(write_path, "trace.so"))
            except Exception as e:
                logger.warning(f"[P3-trace] trace .so/sidecar dump skipped: {e}")
        return key

class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        self.validation_wrapper_name = "validation_wrapper"
        self.validation_binary_name = "validation_binary"
        self.cycle_wrapper_name = "cycle_wrapper"
        self.cycle_binary_name = "cycle_binary"

    def mlir(self, source_code, arg_attributes=[], vectorlane_size=16, tile_size=[], spad_info=None, origins=None, silent_mode=False, **kwargs):
        autotune = kwargs.get('autotune', False)
        def task():
            key = MLIRCodeCache.load(source_code,
                                          validation_wrapper_name=self.validation_wrapper_name,
                                          validation_binary_name=self.validation_binary_name,
                                          arg_attributes=arg_attributes, vectorlane_size=vectorlane_size,
                                          tile_size=tile_size, spad_info=spad_info, origins=origins,
                                          silent_mode=autotune, **kwargs)
            return key
        future = self.submit(task)

        def run_kernel_simulation(*args, autotune_subprocess_timeout_sec=None, **kwargs):
            # Wait for compilation
            key = future.result()
            if not autotune and origins:
                logger.info("[kernel %s] origins: %s",
                            hash_prefix(key), ", ".join(sorted(str(o) for o in origins)))
            from filelock import FileLock
            result_path = os.path.join(extension_config.get_dump_path(), hash_prefix(key))
            lock = FileLock(get_lock_path(result_path), timeout=LOCK_TIMEOUT)
            with lock:
                # Run simulator pass
                # Dump arguments and meta data
                dump_metadata(args, arg_attributes, result_path)
                runtime_path = FunctionalSimulator.get_runtime_dump_path(result_path)
                if extension_config.pytorchsim_functional_mode and not autotune:
                    funcsim = FunctionalSimulator(result_path, key)
                    funcsim.run_spike(args, arg_attributes,
                                    runtime_path, self.validation_binary_name,
                                    vectorlane_size=vectorlane_size, spad_info=spad_info,
                                    silent_mode=autotune)

                if not extension_config.pytorchsim_timing_mode:
                    return [float("inf")]

                # Prepare arguments for launch kernel
                onnx_path = os.path.join(result_path, "tile_graph.onnx")
                attribute_dir = os.path.join(runtime_path, "attribute")
                kernel_attribute_path = TOGSimulator.write_kernel_attribute_file(attribute_dir, args)

                TOGSim = torch.npu.get_tog_simulator()
                if not autotune and TOGSim is not None:
                    torch.npu.launch_kernel(onnx_path, kernel_attribute_path)
                    result = None # No result for non-autotune mode
                else:
                    result_path = TOGSimulator.run_standalone(
                        onnx_path,
                        kernel_attribute_path,
                        autotune_mode=autotune,
                        timeout_sec=autotune_subprocess_timeout_sec,
                    )
                    result = TOGSimulator.get_result_from_file(result_path)
                return result
        return run_kernel_simulation
