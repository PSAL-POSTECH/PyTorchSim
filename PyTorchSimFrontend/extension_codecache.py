import getpass
import tempfile
import os
import re
import shlex
import subprocess

from torch._inductor.codecache import AsyncCompile, get_lock_dir, get_hash, write
from AsmParser.tog_generator import tog_generator
from AsmParser.riscv_parser import riscv_parser
from PyTorchSimFrontend.llvm.llvm_caller_codegen import LLVMKernelCallerCodeGen
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen
from PyTorchSimFrontend import extension_config
from Simulator.simulator import FunctionalSimulator, CycleSimulator, BackendSimulator

LOCK_TIMEOUT = 600

def hash_prefix(hash_value):
    return hash_value[1:5]

def get_write_path(src_code):
    return os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(get_hash(src_code.strip())))

def dump_metadata(args, arg_attributes, path):
    meta_path = os.path.join(path, "meta.txt")
    if os.path.isfile(meta_path):
        return

    with open(meta_path, "a") as file:
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            file.write(f'{arg_name}=({arg_attribute[0]}, {arg.dtype}, {arg.shape})\n')
    return

def llvm_compile_command(input, output):
    opt_output = f"{input[:-3]}_opt.ll"
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/opt --load-pass-plugin={extension_config.CONFIG_TORCHSIM_CUSTOM_PASS_PATH}/libLowerGemminiPass.so -S -march=riscv64 --passes=LowerGemminiPass {input} -o {opt_output}
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v -O2 {opt_output} -o {output}
        """,
    ).strip()]

def mlir_compile_command(filename, vectorlane_size, vlen=256):
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size} vlen={vlen}' \
            -test-memref-to-gemmini="vectorlane={vectorlane_size}" \
            -convert-linalg-to-loops \
            -lower-affine \
            -finalize-memref-to-llvm \
            -lower-vector-multi-reduction \
            -convert-vector-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -reconcile-unrealized-casts \
            {filename}.mlir -o {filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {filename}_llvm.mlir -o {filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc -relocation-model=pic -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v,+xsfvcp,zvl{vlen}b -O2 {filename}.ll -o {filename}.s
        """,
    ).strip()]

def mlir_gem5_compile_command(filename, sample_filename, tog_file, vectorlane_size, vlen=256):
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding='timing_mode=1' \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size} vlen={vlen}' \
            -test-tile-operation-graph='vectorlane={vectorlane_size}' \
            -test-memref-to-gemmini="vectorlane={vectorlane_size} timing=1" \
            -convert-linalg-to-loops \
            -lower-affine \
            -finalize-memref-to-llvm \
            -lower-vector-multi-reduction \
            -convert-vector-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -reconcile-unrealized-casts \
            {filename}.mlir -o {sample_filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {sample_filename}_llvm.mlir -o {sample_filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc -relocation-model=pic -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v,+xsfvcp,zvl{vlen}b -O2 {sample_filename}.ll -o {sample_filename}.s
        """,
    ).strip()]

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
             spad_info=None, origins=None, **kwargs):
        write_path = get_write_path(source_code)
        key, input_path = write(source_code, "mlir", specified_dir=write_path)
        new_input_path = os.path.splitext(input_path)[0]
        raw_tog_path = new_input_path + "_tog.py"
        sample_mlir_path = new_input_path + "_sample"
        gem5_cmds = mlir_gem5_compile_command(new_input_path, sample_mlir_path, raw_tog_path, vectorlane_size)

        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)

        if spad_info is not None:
            link_option = f"-Wl,--section-start=.spad=0x{spad_info['spad_vaddr']:x}"
        else:
            link_option = ""
        # Generate LLVM kernel calller and binary for validation
        if extension_config.CONFIG_TORCHSIM_VALIDATION_MODE:
            # Use custom malloc to avoid size error
            new_link_option = link_option + " -Wl,--wrap=malloc -Wl,--wrap=free"
            cmds = mlir_compile_command(new_input_path, vectorlane_size, vlen=256)
            opt_cmd = shlex.split(cmds[0])
            translate_cmd = shlex.split(cmds[1])
            llc_cmd = shlex.split(cmds[2])
            with lock:
                try:
                    subprocess.check_call(opt_cmd)
                    subprocess.check_call(translate_cmd)
                    subprocess.check_call(llc_cmd)
                except subprocess.CalledProcessError as e:
                    print("Command failed with exit code", e.returncode)
                    print("Error output:", e.output)
                    assert(0)

                val_llvm_caller = MLIRKernelCallerCodeGen(extension_config.CONFIG_TORCHSIM_VALIDATION_MODE, arg_attributes)
                val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
                val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name,
                                                   validation_binary_name, new_link_option)
        # Launch tile graph generator
        gem5_sample_cmd = shlex.split(gem5_cmds[0])
        gem5_translate_cmd = shlex.split(gem5_cmds[1])
        gem5_llc_cmd = shlex.split(gem5_cmds[2])

        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            try:
                result = subprocess.check_output(gem5_sample_cmd)
                with open(raw_tog_path, "wb") as file:
                    file.write(result)
                subprocess.check_call(gem5_translate_cmd)
                subprocess.check_call(gem5_llc_cmd)
            except subprocess.CalledProcessError as e:
                print("Command failed with exit code", e.returncode)
                print("Error output:", e.output)
                assert(0)

        if extension_config.CONFIG_BACKENDSIM_SPIKE_ONLY:
            return key

        # Generate MLIR kernel calller and binary for cycle calculation
        cycle_llvm_caller = MLIRKernelCallerCodeGen(False, arg_attributes, cycle_sim=True)
        cycle_llvm_caller.generate_wrapper_file(write_path, cycle_wrapper_name)
        cycle_llvm_caller.compile_wih_kernel(write_path, key + "_sample", cycle_wrapper_name, cycle_binary_name, link_option)
        array_size = []
        for (arg_name, arg_attribute) in arg_attributes:
            array_size.append(str(arg_attribute[2]))

        # Run cyclesim
        cyclesim = CycleSimulator()
        cycle_list = cyclesim.compile_and_simulate(os.path.join(write_path, cycle_binary_name), " ".join(array_size), vectorlane_size)

        # Create TOG
        offset = vectorlane_size
        if kwargs['loop_size'] is not None and kwargs['loop_size'][0] < vectorlane_size:
            offset = kwargs['loop_size'][0]
        tile_graph_generator = tog_generator(origins)
        tile_graph_generator.load_file(raw_tog_path)
        tile_graph_generator.generate_tile_graph(
            os.path.join(write_path, "tile_graph.onnx"),
            cycle_list=cycle_list,
            offset=offset, # FIXME.
            vector_lane=vectorlane_size
        )
        return key

class LLVMCodeCache:
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
             arg_attributes=[], loop_info={},
             load_tile_info={}, store_tile_info={}, **kwargs):
        write_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(get_hash(source_code.strip())))
        key, input_path = write(source_code, "ll", specified_dir=write_path)
        output_path = input_path[:-2] + "s"

        cmds = llvm_compile_command(input_path, output_path)
        opt_cmd = shlex.split(cmds[0])
        llc_cmd = shlex.split(cmds[1])

        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            # if not os.path.exists(output_path):
            try:
                subprocess.check_call(opt_cmd)
                subprocess.check_call(llc_cmd)
            except subprocess.CalledProcessError as e:
                print("Command failed with exit code", e.returncode)
                print("Error output:", e.output)
                assert(0)   # Todo: make LLVMCompileError

            # Launch tile graph generator
            tile_graph_generator = riscv_parser()
            tile_graph_generator.load_file(output_path,
                                            loop_info=loop_info,
                                            load_tile_info=load_tile_info,
                                            store_tile_info=store_tile_info)
            # Create code for sampling
            tile_graph_generator.dump_sampling_code(output_path[:-2] + "_sample.s")

            # Generate LLVM kernel calller and binary for validation
            if extension_config.CONFIG_TORCHSIM_VALIDATION_MODE:
                val_llvm_caller = LLVMKernelCallerCodeGen(extension_config.CONFIG_TORCHSIM_VALIDATION_MODE, arg_attributes)
                val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
                val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name, validation_binary_name)

            # Generate LLVM kernel calller and binary for cycle calculation
            cycle_llvm_caller = LLVMKernelCallerCodeGen(False, arg_attributes)
            cycle_llvm_caller.generate_wrapper_file(write_path, cycle_wrapper_name)
            cycle_llvm_caller.compile_wih_kernel(write_path, key + "_sample", cycle_wrapper_name, cycle_binary_name)
            array_size = []
            for (arg_name, arg_attribute) in arg_attributes:
                array_size.append(str(arg_attribute[2]))

            # Run cyclesim
            cyclesim = CycleSimulator()
            cycle_list = cyclesim.compile_and_simulate(os.path.join(write_path, cycle_binary_name), " ".join(array_size), vectorlane_size)

            if extension_config.CONFIG_TORCHSIM_DUMP_FILE:
                tile_graph_generator.dump_basic_block_graph(os.path.join(write_path, "basic_block.onnx"))
            tile_graph_generator.cycle_analysis(cycle_list=cycle_list, name=os.path.join(write_path, "tile_graph"))
        return key

class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        self.validation_wrapper_name = "validation_wrapper"
        self.validation_binary_name = "validation_binary"
        self.cycle_wrapper_name = "cycle_wrapper"
        self.cycle_binary_name = "cycle_binary"

    def mlir(self, source_code, arg_attributes=[], vectorlane_size=16, tile_size=[], spad_info=None, origins=None, **kwargs):
        def task():
            key = MLIRCodeCache.load(source_code,
                                          valdiation_wrapper_name=self.validation_binary_name,
                                          validation_binary_name=self.validation_binary_name,
                                          arg_attributes=arg_attributes, vectorlane_size=vectorlane_size,
                                          tile_size=tile_size, spad_info=spad_info, origins=origins, **kwargs)
            return key
        future = self.submit(task)
        def dummy_simulator(*args, **kwargs):
            # Wait for compilation
            key = future.result()

            # Run simulator pass
            result_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key))
            # Dump arguments and meta data
            dump_metadata(args, arg_attributes, result_path)
            if extension_config.CONFIG_TORCHSIM_VALIDATION_MODE:
                funcsim = FunctionalSimulator(result_path, key)
                funcsim.run_spike(args, arg_attributes,
                                  result_path, self.validation_binary_name,
                                  kwargs['intermediate_op'] if 'intermediate_op' in kwargs else None,
                                  vectorlane_size=vectorlane_size, spad_info=spad_info,
                                  cleanup=extension_config.CONFIG_CLEANUP_DUMP_ARGS)
            if extension_config.CONFIG_BACKENDSIM_SPIKE_ONLY:
                return

            onnx_path = os.path.join(result_path, "tile_graph.onnx")
            attribute_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key), "attribute")
            backend_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "PyTorchSimBackend")
            backsim = BackendSimulator(backend_path, extension_config.CONFIG_TORCHSIM_BACKEND_CONFIG)
            attribute_path = backsim.create_attribute_file(attribute_path, args, tile_size=tile_size)
            result_path = backsim.simulation(onnx_path, attribute_path)
            result = BackendSimulator.get_result_from_file(result_path)
            return result

        def dryrun_simulator(*args, **kwargs):
            key = future.result()

        is_dryrun = extension_config.CONFIG_BACKENDSIM_DRYRUN
        target_simulator = dryrun_simulator if is_dryrun else dummy_simulator
        target_simulator.arg_attributes = arg_attributes
        target_simulator.future = future
        return target_simulator

    def llvm(self, source_code, arg_attributes=[], **kwargs):
        def task():
            key = LLVMCodeCache.load(source_code,
                                          valdiation_wrapper_name=self.validation_binary_name,
                                          validation_binary_name=self.validation_binary_name,
                                          arg_attributes=arg_attributes, **kwargs)
            return key
        future = self.submit(task)

        def dummy_simulator(*args, **kwargs):
            # Wait for compilation
            key = future.result()

            # Run simulator pass
            result_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key))
            print("Running dummy simulator!")
            print("OUTPUT PATH > ", result_path)

            # Dump arguments and meta data
            dump_metadata(args, arg_attributes, result_path)
            if extension_config.CONFIG_TORCHSIM_VALIDATION_MODE:
                funcsim = FunctionalSimulator(result_path, key)
                funcsim.run_spike(args, arg_attributes,
                                  os.path.join(result_path, self.validation_binary_name),
                                  kwargs['intermediate_op'] if 'intermediate_op' in kwargs else None)

            assembly_path = os.path.join(result_path, f'{key}.s')
            try:
                with open(assembly_path, 'r') as file:
                    file_contents = file.read()
                    print("Assembly > \n", file_contents)
            except FileNotFoundError:
                print(f'{assembly_path} not found.')
            except Exception as e:
                print(f"Error while reading.")

            onnx_path = os.path.join(result_path, "tile_graph.onnx")
            attribute_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key), "attribute")
            backend_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "PyTorchSimBackend")
            backsim = BackendSimulator(backend_path, extension_config.CONFIG_TORCHSIM_BACKEND_CONFIG)
            attribute_path = backsim.create_attribute_file(attribute_path, args)
            result_path = backsim.simulation(onnx_path, attribute_path)
            result = BackendSimulator.get_result_from_file(result_path)
            return result
        return dummy_simulator
