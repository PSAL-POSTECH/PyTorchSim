import getpass
import tempfile
import os
import re
import shlex
import subprocess
import operator
import functools

import torch
from torch._inductor.codecache import AsyncCompile, get_lock_dir, get_hash, write
from AsmParser.riscv_parser import riscv_parser
from PyTorchSimFrontend.llvm.llvm_caller_codegen import LLVMKernelCallerCodeGen
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen
from Simulator.simulator import FunctionalSimulator, CycleSimulator

LOCK_TIMEOUT = 600
TORCHSIM_DUMP_PATH = os.environ.get('TORCHSIM_DUMP_PATH',
                        default = f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}")
TORCHSIM_DUMP_FILE = int(os.environ.get('TORCHSIM_DUMP_FILE', default="True") == "True")
TORCHSIM_VALIDATION_MODE = int(os.environ.get('TORCHSIM_VALIDATION_MODE', default="True") == "True")
TORCHSIM_LLVM_PATH = os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")
TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
TORCHSIM_CUSTOM_PASS_PATH = os.environ.get('TORCHSIM_CUSTOM_PASS_PATH',
                                           default=f"{TORCHSIM_DIR}/GemminiLowerPass/build")
TORCHSIM_BACKEND_CONFIG = os.environ.get('TORCHSIM_CONFIG',
                                        default=f'{TORCHSIM_DIR}/PyTorchSimBackend/configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json')
GEM5_PATH = os.environ.get('GEM5_PATH',
                           default = f"/workspace/gem5/build/RISCV/gem5.opt")
GEM5_SCRIPT_PATH = os.environ.get('GEM5_SCRIPT_PATH',
                                  default = f"{TORCHSIM_DIR}/gem5_script/script.py")

def hash_prefix(hash_value):
    return hash_value[1:5]

def get_write_path(src_code):
    return os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(get_hash(src_code.strip())))

def dump_metadata(args, arg_attributes, path):
    meta_path = os.path.join(path, "meta.txt")
    if os.path.isfile(meta_path):
        return

    with open(meta_path, "a") as file:
        for (arg_name, arg_attribute), arg in zip(arg_attributes.items(), args):
            file.write(f'{arg_name}=({arg_attribute[0]}, {arg.dtype}, {arg.shape})\n')
    return

def llvm_compile_command(input, output):
    opt_output = f"{input[:-3]}_opt.ll"
    return [re.sub(r"[ \n]+", " ",
        f"""
            {TORCHSIM_LLVM_PATH}/opt --load-pass-plugin={TORCHSIM_CUSTOM_PASS_PATH}/libLowerGemminiPass.so -S -march=riscv64 --passes=LowerGemminiPass {input} -o {opt_output}
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {TORCHSIM_LLVM_PATH}/llc -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v -O2 {opt_output} -o {output}
        """,
    ).strip()]

def mlir_compile_command(filename):
    return [re.sub(r"[ \n]+", " ",
        # f"""
        #     {TORCHSIM_LLVM_PATH}/mlir-opt -lower-affine -test-memref-to-gemmini -finalize-memref-to-llvm -convert-arith-to-llvm -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm -convert-index-to-llvm -convert-vector-to-llvm -reconcile-unrealized-casts {filename}.mlir -o {filename}_llvm.mlir
        # """,
        f"""
        {   TORCHSIM_LLVM_PATH}/mlir-opt \
            -eliminate-empty-tensors \
            -empty-tensor-to-alloc-tensor \
            -one-shot-bufferize="bufferize-function-boundaries" \
            -convert-linalg-to-loops \
            -convert-shape-to-std \
            -fold-memref-alias-ops \
            -lower-affine \
            -finalize-memref-to-llvm \
            -convert-arith-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -canonicalize \
            -split-input-file \
            -cse \
            {filename}.mlir -o {filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {filename}_llvm.mlir -o {filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {TORCHSIM_LLVM_PATH}/llc -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v,+xsfvcp -O2 {filename}.ll -o {filename}.s
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
             arg_attributes={}, loop_info={},
             load_tile_info={}, store_tile_info={}, **kwargs):
        write_path = get_write_path(source_code)
        key, input_path = write(source_code, "mlir", specified_dir=write_path)

        cmds = mlir_compile_command(os.path.splitext(input_path)[0])
        opt_cmd = shlex.split(cmds[0])
        translate_cmd = shlex.split(cmds[1])
        llc_cmd = shlex.split(cmds[2])

        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            try:
                subprocess.check_call(opt_cmd)
                subprocess.check_call(translate_cmd)
                subprocess.check_call(llc_cmd)
            except subprocess.CalledProcessError as e:
                print("Command failed with exit code", e.returncode)
                print("Error output:", e.output)
                assert(0)

            # Generate LLVM kernel calller and binary for validation
            if TORCHSIM_VALIDATION_MODE:
                val_llvm_caller = MLIRKernelCallerCodeGen(TORCHSIM_VALIDATION_MODE, arg_attributes)
                val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
                val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name, validation_binary_name)

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
             arg_attributes={}, loop_info={},
             load_tile_info={}, store_tile_info={}, **kwargs):
        write_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(get_hash(source_code.strip())))
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
            if TORCHSIM_VALIDATION_MODE:
                val_llvm_caller = LLVMKernelCallerCodeGen(TORCHSIM_VALIDATION_MODE, arg_attributes)
                val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
                val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name, validation_binary_name)

            # Generate LLVM kernel calller and binary for cycle calculation
            cycle_llvm_caller = LLVMKernelCallerCodeGen(False, arg_attributes)
            cycle_llvm_caller.generate_wrapper_file(write_path, cycle_wrapper_name)
            cycle_llvm_caller.compile_wih_kernel(write_path, key + "_sample", cycle_wrapper_name, cycle_binary_name)
            array_size = []
            for (arg_name, arg_attribute) in arg_attributes.items():
                array_size.append(str(arg_attribute[2]))

            # Run cyclesim
            cyclesim = CycleSimulator()
            cycle_list = cyclesim.compile_and_simulate(os.path.join(write_path, cycle_binary_name), " ".join(array_size))

            if TORCHSIM_DUMP_FILE:
                tile_graph_generator.dump_basic_block_graph(os.path.join(write_path, "basic_block.onnx"))
            tile_graph_generator.cycle_analysis(cycle_list=cycle_list, name=os.path.join(write_path, "tile_graph"))
        return key

def get_backend_command(model_path):
    base_dir = os.path.join(TORCHSIM_DIR, "PyTorchSimBackend")
    bin = os.path.join(base_dir, "build/bin/Simulator")
    config = os.path.join(base_dir, TORCHSIM_BACKEND_CONFIG)
    cmd = f"{bin} --config {config} --models_list {model_path}"
    return cmd.strip()

class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        self.validation_wrapper_name = "validation_wrapper"
        self.validation_binary_name = "validation_binary"
        self.cycle_wrapper_name = "cycle_wrapper"
        self.cycle_binary_name = "cycle_binary"

    def mlir(self, source_code, arg_attributes={}, **kwargs):
        def task():
            key = MLIRCodeCache.load(source_code,
                                          valdiation_wrapper_name=self.validation_binary_name,
                                          validation_binary_name=self.validation_binary_name,
                                          arg_attributes=arg_attributes, **kwargs)
            return key
        future = self.submit(task)

        def dummy_simulator(*args, **kwargs):
            # Wait for compilation
            key = future.result()

            # Run simulator pass
            result_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key))
            # Dump arguments and meta data
            dump_metadata(args, arg_attributes, result_path)
            if TORCHSIM_VALIDATION_MODE:
                funcsim = FunctionalSimulator(result_path, key)
                funcsim.run_spike(args, arg_attributes,
                                  os.path.join(result_path, self.validation_binary_name),
                                  kwargs['intermediate_op'] if 'intermediate_op' in kwargs else None)
        return dummy_simulator

    def llvm(self, source_code, arg_attributes={}, **kwargs):
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
            result_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key))
            print("Running dummy simulator!")
            print("OUTPUT PATH > ", result_path)

            # Dump arguments and meta data
            dump_metadata(args, arg_attributes, result_path)
            if TORCHSIM_VALIDATION_MODE:
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
            cmd = get_backend_command(onnx_path)

            try:
                subprocess.check_call(shlex.split(cmd))
            except subprocess.CalledProcessError as e:
                print("Command failed with exit code", e.returncode)
                print("Error output:", e.output)
                assert(0)
        return dummy_simulator