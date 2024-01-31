import getpass
import tempfile
import os
import re
import shlex
import subprocess

import sys

import torch
from torch._inductor.codecache import AsyncCompile, get_lock_dir, get_hash, write
from AsmParser.riscv_parser import riscv_parser
from extension_backends.llvm_common import LLVMKernelArgs

LOCK_TIMEOUT = 600
TORCHSIM_DUMP_PATH = os.environ.get('TORCHSIM_DUMP_PATH',
                        default = f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}")
TORCHSIM_DUMP_FILE = int(os.environ.get('TORCHSIM_DUMP_FILE', default="True") == "True")
TORCHSIM_LLVM_PATH = os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")
TORCHSIM_CUSTOM_PASS_PATH = os.environ.get('TORCHSIM_CUSTOM_PASS_PATH', default="./GemminiLowerPass/build")
TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/TorchSim')
TORCHSIM_ONNXIM_CONFIG = os.environ.get('TORCHSIM_CONFIG', default='configs/systolic_ws_8x8_c1_simple_noc.json')

GEM5_DUMP_PATH = os.environ.get('GEM5_DUMP_PATH',
                                default = f"{TORCHSIM_DUMP_PATH}/../gem5/")
GEM5_PATH = os.environ.get('GEM5_PATH',
                           default = f"{GEM5_DUMP_PATH}/build/RISCV/gem5.opt")
GEM5_SCRIPT_PATH = os.environ.get('GEM5_SCRIPT_PATH',
                                  default = f"/workspace/gem5/configs/learning_gem5/part1/script.py")
                                #   default = f"{GEM5_DUMP_PATH}/gem5_script/script.py")
# this will be removed
PREMADE_BINARY_PATH = "/test/test.o"

def compile_and_simulate(asm_file_path, output_binary_path, compiler_path):
    try:
        compile_cmd = [compiler_path, "--target=riscv64", "-march=rv64gcv", "-o2", asm_file_path, "-nostdlib", "-o", output_binary_path]
        subprocess.run(compile_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)

    try:
        if PREMADE_BINARY_PATH is not None:
            gem5_cmd = [GEM5_PATH, GEM5_SCRIPT_PATH, PREMADE_BINARY_PATH]
        else:
            gem5_cmd = [GEM5_PATH, GEM5_SCRIPT_PATH, output_binary_path]

        output = subprocess.check_output(gem5_cmd)
        lines = output.decode('utf-8').split('\n')
        ticks = int(lines[-2] if lines[-1] == '' else lines[-1])
        return ticks
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)

def hash_prefix(hash_value):
    return hash_value[1:5]

def dump_metadata(args, arg_attributes, path):
    meta_path = os.path.join(path, "meta.txt")
    if os.path.isfile(meta_path):
        return

    with open(meta_path, "a") as file:
        for (arg_name, arg_attribute), arg in zip(arg_attributes.items(), args):
            file.write(f'{arg_name}=({arg_attribute}, {arg.dtype}, {arg.shape})\n')
    return

def write_arg(arg, path, name):
    dump_path = os.path.join(path, name)
    os.makedirs(dump_path, exist_ok=True)
    index = len(os.listdir(dump_path))

    if (isinstance(arg, torch.Tensor)):
        data_path = os.path.join(dump_path, f'{index}.raw')
        tensor = arg.cpu()
        t_arr = tensor.numpy().flatten()
        t_arr.tofile(data_path)
    else:
        assert(0)

def dump_args(args, arg_attributes, path):
    for (arg_name, arg_attribute), arg in zip(arg_attributes.items(), args):
        if arg_attribute == LLVMKernelArgs.LLVM_ARGS_IN or arg_attribute == LLVMKernelArgs.LLVM_ARGS_INOUT:
            write_arg(arg, path, arg_name)

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

class LLVMCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)   # Todo: Cache

    @staticmethod
    def _load_library(path):
        pass

    @classmethod
    def load(cls, source_code, loop_info={}, load_tile_info={}, store_tile_info={}, **kwargs):
        global TORCHSIM_DUMP_PATH
        write_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(get_hash(source_code)))
        key, input_path = write(source_code, "ll", specified_dir=write_path)
        output_path = input_path[:-2] + "s"
        
        result_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key))
        assembly_path = os.path.join(result_path, f'{key}.s')
        binary_path = os.path.join(result_path, f'{key}.o')

        cmds = llvm_compile_command(input_path, output_path)
        opt_cmd = shlex.split(cmds[0])
        llc_cmd = shlex.split(cmds[1])

        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            if not os.path.exists(output_path):
                try:
                    subprocess.check_call(opt_cmd)
                    subprocess.check_call(llc_cmd)
                except subprocess.CalledProcessError as e:
                    assert(0)   # Todo: make LLVMCompileError

                # launch tile graph generator
                tile_graph_generator = riscv_parser()
                tile_graph_generator.load_file(output_path,
                                               loop_info=loop_info,
                                               load_tile_info=load_tile_info,
                                               store_tile_info=store_tile_info)
                ticks = compile_and_simulate(assembly_path, binary_path, "clang")
                if TORCHSIM_DUMP_FILE:
                    tile_graph_generator.dump_basic_block_graph(os.path.join(write_path, "basic_block.onnx"))
                tile_graph_generator.cycle_analysis(name=os.path.join(write_path, "tile_graph"))
            else:
                pass
        return key

def get_onnxim_command(model_path):
    base_dir = os.path.join(TORCHSIM_DIR, "ONNXim")
    bin = os.path.join(base_dir, "build/bin/Simulator")
    config = os.path.join(base_dir, TORCHSIM_ONNXIM_CONFIG)
    model_list = os.path.join(base_dir, "models_list.json") # TODO: file format will be changed
    cmd = f"{bin} --config {config} --model {model_path} --models_list {model_list}"
    return cmd.strip()

class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        pass

    def llvm(self, source_code, arg_attributes={}, **kwargs):
        def task():
            self.key = LLVMCodeCache.load(source_code, **kwargs)
            return
        future = self.submit(task)

        def dummy_simulator(*args, **kwargs):
            future.result()
            os.system('echo "Running dummy simulator!"')
            result_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(self.key))
            print("OUTPUT PATH > ", result_path)

            dump_metadata(args, arg_attributes, result_path)
            if TORCHSIM_DUMP_FILE:
                dump_args(args, arg_attributes, result_path)

            assembly_path = os.path.join(result_path, f'{self.key}.s')
            try:
                with open(assembly_path, 'r') as file:
                    file_contents = file.read()
                    print("Assembly > \n", file_contents)
            except FileNotFoundError:
                print(f'{assembly_path} not found.')
            except Exception as e:
                print(f"Error while reading.")
                   
            cmd = get_onnxim_command(result_path)
            subprocess.check_call(shlex.split(cmd))
        return dummy_simulator