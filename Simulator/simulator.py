import os
import shlex
import subprocess

import torch
import numpy as np

from PyTorchSimFrontend.llvm_common import LLVMKernelArgs
import extension_codecache

TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
    torch.bfloat16: np.float16,
}

class FunctionalSimulator():
    def __init__(self, path, key):
        self.path = path
        self.key = key

    def load_tensor(self, arg, arg_name, arg_attribute, path):
        # path = os.path.join(dump_path, arg_name, f'{n_call}.raw')
        with open(path, 'rb') as f:
            np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[arg.dtype])
            src_tensor = torch.from_numpy(np_array).view(dtype=arg.dtype)
            src_tensor = src_tensor.reshape(arg.shape)
            arg.copy_(src_tensor)

    def get_biggest_filename(self, path):
        return len(os.listdir(path))

    def write_arg(self, arg, path, name):
        dump_path = os.path.join(path, name)
        os.makedirs(dump_path, exist_ok=True)
        index = self.get_biggest_filename(dump_path)

        if (isinstance(arg, torch.Tensor)):
            data_path = os.path.join(dump_path, f'{index}.raw')
            tensor = arg.cpu()
            t_arr = tensor.numpy().flatten()
            t_arr.tofile(data_path)
        else:
            assert(0)
        return index

    def dump_args(self, args, arg_attributes, load_path, dump_path):
        array_size = []
        file_path = []
        for (arg_name, arg_attribute), arg in zip(arg_attributes.items(), args):
            array_size.append(arg_attribute[2])
            if LLVMKernelArgs.is_llvm_arg_in(arg_attribute[0]):
                index = self.write_arg(arg, load_path, arg_name)
                file_path.append(os.path.join(load_path, arg_name, f'{index}.raw'))
            elif LLVMKernelArgs.is_llvm_arg_out(arg_attribute[0]):
                path = os.path.join(dump_path, arg_name)
                os.makedirs(path, exist_ok=True)
                file_path.append(os.path.join(path, f'{self.get_biggest_filename(path)}.raw'))
        return array_size, file_path

    def run_spike(self, args, arg_attributes, target_binary, intermediate_op=None):
        load_path = self.path
        dump_path = self.path

        if intermediate_op is not None:
            os.makedirs(os.path.join(self.path, "intermediate"), exist_ok=True)
            if intermediate_op & 0b10: # input comes from intermediate
                load_path = os.path.join(self.path, "intermediate")
            if intermediate_op & 0b01: # output dumps to intermediate
                dump_path = os.path.join(self.path, "intermediate")
                for name, attr in arg_attributes.items():
                    if attr[0] == 2:
                        os.makedirs(os.path.join(dump_path, name), exist_ok=True)

        array_size, file_path = self.dump_args(args, arg_attributes, load_path, dump_path)
        array_size_str = ' '.join(map(str, array_size))
        file_path_str = ' '.join(file_path)
 
        run = f'spike --isa rv64gcv /workspace/riscv-pk/build/pk {target_binary} {array_size_str} {file_path_str}'
        print("Spike cmd > ", run)
        run_cmd = shlex.split(run)
        try:
            subprocess.check_call(run_cmd)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("Error output:", e.output)
            assert(0)

        for (arg_name, arg_attribute), arg, path in zip(arg_attributes.items(), args, file_path):
            if LLVMKernelArgs.is_llvm_arg_out(arg_attribute[0]):
                self.load_tensor(arg, arg_name, arg_attribute, path)

class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary):
        try:
            gem5_cmd = [extension_codecache.GEM5_PATH, extension_codecache.GEM5_SCRIPT_PATH, target_binary]
            output = subprocess.check_output(gem5_cmd)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("Error output:", e.output)
            assert(0)

        with open("m5out/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        return cycle_list