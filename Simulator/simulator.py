import os
import shlex
import subprocess
import re
import sys
import json
import time
import threading

import torch
import numpy as np

from PyTorchSimFrontend.llvm.llvm_common import LLVMKernelArgs

TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.uint8,
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
            if (arg.dtype == torch.bool):
                np_array = np.unpackbits(np_array)
            src_tensor = torch.as_strided(torch.from_numpy(np_array), arg.size(), arg.stride())
            arg.copy_(src_tensor.to(dtype=arg.dtype))

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
            if (tensor.dtype == torch.bool):
                t_arr = np.packbits(t_arr)
            t_arr.tofile(data_path)
        else:
            assert(0)
        return index

    def dump_args(self, args, arg_attributes, load_path, dump_path):
        array_size = []
        file_path = []
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            size = arg_attribute[2] if arg_attribute[1] != torch.bool else (arg_attribute[2] + 7) // 8
            array_size.append(size)
            if LLVMKernelArgs.is_llvm_arg_in(arg_attribute[0]):
                index = self.write_arg(arg, load_path, arg_name)
                file_path.append(os.path.join(load_path, arg_name, f'{index}.raw'))
            elif LLVMKernelArgs.is_llvm_arg_out(arg_attribute[0]):
                path = os.path.join(dump_path, arg_name)
                os.makedirs(path, exist_ok=True)
                file_path.append(os.path.join(path, f'{self.get_biggest_filename(path)}.raw'))

        return array_size, file_path

    def run_spike(self, args, arg_attributes, path, binary, intermediate_op=None, vectorlane_size=4, spad_info=None):
        load_path = self.path
        dump_path = self.path

        target_binary = os.path.join(path, binary)
        objdump = f"riscv64-unknown-elf-objdump -d {target_binary} > {os.path.join(path, 'binary.dump')}"
        kernel_start = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}'"
        kernel_end = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}' | xargs -I {{}} awk '/{{}}/,0' {os.path.join(path, 'binary.dump')} | grep ret | awk 'NR==1 {{print $1}}' | awk '{{gsub(/:$/, \"\"); print}}'"

        subprocess.run(objdump, shell=True)
        kernel_start_addr = subprocess.run(kernel_start, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')
        kernel_end_addr = subprocess.run(kernel_end, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')

        if intermediate_op is not None:
            os.makedirs(os.path.join(self.path, "intermediate"), exist_ok=True)
            if intermediate_op & 0b10: # input comes from intermediate
                load_path = os.path.join(self.path, "intermediate")
            if intermediate_op & 0b01: # output dumps to intermediate
                dump_path = os.path.join(self.path, "intermediate")
                for name, attr in arg_attributes:
                    if attr[0] == 2:
                        os.makedirs(os.path.join(dump_path, name), exist_ok=True)

        _, file_path = self.dump_args(args, arg_attributes, load_path, dump_path)
        file_path_str = ' '.join(file_path)

        # Set hardware information
        spad_option = f"--scratchpad-base-paddr={spad_info['spad_paddr']} " + \
            f"--scratchpad-base-vaddr={spad_info['spad_vaddr']} " + \
            f"--scratchpad-size={spad_info['spad_size']}"
        vectorlane_option = f"--vectorlane-size={vectorlane_size}"
        kernel_address = f"--kernel-addr={kernel_start_addr}:{kernel_end_addr}"
        run = f'spike --isa rv64gcv {vectorlane_option} {spad_option} {kernel_address} /workspace/riscv-pk/build/pk {target_binary} {file_path_str}'

        print("[SpikeSimulator] cmd> ", run)
        run_cmd = shlex.split(run)
        try:
            subprocess.check_call(run_cmd)
        except subprocess.CalledProcessError as e:
            print("[SpikeSimulator] Command failed with exit code", e.returncode)
            print("[SpikeSimulator] Error output:", e.output)
            assert(0)

        for (arg_name, arg_attribute), arg, path in zip(arg_attributes, args, file_path):
            if LLVMKernelArgs.is_llvm_arg_out(arg_attribute[0]):
                self.load_tensor(arg, arg_name, arg_attribute, path)

class CycleSimulator():
    GEM5_PATH = os.environ.get('GEM5_PATH',
                           default = f"/workspace/gem5/build/RISCV/gem5.opt")
    GEM5_SCRIPT_PATH = os.environ.get('GEM5_SCRIPT_PATH',
                                  default = f"{TORCHSIM_DIR}/gem5_script/script.py")
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, array_size, vectorlane_size):
        def show_progress():
            i = 0
            while not finished:
                i = (i + 1) % 3
                tail = "." * i + " " * (3-i)
                sys.stdout.write("\r[Gem5Simulator] Simulation is still running." + tail)
                time.sleep(1)
            print("")
        # Create progress thread
        finished = False
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.start()

        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        try:
            gem5_cmd = [self.GEM5_PATH, "-d", dir_path, self.GEM5_SCRIPT_PATH, "-c", target_binary, "--vlane", str(vectorlane_size)]
            output = subprocess.check_output(gem5_cmd, stderr=subprocess.DEVNULL)
            finished = True
            progress_thread.join()
        except subprocess.CalledProcessError as e:
            print("[Gem5Simulator] Command failed with exit code", e.returncode)
            print("[Gem5Simulator] Error output:", e.output)
            finished = True
            progress_thread.join()
            assert(0)

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) / cycle_per_tick for line in raw_list if "system.cpu.numCycles" in line]
        #cycle_list = [cycle_list[i+1] - cycle_list[i] for i in range(len(cycle_list)-1)]
        cycle_list = [128 for i in range(len(cycle_list))] # FIXME.
        return cycle_list

class BackendSimulator():
    BACKEND_RESULT_PATH_KEY = "BACKEND_RESULT_PATH"
    BACKENDSIM_DRYRUN = "BACKENDSIM_DRYRUN"
    BACKENDSIM_EAGER_MODE = "BACKENDSIM_EAGER_MODE"
    FINISH_STR = "Simulation Finished"
    def __init__(self, backend_path, config_path) -> None:
        self.base_dir = backend_path
        self.config_path = config_path
        self.process = None

    def get_backend_command(self):
        bin = os.path.join(self.base_dir, "build/bin/Simulator")
        config = os.path.join(self.base_dir, self.config_path)
        cmd = f"{bin} --config {config}"
        return cmd

    def simulation(self, model_path, attribute_path=""):
        def show_progress():
            i = 0
            while not finished:
                i = (i + 1) % 3
                tail = "." * i + " " * (3-i)
                sys.stdout.write("\r[BackendSimulator] Simulation is still running." + tail)
                time.sleep(1)
            print("")
        cmd = f"{self.get_backend_command()} --models_list {model_path}"
        if attribute_path:
            cmd = f"{cmd} --attributes_list {attribute_path}"
        print("[BackendSimulator] cmd> ", cmd)

        # Create progress thread
        finished = False
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.start()
        try:
            result = subprocess.check_output(shlex.split(cmd))
            finished = True
            progress_thread.join()
        except subprocess.CalledProcessError as e:
            finished = True
            progress_thread.join()
            print("[BackendSimulator] Command failed with exit code", e.returncode)
            print("[BackendSimulator] Error output:", e.output)
            assert(0)

        result_path = os.getenv(self.BACKEND_RESULT_PATH_KEY, os.path.join(os.path.dirname(model_path), "backendsim_result"))

        # Save result to result_path
        os.makedirs(result_path, exist_ok=True)
        file_name = str(len(os.listdir(result_path)))
        result_path = os.path.join(result_path, file_name)
        with open(result_path, "w") as f:
            f.write(result.decode())
            print(f'[BackendSimulator] Simulation of "{model_path}" is stored to "{result_path}"')
        return result_path

    def interactive_simulation(self):
        cmd = f"{self.get_backend_command()} --mode interactive"
        if self.process is None:
            self.process = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            print("[BackendSimulator] Simulator is already running.")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process = None

    def send_command(self, command):
        if self.process:
            try:
                print(command)
                self.process.stdin.write(command + '\n')
                self.process.stdin.flush()
                ret = self.process.stderr.readline().strip()
                return ret
            except BrokenPipeError:
                err = self.process.stderr.readlines()
                for line in err:
                    print(line)
                self.process = None
                exit(1)
        else:
            print("Simulator is not running.")
            return None

    def launch(self, onnx_path, attribute_path, arrival_time=0, partion_id=0):
        command = f"launch {onnx_path} {attribute_path} {arrival_time} {partion_id}"
        ret = self.send_command(command)
        return 0

    def cycle(self):
        ret = self.send_command("cycle")
        return int(ret.split(" ")[-1])

    def until(self, until_cycle):
        command = f"until {until_cycle}"
        ret = self.send_command(command)
        return int(ret.split(" ")[-1])

    def create_attribute_file(self, attribute_path, inputs):
        address_info = {}
        os.makedirs(attribute_path, exist_ok=True)
        index = str(len(os.listdir(attribute_path)))
        attribute_path = os.path.join(attribute_path, index)

        for idx, tensor in enumerate(inputs):
            address_info[f"arg{idx}"] = tensor.data_ptr()

        with open(attribute_path, "w") as f:
            json.dump({"address_info" : address_info}, f, indent=4)
        return attribute_path

    @staticmethod
    def get_result_from_file(result_path):
        core_metrics = {}
        dram_channel_bw = {}
        avg_dram_bw = None
        simulation_time = None

        # Read and find total stat position
        with open(result_path, "r") as f:
            lines = f.readlines()

        simulation_finished_idx = -1
        simulation_finished = False
        for idx, line in enumerate(lines):
            if BackendSimulator.FINISH_STR in line:
                simulation_finished = True
                simulation_finished_idx = idx
                break

        if simulation_finished_idx == -1:
            print("[BackendSimulator] Treid to parsing wrong formated output file!")
            return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time

        total_stat_lines = lines[simulation_finished_idx:]

        for line in total_stat_lines:
            # Parse core metrics (MatMul active cycle, Vector active cycle, etc.)
            if 'Core' in line:
                if 'MatMul active cycle' in line:
                    matmul_cycle = re.search(r'MatMul active cycle (\d+)', line).group(1)
                    vector_cycle = re.search(r'Vector active cycle (\d+)', line).group(1)
                    core_metrics['MatMul_active_cycle'] = int(matmul_cycle)
                    core_metrics['Vector_active_cycle'] = int(vector_cycle)
                elif 'Systolic Array Utilization' in line:
                    systolic_util = re.search(r'Systolic Array Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    vector_util = re.search(r'Vector Unit Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    total_cycle = re.search(r'Total cycle: (\d+)', line).group(1)
                    core_metrics['Systolic_Array_Utilization'] = float(systolic_util)
                    core_metrics['Vector_Unit_Utilization'] = float(vector_util)
                    core_metrics['Total_cycle'] = int(total_cycle)

            # Parse DRAM channel bandwidth utilization
            if 'DRAM CH' in line:
                channel = re.search(r'DRAM CH\[(\d+)\]', line).group(1)
                bw_util = re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1)
                dram_channel_bw[f'CH[{channel}]'] = float(bw_util)

            # Parse average DRAM bandwidth
            if 'DRAM: AVG BW Util' in line:
                avg_dram_bw = float(re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1))

            # Parse total simulation time
            if 'Simulation time' in line:
                simulation_time = float(re.search(r'Simulation time: (\d+\.?\d*) seconds', line).group(1))
        return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time

if __name__ == "__main__":
    sim = BackendSimulator("/workspace/PyTorchSim/PyTorchSimBackend", "/workspace/PyTorchSim/PyTorchSimBackend/configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json")
    sim.interactive_simulation()
    sim.until(4000)