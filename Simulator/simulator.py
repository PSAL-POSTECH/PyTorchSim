import os
import shlex
import ctypes
import subprocess
import re
import sys
import yaml
import time
import datetime
import threading
from pathlib import Path
import uuid

import torch
import numpy as np

from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend import extension_config

# Configure logger for Simulator module
logger = extension_config.setup_logger()
from tqdm import tqdm


class ProgressBar:
    def __init__(self, desc, silent_mode=False, update_interval=0.5):
        self.desc = desc
        self.silent_mode = silent_mode
        self.update_interval = update_interval
        self.pbar = None
        self.finished = False
        self.progress_thread = None

    def __enter__(self):
        if not self.silent_mode:
            self.pbar = tqdm(
                desc=self.desc,
                bar_format='{desc}: {elapsed}',
                leave=False,  # Don't leave the bar when done (it will disappear)
                ncols=80,
                disable=False,
                total=100,  # Use a total for smooth animation
            )
            # Update progress bar in a separate thread
            def update_progress():
                while not self.finished:
                    self.pbar.update(1)
                    time.sleep(self.update_interval)

            self.progress_thread = threading.Thread(target=update_progress, daemon=True)
            self.progress_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished = True
        if not self.silent_mode and self.pbar is not None:
            self.pbar.close()
        return False


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
    torch.float16: np.float16,
}

class FunctionalSimulator():
    def __init__(self, path, key):
        self.path = path
        self.key = key

    def load_tensor(self, arg, arg_name, arg_attribute, path):
        # path = os.path.join(dump_path, arg_name, f'{n_call}.raw')
        with open(path, 'rb') as f:
            np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[arg.dtype])
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
            tensor = arg.cpu().detach()
            buffer_size = tensor.untyped_storage().size()
            buffer = (ctypes.c_char * buffer_size).from_address(tensor.data_ptr())
            t_arr = np.frombuffer(buffer, dtype=TORCH_TO_NUMPY[tensor.dtype], count=buffer_size // tensor.element_size())
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
            if MLIRKernelArgs.is_mlir_arg_in(arg_attribute[0]):
                index = self.write_arg(arg, load_path, arg_name)
                file_path.append(os.path.join(load_path, arg_name, f'{index}.raw'))
            elif MLIRKernelArgs.is_mlir_arg_out(arg_attribute[0]):
                path = os.path.join(dump_path, arg_name)
                os.makedirs(path, exist_ok=True)
                file_path.append(os.path.join(path, f'{self.get_biggest_filename(path)}.raw'))

        return array_size, file_path

    def run_spike(self, args, arg_attributes, runtime_path, binary, vectorlane_size=4, spad_info=None, cleanup=False, silent_mode=False):
        load_path = runtime_path
        dump_path = runtime_path

        target_binary = os.path.join(self.path, binary)
        objdump = f"riscv64-unknown-elf-objdump -d {target_binary} > {os.path.join(self.path, 'binary.dump')}"
        kernel_start = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}'"
        kernel_end = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}' | xargs -I {{}} awk '/{{}}/,0' {os.path.join(self.path, 'binary.dump')} | grep ret | awk 'NR==1 {{print $1}}' | awk '{{gsub(/:$/, \"\"); print}}'"

        subprocess.run(objdump, shell=True)
        kernel_start_addr = subprocess.run(kernel_start, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')
        kernel_end_addr = subprocess.run(kernel_end, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')

        _, file_path = self.dump_args(args, arg_attributes, load_path, dump_path)
        file_path_str = ' '.join(file_path)

        # Set hardware information
        spad_option = f"-m0x{0x80000000:x}:0x{100<<30:x},0x{spad_info['spad_paddr']:x}:0x{spad_info['spad_size']*vectorlane_size:x} " + \
            f"--scratchpad-base-paddr={spad_info['spad_paddr']} " + \
            f"--scratchpad-base-vaddr={spad_info['spad_vaddr']} " + \
            f"--scratchpad-size={spad_info['spad_size']} "
        vectorlane_option = f"--vectorlane-size={vectorlane_size}"
        kernel_address = f"--kernel-addr={kernel_start_addr}:{kernel_end_addr}"
        base_path= f"--base-path={runtime_path}"
        os.makedirs(os.path.join(runtime_path, "indirect_access"), exist_ok=True)
        os.makedirs(os.path.join(runtime_path, "dma_access"), exist_ok=True)
        run = f'spike --isa rv64gcv_zfh --varch=vlen:256,elen:64 {vectorlane_option} {spad_option} {kernel_address} {base_path} /workspace/riscv-pk/build/pk {target_binary} {file_path_str}'
        if not silent_mode:
            logger.debug(f"[Spike] cmd> {run}")
            logger.info("[Spike] Running Spike simulator")
        run_cmd = shlex.split(run)
        try:
            stdout_setting = subprocess.DEVNULL if silent_mode else None
            stderr_setting = subprocess.DEVNULL if silent_mode else None
            with ProgressBar("[Spike] Running simulation", silent_mode=silent_mode):
                subprocess.check_call(run_cmd, stdout=stdout_setting, stderr=stderr_setting)
        except subprocess.CalledProcessError as e:
            if not silent_mode:
                logger.error(f"[Spike] Command failed with exit code {e.returncode}")
            error_msg = ""
            if e.returncode == 200:
                error_msg = "INVALID_SPAD_ACCESS"
            elif e.returncode == 201:
                error_msg = "STACK_OVERFLOW"
            else:
                error_msg = "UNKNOWN_ERROR"
            raise RuntimeError(f"{error_msg}")

        for (arg_name, arg_attribute), arg, path in zip(arg_attributes, args, file_path):
            if MLIRKernelArgs.is_mlir_arg_out(arg_attribute[0]):
                self.load_tensor(arg, arg_name, arg_attribute, path)

        if cleanup:
            for path in file_path:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def get_runtime_dump_path(base_path, prefix="runtime", zfill=4):
        indices = [
            int(match.group(1))
            for d in os.listdir(base_path)
            if (match := re.fullmatch(rf"{prefix}_(\d{{{zfill}}})", d))
        ]

        max_index = max(indices, default=-1)
        next_index = max_index + 1
        folder_name = f"{prefix}_{str(next_index).zfill(zfill)}"
        full_path = os.path.join(base_path, folder_name)

        os.makedirs(full_path)
        return full_path

class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, vectorlane_size, silent_mode=False):
        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        gem5_script_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "gem5_script/script_systolic.py")
        gem5_cmd = [extension_config.CONFIG_GEM5_PATH, "-r", "--stdout-file=sto.log", "-d", dir_path, gem5_script_path, "-c", target_binary, "--vlane", str(vectorlane_size)]

        if not silent_mode:
            logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
            logger.info("[Gem5] Gem5 simulation started")

        try:
            #with ProgressBar("[Gem5] Running simulation", silent_mode=is_dryrun):
            output = subprocess.check_output(gem5_cmd, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            output_error = e.output.decode() if isinstance(e.output, bytes) else str(e.output)
            logger.debug(f"[Gem5] Gem5 simulation failed with error: \"{output_error}\"")
            raise RuntimeError(f"Gem5 Simulation Failed: \"{output_error}\"")

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        cycle_list = cycle_list[:-1]
        return cycle_list

class TOGSimulator():
    TOGSIM_RESULT_PATH_KEY = "TOGSIM_RESULT_PATH"
    FINISH_STR = "Simulation finished"
    ALLOC_POOL = dict() # For eagermode buffer plan
    def __init__(self, config_path=None, togsim_path=None) -> None:
        if config_path is None:
            config_path = extension_config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")

        self.base_dir = togsim_path
        self.config_path = config_path
        self.config_yaml = self.load_yaml(self.config_path)
        self.process = None
        self._next_kernel_id = 0  # Auto-incrementing kernel ID

        # Create FIFOs for command and event communication
        self.fifo_dir = os.path.join("/tmp", f"togsim_fifo_{os.getpid()}")
        os.makedirs(self.fifo_dir, exist_ok=True)
        self.trace_file_path = os.path.join(self.fifo_dir, "cmd_fifo")
        self.trace_log = "# command_type, kernel_id, device_index, stream_index, tog_path, attribute_path, timestamp\n"

        # Create FIFOs if they don't exist
        if os.path.exists(self.trace_file_path):
            os.remove(self.trace_file_path)
        os.mkfifo(self.trace_file_path)

        # Start TOGSim process
        self._start_process()

        # Open trace file FIFO once and keep it open (after process starts)
        self._trace_file_lock = threading.Lock()
        try:
            self._trace_file_handle = open(self.trace_file_path, 'w')
        except IOError as e:
            logger.error(f"[TOGSim] Failed to open trace file: {e}")
            raise RuntimeError(f"Failed to open trace file: {e}")

    def __enter__(self):
        """Context manager entry."""
        # Set this simulator instance as the global TOGSimulator
        self.old_tog_simulator = torch.npu.get_tog_simulator()
        torch.npu.set_tog_simulator(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup."""
        # Reset global TOGSimulator to None
        self.until()
        torch.npu.set_tog_simulator(self.old_tog_simulator)

    def _start_process(self):
        cmd = f"{self.get_togsim_command(self.config_path, self.base_dir)} --models_list {self.trace_file_path}"
        if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
            cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

        logger.debug(f"[TOGSim] cmd> {cmd}")
        if self.process is None:
            self.process = subprocess.Popen(
                shlex.split(cmd),
                #stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            logger.warning("[TOGSim] Simulator is already running.")

    def _cleanup_fifos(self):
        """Clean up FIFO files"""
        try:
            if os.path.exists(self.trace_file_path):
                os.remove(self.trace_file_path)
            if os.path.exists(self.fifo_dir):
                os.rmdir(self.fifo_dir)
        except OSError as e:
            logger.warning(f"[TOGSim] Failed to clean up FIFOs: {e}")

    def _send_command(self, command_type, device_index, stream_index, tog_path="", attribute_path="", timestamp=0):
        """
        Internal method to send a command to TOGSim via FIFO.

        Args:
            command_type: Type of command ("LAUNCH_KERNEL" or "DEVICE_SYNC")
            device_index: Device index
            stream_index: Stream index
            tog_path: Path to TOG file (ONNX model) - empty for DEVICE_SYNC
            attribute_path: Path to attribute file - empty for DEVICE_SYNC
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The kernel ID assigned to this command
        """
        if self.process is None:
            raise RuntimeError("[TOGSim] Simulator process is not running")

        if self.process.poll() is not None:
            raise RuntimeError("[TOGSim] Simulator process has terminated")

        # Get and increment kernel ID
        kernel_id = self._next_kernel_id
        self._next_kernel_id += 1

        # Format command: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
        command = f"{command_type},{kernel_id},{device_index},{stream_index},{tog_path},{attribute_path},{timestamp}"

        with self._trace_file_lock:
            # Write command to TOGSim
            try:
                self._trace_file_handle.write(command + '\n')
                self._trace_file_handle.flush()
                self.trace_log += command + '\n'
                logger.debug(f"[TOGSim] Sent command: {command}")
            except IOError as e:
                logger.error(f"[TOGSim] Failed to write to trace file: {e}")
                raise RuntimeError(f"Failed to send command to TOGSim: {e}")
        return kernel_id

    def until(self):
        # Make sure that all kernels in the stream are finished
        torch.npu.synchronize()

        # Close trace file handle if open
        if self._trace_file_handle is not None:
            try:
                self._trace_file_handle.close()
            except:
                pass
            self._trace_file_handle = None

        if self.process:
            self.process.wait()

            # Read output streams
            stdout_output = ""
            stderr_output = ""
            if self.process.stdout:
                stdout_output = self.process.stdout.read()
            if self.process.stderr:
                stderr_output = self.process.stderr.read()

            # Print stderr immediately if there's any error output
            if stderr_output:
                sys.stderr.write(stderr_output)
                sys.stderr.flush()

            # Save stdout to result file
            if stdout_output:
                result_path = extension_config.CONFIG_TORCHSIM_LOG_PATH
                os.makedirs(result_path, exist_ok=True)
                file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".log"
                result_path = os.path.join(result_path, file_name)
                with open(result_path, "w") as f:
                    f.write(stdout_output)
                logger.info(f'[TOGSim] Simulation log is stored to "{result_path}"')
            self.process = None

        # Save trace_log with same name but .trace extension
        if self.trace_log:
            result_path = extension_config.CONFIG_TORCHSIM_LOG_PATH
            os.makedirs(result_path, exist_ok=True)
            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".trace"
            trace_path = os.path.join(result_path, file_name)
            with open(trace_path, "w") as f:
                f.write(self.trace_log)
            logger.info(f'[TOGSim] Trace log is stored to "{trace_path}"')

        # Clean up FIFOs
        self._cleanup_fifos()

    def launch_kernel(self, device_index, stream_index, tog_path, attribute_path, timestamp=0):
        """
        Launch a kernel via FIFO communication.

        Args:
            device_index: Device index
            stream_index: Stream index
            tog_path: Path to TOG file (ONNX model)
            attribute_path: Path to attribute file
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The kernel ID assigned to this launch
        """
        return self._send_command("LAUNCH_KERNEL", device_index, stream_index, tog_path, attribute_path, timestamp)

    def device_synchronize(self, device_index):
        """
        Synchronize all streams on a device via FIFO communication.

        Args:
            device_index: Device index to synchronize
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The command ID assigned to this synchronization
        """
        # For device_synchronize, stream_index is not meaningful, use 0
        return self._send_command("DEVICE_SYNC", device_index, 0, "", "", 0)

    @classmethod
    def sram_alloc(cls, buf_name, addr_range):
        cls.ALLOC_POOL[buf_name] = addr_range

    @classmethod
    def sram_dealloc(cls, buf_name, addr_range):
        if buf_name in cls.ALLOC_POOL:
            del cls.ALLOC_POOL[buf_name]

    def create_attribute_file(self, attribute_path, inputs, **kwargs):
        address_info = {}
        sram_buffer = {}
        yaml_content = {}

        os.makedirs(attribute_path, exist_ok=True)
        index = str(len(os.listdir(attribute_path)))
        attribute_path = os.path.join(attribute_path, index)

        for idx, tensor in enumerate(inputs):
            address_info[f"arg{idx}"] = tensor.data_ptr()
        yaml_content["address_info"] = address_info

        for buf_name, range in self.ALLOC_POOL.items():
            sram_buffer[buf_name] = range
        yaml_content["sram_alloc"] = sram_buffer

        with open(attribute_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            f.flush()
            os.fsync(f.fileno()) # There could be a race condition.
        return attribute_path

    def load_yaml(self, config_path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"YAML file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                data = yaml.safe_load(file)
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

    def get_core_freq(self):
        if "core_freq_mhz" in self.config_yaml:
            return self.config_yaml["core_freq_mhz"] * 1000 * 1000 # MHz
        else:
            raise KeyError("Key 'core_freq' not found in JSON.")

    @staticmethod
    def get_togsim_command(config_path, togsim_path=None):
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")
        bin = os.path.join(togsim_path, "build/bin/Simulator")
        config = os.path.join(togsim_path, config_path)
        cmd = f"{bin} --config {config}"
        return cmd

    @staticmethod
    def run_standalone(model_path, attribute_path="", autotune_mode=False, config_path=None, togsim_path=None):
        """
        Run a single kernel simulation in standalone mode.
        This method starts a new TOGSim process, runs the kernel, and waits for completion.
        For streaming multiple kernels, use launch_kernel() instead.

        Args:
            model_path: Path to TOG file (ONNX model)
            attribute_path: Path to attribute file
            autotune_mode: If True, run in autotune mode (silent)
            config_path: Path to TOGSim config file (required)
            togsim_path: Path to TOGSim directory (optional, defaults to CONFIG_TORCHSIM_DIR/TOGSim)

        Returns:
            Path to the simulation result log file
        """
        if config_path is None:
            config_path = extension_config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")

        # Create result path with appropriate filename
        if autotune_mode:
            base_dir = Path(model_path).parent / "togsim_result"
        else:
            base_dir = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)

        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        result_path = base_dir / f"{file_name}.log"
        trace_file_path = base_dir / f"{file_name}.trace"

        # Create trace file in result directory
        kernel_id, device_index, stream_index, timestamp = 0, 0, 0, 0
        command = f"LAUNCH_KERNEL,{kernel_id},{device_index},{stream_index},{model_path},{attribute_path},{timestamp}\n"
        with open(trace_file_path, 'w') as trace_file:
            trace_file.write(command)
            trace_file.flush()
            os.fsync(trace_file.fileno())

        try:
            cmd = f"{TOGSimulator.get_togsim_command(config_path, togsim_path)} --models_list {trace_file_path}"
            if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
                cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

            if not autotune_mode:
                logger.debug(f"[TOGSim] cmd> {cmd}")
                logger.info("[TOGSim] TOGSim simulation started")
            with ProgressBar("[TOGSim] Running simulation", silent_mode=autotune_mode):
                result = subprocess.check_output(shlex.split(cmd))
        except subprocess.CalledProcessError as e:
            logger.error(f"[TOGSim] Command failed with exit code {e.returncode}")
            logger.error(f"[TOGSim] Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
            assert 0

        # Prevent race condition
        with open(result_path, "w") as f:
            f.write(result.decode())
            f.flush()
            os.fsync(f.fileno())

        if not autotune_mode:
            import logging as _logging
            model_path_log = f' of "{model_path}" ' if logger.isEnabledFor(_logging.DEBUG) else " "
            logger.info(f'[TOGSim] Simulation log{model_path_log}is stored to "{result_path}"')
        return result_path

    @staticmethod
    def get_result_from_file(result_path):
        core_metrics = {}
        dram_channel_bw = {}
        avg_dram_bw = 0.0
        simulation_time = float("inf")
        total_cycle = float("inf")

        # Read and find total stat position
        with open(result_path, "r") as f:
            lines = f.readlines()

        simulation_finished_idx = -1
        simulation_finished = False
        for idx, line in enumerate(lines):
            if TOGSimulator.FINISH_STR in line:
                simulation_finished = True
                simulation_finished_idx = idx
                break

        if simulation_finished_idx == -1:
            logger.warning(f"[TOGSim] Warning: Unable to parse the output file ({result_path}). The file may be improperly formatted.")
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

            if 'Total execution cycles' in line:
                total_cycle = int(re.search(r'Total execution cycles: (\d+)', line).group(1))

            # Parse total simulation time
            if 'Wall-clock time for simulation' in line:
                simulation_time = float(re.search(r'Wall-clock time for simulation: (\d+\.?\d*) seconds', line).group(1))
        return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle

if __name__ == "__main__":
    # Example paths (adjust these to your actual test files)
    test_tog_path = "/workspace/PyTorchSim/outputs/6vxl6mwzhfl/tile_graph.onnx"
    test_attribute_path = "/workspace/PyTorchSim/outputs/6vxl6mwzhfl/runtime_0001/attribute/0"

    # Test: Launch multiple kernels
    sim = TOGSimulator(config_path="/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
    with sim:
        try:
            id1 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id2 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id3 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
        except Exception as e:
            print(f"Error during kernel launch: {e}")

        try:
            id2 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id1 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id3 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
        except Exception as e:
            print(f"Error during kernel launch: {e}")
    print(sim.trace_log)