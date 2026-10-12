"""TOGSim as a process: started once, fed kernels over a FIFO, read back from its log.

Also the multi-tenant entry points built on it, and the Poisson arrivals that drive them.
"""
import os
import numpy as np
import shlex
import subprocess
import re
import sys
import yaml
import time
import datetime
import threading
from pathlib import Path
import uuid

from pytorchsim import config

# Configure logger for Simulator module
logger = config.setup_logger()
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


class TOGSimulator():
    TOGSIM_RESULT_PATH_KEY = "TOGSIM_RESULT_PATH"
    FINISH_STR = "Simulation finished"
    def __init__(self, config_path=None, togsim_path=None) -> None:
        if config_path is None:
            config_path = config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = config.CONFIG_TOGSIM_DIR

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

    def _start_process(self):
        cmd = f"{self.get_togsim_command(self.config_path, self.base_dir)} --models_list {self.trace_file_path}"
        if config.CONFIG_TOGSIM_DEBUG_LEVEL:
            cmd += f" --log_level {config.CONFIG_TOGSIM_DEBUG_LEVEL}"

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
            tog_path: kernel-dir handle; TOGSim derives tog.so/tog_cycles.tsv from
                its directory - empty for DEVICE_SYNC
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

    def shutdown(self):
        """Close the FIFO, wait for TOGSim to exit, and write its log and trace.

        Assumes every queued kernel has already been sent; draining the stream
        belongs to whoever owns it.
        """
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
                result_path = config.CONFIG_TORCHSIM_LOG_PATH
                os.makedirs(result_path, exist_ok=True)
                file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".log"
                result_path = os.path.join(result_path, file_name)
                with open(result_path, "w") as f:
                    f.write(stdout_output)
                logger.info(f'[TOGSim] Simulation log is stored to "{result_path}"')
            self.process = None

        # Save trace_log with same name but .trace extension
        if self.trace_log:
            result_path = config.CONFIG_TORCHSIM_LOG_PATH
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
            tog_path: kernel-dir handle; TOGSim derives tog.so from its directory
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
        """The Simulator of the TOGSim checkout at `togsim_path` (TOGSIM_DIR's by default), on `config_path`."""
        if togsim_path is None:
            togsim_path = config.CONFIG_TOGSIM_DIR
        bin = config.togsim_simulator(togsim_path)
        if not os.path.isfile(bin):
            raise FileNotFoundError(
                f"{bin} not found: TOGSIM_DIR must name a TOGSim checkout that has been built "
                f"(the compiler's setup/toolchain.sh togsim builds the one it carries)")
        config_file = os.path.join(togsim_path, config_path)
        cmd = f"{bin} --config {config_file}"
        return cmd

    @staticmethod
    def run_standalone(
        model_path,
        attribute_path="",
        autotune_mode=False,
        config_path=None,
        togsim_path=None,
        timeout_sec=None,
    ):
        """
        Run a single kernel simulation in standalone mode.
        This method starts a new TOGSim process, runs the kernel, and waits for completion.
        For streaming multiple kernels, use launch_kernel() instead.

        Args:
            model_path: kernel-dir handle; tog.so/tog_cycles.tsv are derived from
                its directory
            attribute_path: Path to attribute file
            autotune_mode: If True, run in autotune mode (silent)
            config_path: Path to TOGSim config file (required)
            togsim_path: TOGSim repository root (optional, defaults to config.CONFIG_TOGSIM_DIR)
            timeout_sec: If set, terminate the Simulator subprocess after this many seconds
                (autotune uses this to skip very slow tile candidates).

        Returns:
            Path to the simulation result log file

        Raises:
            RuntimeError: the Simulator timed out, or exited non-zero (a kernel it refuses);
                the message carries its exit status and the last line it printed
        """
        if config_path is None:
            config_path = config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = config.CONFIG_TOGSIM_DIR

        # Create result path with appropriate filename
        if autotune_mode:
            base_dir = Path(model_path).parent / "togsim_result"
        else:
            base_dir = Path(config.CONFIG_TORCHSIM_LOG_PATH)

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
            tog_so = os.path.join(os.path.dirname(str(model_path)), "tog.so")
            cycle_tsv = os.path.join(os.path.dirname(str(model_path)), "tog_cycles.tsv")
            if not os.path.exists(tog_so):
                raise FileNotFoundError(f"{tog_so} not found -- the kernel has no TOG to simulate")
            base_cmd = TOGSimulator.get_togsim_command(config_path, togsim_path)
            unit_tsv = os.path.join(os.path.dirname(str(model_path)), "tog_units.tsv")
            cmd = f"{base_cmd} --tog_so {tog_so} --cycle_table {cycle_tsv} --unit_table {unit_tsv}"
            if config.CONFIG_TOGSIM_DEBUG_LEVEL:
                cmd += f" --log_level {config.CONFIG_TOGSIM_DEBUG_LEVEL}"

            if not autotune_mode:
                logger.debug(f"[TOGSim] cmd> {cmd}")
                logger.info("[TOGSim] TOGSim simulation started")
            with ProgressBar("[TOGSim] Running simulation", silent_mode=autotune_mode):
                completed = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True,
                    check=True,
                    timeout=timeout_sec,
                )
                result = completed.stdout
        except subprocess.TimeoutExpired as e:
            logger.warning(
                "[TOGSim] Simulator subprocess exceeded timeout (%.1f s); terminating.",
                float(timeout_sec) if timeout_sec is not None else -1.0,
            )
            raise RuntimeError("TOGSim subprocess timeout") from e
        except subprocess.CalledProcessError as e:
            said = (e.stdout + e.stderr).decode(errors="replace").strip().splitlines()
            if not autotune_mode:
                logger.error("[TOGSim] Simulator exited with status %d:\n%s", e.returncode, "\n".join(said))
            raise RuntimeError(f"TOGSim Simulator exited with status {e.returncode}: "
                               f"{said[-1] if said else 'it printed nothing'}") from e

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

    _UNIT_LINE = re.compile(r"\] (Core \[\d+\]|Total) : (.+?) [Uu]tilization\(%\): (\d+\.?\d*), "
                            r"active[ _]cycles?: (\d+), idle[ _]cycles?: (\d+)")
    _DRAM_CHANNEL = re.compile(r"\[DRAM\] channel (\d+) \| .*?(\d+\.?\d*)% of utilization")
    _DRAM_COMBINED = re.compile(r"\[DRAM\] channels .* combined \| .*?(\d+\.?\d*)% of utilization")
    _ARRAY_UNITS = ("Systolic", "Msa")

    @staticmethod
    def get_result_from_file(result_path):
        """(core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle) of a run's log.
        core_metrics["units"] holds every 'Core [i]' and 'Total' unit row; the flat keys read the
        array and the vector unit off the Total row, else the one core."""
        core_metrics = {}
        dram_channel_bw = {}
        avg_dram_bw = 0.0
        simulation_time = float("inf")
        total_cycle = float("inf")

        with open(result_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if TOGSimulator.FINISH_STR in line:
                break
        else:
            logger.warning(f"[TOGSim] Warning: Unable to parse the output file ({result_path}). The file may be improperly formatted.")
            return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle

        units = {}
        combined = None
        for line in lines[idx:]:
            m = TOGSimulator._UNIT_LINE.search(line)
            if m:
                units.setdefault(m.group(1), {})[m.group(2)] = {
                    "utilization": float(m.group(3)), "active_cycles": int(m.group(4)),
                    "idle_cycles": int(m.group(5))}
                continue
            m = TOGSimulator._DRAM_CHANNEL.search(line)
            if m:
                dram_channel_bw[f"CH[{m.group(1)}]"] = float(m.group(2))
                continue
            m = TOGSimulator._DRAM_COMBINED.search(line)
            if m:
                combined = float(m.group(1))
            elif 'Total execution cycles' in line:
                total_cycle = int(re.search(r'Total execution cycles: (\d+)', line).group(1))
            elif 'Wall-clock time for simulation' in line:
                simulation_time = float(re.search(r'Wall-clock time for simulation: (\d+\.?\d*) seconds', line).group(1))

        if combined is not None:
            avg_dram_bw = combined
        elif dram_channel_bw:
            avg_dram_bw = sum(dram_channel_bw.values()) / len(dram_channel_bw)

        if units:
            core_metrics["units"] = units
            whole = units.get("Total") or next(iter(units.values()))
            table = [whole[name] for name in TOGSimulator._ARRAY_UNITS if name in whole]
            arrays = ([max(table, key=lambda u: u["active_cycles"])] if table else
                      [u for name, u in whole.items() if name.startswith("Systolic array [")])
            vector = whole.get("Vpu") or whole.get("Vector unit")
            if arrays:
                core_metrics["Systolic_Array_Utilization"] = sum(u["utilization"] for u in arrays) / len(arrays)
                core_metrics["MatMul_active_cycle"] = sum(u["active_cycles"] for u in arrays)
            if vector:
                core_metrics["Vector_Unit_Utilization"] = vector["utilization"]
                core_metrics["Vector_active_cycle"] = vector["active_cycles"]
            if total_cycle != float("inf"):
                core_metrics["Total_cycle"] = total_cycle
        return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle


def poisson_request_generator(lambda_requests, max_msec_time=None):
    """Yield synthetic arrival times in milliseconds (first sample is 0)."""
    current_time = 0.0  # msec

    yield 0
    while max_msec_time is None or current_time < max_msec_time:
        inter_arrival_time = np.random.exponential(scale=1000 / lambda_requests)
        current_time += inter_arrival_time

        if max_msec_time is not None and current_time > max_msec_time:
            break

        yield current_time

if __name__ == "__main__":
    """Drive the backend with no PyTorch in the process, which is the point of it."""
    import argparse

    ap = argparse.ArgumentParser(description="Send kernels to TOGSim directly.")
    ap.add_argument("kernel_dir", help="directory holding tog.so and tog_cycles.tsv")
    ap.add_argument("attribute", help="kernel attribute file")
    ap.add_argument("--config", default=None, help="TOGSim hardware YAML")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    sim = TOGSimulator(config_path=args.config)
    for _ in range(args.repeat):
        sim.launch_kernel(0, 0, args.kernel_dir, args.attribute)
    sim.device_synchronize(0)
    sim.shutdown()
    print(sim.trace_log)
