"""TOGSim as a process: started once, fed kernels over a FIFO, read back from the result file it writes."""
import datetime
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

import yaml
from tqdm import tqdm

from pytorchsim import config

logger = config.setup_logger()



class Cut(RuntimeError):
    """A standalone run its caller stopped; the reason is the message."""


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
                leave=False,
                ncols=80,
                disable=False,
                total=100,
            )
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


class TOGSimProcess():
    def __init__(self, config_path=None, togsim_path=None) -> None:
        if config_path is None:
            config_path = config.CONFIG_MACHINE_FILE
        if togsim_path is None:
            togsim_path = config.CONFIG_TOGSIM_DIR

        self.base_dir = togsim_path
        self.config_path = config_path
        self.config_yaml = self.load_yaml(self.config_path)
        self.process = None
        self._next_kernel_id = 0
        os.makedirs(config.CONFIG_TORCHSIM_LOG_PATH, exist_ok=True)
        stem = os.path.join(config.CONFIG_TORCHSIM_LOG_PATH, _stamp())
        self.log_path, self.trace_path, self.result_path = stem + ".log", stem + ".trace", stem + ".json"
        self._log_handle = None

        self.fifo_dir = os.path.join("/tmp", f"togsim_fifo_{os.getpid()}")
        os.makedirs(self.fifo_dir, exist_ok=True)
        self.trace_file_path = os.path.join(self.fifo_dir, "cmd_fifo")
        self.trace_log = "# command_type, kernel_id, device_index, stream_index, tog_path, attribute_path, timestamp\n"

        if os.path.exists(self.trace_file_path):
            os.remove(self.trace_file_path)
        os.mkfifo(self.trace_file_path)

        self._start_process()

        self._trace_file_lock = threading.Lock()
        try:
            self._trace_file_handle = open(self.trace_file_path, 'w')
        except IOError as e:
            logger.error(f"[TOGSim] Failed to open trace file: {e}")
            raise RuntimeError(f"Failed to open trace file: {e}")

    def _start_process(self):
        cmd = (f"{self.get_togsim_command(self.config_path, self.base_dir)} --models_list {self.trace_file_path}"
               f" --result {self.result_path}")
        if config.CONFIG_TOGSIM_DEBUG_LEVEL:
            cmd += f" --log_level {config.CONFIG_TOGSIM_DEBUG_LEVEL}"

        logger.debug(f"[TOGSim] cmd> {cmd}")
        if self.process is None:
            self._log_handle = open(self.log_path, "w")
            self.process = subprocess.Popen(shlex.split(cmd), stdout=self._log_handle, universal_newlines=True)
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
        """Write one command (LAUNCH_KERNEL or DEVICE_SYNC) to TOGSim's FIFO. Returns its kernel id.
        tog_path is the kernel-dir handle TOGSim derives tog.so from; both paths are empty for DEVICE_SYNC."""
        if self.process is None:
            raise RuntimeError("[TOGSim] Simulator process is not running")

        if self.process.poll() is not None:
            raise RuntimeError("[TOGSim] Simulator process has terminated")

        kernel_id = self._next_kernel_id
        self._next_kernel_id += 1

        command = f"{command_type},{kernel_id},{device_index},{stream_index},{tog_path},{attribute_path},{timestamp}"

        with self._trace_file_lock:
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
        """Close the FIFO, wait for TOGSim to exit, and write its log and trace; a Simulator that
        exited non-zero (a kernel it refuses ends the session) is a RuntimeError raised after that.
        Assumes every queued kernel has already been sent: draining the stream is its owner's."""
        if self._trace_file_handle is not None:
            try:
                self._trace_file_handle.close()
            except BrokenPipeError:
                pass
            self._trace_file_handle = None

        status = 0
        if self.process:
            status = self.process.wait()
            self._log_handle.close()
            logger.info(f'[TOGSim] Simulation log is stored to "{self.log_path}"')
            if not status:
                logger.info(f'[TOGSim] Simulation result is stored to "{self.result_path}"')
            self.process = None

        with open(self.trace_path, "w") as f:
            f.write(self.trace_log)
        logger.info(f'[TOGSim] Trace log is stored to "{self.trace_path}"')

        self._cleanup_fifos()
        if status:
            raise RuntimeError(f"TOGSim Simulator exited with status {status}: the session did not run "
                               f"every kernel it was sent, and the Simulator's own output says why")

    def launch_kernel(self, device_index, stream_index, tog_path, attribute_path, timestamp=0):
        """Queue one kernel on `stream_index` at `timestamp` ns. Returns its kernel id; TOGSim derives
        tog.so from tog_path's directory."""
        return self._send_command("LAUNCH_KERNEL", device_index, stream_index, tog_path, attribute_path, timestamp)

    def device_synchronize(self, device_index):
        """Queue a DEVICE_SYNC for every stream of `device_index`. Returns its command id."""
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
        cut=None,
    ):
        """Run one kernel in a TOGSim of its own and wait. Returns the path of the result file it wrote
        (--result), beside its log. `cut`, polled while it runs, kills it with a reason (raised as Cut); a
        Simulator that exits non-zero is a RuntimeError carrying its status and its last line."""
        if config_path is None:
            config_path = config.CONFIG_MACHINE_FILE
        if togsim_path is None:
            togsim_path = config.CONFIG_TOGSIM_DIR

        if autotune_mode:
            base_dir = Path(model_path).parent / "togsim_result"
        else:
            base_dir = Path(config.CONFIG_TORCHSIM_LOG_PATH)

        base_dir.mkdir(parents=True, exist_ok=True)
        file_name = _stamp()
        log_path = base_dir / f"{file_name}.log"
        result_path = base_dir / f"{file_name}.json"
        trace_file_path = base_dir / f"{file_name}.trace"

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
            base_cmd = TOGSimProcess.get_togsim_command(config_path, togsim_path)
            unit_tsv = os.path.join(os.path.dirname(str(model_path)), "tog_units.tsv")
            cmd = (f"{base_cmd} --tog_so {tog_so} --cycle_table {cycle_tsv} --unit_table {unit_tsv}"
                   f" --result {result_path}")
            if attribute_path:
                cmd += f" --attribute_path {attribute_path}"
            if config.CONFIG_TOGSIM_DEBUG_LEVEL:
                cmd += f" --log_level {config.CONFIG_TOGSIM_DEBUG_LEVEL}"

            if not autotune_mode:
                logger.debug(f"[TOGSim] cmd> {cmd}")
                logger.info("[TOGSim] TOGSim simulation started")
            with ProgressBar("[TOGSim] Running simulation", silent_mode=autotune_mode):
                proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                while True:
                    try:
                        result, err = proc.communicate(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        why = cut() if cut else None
                        if why:
                            proc.kill()
                            proc.communicate()
                            raise Cut(why) from None
                if proc.returncode:
                    raise subprocess.CalledProcessError(proc.returncode, cmd, result, err)
        except subprocess.CalledProcessError as e:
            said = (e.stdout + e.stderr).decode(errors="replace").strip().splitlines()
            if not autotune_mode:
                logger.error("[TOGSim] Simulator exited with status %d:\n%s", e.returncode, "\n".join(said))
            raise RuntimeError(f"TOGSim Simulator exited with status {e.returncode}: "
                               f"{said[-1] if said else 'it printed nothing'}") from e

        with open(log_path, "w") as f:
            f.write(result.decode())
            f.flush()
            os.fsync(f.fileno())

        if not autotune_mode:
            import logging as _logging
            model_path_log = f' of "{model_path}" ' if logger.isEnabledFor(_logging.DEBUG) else " "
            logger.info(f'[TOGSim] Simulation log{model_path_log}is stored to "{log_path}"')
        return result_path

    @staticmethod
    def read_result(result_path):
        """The results TOGSim wrote with --result: total_cycles, wall_clock_s, core_cycles, units, dma, dram
        and kernels, each as its end-of-run line prints it (TOGSim's docs/output-lines.md)."""
        with open(result_path) as f:
            return json.load(f)


def _stamp():
    """The name a run's .log, .trace and .json share: when it started, and a random suffix."""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f"_{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    """Drive the backend with no PyTorch in the process, which is the point of it."""
    import argparse

    ap = argparse.ArgumentParser(description="Send kernels to TOGSim directly.")
    ap.add_argument("kernel_dir", help="directory holding tog.so and tog_cycles.tsv")
    ap.add_argument("attribute", help="kernel attribute file")
    ap.add_argument("--config", default=None, help="TOGSim hardware YAML")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    sim = TOGSimProcess(config_path=args.config)
    for _ in range(args.repeat):
        sim.launch_kernel(0, 0, args.kernel_dir, args.attribute)
    sim.device_synchronize(0)
    sim.shutdown()
    print(sim.trace_log)
