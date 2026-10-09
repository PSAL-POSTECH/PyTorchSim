"""Gem5 as what it is here: a compile-time measurement, not a running device.

One tile binary in, its cycle count out. `timing.py` keys the cycle table by
the result; nothing in this file runs while the model does.
"""

import os
import re
import subprocess

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()


#: What gem5 says when it dies, as opposed to the libc backtrace that follows it.
_GEM5_SIGNAL = re.compile(
    r"^(?:.*\b(?:panic|fatal):.*|.*Assertion.*|Program aborted at tick.*)$", re.M)


def _gem5_failure(proc, log_path):
    """The message a failed gem5 run should raise, from the log it redirects to.

    `-r` sends stdout AND stderr to `--stdout-file`, so the pipes are empty and
    a return code on its own says nothing: gem5 panics and aborts.
    """
    how = (f"signal {-proc.returncode}" if proc.returncode < 0
           else f"exit {proc.returncode}")
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        text = (proc.stderr or "") + (proc.stdout or "")
    hits = [h.strip() for h in _GEM5_SIGNAL.findall(text)]
    if not hits:
        hits = [l.strip() for l in text.strip().splitlines()[-3:]]
    if not hits:
        hits = [f"gem5 left no diagnostic; see {log_path}"]
    return (f"Gem5 Simulation Failed ({how}); full log in {log_path}\n  "
            + "\n  ".join(h[:300] for h in hits[:3]))


class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, vectorlane_size, silent_mode=False):
        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        gem5_cmd = [extension_config.CONFIG_GEM5_PATH, "-r", "--stdout-file=sto.log", "-d", dir_path,
                    extension_config.CONFIG_GEM5_SCRIPT, "-c", target_binary,
                    "--model", extension_config.CONFIG_VCIX_MODEL, "--vlane", str(vectorlane_size),
                    "--vlen", str(extension_config.vpu_vector_length_bits)]

        if not silent_mode:
            logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
            logger.info("[Gem5] Gem5 simulation started")

        proc = subprocess.run(gem5_cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(_gem5_failure(proc, os.path.join(dir_path, "sto.log")))

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        cycle_list = cycle_list[:-1]
        return cycle_list
