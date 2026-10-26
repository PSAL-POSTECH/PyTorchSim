"""npuwattch.json from a TPUv3 launch on BookSim2 (a 256^3 bf16 matmul and its copy node), kept as data:
the counters are totalled by the TOG's rows into the v1.1 form, refreshes are counted in all-bank units,
and the file is written only when TORCHSIM_NPUWATTCH asks and grows by a kernel per launch."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import yaml

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

from pytorchsim.simulator.timing import npuwattch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "npuwattch_tpuv3")
KERNEL = os.path.join(DATA, "kernel")
MAIN = os.path.join(DATA, "togsim", "main.json")
COPY = os.path.join(DATA, "togsim", "copy.json")


def _machine():
    path = os.path.join(DATA, "machine.yml")
    with open(path) as f:
        return {"path": path, "dir": DATA, "yaml": yaml.safe_load(f)}


def _entry():
    machine = _machine()
    cfg = npuwattch.run_config(machine, MAIN)
    return cfg, npuwattch.kernel_entry(KERNEL, KERNEL, machine, cfg, MAIN, [COPY], 0)


def test_config_states_the_machine():
    cfg, _ = _entry()
    assert cfg["vpu_num_lanes"] == 128 and cfg["vpu_vector_length_bits"] == 256
    assert cfg["systolic_array"] == {"width": 128, "height": 128, "per_core": 2}
    assert cfg["dram"] == {"type": "hbm2", "channels": 16, "req_size_byte": 32, "freq_mhz": 940.0}
    assert cfg["noc"]["topology"] == "fly" and cfg["noc"]["k"] == 32 and cfg["noc"]["n"] == 1


def test_kernel_entry_totals_the_launch():
    _, e = _entry()
    assert e["operation"] == "mm" and e["has_matmul"]
    assert e["dtype"] == {"operand": "bf16", "accumulator": "fp32", "source": "mlir"}
    assert e["useful_macs"] == 256 ** 3
    assert [t["shape"] for t in e["tensors"]["inputs"]] == [[256, 256], [256, 256]]
    assert "notes" not in e
    core = e["cores"][0]
    assert sum(a["active_cycles"] for a in core["systolic"]) == 1024
    assert core["inst_count"]["movin"] == 2
    with open(MAIN) as f, open(COPY) as g:
        assert e["total_cycles"] == json.load(f)["total_cycles"] + json.load(g)["total_cycles"]


def test_matmul_classes_carry_the_data_volumes():
    _, e = _entry()
    classes = e["gem5"]["committed_inst_by_class"]
    assert classes["CustomMatMuliVpush"] == 1024 * 128 / (128 * 16)
    assert classes["CustomMatMulwVpush"] == 256 * 256 / (128 * 16)
    assert classes["CustomMatMulvpop"] == 1024 * 128 * 4 / (128 * 16 * 2)


def test_per_bank_refreshes_are_all_bank_units():
    _, e = _entry()
    with open(MAIN.replace(".json", ".log")) as f, open(COPY.replace(".json", ".log")) as g:
        texts = f.read(), g.read()
    raw = [npuwattch._launch_counters(p)["channels"][0]["num_maintenance_reqs"] for p in (MAIN, COPY)]
    assert all('"impl": "PerBank"' in t for t in texts)
    assert e["dram"]["per_channel"][0]["refresh_cmds"] == sum(raw) / 16


def test_booksim_flits_are_two_per_request():
    _, e = _entry()
    reqs = sum(c["reads"] + c["writes"] for c in e["dram"]["per_channel"])
    assert e["noc"]["injected_flits"] == 2 * reqs and e["noc"]["avg_packet_length"] == 1.0


def _record(env):
    log = tempfile.mkdtemp()
    target = SimpleNamespace(file=os.path.join(DATA, "machine.yml"))
    with mock.patch.dict(os.environ, {"TORCHSIM_LOG_PATH": log, npuwattch.ENV: env}), \
            mock.patch("pytorchsim.backend.compiler_bridge.target", return_value=target), \
            mock.patch.object(npuwattch, "RUN", npuwattch.Run()):
        npuwattch.record(KERNEL, KERNEL, MAIN, [COPY])
        npuwattch.record(KERNEL, KERNEL, MAIN, [COPY])
    return [os.path.join(log, n) for n in os.listdir(log)]


def test_off_writes_nothing():
    assert _record("0") == []


def test_on_writes_a_kernel_per_launch():
    files = _record("1")
    assert len(files) == 1 and os.path.basename(files[0]).startswith("npuwattch_")
    with open(files[0]) as f:
        doc = json.load(f)
    assert doc["schema"] == "pytorchsim.run_stats" and doc["schema_version"] == "1.1"
    assert [k["order"] for k in doc["kernels"]] == [0, 1]
