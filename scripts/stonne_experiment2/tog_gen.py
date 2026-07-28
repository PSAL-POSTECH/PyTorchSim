import os
import sys
import re
import glob
from collections import defaultdict
sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from AsmParser.tog_generator import tog_generator
from Simulator.simulator import TOGSimulator
from PyTorchSimFrontend import extension_config

def extract_simulation_stats(result_path):
    with open(result_path, "r") as f:
        lines = f.readlines()[-4:]

    nr_multiplications = None
    total_cycle = None
    sim_time = None

    for line in lines:
        if "nr_multiplications" in line:
            nr_multiplications = line.strip().split(":")[-1].strip()
        elif "Total execution cycles" in line:
            total_cycle = line.strip().split(":")[-1].strip()
        elif "Wall-clock time for simulation" in line:
            sim_time = line.strip().split(":")[-1].replace("seconds", "").strip()
    return nr_multiplications, total_cycle, sim_time

if __name__ == "__main__":
    base_dir = "/home/workspace/stonneResult"
    trace_mode_paths = []
    perf_mode_paths = []
    for root, dirs, files in os.walk(base_dir):
        if "raw_tog.py" in files:
            raw_tog_path = os.path.join(root, "raw_tog.py")
            tog_path = os.path.join(root, "tile_graph.onnx")
            if not os.path.exists(tog_path):
                tile_graph_generator = tog_generator([root])
                tile_graph_generator.load_file(raw_tog_path)
                tile_graph_generator.generate_tile_graph(
                    tog_path,
                    cycle_list=[0],
                    x_offset=0,
                    w_offset=0,
                    vector_lane=0,
                    stonneGraph=True
                )
                print(f"TOG genereted at {tog_path}")
            rel_depth = os.path.relpath(root, base_dir).count(os.sep)
            if rel_depth == 0:
                trace_mode_paths.append(root)
            else:
                perf_mode_paths.append(root)
    cycle_list = {}
    simul_list = defaultdict(list)
    for path in perf_mode_paths:
        parent = os.path.dirname(path)
        counter_files = glob.glob(os.path.join(path, "*.counters"))
        for counter_file in counter_files:
            with open(counter_file, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                if first_line.startswith("CYCLES="):
                    cycle = int(first_line.split("=")[1])
                    cycle_list[parent] = cycle
                if second_line.startswith("Simulation time="):
                    match = re.search(r'Simulation time=([0-9.]+)', second_line)
                    simul_list[parent].append(float(match.group(1)))

    print("\n=== Run TLS simulation ===")
    for path in trace_mode_paths:
        if "outerPro" in path:
            continue
        tog_path = os.path.join(path, "tile_graph.onnx")
        stonne_config_path = f'{extension_config.CONFIG_TORCHSIM_DIR}/configs/stonne_validation_c1_simple_noc.yml'
        result_path = TOGSimulator.run_standalone(tog_path, config_path=stonne_config_path)
        nr_multiplications, total_cycle, sim_time = extract_simulation_stats(result_path)
        sim_time, total_cycle = float(sim_time), int(total_cycle)
        print(f"[TLS] Cycle={total_cycle} Sim time={sim_time} nr_multiplications={nr_multiplications}")
        avg_simul = sum(simul_list[path]) / len(simul_list[path])
        print(f"[ILS] Cycle={cycle_list[path]} Sim time= {avg_simul} at {path}")
        speedup = avg_simul / sim_time if avg_simul != 0 else float('inf')
        error_rate = abs(cycle_list[path] - int(total_cycle)) / total_cycle if total_cycle != 0 else float('inf')
        print(f"[EVAL] Speedup={speedup:.3f}x Error rate={error_rate:.4%}")