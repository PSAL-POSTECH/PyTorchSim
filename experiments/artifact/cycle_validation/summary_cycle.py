import os
import math
import csv
import re
import matplotlib.pyplot as plt
import numpy as np

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", ".")
LOG_DIR = os.path.join(TORCHSIM_DIR, "experiments/artifact/logs")
BASELINE_CSV = os.path.join(TORCHSIM_DIR, "experiments/artifact/baseline_cycle.csv")

def plot_error_bars(data: dict, filename: str):
    colors = {
        'SCALE-Sim v3': 'gold',
        'mNPUSim': 'orange',
        'Timeloop': 'green',
        'Maestro': 'violet',
        'PyTorchSim-SN': 'royalblue',
    }

    labels = list(data.keys())
    num_sims = len(colors)
    bar_width = 1
    fig, ax = plt.subplots(figsize=(48, 8))

    grouped_data = {sim: [[], []] for sim in colors}
    x_pos = []
    x_offset = 0

    for key, value in data.items():
        for i, (sim, color) in enumerate(colors.items()):
            grouped_data[sim][0].append(value[i])
            grouped_data[sim][1].append(x_offset + bar_width * i)
        x_pos.append(x_offset + bar_width * (num_sims // 2))
        x_offset += bar_width * (num_sims + 2)

    for sim, (heights, xpos) in grouped_data.items():
        bars = ax.bar(xpos, heights, width=bar_width, color=colors[sim], label=sim, edgecolor='black')
        mae_val = heights[-1]
        ax.text(
            xpos[-1],
            mae_val + 2 if mae_val >= 0 else mae_val - 6,
            f'{mae_val:.1f}%',
            ha='center',
            va='bottom' if mae_val >= 0 else 'top',
            fontsize=9,
            rotation=90
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(-100, 150)
    ax.set_yticks(np.arange(-100, 151, 50))
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()

    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def format_with_error(value, ref, error_list=None):
    try:
        if value == "" or ref == "" or float(ref) == 0:
            return "N/A", 0.0
        val = float(value)
        ref = float(ref)
        err = ((val - ref) / ref) * 100
        if error_list is not None:
            error_list.append(abs(err))
        val_str = f"{int(val):>7}"
        err_str = f"{err:+.2f}%"
        return f"{val_str} ({err_str:>8})", err
    except (ValueError, TypeError):
        return "N/A", 0.0

def compute_mae(errors):
    if not errors:
        return "N/A"
    abs_errors = [abs(err) for err in errors]
    return sum(abs_errors) / len(errors)

if __name__ == "__main__":
    # 1. Generate cycle_map
    cycle_map = {}
    for file in os.listdir(LOG_DIR):
        if file.endswith(".log"):
            full_path = os.path.join(LOG_DIR, file)
            name = file[:-4]
            with open(full_path, errors="ignore") as f:
                for line in f:
                    match = re.search(r"Total execution cycles:\s*([0-9]+)", line)
                    if match:
                        cycle_map[name] = int(match.group(1))
                        break

    # Error list init
    mnpusim_errors = []
    timeloop_errors = []
    maestro_errors = []
    scalesim_errors = []
    togsim_errors = []

    # Plot data
    plot_data ={}

    # Header
    print("[*] Summary of Total Execution Cycles with TPUv3-relative (%) Error")
    print("=" * 190)
    print(f"{'Workload':>30} {'TPUv3':>25} {'mNPUSim':>25} {'Timeloop':>25} {'Maestro':>25} {'SCALE-Sim v3':>25} {'TOGSim(Ours)':>25}")
    print("=" * 190)

    with open(BASELINE_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            workload = row["Workload"].lstrip('\ufeff')
            tpv3 = row["TPUv3"]
    
            mnpusim, mnpusim_err   = format_with_error(row["mNPUSim"], tpv3, mnpusim_errors)
            timeloop, timeloop_err = format_with_error(row["Timeloop"], tpv3, timeloop_errors)
            maestro, maestro_err   = format_with_error(row["Maestro"], tpv3, maestro_errors)
            scalesim, scalesim_err = format_with_error(row["SCALE-Sim v3"], tpv3, scalesim_errors)
    
            togsim_val = cycle_map.get(workload, "")
            if "softmax" in workload or "layernorm" in workload:
                togsim_str, togsim_err = format_with_error(str(togsim_val), tpv3, [])
            else:
                togsim_str, togsim_err = format_with_error(str(togsim_val), tpv3, togsim_errors)
            plot_data[workload] = [scalesim_err, mnpusim_err, timeloop_err, maestro_err, togsim_err]
            print(f"{workload:>30} {tpv3:>25} {mnpusim:>25} {timeloop:>25} {maestro:>25} {scalesim:>25} {togsim_str:>25}")

    # MAE row
    mae_mnpusim = compute_mae(mnpusim_errors)
    mae_timeloop = compute_mae(timeloop_errors)
    mae_maestro = compute_mae(maestro_errors)
    mae_scalesim = compute_mae(scalesim_errors)
    mae_togsim = compute_mae(togsim_errors)
    plot_data["MAE"] = [mae_scalesim, mae_mnpusim, mae_timeloop, mae_maestro, mae_togsim]
    print("=" * 190)
    print(f"{'[*] Mean Absolute Error(%)':>30} {'0.00%':>25} {mae_mnpusim:>24.2f}% {mae_timeloop:>24.2f}% {mae_maestro:>24.2f}% {mae_scalesim:>24.2f}% {mae_togsim:>24.2f}%")

    # Plot the error bars
    path = os.path.join(TORCHSIM_DIR, "experiments/artifact/cycle_validation/cycle_validation.png")
    plot_error_bars(plot_data, path)
