#!/bin/bash
set -e

LOG_DIR=$TORCHSIM_DIR/experiments/artifact/logs
CONFIG_DIR="$TORCHSIM_DIR/configs"
# CI: e.g. SKIP_ILS=1, SPEEDUP_ITERS=1 (shorter, no ILS re-runs)
: "${SKIP_ILS:=0}"
: "${SPEEDUP_ITERS:=5}"

configs=(
    "systolic_ws_128x128_c2_simple_noc_tpuv3.yml"
    "systolic_ws_128x128_c2_booksim_tpuv3.yml"
)

target_list=(
  "gemm_512x512x512"
  "gemm_1024x1024x1024"
  "gemm_2048x2048x2048"
  "conv_1x56x56x64x64x3x1x1"
  "conv_1x28x28x128x128x3x1x1"
  "conv_1x14x14x256x256x3x1x1"
  "conv_1x7x7x512x512x3x1x1"
  "resnet50"
  "bert_large"
)

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
output_dir="$TORCHSIM_DIR/experiments/artifact/speedup/results"
mkdir -p "$output_dir"

echo "[*] Scanning log files in: $LOG_DIR"
echo "[*] Extracting Simulator command and trace path from logs ([TOGSim] Run command|Command line, Trace log is stored to)"
echo ""

for log_file in "$LOG_DIR"/*.log; do
  [[ -f "$log_file" ]] || continue
  filename=$(basename "$log_file")
  workload="${filename%.log}"

  if [[ ! " ${target_list[@]} " =~ " ${workload} " ]]; then
    continue
  fi
  echo "==> Workload: $workload"

  # === Extract Simulator invocation from log (TOGSim renamed the log tag) ===
  # Legacy logs: "[TOGSim] Run command: ..."  Current TOGSim/main.cc: "[TOGSim] Command line: ..."
  base_cmd=$(grep -E "\[TOGSim\] (Run command|Command line):" "$log_file" 2>/dev/null | sed -E 's/.*\[TOGSim\] (Run command|Command line): //' | head -1)
  if [[ -z "$base_cmd" ]]; then
    echo "    Skipping: no [TOGSim] Run command / Command line found in $log_file"
    continue
  fi

  # === Trace file: PyTorchSim logs it as "[TOGSim] Trace log is stored to \"<path>.trace\"" (often on stderr, now merged in cycle logs) ===
  trace_line=$(grep -F '[TOGSim] Trace log is stored to' "$log_file" 2>/dev/null | tail -n 1) || true
  if [[ -z "$trace_line" ]]; then
    echo "    Skipping: no [TOGSim] Trace log is stored to ... line in $log_file"
    continue
  fi
  trace_file="${trace_line#*Trace log is stored to \"}"
  trace_file="${trace_file%%\"*}"
  if [[ -z "$trace_file" || ! -f "$trace_file" ]]; then
    echo "    Skipping: trace path missing or not a file: ${trace_file:-<empty>} (from $log_file)"
    continue
  fi

  # Normal configs
  for config in "${configs[@]}"; do
    output_file="$output_dir/${workload}_${config}.txt"
    echo "===== config=$config | model=$workload =====" > "$output_file"
    sum_all_iters=0.0
    iter_count=0

    for iter in $(seq 1 "${SPEEDUP_ITERS}"); do
      echo "[Iter $iter] Running simulation for workload=$workload config=$config"
      # Build command: replace --config and --models_list in base_cmd with our config and trace
      cmd=$(echo "$base_cmd" | sed -E "s|--config [^ ]+|--config $CONFIG_DIR/$config|" | sed -E "s|--models_list [^ ]+|--models_list $trace_file|")
      echo "$cmd"
      output=$(bash -c "$cmd" 2>&1) || true
      sim_time=$(echo "$output" | grep "Wall-clock time for simulation:" | sed -E 's/.*Wall-clock time for simulation: ([0-9]+\.[0-9]+) seconds.*/\1/')

      if [[ -n "$sim_time" ]]; then
        echo "Iteration $iter: simulation_time = $sim_time" >> "$output_file"
        sum_all_iters=$(awk -v a="$sum_all_iters" -v b="$sim_time" 'BEGIN {printf "%.6f", a + b}')
        iter_count=$((iter_count + 1))
      else
        echo "Iteration $iter: No simulation time found." >> "$output_file"
      fi
    done

    # === Final average ===
    if [[ $iter_count -gt 0 ]]; then
      avg=$(awk -v total="$sum_all_iters" -v n="$iter_count" 'BEGIN {printf "%.6f", total / n}')
      echo "Average simulation time for $workload with config $config: $avg seconds"
      echo "Average simulation time = $avg" >> "$output_file"
    else
      echo "No valid simulation times found for config $config"
      echo "Average simulation time = NA" >> "$output_file"
    fi
  done
done

# ILS: optional (skip in CI; slow and separate from simple-noc / booksim re-sims)
if [[ "$SKIP_ILS" != "1" ]]; then
  $TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_matmul.sh
  $TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_conv.sh
  $TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_bert.sh
  $TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_resnet.sh
else
  echo "[*] SKIP_ILS=1 — skipping ILS matmul/conv/bert/resnet."
fi

python3 $TORCHSIM_DIR/experiments/artifact/speedup/summary_speedup.py | tee "$TORCHSIM_DIR/experiments/artifact/speedup/summary_speedup.log"