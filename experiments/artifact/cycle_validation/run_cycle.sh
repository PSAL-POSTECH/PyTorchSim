#!/bin/bash
# pipefail so a failing "python3 ... | tee log" aborts instead of being masked by
# tee's exit code (otherwise a broken model -- e.g. BERT -- fails silently and the
# job stays green).
set -eo pipefail

usage() {
  cat <<'EOF'
Usage: run_cycle.sh [--only SECTION[,SECTION...]]

  Run cycle validation benchmarks. Default: all sections + summary.

  SECTION (comma-separated for --only):
    matmul      GEMM sizes
    conv        Conv2d sizes
    layernorm   LayerNorm sizes
    softmax     Softmax sizes
    attention   Attention sizes
    resnet      resnet18, resnet50
    bert        BERT base/large/xlarge
    summary     summary_cycle.py (reads logs under experiments/artifact/logs)

Examples:
  ./run_cycle.sh
  ./run_cycle.sh --only matmul
  ./run_cycle.sh --only matmul,conv,summary
EOF
}

ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="${2:-}"
      if [[ -z "$ONLY" ]]; then echo "error: --only needs a value"; exit 1; fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# If ONLY is set, run section NAME only when ",$NAME," appears in ",$ONLY,"
should_run() {
  local name=$1
  if [[ -z "$ONLY" ]]; then
    return 0
  fi
  [[ ",${ONLY}," == *",${name},"* ]]
}

# The ONLY v3 pin left: this script's reference column (baseline_cycle.csv) was
# measured on a TPUv3, so simulating v6e here would compare two different machines.
# Everything else validates v6e. Override TOGSIM_CONFIG if you want a v6e run anyway.
export TOGSIM_CONFIG="${TOGSIM_CONFIG:-$TORCHSIM_DIR/configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml}"
LOG_DIR=$TORCHSIM_DIR/experiments/artifact/logs
mkdir -p $LOG_DIR

# Matmul
if should_run matmul; then
  for sz in "256 256 256" "512 512 512" "1024 1024 1024" "2048 2048 2048"; do
    name="gemm_${sz// /x}"
    echo ""
    echo "==================================================="
    echo "[*] Running Matmul size=$sz"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/gemm.py --size $sz 2>&1 | tee $LOG_DIR/${name}.log
  done
fi

# Conv
if should_run conv; then
  for sz in \
    "1 56 56 64 64 3 1 1" \
    "1 28 28 128 128 3 1 1" \
    "1 14 14 256 256 3 1 1" \
    "1 7 7 512 512 3 1 1" \
    "64 56 56 64 64 3 1 1" \
    "64 28 28 128 128 3 1 1" \
    "64 14 14 256 256 3 1 1" \
    "64 7 7 512 512 3 1 1"; do
    name="conv_${sz// /x}"
    echo ""
    echo "==================================================="
    echo "[*] Running Conv size=$sz"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/conv.py --size $sz 2>&1 | tee $LOG_DIR/${name}.log
  done
fi

# LayerNorm
if should_run layernorm; then
  for sz in "512 768" "2048 768" "8192 768"; do
    name="layernorm_${sz// /x}"
    echo ""
    echo "==================================================="
    echo "[*] Running LayerNorm size=$sz"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/layernorm.py --size $sz 2>&1 | tee $LOG_DIR/${name}.log
  done
fi

# Softmax
if should_run softmax; then
  for sz in "512 512" "2048 2048" "8192 8192"; do
    name="softmax_${sz// /x}"
    echo ""
    echo "==================================================="
    echo "[*] Running Softmax size=$sz"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/softmax.py --size $sz 2>&1 | tee $LOG_DIR/${name}.log
  done
fi

# Attention
if should_run attention; then
  for sz in "12 512 64" "16 512 64" "32 512 64"; do
    name="attention_${sz// /x}"
    echo ""
    echo "==================================================="
    echo "[*] Running Attention size=$sz"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/attention.py --size $sz 2>&1 | tee $LOG_DIR/${name}.log
  done
fi

# ResNet
if should_run resnet; then
  for model in "resnet18" "resnet50"; do
    echo ""
    echo "==================================================="
    echo "[*] Running $model"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/${model}.py 2>&1 | tee $LOG_DIR/${model}.log
  done
fi

# BERT
if should_run bert; then
  for model in "base" "large" "xlarge"; do
    echo ""
    echo "==================================================="
    echo "[*] Running BERT size=$model"
    echo "==================================================="
    python3 $TORCHSIM_DIR/experiments/BERT.py --size $model 2>&1 | tee $LOG_DIR/bert_${model}.log
  done
fi

# Cycle Summary
if should_run summary; then
  python3 $TORCHSIM_DIR/experiments/artifact/cycle_validation/summary_cycle.py 2>&1 | tee "$TORCHSIM_DIR/experiments/artifact/cycle_validation/summary_cycle.out"
fi
