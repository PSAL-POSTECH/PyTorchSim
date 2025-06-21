#!/bin/bash

if [ -z "$TORCHSIM_DIR" ]; then
    echo "Error: TORCHSIM_DIR environment variable is not set."
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 GEMM_PATH [ATTRIBUTE_FILE]"
    echo "  GEMM_PATH: Path to the gemm directory (e.g., ../../gemmx1024x1024x1024)"
    echo "  ATTRIBUTE_FILE: Optional path to the attribute file"
    exit 1
fi

GEMM_PATH="$1"
INDEX_NAME="$2"
SIMULATOR_PATH="$TORCHSIM_DIR/PyTorchSimBackend/build/bin/Simulator"
GEMM_DIR_NAME=$(basename "$GEMM_PATH")
echo "GEMM Directory Name: $GEMM_DIR_NAME"

CONFIG_LIST=(
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_chiplet_tpuv3.json"
)
CONFIG_LIST2=(
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_booksim_tpuv3.json"
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_chiplet_tpuv3_xnuma.json"
)
shift
shift
for ATTRIBUTE in "$@"; do
    ATTRIBUTE_FILE="$GEMM_PATH/runtime_0000/attribute/$ATTRIBUTE"
    if [ ! -f "$ATTRIBUTE_FILE" ]; then
        echo "Error: Attribute file '$ATTRIBUTE_FILE' does not exist."
        exit 1
    fi
    ATTRIBUTE_FILES+=("$ATTRIBUTE_FILE")
done
MODELS_LIST="$GEMM_PATH/tile_graph.onnx"
ATTRIBUTE_PATH="$GEMM_PATH/runtime_0000/attribute"

for CONFIG in "${CONFIG_LIST[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG" .json)

    for ATTRIBUTE_FILE in "${ATTRIBUTE_FILES[@]}"; do
        ATTRIBUTE_NAME=$(basename "$ATTRIBUTE_FILE")

        RESULTS_DIR="./chiplet_results$INDEX_NAME/$GEMM_DIR_NAME/$ATTRIBUTE_NAME"
        mkdir -p "$RESULTS_DIR"
        OUTPUT_FILE="$RESULTS_DIR/${CONFIG_NAME}_result.txt"

        # Run Simulator
        echo "$SIMULATOR_PATH" --config "$CONFIG" --models_list "$MODELS_LIST" --attributes_list "$ATTRIBUTE_PATH/$ATTRIBUTE_NAME"
        "$SIMULATOR_PATH" --config "$CONFIG" --models_list "$MODELS_LIST" --log_level trace --attributes_list "$ATTRIBUTE_PATH/$ATTRIBUTE_NAME" > "$OUTPUT_FILE" &
        echo "[BackendSimulator] for $CONFIG stored to \"$(pwd)/$OUTPUT_FILE\""
    done
done

for CONFIG in "${CONFIG_LIST2[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG" .json)
    ATTRIBUTE_NAME=0
    RESULTS_DIR="./chiplet_results$INDEX_NAME/$GEMM_DIR_NAME/$ATTRIBUTE_NAME"
    mkdir -p "$RESULTS_DIR"
    OUTPUT_FILE="$RESULTS_DIR/${CONFIG_NAME}_result.txt"

    # Run Simulator
    # echo "$SIMULATOR_PATH" --config "$CONFIG" --models_list "$MODELS_LIST" --attributes_list "$ATTRIBUTE_PATH/$ATTRIBUTE_NAME"
    "$SIMULATOR_PATH" --config "$CONFIG" --models_list "$MODELS_LIST" --log_level trace --attributes_list "$ATTRIBUTE_PATH/$ATTRIBUTE_NAME" > "$OUTPUT_FILE" &
    echo "[BackendSimulator] for $CONFIG stored to \"$(pwd)/$OUTPUT_FILE\""
done
wait