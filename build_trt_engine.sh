#!/bin/bash
set -e

# Default arguments
RUN_DIR="runs/sudoku/run_3"
BATCH_SIZE="128"
CHECKPOINT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_dir) RUN_DIR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================="
echo " Starting TensorRT Build Pipeline        "
echo " Run Directory : $RUN_DIR "
echo " Batch Size    : $BATCH_SIZE"
echo "========================================="

# Move to the root of the project
cd "$(dirname "$0")"

export PYTHONPATH="."

# 1. Export ONNX (forces opset 17 + static batch size)
echo "[1/4] Exporting PyTorch model to ONNX..."
if [ -n "$CHECKPOINT" ]; then
    conda run -n ebt_policy python3 tensorrt/export_onnx.py --run_dir "$RUN_DIR" --batch_size "$BATCH_SIZE" --checkpoint "$CHECKPOINT"
else
    conda run -n ebt_policy python3 tensorrt/export_onnx.py --run_dir "$RUN_DIR" --batch_size "$BATCH_SIZE"
fi

# 2. Export Test Data
echo -e "\n[2/4] Generating Real Sudoku binary test data..."
conda run -n ebt_policy python3 tensorrt/export_test_data.py --batch_size "$BATCH_SIZE"
mv sudoku_test_boards.bin tensorrt/sudoku_test_boards.bin

# 3. Docker Compile Engine & Run Evaluation
echo -e "\n[3/4] Launching NVIDIA TRT Container for Native Compilation and TRTEXEC..."
# Paths for the docker container to map internally
ONNX_PATH="$RUN_DIR/model_b${BATCH_SIZE}.onnx"
ENGINE_PATH="$RUN_DIR/model_b${BATCH_SIZE}.engine"

# We run trtexec and then compile/run the C++ benchmark script back-to-back inside the same container
# (Removed -i to prevent TTY pipeline hangs, Added memPoolSize to prevent stalling when VRAM is contested by PyTorch)
docker run --gpus all --rm -v "$(pwd):/workspace" -w /workspace nvcr.io/nvidia/tensorrt:25.11-py3 bash -c "
    echo '>> Compiling TRT Engine...'
    trtexec --onnx=$ONNX_PATH --saveEngine=$ENGINE_PATH --memPoolSize=workspace:2048
    
    echo -e '\n>> Building C++ Benchmark Script...'
    cd tensorrt
    mkdir -p build && cd build
    cmake .. && make
    
    echo -e '\n>> Executing C++ Evaluation against Engine...'
    ./trt_infer ../../$ENGINE_PATH $BATCH_SIZE
"

echo -e "\n[4/4] Pipeline Complete!"
echo "Your TRT engine is saved dynamically at: $ENGINE_PATH"
