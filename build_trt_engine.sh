#!/bin/bash
set -e

# Default arguments
RUN_DIR="./runs/robomimic/run_square"
BATCH_SIZE="1"
CHECKPOINT=""
IMAGE_SIZE="84"
NUM_CAMERAS="2"
OBS_HORIZON="2"
PROPRIO_DIM="7"
RUN_CPP_INFER="1"
CPP_ITERS="100"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_dir) RUN_DIR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --image_size) IMAGE_SIZE="$2"; shift ;;
        --num_cameras) NUM_CAMERAS="$2"; shift ;;
        --obs_horizon) OBS_HORIZON="$2"; shift ;;
        --proprio_dim) PROPRIO_DIM="$2"; shift ;;
        --run_cpp_infer) RUN_CPP_INFER="$2"; shift ;;
        --cpp_iters) CPP_ITERS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================="
echo " Starting TensorRT Build Pipeline        "
echo " Run Directory : $RUN_DIR "
echo " Batch Size    : $BATCH_SIZE"
echo " Image Size    : $IMAGE_SIZE"
echo " Num Cameras   : $NUM_CAMERAS"
echo " Obs Horizon   : $OBS_HORIZON"
echo " Proprio Dim   : $PROPRIO_DIM"
echo "========================================="

# Move to the root of the project
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

RUN_DIR_ABS="$(realpath "$RUN_DIR")"
PROJECT_ROOT_ABS="$(realpath "$PROJECT_ROOT")"
if [[ "$RUN_DIR_ABS" != "$PROJECT_ROOT_ABS"* ]]; then
    echo "Error: --run_dir must be inside project root for Docker mount mapping."
    echo "  run_dir      : $RUN_DIR_ABS"
    echo "  project root : $PROJECT_ROOT_ABS"
    exit 1
fi

RUN_DIR_REL="${RUN_DIR_ABS#$PROJECT_ROOT_ABS/}"

export PYTHONPATH="."

# 1. Export ONNX with fixed batch size / camera count.
echo "[1/2] Exporting Robomimic policy to ONNX..."
if [ -n "$CHECKPOINT" ]; then
    conda run -n ebt_policy python3 tensorrt/export_onnx.py \
        --run_dir "$RUN_DIR_ABS" \
        --batch_size "$BATCH_SIZE" \
        --checkpoint "$CHECKPOINT" \
        --image_size "$IMAGE_SIZE" \
        --num_cameras "$NUM_CAMERAS"
else
    conda run -n ebt_policy python3 tensorrt/export_onnx.py \
        --run_dir "$RUN_DIR_ABS" \
        --batch_size "$BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --num_cameras "$NUM_CAMERAS"
fi

# 2. Docker build TensorRT engine
echo -e "\n[2/2] Launching NVIDIA TRT Container for trtexec engine build..."
# Paths for the docker container to map internally
ONNX_PATH_HOST="$RUN_DIR_ABS/model_b${BATCH_SIZE}.onnx"
ENGINE_PATH_HOST="$RUN_DIR_ABS/model_b${BATCH_SIZE}.engine"
ONNX_PATH_CONTAINER="/workspace/$RUN_DIR_REL/model_b${BATCH_SIZE}.onnx"
ENGINE_PATH_CONTAINER="/workspace/$RUN_DIR_REL/model_b${BATCH_SIZE}.engine"

# Build engine with fixed input shapes exported at batch size BATCH_SIZE.
docker run --gpus all --rm -v "$(pwd):/workspace" -w /workspace nvcr.io/nvidia/tensorrt:25.11-py3 bash -c "
    echo '>> Compiling TRT Engine from ONNX...'
    trtexec --onnx=$ONNX_PATH_CONTAINER --saveEngine=$ENGINE_PATH_CONTAINER --memPoolSize=workspace:2048
"

if [[ "$RUN_CPP_INFER" == "1" ]]; then
    echo -e "\n[3/3] Building and running C++ TensorRT inference..."
    docker run --gpus all --rm -v "$(pwd):/workspace" -w /workspace nvcr.io/nvidia/tensorrt:25.11-py3 bash -c "
        cd tensorrt
        mkdir -p build && cd build
        cmake .. && make -j
        ./trt_infer $ENGINE_PATH_CONTAINER $BATCH_SIZE $OBS_HORIZON $NUM_CAMERAS $IMAGE_SIZE $PROPRIO_DIM $CPP_ITERS
    "
else
    echo -e "\n[3/3] Skipping C++ TensorRT inference benchmark (--run_cpp_infer $RUN_CPP_INFER)"
fi

echo -e "\nPipeline Complete!"
echo "Your TRT engine is saved at: $ENGINE_PATH_HOST"
