#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/launch_pe_ablation.sh \
#   --config-name maestro_standard \
#   --run-prefix maestro_pe_ablation \
#   --gpu-baseline 0 \
#   --gpu-v1 1 \
#   --gpu-v2 2

CONFIG_NAME="maestro_standard"
RUN_PREFIX="maestro_pe_ablation"
GPU_BASELINE="0"
GPU_V1="1"
GPU_V2="2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --run-prefix)
            RUN_PREFIX="$2"
            shift 2
            ;;
        --gpu-baseline)
            GPU_BASELINE="$2"
            shift 2
            ;;
        --gpu-v1)
            GPU_V1="$2"
            shift 2
            ;;
        --gpu-v2)
            GPU_V2="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

[[ -n "$GPU_BASELINE" ]] || { echo "--gpu-baseline is required"; exit 1; }
[[ -n "$GPU_V1" ]] || { echo "--gpu-v1 is required"; exit 1; }
[[ -n "$GPU_V2" ]] || { echo "--gpu-v2 is required"; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

launch_one() {
    local exp_name="$1"
    local gpu_id="$2"
    shift 2
    mkdir -p "logs/${RUN_PREFIX}"
    local log_path="logs/${RUN_PREFIX}/${exp_name}.log"

    nohup python train.py \
        --config-name "$CONFIG_NAME" \
        "run_name=${RUN_PREFIX}_${exp_name}" \
        "train.device=cuda:${gpu_id}" \
        "$@" \
        > "$log_path" 2>&1 &

    local pid=$!
    echo "[$exp_name] gpu=${gpu_id} pid=${pid} log=${log_path}"
}

# baseline: ordinary RoPE
launch_one "baseline_ordinary_rope" "$GPU_BASELINE" \
    "llm.time_aware_rope.enable=False"

# v1: 1D time-aware RoPE
launch_one "v1_time_aware_1d" "$GPU_V1" \
    "llm.time_aware_rope.enable=True" \
    "llm.time_aware_rope.mode=time_aware" \
    "llm.time_aware_rope.use_linear=False"

# v2: 2D time-aware RoPE
launch_one "v2_time_aware_2d" "$GPU_V2" \
    "llm.time_aware_rope.enable=True" \
    "llm.time_aware_rope.mode=time_aware_2d"
