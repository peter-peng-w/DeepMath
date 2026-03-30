#!/bin/bash
set -e

# =============================================================================
# MCQA Evaluation Script
# Evaluates a model on MCQA benchmarks (Science QA, GPQA Diamond)
#
# Usage:
#   bash scripts/eval/mcqa_eval.sh --model <model_path> --dataset <dataset> [options]
#
# Examples:
#   bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset science_qa
#   bash scripts/eval/mcqa_eval.sh --model exp/qwen3-0.6b/... --dataset gpqa_diamond
#   bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset all
#   bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset science_qa --enable-thinking
# =============================================================================

# Defaults
MODEL=""
DATASET="all"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
GPU="${GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
ENABLE_THINKING=""
OUTPUT_BASE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --enable-thinking) ENABLE_THINKING="--enable_thinking True"; shift ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    echo "Usage: bash scripts/eval/mcqa_eval.sh --model <model_path> --dataset <dataset>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Derive model name for output directory
MODEL_NAME=$(basename "$MODEL")

# Set output base directory
if [ -z "$OUTPUT_BASE" ]; then
    OUTPUT_BASE="exp/local_eval/${MODEL_NAME}"
fi

# Dataset configs
SCIENCE_QA_DATA="${REPO_ROOT}/data/science_qa/test.json"
GPQA_DATA="${REPO_ROOT}/data/gpqa_diamond/test.json"

export CUDA_VISIBLE_DEVICES="$GPU"

run_eval() {
    local data_path="$1"
    local data_name="$2"
    local answer_key="$3"
    local category_keys="$4"
    local output_dir="${OUTPUT_BASE}/${data_name}/temp_${TEMPERATURE}_top_p_${TOP_P}_n_${N}"

    if [ ! -f "$data_path" ]; then
        echo "WARNING: Data file not found: $data_path (skipping ${data_name})"
        return
    fi

    echo "============================================================"
    echo "Evaluating: ${data_name}"
    echo "  Model:       ${MODEL}"
    echo "  Data:        ${data_path}"
    echo "  Samples (n): ${N}"
    echo "  Temperature: ${TEMPERATURE}"
    echo "  GPU:         ${GPU}"
    echo "  Output:      ${output_dir}"
    echo "============================================================"

    python mcqa_eval.py \
        --base_model "$MODEL" \
        --local_data_path "$data_path" \
        --problem_key_override "problem" \
        --answer_key_override "$answer_key" \
        --category_keys_override "$category_keys" \
        --output_dir "$output_dir" \
        --n "$N" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_model_len "$MAX_MODEL_LEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --bf16 \
        $ENABLE_THINKING
}

if [ "$DATASET" = "science_qa" ] || [ "$DATASET" = "all" ]; then
    run_eval "$SCIENCE_QA_DATA" "science_qa" "solution" "source"
fi

if [ "$DATASET" = "gpqa_diamond" ] || [ "$DATASET" = "all" ]; then
    run_eval "$GPQA_DATA" "gpqa_diamond" "solution" "domain"
fi

if [ "$DATASET" != "science_qa" ] && [ "$DATASET" != "gpqa_diamond" ] && [ "$DATASET" != "all" ]; then
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Available: science_qa, gpqa_diamond, all"
    exit 1
fi

echo "Done!"
