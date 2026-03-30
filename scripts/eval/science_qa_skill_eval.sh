#!/bin/bash
set -e

# =============================================================================
# Science QA Skill-level Evaluation
# Runs inference on training set with multiple samples to analyze
# per-skill accuracy variance (validates skill taxonomy for SCPO)
# =============================================================================

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
GPUS="${GPUS:-0,1,2,3}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/data/science_qa/train_with_skills.jsonl"

echo "============================================================"
echo "Science QA Skill Evaluation"
echo "  Model:       ${MODEL}"
echo "  Samples (n): ${N}"
echo "  Temperature: ${TEMPERATURE}"
echo "  GPUs:        ${GPUS}"
echo "  Data:        ${DATA_PATH}"
echo "============================================================"

cd "${REPO_ROOT}"

bash scripts/eval/local_dataset_eval.sh \
    --parallel \
    --gpus "${GPUS}" \
    --model "${MODEL}" \
    --data "${DATA_PATH}" \
    --problem-key "problem" \
    --answer-key "answer" \
    --category-keys "primary_skill,source" \
    --n "${N}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-model-len 4096 \
    --max-new-tokens 3072
