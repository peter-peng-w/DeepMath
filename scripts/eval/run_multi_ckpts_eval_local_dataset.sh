#!/bin/bash
set -e

# =============================================================================
# Local Dataset Evaluation Script
# Evaluates models on local training data with both sampling and greedy modes
# =============================================================================

# Configuration
# MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH="data/math500/lvl3to5_8k/skills/train.json"
PROBLEM_KEY="problem"
ANSWER_KEY="answer"
CATEGORY_KEYS="level,subject"
OUTPUT_BASE="./exp/math_lvl3to5_8k"
GPUS="0,1,2,3"

echo "=========================================="
echo "Local Dataset Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "GPUs: $GPUS"
echo "=========================================="

# -----------------------------------------------------------------------------
# 1. Sampling Evaluation (temperature=0.6, top_p=0.95, n=8)
# -----------------------------------------------------------------------------
# This generates 8 rollouts per question with sampling
# Output: generation.jsonl with 8 responses and 8 correctness values per question
echo ""
echo ">>> Running SAMPLING evaluation (temp=0.6, top_p=0.95, n=8)..."
echo ""

./scripts/eval/local_dataset_eval.sh \
   --parallel \
   --gpus "$GPUS" \
   --model "$MODEL" \
   --data "$DATA_PATH" \
   --problem-key "$PROBLEM_KEY" \
   --answer-key "$ANSWER_KEY" \
   --category-keys "$CATEGORY_KEYS" \
   --output-dir "${OUTPUT_BASE}/sampling" \
   --temperature 0.6 \
   --top-p 0.95 \
   --n 8

# -----------------------------------------------------------------------------
# 2. Greedy Evaluation (temperature=0.0, n=1)
# -----------------------------------------------------------------------------
# This generates 1 deterministic response per question (greedy decoding)
# Output: generation.jsonl with 1 response and 1 correctness value per question
echo ""
echo ">>> Running GREEDY evaluation (temp=0.0, n=1)..."
echo ""

./scripts/eval/local_dataset_eval.sh \
   --parallel \
   --gpus "$GPUS" \
   --model "$MODEL" \
   --data "$DATA_PATH" \
   --problem-key "$PROBLEM_KEY" \
   --answer-key "$ANSWER_KEY" \
   --category-keys "$CATEGORY_KEYS" \
   --output-dir "${OUTPUT_BASE}/greedy" \
   --temperature 0.0 \
   --top-p 1.0 \
   --n 1

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Sampling: ${OUTPUT_BASE}/sampling/"
echo "  Greedy:   ${OUTPUT_BASE}/greedy/"
echo ""
echo "Each directory contains:"
echo "  - generation.jsonl: Per-question results with rollouts and rewards"
echo "  - result.log: Aggregated statistics by category (level, subject)"
echo ""