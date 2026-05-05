#!/bin/bash
set -e

# =============================================================================
# Multi-Checkpoint Local Dataset Evaluation
# Evaluates a list of RL checkpoints sequentially on a local training dataset,
# running both sampling (temp=0.6, top_p=0.95, n=8) and greedy (temp=0.0, n=1)
# for each checkpoint using all available GPUs in parallel per checkpoint.
#
# USAGE:
#   Edit the configuration block below and run:
#     ./scripts/eval/run_multi_ckpts_eval_local_dataset.sh
#
# CHECKPOINT INPUT FORMATS (set CHECKPOINTS_INPUT to one of):
#   - Text file: path to a .txt file with one checkpoint path per line
#                (lines starting with # are treated as comments)
#   - Directory: path to a training run directory containing checkpoint-* subdirs
#                (auto-discovered and sorted by step number)
#   - Comma-separated: "path/ckpt-100,path/ckpt-200,Org/Model-HF-ID"
# =============================================================================

# =============================================================================
# Configuration — edit this block for different models and settings
# =============================================================================

## Qwen2.5-Math-1.5B
# CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std_on_policy_cluster_value_no_drift_penalty/checkpoints_to_eval.txt"
# CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std_on_policy_cluster_value_no_drift_penalty/checkpoints_to_eval-split4.txt"
# OUTPUT_BASE="./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std_on_policy_cluster_value_no_drift_penalty/training_dataset"

CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std_mixed_policy_cluster_value_no_drift_penalty/checkpoints_to_eval.txt"
OUTPUT_BASE="./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std_mixed_policy_cluster_value_no_drift_penalty/training_dataset"

# CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mixed_policy_cluster_value_no_drift_penalty/checkpoints_to_eval.txt"
# OUTPUT_BASE="./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mixed_policy_cluster_value_no_drift_penalty/training_dataset"

# CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"
# OUTPUT_BASE="./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/training_dataset"

# CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_off_policy_cluster_value/checkpoints_to_eval.txt"
# OUTPUT_BASE="./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_off_policy_cluster_value/training_dataset"

# CHECKPOINTS_INPUT="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_onpolicy_lambda_no_drift/checkpoints_to_eval.txt"
# OUTPUT_BASE="./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_onpolicy_lambda_no_drift/training_dataset"

DATA_PATH="data/math500/lvl3to5_8k/skills/train.json"
PROBLEM_KEY="problem"
ANSWER_KEY="answer"
CATEGORY_KEYS="level,subject"
GPUS="0,1,2,3"
SKIP_EXISTING="true"

# Which eval modes to run: "both", "sampling", or "greedy"
EVAL_MODES="both"

# Sampling config
SAMPLING_TEMPERATURE="0.6"
SAMPLING_TOP_P="0.95"
SAMPLING_N="8"

# Greedy config
GREEDY_TEMPERATURE="0.0"
GREEDY_TOP_P="1.0"
GREEDY_N="1"

# =============================================================================
# Parse checkpoint list
# =============================================================================

CHECKPOINT_LIST=()

if [[ -f "$CHECKPOINTS_INPUT" ]]; then
    echo "Reading checkpoints from file: $CHECKPOINTS_INPUT"
    while IFS= read -r line; do
        # Skip comment lines and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        CHECKPOINT_LIST+=("$line")
    done < "$CHECKPOINTS_INPUT"

elif [[ -d "$CHECKPOINTS_INPUT" ]]; then
    echo "Discovering checkpoints in directory: $CHECKPOINTS_INPUT"
    while IFS= read -r d; do
        CHECKPOINT_LIST+=("$d")
    done < <(find "$CHECKPOINTS_INPUT" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

else
    echo "Parsing comma-separated checkpoints"
    IFS=',' read -ra CHECKPOINT_LIST <<< "$CHECKPOINTS_INPUT"
fi

if [[ ${#CHECKPOINT_LIST[@]} -eq 0 ]]; then
    echo "Error: No checkpoints found from: $CHECKPOINTS_INPUT"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================

DATA_BASENAME=$(basename "$DATA_PATH" | sed 's/\.[^.]*$//')

echo "=========================================="
echo "Multi-Checkpoint Local Dataset Evaluation"
echo "=========================================="
echo "Checkpoints:  ${#CHECKPOINT_LIST[@]} found"
for ckpt in "${CHECKPOINT_LIST[@]}"; do
    echo "  - $(basename "$ckpt")"
done
echo "Data:         $DATA_PATH"
echo "Output base:  $OUTPUT_BASE"
echo "GPUs:         $GPUS"
echo "Eval modes:   $EVAL_MODES"
echo "Skip existing: $SKIP_EXISTING"
echo "=========================================="

# =============================================================================
# Helper: check if output already exists
# =============================================================================

check_existing() {
    local ckpt="$1"
    local temperature="$2"
    local top_p="$3"
    local n="$4"
    local ckpt_name
    ckpt_name=$(basename "$ckpt")
    local expected="${OUTPUT_BASE}/${ckpt_name}/${DATA_BASENAME}/temp_${temperature}_top_p_${top_p}_n_${n}/generation.jsonl"
    if [[ -f "$expected" ]]; then
        return 0  # exists
    fi
    return 1  # does not exist
}

# =============================================================================
# Main loop — process each checkpoint sequentially
# =============================================================================

TOTAL=${#CHECKPOINT_LIST[@]}
CURRENT=0

for ckpt in "${CHECKPOINT_LIST[@]}"; do
    CURRENT=$((CURRENT + 1))
    CKPT_NAME=$(basename "$ckpt")

    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Checkpoint: $CKPT_NAME"
    echo "=========================================="

    # ------------------------------------------------------------------
    # Sampling evaluation
    # ------------------------------------------------------------------
    if [[ "$EVAL_MODES" == "both" || "$EVAL_MODES" == "sampling" ]]; then
        if [[ "$SKIP_EXISTING" == "true" ]] && check_existing "$ckpt" "$SAMPLING_TEMPERATURE" "$SAMPLING_TOP_P" "$SAMPLING_N"; then
            echo "  [SKIP] Sampling results already exist for $CKPT_NAME"
        else
            echo "  >>> Running SAMPLING (temp=$SAMPLING_TEMPERATURE, top_p=$SAMPLING_TOP_P, n=$SAMPLING_N)..."
            ./scripts/eval/local_dataset_eval.sh \
                --parallel \
                --gpus "$GPUS" \
                --model "$ckpt" \
                --data "$DATA_PATH" \
                --problem-key "$PROBLEM_KEY" \
                --answer-key "$ANSWER_KEY" \
                --category-keys "$CATEGORY_KEYS" \
                --output-dir "$OUTPUT_BASE" \
                --temperature "$SAMPLING_TEMPERATURE" \
                --top-p "$SAMPLING_TOP_P" \
                --n "$SAMPLING_N"
        fi
    fi

    # ------------------------------------------------------------------
    # Greedy evaluation
    # ------------------------------------------------------------------
    if [[ "$EVAL_MODES" == "both" || "$EVAL_MODES" == "greedy" ]]; then
        if [[ "$SKIP_EXISTING" == "true" ]] && check_existing "$ckpt" "$GREEDY_TEMPERATURE" "$GREEDY_TOP_P" "$GREEDY_N"; then
            echo "  [SKIP] Greedy results already exist for $CKPT_NAME"
        else
            echo "  >>> Running GREEDY (temp=$GREEDY_TEMPERATURE, top_p=$GREEDY_TOP_P, n=$GREEDY_N)..."
            ./scripts/eval/local_dataset_eval.sh \
                --parallel \
                --gpus "$GPUS" \
                --model "$ckpt" \
                --data "$DATA_PATH" \
                --problem-key "$PROBLEM_KEY" \
                --answer-key "$ANSWER_KEY" \
                --category-keys "$CATEGORY_KEYS" \
                --output-dir "$OUTPUT_BASE" \
                --temperature "$GREEDY_TEMPERATURE" \
                --top-p "$GREEDY_TOP_P" \
                --n "$GREEDY_N"
        fi
    fi

    echo "  Done: $CKPT_NAME"
done

echo ""
echo "=========================================="
echo "All checkpoints completed!"
echo "Results saved under: $OUTPUT_BASE/"
echo ""
echo "Output structure:"
echo "  ${OUTPUT_BASE}/<checkpoint-name>/${DATA_BASENAME}/temp_${SAMPLING_TEMPERATURE}_top_p_${SAMPLING_TOP_P}_n_${SAMPLING_N}/generation.jsonl  (sampling)"
echo "  ${OUTPUT_BASE}/<checkpoint-name>/${DATA_BASENAME}/temp_${GREEDY_TEMPERATURE}_top_p_${GREEDY_TOP_P}_n_${GREEDY_N}/generation.jsonl    (greedy)"
echo "=========================================="
