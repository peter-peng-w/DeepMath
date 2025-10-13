#!/bin/bash
set -e
set -u

export CUDA_VISIBLE_DEVICES=3

# =============================================================================
# Composable Evaluation Script for Qwen2.5-Math-1.5B
# =============================================================================

# Model configuration (shared across all evaluations)
#### 1. Rollouts size = 8, Semantic Group size = 8, Combinatorial Group size = 8*8, Using Cosine Similarity to estimate the advantage
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-160"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-320"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-480"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-640"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-800"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-960"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-1120"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-1280"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-1440"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-1600"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-1760"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-1920"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-2080"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-2240"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-2400"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-2560"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-2720"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-2880"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3040"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3200"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3360"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3520"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3680"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3840"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-4000"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-6560"


#### 2. Rollouts size = 4, Semantic Group size = 8, Combinatorial Group size = 4*8, Using Cosine Similarity to estimate the advantage
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-80"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-160"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-240"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-320"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-400"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-480"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-560"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-640"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-720"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-800"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-880"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-960"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-1040"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-1120"
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-1200"
BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-1280"




#### 3. Rollouts size = 4, Semantic Group size = 2, Combinatorial Group size = 4*2, Using Cosine Similarity to estimate the advantage
# BASE_MODEL="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo"


#### 4. Rollouts size = 2, Semantic Group size = 8, Combinatorial Group size = 2*8, Using Cosine Similarity to estimate the advantage

CHAT_TEMPLATE_NAME="default"
SYSTEM_PROMPT_NAME="disabled"
BF16="True"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="3072"

# VLLM environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Function to run evaluation
run_evaluation() {
    local data_id=$1
    local split=$2
    local temperature=$3
    local top_p=$4
    local n=$5
    local output_subdir=$6
    
    # Extract dataset name from data_id for output directory
    local dataset_name=$(echo "$data_id" | cut -d'/' -f2)
    local model_name="qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoint-1280"
    # local model_name="qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-3840"
    # local model_name="qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoint-10560"
    # local model_name="qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo"
    # local model_name="qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts/checkpoint-5280"
    local output_dir="./exp/${dataset_name}/${model_name}/${output_subdir}"
    
    echo "=========================================="
    echo "Running evaluation on: $data_id"
    echo "Output directory: $output_dir"
    echo "Config: temp=$temperature, top_p=$top_p, n=$n"
    echo "=========================================="
    
    python3 uni_eval.py \
        --base_model "$BASE_MODEL" \
        --chat_template_name "$CHAT_TEMPLATE_NAME" \
        --system_prompt_name "$SYSTEM_PROMPT_NAME" \
        --output_dir "$output_dir" \
        --bf16 "$BF16" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --data_id "$data_id" \
        --split "$split" \
        --max_model_len "$MAX_MODEL_LEN" \
        --temperature "$temperature" \
        --top_p "$top_p" \
        --n "$n"
    
    echo "Completed evaluation on: $data_id"
    echo ""
}

# =============================================================================
# Evaluation Configurations
# =============================================================================

# Define datasets to evaluate with their corresponding splits
# Format: "data_id|split"
DATASETS=(
    # "zwhe99/MATH|math500"                     # MATH500
    "zwhe99/amc23|test"                       # AMC23
    # "zwhe99/simplerl-OlympiadBench|test"      # OlympiadBench
    # "zwhe99/simplerl-minerva-math|test"       # MinervaMath
    "zwhe99/aime90|2024"                      # AIME24
    "math-ai/aime25|test"                     # AIME25
    # Add more datasets here as needed
)

# Define evaluation configurations
declare -A CONFIGS
# Sampling: Use multiple samples (n=16) for pass@k metrics with k>1
# CONFIGS["sampling"]="0.6 0.95 16"    # temperature top_p n
CONFIGS["sampling"]="0.6 0.95 64"    # temperature top_p n
# Greedy: Use single sample (n=1) since greedy decoding is deterministic
# Using n>1 for greedy would be wasteful as all samples would be identical
CONFIGS["greedy"]="0.0 1.0 1"        # temperature top_p n

# =============================================================================
# Main Execution
# =============================================================================

# Parse command line arguments
CONFIG_TYPE=${1:-"sampling"}  # Default to sampling if no argument provided
SPECIFIC_DATASET=${2:-""}     # Optional: run on specific dataset only

if [[ ! ${CONFIGS[$CONFIG_TYPE]+_} ]]; then
    echo "Error: Unknown configuration type '$CONFIG_TYPE'"
    echo "Available configurations: ${!CONFIGS[@]}"
    echo ""
    echo "Usage: $0 [CONFIG_TYPE] [SPECIFIC_DATASET]"
    echo "  CONFIG_TYPE: ${!CONFIGS[@]}"
    echo "  SPECIFIC_DATASET (optional): Run on specific dataset only"
    echo ""
    echo "Available datasets:"
    for dataset_entry in "${DATASETS[@]}"; do
        IFS='|' read -r data_id split <<< "$dataset_entry"
        echo "  - $data_id (split: $split)"
    done
    exit 1
fi

# Extract configuration parameters
IFS=' ' read -r TEMPERATURE TOP_P N <<< "${CONFIGS[$CONFIG_TYPE]}"

# Determine output subdirectory name
if [[ "$CONFIG_TYPE" == "greedy" ]]; then
    OUTPUT_SUBDIR="greedy"
else
    OUTPUT_SUBDIR="temperature_${TEMPERATURE}_top_p_${TOP_P}_rollouts_${N}"
fi

echo "Starting evaluations with configuration: $CONFIG_TYPE"
echo "Parameters: temperature=$TEMPERATURE, top_p=$TOP_P, n=$N"
echo ""

# Run evaluations
if [[ -n "$SPECIFIC_DATASET" ]]; then
    # Run on specific dataset only
    echo "Running evaluation on specific dataset: $SPECIFIC_DATASET"
    
    # Find the dataset and its split
    found=false
    for dataset_entry in "${DATASETS[@]}"; do
        IFS='|' read -r data_id split <<< "$dataset_entry"
        if [[ "$data_id" == "$SPECIFIC_DATASET" ]]; then
            run_evaluation "$data_id" "$split" "$TEMPERATURE" "$TOP_P" "$N" "$OUTPUT_SUBDIR"
            found=true
            break
        fi
    done
    
    if [[ "$found" == false ]]; then
        echo "Error: Dataset '$SPECIFIC_DATASET' not found in predefined datasets."
        echo "Available datasets:"
        for dataset_entry in "${DATASETS[@]}"; do
            IFS='|' read -r data_id split <<< "$dataset_entry"
            echo "  - $data_id (split: $split)"
        done
        exit 1
    fi
else
    # Run on all datasets
    for dataset_entry in "${DATASETS[@]}"; do
        IFS='|' read -r data_id split <<< "$dataset_entry"
        run_evaluation "$data_id" "$split" "$TEMPERATURE" "$TOP_P" "$N" "$OUTPUT_SUBDIR"
    done
fi

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="