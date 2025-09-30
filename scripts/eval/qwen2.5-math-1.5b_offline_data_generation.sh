#!/bin/bash
set -e
set -u

export CUDA_VISIBLE_DEVICES=1

# =============================================================================
# Composable Evaluation Script for Qwen2.5-Math-1.5B
# =============================================================================

# Model configuration (shared across all evaluations)
BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
CHAT_TEMPLATE_NAME="default"
SYSTEM_PROMPT_NAME="disabled"
BF16="True"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="3072"
START_INDEX="7000"
END_INDEX="8000"

# VLLM environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_FLASHINFER_SAMPLER=0
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
    local model_name="qwen2.5-math-1.5b"
    local output_dir="./exp/${dataset_name}/${model_name}/synthesize/lvl3to5_8k/${output_subdir}/index_${START_INDEX}_${END_INDEX}"
    
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
        --n "$n" \
        --start_idx "$START_INDEX" \
        --end_idx "$END_INDEX"
    
    echo "Completed evaluation on: $data_id"
    echo ""
}

# =============================================================================
# Evaluation Configurations
# =============================================================================

# Define datasets to evaluate with their corresponding splits
# Format: "data_id|split"

#### Evaluation Datasets ####
# DATASETS=(
#     "zwhe99/MATH|math500"                     # MATH500
#     "zwhe99/amc23|test"                       # AMC23
#     "zwhe99/simplerl-OlympiadBench|test"      # OlympiadBench
#     "zwhe99/simplerl-minerva-math|test"       # MinervaMath
#     "zwhe99/aime90|2024"                      # AIME24
#     "math-ai/aime25|test"                     # AIME25
#     # Add more datasets here as needed
# )
#### Training Datasets ####
DATASETS=(
    "stillarrow/MATH|train_lvl3to5"                     # train_lvl3to5
    # Add more datasets here as needed
)

# Define evaluation configurations
declare -A CONFIGS
# Sampling: Use multiple samples (n=16) for pass@k metrics with k>1
CONFIGS["sampling"]="1.0 1.0 128"    # temperature top_p n
# CONFIGS["sampling"]="0.6 0.95 16"    # temperature top_p n
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