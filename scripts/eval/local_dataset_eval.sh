#!/bin/bash
set -e
set -u

# =============================================================================
# Local Dataset Evaluation Script
# Evaluates model checkpoints on local JSON/JSONL datasets
# Supports parallel multi-GPU evaluation with automatic data sharding
# =============================================================================

# Default configuration
export DEFAULT_CHAT_TEMPLATE_NAME="default"
export DEFAULT_SYSTEM_PROMPT_NAME="disabled"
export DEFAULT_BF16="True"
export DEFAULT_TENSOR_PARALLEL_SIZE="1"
export DEFAULT_MAX_MODEL_LEN="4096"
export DEFAULT_MAX_NEW_TOKENS="3072"

# VLLM environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# =============================================================================
# Functions
# =============================================================================

show_help() {
    cat << EOF
Local Dataset Evaluation Script

USAGE:
    # Single GPU evaluation
    $0 --model MODEL_PATH --data LOCAL_DATA_PATH --problem-key KEY --answer-key KEY [OPTIONS]
    
    # Parallel multi-GPU evaluation
    $0 --parallel --gpus "0,1,2,3" --model MODEL_PATH --data LOCAL_DATA_PATH --problem-key KEY --answer-key KEY [OPTIONS]
    
    # Merge results from parallel runs
    $0 --merge --output-dir DIR

REQUIRED:
    --model MODEL_PATH      Path to the model checkpoint or HuggingFace model ID
    --data LOCAL_DATA_PATH  Path to local JSON/JSONL file
    --problem-key KEY       Key in JSON for the problem text (e.g., "problem", "question")
    --answer-key KEY        Key in JSON for the expected answer (e.g., "answer", "expected_answer")

OPTIONS:
    --category-keys KEYS    Comma-separated list of category keys for reporting (e.g., "level,subject")
    --output-dir DIR        Output directory (default: ./exp/local_eval)
    --gpu GPU_ID            GPU ID to use for single GPU mode (default: 0)
    --temperature TEMP      Sampling temperature (default: 0.6)
    --top-p TOP_P           Top-p sampling (default: 0.95)
    --n N                   Number of samples per problem (default: 1)
    --max-model-len LEN     Maximum model context window (prompt+response) (default: 16384)
    --max-new-tokens LEN    Maximum tokens to generate (response only) (default: 3072)
    --start-idx IDX         Start index for dataset slicing
    --end-idx IDX           End index for dataset slicing
    --help                  Show this help message

PARALLEL MODE OPTIONS:
    --parallel              Enable parallel multi-GPU evaluation
    --gpus GPU_IDS          Comma-separated list of GPU IDs (e.g., "0,1,2,3")
    --total-samples N       Total number of samples in dataset (auto-detected if not provided)

MERGE MODE OPTIONS:
    --merge                 Merge results from parallel runs
    --output-dir DIR        Directory containing shard subdirectories to merge

EXAMPLES:
    # Basic single GPU evaluation
    $0 --model Qwen/Qwen2.5-Math-1.5B \\
       --data data/math500/lvl3to5_8k/skills/train.json \\
       --problem-key problem \\
       --answer-key answer

    # With category reporting and multiple samples
    $0 --model /path/to/checkpoint \\
       --data data/math500/lvl3to5_8k/skills/train.json \\
       --problem-key problem \\
       --answer-key answer \\
       --category-keys "level,subject" \\
       --n 16 \\
       --temperature 0.6

    # Evaluate subset of data on single GPU
    $0 --model Qwen/Qwen2.5-Math-1.5B \\
       --data data/math500/lvl3to5_8k/skills/train.json \\
       --problem-key problem \\
       --answer-key answer \\
       --start-idx 0 \\
       --end-idx 100

    # Parallel evaluation on 4 GPUs
    $0 --parallel --gpus "0,1,2,3" \\
       --model Qwen/Qwen2.5-Math-1.5B \\
       --data data/math500/lvl3to5_8k/skills/train.json \\
       --problem-key problem \\
       --answer-key answer \\
       --category-keys "level,subject"

    # Merge results after parallel evaluation
    $0 --merge --output-dir ./exp/local_eval/Qwen2.5-Math-1.5B/train/temp_0.6_top_p_0.95_n_1

SUPPORTED DATASET FORMATS:
    - JSON Lines (.jsonl): One JSON object per line
    - JSON array (.json): Array of JSON objects
    - JSON Lines in .json file: Multiple JSON objects without array wrapper

    Example JSON structure:
    {
        "problem": "What is 2+2?",
        "answer": "\\\\boxed{4}",
        "level": 1,
        "subject": "Arithmetic"
    }

EOF
}

# Count lines in JSON/JSONL file
count_dataset_samples() {
    local data_path="$1"
    if [[ "$data_path" == *.jsonl ]]; then
        wc -l < "$data_path"
    else
        # For JSON files, check if it's a JSON array or JSON lines
        local first_char=$(head -c 1 "$data_path")
        if [[ "$first_char" == "[" ]]; then
            # JSON array - count elements
            python3 -c "import json; print(len(json.load(open('$data_path'))))"
        else
            # JSON lines format
            wc -l < "$data_path"
        fi
    fi
}

# Merge results from multiple shards
merge_results() {
    local output_dir="$1"
    
    echo "=========================================="
    echo "Merging Results"
    echo "=========================================="
    echo "Output directory: $output_dir"
    
    # Find all shard directories
    local shard_dirs=()
    for d in "$output_dir"/shard_*; do
        if [[ -d "$d" ]]; then
            shard_dirs+=("$d")
        fi
    done
    
    if [[ ${#shard_dirs[@]} -eq 0 ]]; then
        echo "Error: No shard directories found in $output_dir"
        exit 1
    fi
    
    echo "Found ${#shard_dirs[@]} shard directories"
    
    # Check all shards have generation.jsonl
    for shard_dir in "${shard_dirs[@]}"; do
        if [[ ! -f "$shard_dir/generation.jsonl" ]]; then
            echo "Error: Missing generation.jsonl in $shard_dir"
            exit 1
        fi
    done
    
    # Merge generation.jsonl files
    local merged_file="$output_dir/generation.jsonl"
    echo "Merging generation files..."
    
    # Sort shard directories by their index range to maintain order
    IFS=$'\n' sorted_dirs=($(printf '%s\n' "${shard_dirs[@]}" | sort -t'_' -k2 -n))
    unset IFS
    
    > "$merged_file"  # Clear/create the merged file
    for shard_dir in "${sorted_dirs[@]}"; do
        cat "$shard_dir/generation.jsonl" >> "$merged_file"
        echo "  Added $(wc -l < "$shard_dir/generation.jsonl") samples from $(basename "$shard_dir")"
    done
    
    local total_samples=$(wc -l < "$merged_file")
    echo "Total merged samples: $total_samples"
    
    # Generate merged result.log using Python
    echo "Generating merged result.log..."
    python3 << PYEOF
import json
import math

def pass_at_k(correct_lst, k):
    assert k > 0, "k must be greater than 0"
    assert k <= len(correct_lst), "k must be less than or equal to the length of correct_lst"
    num_samples = len(correct_lst)
    num_correct = sum(correct_lst)
    if num_correct == 0:
        return 0.0
    elif (num_samples - num_correct) < k:
        return 1.0
    else:
        log_ratio = 0.0
        for i in range(k):
            log_ratio += math.log(num_samples - num_correct - i) - math.log(num_samples - i)
        return 1.0 - math.exp(log_ratio)

def mean_at_k(correct_lst, k):
    assert k > 0, "k must be greater than 0"
    assert k <= len(correct_lst), "k must be less than or equal to the length of correct_lst"
    top_k_correct = correct_lst[:k]
    return sum(top_k_correct) / k

# Load merged generations
generations = []
with open("$merged_file", "r") as f:
    for line in f:
        generations.append(json.loads(line))

print(f"Loaded {len(generations)} samples")

# Detect n from the first sample
n = len(generations[0]["response"]) if generations else 1

# Compute ks
ks = [2 ** e for e in range(0, 10)]
ks_pass = [k for k in ks if (2 * k) <= n or k == 1]
ks_mean = [k for k in ks if k <= n or k == 1]

# Try to detect category keys from first sample
category_keys = []
for key in ["level", "type", "subject", "domain"]:
    if key in generations[0]:
        category_keys.append(key)

# Write result.log
with open("$output_dir/result.log", "w") as f:
    for k in ks_pass:
        f.write(f"pass@{k} >>>\n")
        if category_keys:
            for ck in category_keys:
                all_cate = sorted(list(set([str(g[ck]) for g in generations if ck in g])))
                for cate in all_cate:
                    pass_prob_lst = [g.get(f"pass@{k}", pass_at_k(g["correct"], k)) for g in generations if str(g.get(ck)) == cate]
                    if pass_prob_lst:
                        pass_prob_avg = sum(pass_prob_lst) / len(pass_prob_lst)
                        f.write(f"{cate}: {pass_prob_avg * 100:.1f} ({len(pass_prob_lst)} samples)\n")
        
        # overall
        pass_prob_lst = [g.get(f"pass@{k}", pass_at_k(g["correct"], k)) for g in generations]
        pass_prob_avg = sum(pass_prob_lst) / len(pass_prob_lst)
        f.write(f"Overall: {pass_prob_avg * 100:.1f} ({len(pass_prob_lst)} samples)\n\n")

    for k in ks_mean:
        f.write(f"mean@{k} >>>\n")
        if category_keys:
            for ck in category_keys:
                all_cate = sorted(list(set([str(g[ck]) for g in generations if ck in g])))
                for cate in all_cate:
                    mean_prob_lst = [g.get(f"mean@{k}", mean_at_k(g["correct"], k)) for g in generations if str(g.get(ck)) == cate]
                    if mean_prob_lst:
                        mean_prob_avg = sum(mean_prob_lst) / len(mean_prob_lst)
                        f.write(f"{cate}: {mean_prob_avg * 100:.1f} ({len(mean_prob_lst)} samples)\n")
        
        # overall
        mean_prob_lst = [g.get(f"mean@{k}", mean_at_k(g["correct"], k)) for g in generations]
        mean_prob_avg = sum(mean_prob_lst) / len(mean_prob_lst)
        f.write(f"Overall: {mean_prob_avg * 100:.1f} ({len(mean_prob_lst)} samples)\n\n")

print("Results written to $output_dir/result.log")
PYEOF

    # Print the merged result
    echo ""
    echo "=========================================="
    echo "Merged Results"
    echo "=========================================="
    cat "$output_dir/result.log"
    
    echo ""
    echo "=========================================="
    echo "Merge completed!"
    echo "Merged generation file: $merged_file"
    echo "Merged result file: $output_dir/result.log"
    echo "=========================================="
}

# =============================================================================
# Parse Arguments
# =============================================================================

MODEL=""
DATA_PATH=""
PROBLEM_KEY=""
ANSWER_KEY=""
CATEGORY_KEYS=""
OUTPUT_DIR="./exp/local_eval"
GPU_ID="0"
TEMPERATURE="0.6"
TOP_P="0.95"
N="1"
MAX_MODEL_LEN="$DEFAULT_MAX_MODEL_LEN"
MAX_NEW_TOKENS="$DEFAULT_MAX_NEW_TOKENS"
START_IDX=""
END_IDX=""

# Parallel mode options
PARALLEL_MODE="false"
GPUS=""
TOTAL_SAMPLES=""

# Merge mode options
MERGE_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --problem-key)
            PROBLEM_KEY="$2"
            shift 2
            ;;
        --answer-key)
            ANSWER_KEY="$2"
            shift 2
            ;;
        --category-keys)
            CATEGORY_KEYS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --end-idx)
            END_IDX="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_MODE="true"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --total-samples)
            TOTAL_SAMPLES="$2"
            shift 2
            ;;
        --merge)
            MERGE_MODE="true"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Handle Merge Mode
# =============================================================================

if [[ "$MERGE_MODE" == "true" ]]; then
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "Error: --output-dir is required for merge mode"
        exit 1
    fi
    merge_results "$OUTPUT_DIR"
    exit 0
fi

# =============================================================================
# Validate Required Arguments
# =============================================================================

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    show_help
    exit 1
fi

if [[ -z "$DATA_PATH" ]]; then
    echo "Error: --data is required"
    show_help
    exit 1
fi

if [[ -z "$PROBLEM_KEY" ]]; then
    echo "Error: --problem-key is required"
    show_help
    exit 1
fi

if [[ -z "$ANSWER_KEY" ]]; then
    echo "Error: --answer-key is required"
    show_help
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Validate parallel mode arguments
if [[ "$PARALLEL_MODE" == "true" ]]; then
    if [[ -z "$GPUS" ]]; then
        echo "Error: --gpus is required when using --parallel"
        exit 1
    fi
fi

# =============================================================================
# Build Base Output Directory
# =============================================================================

DATA_BASENAME=$(basename "$DATA_PATH" | sed 's/\.[^.]*$//')
MODEL_BASENAME=$(basename "$MODEL")
BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_BASENAME}/${DATA_BASENAME}/temp_${TEMPERATURE}_top_p_${TOP_P}_n_${N}"

# =============================================================================
# Handle Parallel Mode
# =============================================================================

if [[ "$PARALLEL_MODE" == "true" ]]; then
    echo "=========================================="
    echo "Parallel Multi-GPU Evaluation"
    echo "=========================================="
    echo "Model: $MODEL"
    echo "Data: $DATA_PATH"
    echo "Problem Key: $PROBLEM_KEY"
    echo "Answer Key: $ANSWER_KEY"
    if [[ -n "$CATEGORY_KEYS" ]]; then
        echo "Category Keys: $CATEGORY_KEYS"
    fi
    echo "GPUs: $GPUS"
    echo "Base Output: $BASE_OUTPUT_DIR"
    echo "Config: temp=$TEMPERATURE, top_p=$TOP_P, n=$N"
    echo "=========================================="
    
    # Parse GPU IDs
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "Number of GPUs: $NUM_GPUS"
    
    # Get total samples if not provided
    if [[ -z "$TOTAL_SAMPLES" ]]; then
        echo "Counting dataset samples..."
        TOTAL_SAMPLES=$(count_dataset_samples "$DATA_PATH")
    fi
    echo "Total samples: $TOTAL_SAMPLES"
    
    # Calculate shard sizes
    SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))
    REMAINDER=$((TOTAL_SAMPLES % NUM_GPUS))
    
    echo "Samples per GPU: ~$SAMPLES_PER_GPU"
    echo ""
    
    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    
    # Launch processes for each GPU
    PIDS=()
    CURRENT_IDX=0
    
    for i in "${!GPU_ARRAY[@]}"; do
        GPU_ID="${GPU_ARRAY[$i]}"
        SHARD_START=$CURRENT_IDX
        
        # Add one extra sample to first REMAINDER GPUs to distribute remainder
        if [[ $i -lt $REMAINDER ]]; then
            SHARD_SIZE=$((SAMPLES_PER_GPU + 1))
        else
            SHARD_SIZE=$SAMPLES_PER_GPU
        fi
        
        SHARD_END=$((SHARD_START + SHARD_SIZE))
        CURRENT_IDX=$SHARD_END
        
        SHARD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/shard_${SHARD_START}_${SHARD_END}"
        
        echo "GPU $GPU_ID: samples $SHARD_START to $SHARD_END ($(($SHARD_END - $SHARD_START)) samples)"
        echo "  Output: $SHARD_OUTPUT_DIR"
        
        # Build command for this shard
        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python3 uni_eval.py"
        CMD="$CMD --base_model \"$MODEL\""
        CMD="$CMD --local_data_path \"$DATA_PATH\""
        CMD="$CMD --problem_key_override \"$PROBLEM_KEY\""
        CMD="$CMD --answer_key_override \"$ANSWER_KEY\""
        CMD="$CMD --output_dir \"$SHARD_OUTPUT_DIR\""
        CMD="$CMD --chat_template_name \"$DEFAULT_CHAT_TEMPLATE_NAME\""
        CMD="$CMD --system_prompt_name \"$DEFAULT_SYSTEM_PROMPT_NAME\""
        CMD="$CMD --bf16 $DEFAULT_BF16"
        CMD="$CMD --tensor_parallel_size $DEFAULT_TENSOR_PARALLEL_SIZE"
        CMD="$CMD --max_model_len $MAX_MODEL_LEN"
        CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
        CMD="$CMD --temperature $TEMPERATURE"
        CMD="$CMD --top_p $TOP_P"
        CMD="$CMD --n $N"
        CMD="$CMD --start_idx $SHARD_START --end_idx $SHARD_END"
        
        # Add optional category keys
        if [[ -n "$CATEGORY_KEYS" ]]; then
            CATEGORY_KEYS_JSON="[$(echo "$CATEGORY_KEYS" | sed 's/,/","/g' | sed 's/^/"/; s/$/"/')]"
            CMD="$CMD --category_keys_override '$CATEGORY_KEYS_JSON'"
        fi
        
        # Launch in background
        LOG_FILE="${SHARD_OUTPUT_DIR}.log"
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "  Log: $LOG_FILE"
        eval "$CMD" > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done
    
    echo ""
    echo "Launched ${#PIDS[@]} parallel processes"
    echo "PIDs: ${PIDS[*]}"
    echo ""
    echo "Waiting for all processes to complete..."
    
    # Wait for all processes and track failures
    FAILED=0
    for i in "${!PIDS[@]}"; do
        PID="${PIDS[$i]}"
        GPU_ID="${GPU_ARRAY[$i]}"
        if wait $PID; then
            echo "  GPU $GPU_ID (PID $PID): Completed successfully"
        else
            echo "  GPU $GPU_ID (PID $PID): FAILED"
            FAILED=$((FAILED + 1))
        fi
    done
    
    if [[ $FAILED -gt 0 ]]; then
        echo ""
        echo "WARNING: $FAILED process(es) failed!"
        echo "Check individual log files for details."
        exit 1
    fi
    
    echo ""
    echo "All processes completed successfully!"
    echo ""
    
    # Automatically merge results
    merge_results "$BASE_OUTPUT_DIR"
    
    exit 0
fi

# =============================================================================
# Single GPU Mode
# =============================================================================

# Build output directory with index range if specified
if [[ -n "$START_IDX" ]] && [[ -n "$END_IDX" ]]; then
    FULL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/shard_${START_IDX}_${END_IDX}"
else
    FULL_OUTPUT_DIR="$BASE_OUTPUT_DIR"
fi

echo "=========================================="
echo "Local Dataset Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "Problem Key: $PROBLEM_KEY"
echo "Answer Key: $ANSWER_KEY"
if [[ -n "$CATEGORY_KEYS" ]]; then
    echo "Category Keys: $CATEGORY_KEYS"
fi
echo "Output: $FULL_OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Config: temp=$TEMPERATURE, top_p=$TOP_P, n=$N"
if [[ -n "$START_IDX" ]] && [[ -n "$END_IDX" ]]; then
    echo "Index Range: $START_IDX to $END_IDX"
fi
echo "=========================================="

# Build the command
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python3 uni_eval.py"
CMD="$CMD --base_model \"$MODEL\""
CMD="$CMD --local_data_path \"$DATA_PATH\""
CMD="$CMD --problem_key_override \"$PROBLEM_KEY\""
CMD="$CMD --answer_key_override \"$ANSWER_KEY\""
CMD="$CMD --output_dir \"$FULL_OUTPUT_DIR\""
CMD="$CMD --chat_template_name \"$DEFAULT_CHAT_TEMPLATE_NAME\""
CMD="$CMD --system_prompt_name \"$DEFAULT_SYSTEM_PROMPT_NAME\""
CMD="$CMD --bf16 $DEFAULT_BF16"
CMD="$CMD --tensor_parallel_size $DEFAULT_TENSOR_PARALLEL_SIZE"
CMD="$CMD --max_model_len $MAX_MODEL_LEN"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --n $N"

# Add optional category keys
if [[ -n "$CATEGORY_KEYS" ]]; then
    # Convert comma-separated to JSON array format
    CATEGORY_KEYS_JSON="[$(echo "$CATEGORY_KEYS" | sed 's/,/","/g' | sed 's/^/"/; s/$/"/')]"
    CMD="$CMD --category_keys_override '$CATEGORY_KEYS_JSON'"
fi

# Add optional start/end indices
if [[ -n "$START_IDX" ]] && [[ -n "$END_IDX" ]]; then
    CMD="$CMD --start_idx $START_IDX --end_idx $END_IDX"
fi

# Execute the command
echo ""
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $FULL_OUTPUT_DIR"
echo "=========================================="
