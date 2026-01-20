#!/bin/bash
set -e
set -u

# =============================================================================
# Multi-Checkpoint Evaluation Script
# Evaluates multiple model checkpoints across datasets
# =============================================================================

# Default configuration
export DEFAULT_CHAT_TEMPLATE_NAME="default"
export DEFAULT_SYSTEM_PROMPT_NAME="disabled"
export DEFAULT_BF16="True"
export DEFAULT_TENSOR_PARALLEL_SIZE="1"
export DEFAULT_MAX_MODEL_LEN="4096"             # Qwen2.5-Math: 4096, Qwen3: 16384
export DEFAULT_MAX_NEW_TOKENS="3072"

# VLLM environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Define datasets to evaluate with their corresponding splits
# Format: "data_id|split"
DATASETS=(
    # "zwhe99/MATH|math500"                     # MATH500
    "zwhe99/amc23|test"                       # AMC23
    # "zwhe99/simplerl-OlympiadBench|test"      # OlympiadBench
    # "zwhe99/simplerl-minerva-math|test"       # MinervaMath
    "zwhe99/aime90|2024"                      # AIME24
    "math-ai/aime25|test"                     # AIME25
)

# =============================================================================
# Define evaluation configurations
# =============================================================================
# For difficult datasets including AIME and AMC, we need to sample more rollouts per question (e.g., 64) to get a stable performance,
# as we don't have enough samples in these test benchmarks.
# For easier datasets such as MATH500 and minerva, we can use less rollouts per question (e.g., 16) as they have enough testing samples.
declare -A CONFIGS
# CONFIGS["sampling"]="0.6 0.95 4"    # temperature top_p n
# CONFIGS["sampling"]="0.6 0.95 16"    # temperature top_p n
CONFIGS["sampling"]="0.6 0.95 64"    # temperature top_p n
CONFIGS["greedy"]="0.0 1.0 1"        # temperature top_p n

# =============================================================================
# Functions
# =============================================================================

show_help() {
    cat << EOF
Multi-Checkpoint Evaluation Script

USAGE:
    $0 --checkpoints CHECKPOINT_LIST [OPTIONS]

REQUIRED:
    --checkpoints INPUT     One of:
                           - File containing checkpoint paths (one per line)
                           - Comma-separated list of checkpoint paths  
                           - Directory containing checkpoint subdirectories
                           - Directory pattern with wildcards

OPTIONS:
    --config TYPE          Evaluation config: sampling|greedy (default: both)
    --datasets DATASETS    Comma-separated dataset IDs (default: all)
    --output-base DIR      Base output directory (default: ./exp)
    --gpus GPUS            Comma-separated list of GPU IDs for parallel execution (e.g. "0,1,2,3")
    --skip-existing        Skip checkpoints that already have results
    --pattern PATTERN      Checkpoint directory pattern (default: checkpoint-*)
    --sort-by TYPE         Sort checkpoints by: name|step|time (default: step)
    --limit N              Limit to first N checkpoints after sorting
    --latest N             Get the latest N checkpoints
    --help                 Show this help message

EXAMPLES:
    # Evaluate all checkpoints in a directory
    $0 --checkpoints /path/to/training/run/

    # Evaluate checkpoints matching pattern
    $0 --checkpoints /path/to/training/run/ --pattern "checkpoint-*"

    # Evaluate only recent checkpoints  
    $0 --checkpoints /path/to/training/run/ --sort-by step --limit 5

    # Evaluate latest 3 checkpoints
    $0 --checkpoints /path/to/training/run/ --latest 3

    # Evaluate from file list
    $0 --checkpoints checkpoints.txt

    # Evaluate specific checkpoints with greedy only
    $0 --checkpoints "path1,path2,path3" --config greedy

    # Evaluate on specific datasets
    $0 --checkpoints /path/to/training/run/ --datasets "zwhe99/MATH,zwhe99/amc23"

CHECKPOINT LIST FORMAT:
    # checkpoints.txt example:
    /path/to/checkpoint-1000
    /path/to/checkpoint-2000
    /path/to/checkpoint-3000
    # Lines starting with # are ignored

HUGGINGFACE MODELS:
    You can also evaluate HuggingFace models directly by specifying their model IDs:
    
    # Single HF model via command line
    $0 --checkpoints "Qwen/Qwen2.5-Math-1.5B" --config greedy
    
    # Multiple HF models via command line
    $0 --checkpoints "Qwen/Qwen2.5-Math-1.5B,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # HF models via file (hf_models.txt):
    Qwen/Qwen2.5-Math-1.5B
    Qwen/Qwen2.5-Math-7B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

EOF
}

parse_checkpoints() {
    local checkpoint_input="$1"
    local pattern="$2"
    local sort_by="$3"
    local limit="$4"
    local latest="$5"
    
    if [[ -f "$checkpoint_input" ]]; then
        # Read from file, ignore comments and empty lines
        CHECKPOINT_LIST=($(grep -v '^#' "$checkpoint_input" | grep -v '^$' | tr '\n' ' '))
    elif [[ -d "$checkpoint_input" ]]; then
        # Directory containing checkpoints - discover automatically
        echo "Discovering checkpoints in directory: $checkpoint_input"
        echo "Pattern: $pattern"
        
        # Find all matching directories
        local found_checkpoints=()
        while IFS= read -r -d '' checkpoint_dir; do
            # Verify it's a valid checkpoint (has config.json or pytorch_model.bin)
            if [[ -f "$checkpoint_dir/config.json" ]] || [[ -f "$checkpoint_dir/pytorch_model.bin" ]] || [[ -f "$checkpoint_dir/model.safetensors" ]]; then
                found_checkpoints+=("$checkpoint_dir")
            fi
        done < <(find "$checkpoint_input" -maxdepth 1 -type d -name "$pattern" -print0)
        
        if [[ ${#found_checkpoints[@]} -eq 0 ]]; then
            echo "Error: No valid checkpoints found in $checkpoint_input matching pattern '$pattern'"
            exit 1
        fi
        
        echo "Found ${#found_checkpoints[@]} checkpoints"
        
        # Sort checkpoints
        case "$sort_by" in
            "step")
                # Sort by step number extracted from checkpoint name
                CHECKPOINT_LIST=($(printf '%s\n' "${found_checkpoints[@]}" | sort -V))
                ;;
            "time")
                # Sort by modification time (newest first)
                CHECKPOINT_LIST=($(printf '%s\n' "${found_checkpoints[@]}" | xargs ls -dt))
                ;;
            "name")
                # Sort alphabetically
                CHECKPOINT_LIST=($(printf '%s\n' "${found_checkpoints[@]}" | sort))
                ;;
            *)
                CHECKPOINT_LIST=("${found_checkpoints[@]}")
                ;;
        esac
        
        # Apply limit if specified
        if [[ "$limit" -gt 0 && "$limit" -lt ${#CHECKPOINT_LIST[@]} ]]; then
            if [[ "$latest" -gt 0 ]]; then
                echo "Limiting to latest $latest checkpoints"
                # Get the last N checkpoints (reverse the array and take first N)
                CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | tac | head -n "$latest"))
                # Reverse back to show in ascending order
                CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | tac))
            else
                echo "Limiting to first $limit checkpoints after sorting"
                CHECKPOINT_LIST=("${CHECKPOINT_LIST[@]:0:$limit}")
            fi
        fi
        
    else
        # Parse comma-separated list
        IFS=',' read -ra CHECKPOINT_LIST <<< "$checkpoint_input"
    fi
    
    if [[ ${#CHECKPOINT_LIST[@]} -eq 0 ]]; then
        echo "Error: No checkpoints found"
        exit 1
    fi
    
    echo "Checkpoints to evaluate:"
    for checkpoint in "${CHECKPOINT_LIST[@]}"; do
        echo "  - $(basename "$checkpoint")"
    done
    echo ""
}

get_checkpoint_name() {
    local checkpoint_path="$1"
    # Extract meaningful name from path
    # e.g., /path/to/model/checkpoint-1000 -> checkpoint-1000
    # e.g., /path/to/final_model -> final_model
    basename "$checkpoint_path"
}

acquire_gpu() {
    local gpu_pool_dir="$1"
    local timeout=3600  # 1 hour timeout
    local wait_time=0
    local sleep_interval=5
    
    # Parse GPU_IDS_STR into array (bash arrays can't be exported to subshells)
    IFS=',' read -ra AVAILABLE_GPUS <<< "$GPU_IDS_STR"
    
    while true; do
        # Try to acquire a GPU lock
        for gpu_id in "${AVAILABLE_GPUS[@]}"; do
            local lock_file="${gpu_pool_dir}/gpu_${gpu_id}.lock"
            local lock_dir="${gpu_pool_dir}/gpu_${gpu_id}.lock.d"
            
            # Check for stale locks (process no longer exists)
            if [[ -f "$lock_file" ]] && [[ -d "$lock_dir" ]]; then
                local lock_pid=$(cat "$lock_file" 2>/dev/null)
                if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up stale lock for GPU $gpu_id (PID $lock_pid)" >&2
                    rm -f "$lock_file"
                    rmdir "$lock_dir" 2>/dev/null
                fi
            fi
            
            # Try to create lock file atomically (using mkdir for atomicity)
            if mkdir "$lock_dir" 2>/dev/null; then
                # Successfully acquired lock
                echo $$ > "$lock_file"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $$ acquired GPU $gpu_id" >&2
                echo "$gpu_id"
                return 0
            fi
        done
        
        # No GPU available, wait and retry
        if [[ $wait_time -eq 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $$ waiting for available GPU (pool: $GPU_IDS_STR)..." >&2
        fi
        sleep $sleep_interval
        wait_time=$((wait_time + sleep_interval))
        
        if [[ $wait_time -ge $timeout ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: PID $$ timeout waiting for GPU after ${timeout}s" >&2
            return 1
        fi
    done
}

release_gpu() {
    local gpu_pool_dir="$1"
    local gpu_id="$2"
    
    local lock_file="${gpu_pool_dir}/gpu_${gpu_id}.lock"
    local lock_dir="${gpu_pool_dir}/gpu_${gpu_id}.lock.d"
    
    # Remove lock
    rm -f "$lock_file"
    rmdir "$lock_dir" 2>/dev/null
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $$ released GPU $gpu_id" >&2
}

run_single_evaluation() {
    local checkpoint_path="$1"
    local data_id="$2"
    local split="$3"
    local config_type="$4"
    local temperature="$5"
    local top_p="$6"
    local n="$7"
    local output_base="$8"
    local gpu_pool_dir="$9"
    
    # Dynamically acquire an available GPU
    local gpu_id
    gpu_id=$(acquire_gpu "$gpu_pool_dir")
    local acquire_status=$?
    
    if [[ $acquire_status -ne 0 ]]; then
        echo "Failed to acquire GPU, skipping job: $checkpoint_path - $data_id"
        return 1
    fi
    
    # Ensure GPU is released on exit (success or failure)
    trap "release_gpu '$gpu_pool_dir' '$gpu_id'" EXIT INT TERM
    
    # Create meaningful names
    local checkpoint_name=$(get_checkpoint_name "$checkpoint_path")
    local dataset_name=$(echo "$data_id" | cut -d'/' -f2)
    
    # Determine output subdirectory name
    local output_subdir
    if [[ "$config_type" == "greedy" ]]; then
        output_subdir="greedy"
    else
        output_subdir="temperature_${temperature}_top_p_${top_p}_rollouts_${n}"
    fi
    
    # Organize by checkpoint, then dataset
    local output_dir="${output_base}/${checkpoint_name}/${dataset_name}/${output_subdir}"
    
    echo "=========================================="
    echo "Checkpoint: $checkpoint_name"
    echo "Dataset: $data_id"
    echo "Config: $config_type (temp=$temperature, top_p=$top_p, n=$n)"
    echo "Output: $output_dir"
    echo "GPU: $gpu_id (PID: $$)"
    echo "=========================================="
    
    # Run evaluation with error handling
    local exit_code=0
    CUDA_VISIBLE_DEVICES=$gpu_id python3 uni_eval.py \
        --base_model "$checkpoint_path" \
        --chat_template_name "$DEFAULT_CHAT_TEMPLATE_NAME" \
        --system_prompt_name "$DEFAULT_SYSTEM_PROMPT_NAME" \
        --output_dir "$output_dir" \
        --bf16 "$DEFAULT_BF16" \
        --tensor_parallel_size "$DEFAULT_TENSOR_PARALLEL_SIZE" \
        --data_id "$data_id" \
        --split "$split" \
        --max_model_len "$DEFAULT_MAX_MODEL_LEN" \
        --max_new_tokens "$DEFAULT_MAX_NEW_TOKENS" \
        --temperature "$temperature" \
        --top_p "$top_p" \
        --n "$n" || exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✓ Completed: $checkpoint_name on $dataset_name ($config_type) [GPU $gpu_id]"
    else
        echo "✗ Failed: $checkpoint_name on $dataset_name ($config_type) [GPU $gpu_id] (exit code: $exit_code)"
    fi
    echo ""
    
    # GPU will be released by the trap
    return $exit_code
}

check_existing_results() {
    local checkpoint_path="$1"
    local data_id="$2"
    local config_type="$3"
    local output_base="$4"
    
    local checkpoint_name=$(get_checkpoint_name "$checkpoint_path")
    local dataset_name=$(echo "$data_id" | cut -d'/' -f2)
    
    local output_subdir
    if [[ "$config_type" == "greedy" ]]; then
        output_subdir="greedy"
    else
        IFS=' ' read -r temperature top_p n <<< "${CONFIGS[$config_type]}"
        output_subdir="temperature_${temperature}_top_p_${top_p}_rollouts_${n}"
    fi
    
    local output_dir="${output_base}/${checkpoint_name}/${dataset_name}/${output_subdir}"
    local result_file="${output_dir}/result.log"
    
    [[ -f "$result_file" ]]
}

create_summary_report() {
    local output_base="$1"
    local summary_file="${output_base}/checkpoint_comparison.md"
    
    echo "# Checkpoint Evaluation Summary" > "$summary_file"
    echo "" >> "$summary_file"
    echo "Generated on: $(date)" >> "$summary_file"
    echo "" >> "$summary_file"
    
    # Find all result files
    find "$output_base" -name "result.log" -type f | sort | while read -r result_file; do
        # Extract path components
        local rel_path=${result_file#$output_base/}
        local checkpoint_name=$(echo "$rel_path" | cut -d'/' -f1)
        local dataset_name=$(echo "$rel_path" | cut -d'/' -f2)
        local config_name=$(echo "$rel_path" | cut -d'/' -f3)
        
        echo "## $checkpoint_name - $dataset_name - $config_name" >> "$summary_file"
        echo "" >> "$summary_file"
        echo '```' >> "$summary_file"
        cat "$result_file" >> "$summary_file"
        echo '```' >> "$summary_file"
        echo "" >> "$summary_file"
    done
    
    echo "Summary report created: $summary_file"
}

# =============================================================================
# Main Script
# =============================================================================

# Parse command line arguments
CHECKPOINTS=""
CONFIG_TYPE="both"
SELECTED_DATASETS=""
OUTPUT_BASE="./exp"
GPUS="0"
SKIP_EXISTING=false
PATTERN="checkpoint-*"
SORT_BY="step"
LIMIT=0
LATEST=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoints)
            CHECKPOINTS="$2"
            shift 2
            ;;
        --config)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --datasets)
            SELECTED_DATASETS="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --sort-by)
            SORT_BY="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --latest)
            LATEST="$2"
            shift 2
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

# Handle --latest option
if [[ "$LATEST" -gt 0 ]]; then
    LIMIT="$LATEST"
    SORT_BY="step"
fi

# Validate required arguments
if [[ -z "$CHECKPOINTS" ]]; then
    echo "Error: --checkpoints is required"
    show_help
    exit 1
fi

# Validate config type
if [[ "$CONFIG_TYPE" != "both" && "$CONFIG_TYPE" != "sampling" && "$CONFIG_TYPE" != "greedy" ]]; then
    echo "Error: --config must be 'both', 'sampling', or 'greedy'"
    exit 1
fi

# Parse checkpoints
parse_checkpoints "$CHECKPOINTS" "$PATTERN" "$SORT_BY" "$LIMIT" "$LATEST"
echo "Final list: ${#CHECKPOINT_LIST[@]} checkpoints to evaluate"

# Parse datasets if specified
if [[ -n "$SELECTED_DATASETS" ]]; then
    IFS=',' read -ra SELECTED_DATASET_LIST <<< "$SELECTED_DATASETS"
    # Validate selected datasets exist
    FILTERED_DATASETS=()
    for selected in "${SELECTED_DATASET_LIST[@]}"; do
        found=false
        for dataset_entry in "${DATASETS[@]}"; do
            IFS='|' read -r data_id split <<< "$dataset_entry"
            if [[ "$data_id" == "$selected" ]]; then
                FILTERED_DATASETS+=("$dataset_entry")
                found=true
                break
            fi
        done
        if [[ "$found" == false ]]; then
            echo "Warning: Dataset '$selected' not found in predefined list"
        fi
    done
    DATASETS=("${FILTERED_DATASETS[@]}")
fi

# Determine configurations to run
CONFIGS_TO_RUN=()
if [[ "$CONFIG_TYPE" == "both" ]]; then
    CONFIGS_TO_RUN=("greedy" "sampling")
elif [[ "$CONFIG_TYPE" == "sampling" || "$CONFIG_TYPE" == "greedy" ]]; then
    CONFIGS_TO_RUN=("$CONFIG_TYPE")
fi

# Set up GPUs for parallel execution
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}
GPU_IDS_STR="$GPUS"  # Keep original string format for export (arrays can't be exported)
echo "Using $NUM_GPUS GPUs for parallel evaluation: ${GPU_IDS[*]}"
echo ""

echo "Will evaluate:"
echo "  Checkpoints: ${#CHECKPOINT_LIST[@]}"
echo "  Datasets: ${#DATASETS[@]}"
echo "  Configurations: ${CONFIGS_TO_RUN[@]}"
echo "  Output base: $OUTPUT_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Create GPU pool directory for lock management
GPU_POOL_DIR=$(mktemp -d "${OUTPUT_BASE}/.gpu_pool.XXXXXX")
echo "GPU pool directory: $GPU_POOL_DIR"

# Cleanup function to remove GPU pool on exit
cleanup_gpu_pool() {
    echo "Cleaning up GPU pool..."
    rm -rf "$GPU_POOL_DIR"
}
trap cleanup_gpu_pool EXIT INT TERM

# Export functions and variables to be used by xargs
export -f run_single_evaluation get_checkpoint_name acquire_gpu release_gpu
export GPU_IDS_STR  # Export as string (bash arrays can't be exported to subshells)
export GPU_POOL_DIR

# Create a list of all jobs to run
declare -a JOBS
for checkpoint_path in "${CHECKPOINT_LIST[@]}"; do
    # Check if it's a HuggingFace model identifier (format: org/model or user/model)
    # HF model IDs contain a "/" but don't start with "/" (not absolute paths)
    is_hf_model=false
    if [[ "$checkpoint_path" =~ ^[^/]+/[^/]+$ ]] && [[ ! "$checkpoint_path" =~ ^/ ]]; then
        is_hf_model=true
        echo "Detected HuggingFace model: $checkpoint_path"
    fi
    
    # Skip local validation for HuggingFace models
    if [[ "$is_hf_model" == false ]]; then
        if [[ ! -d "$checkpoint_path" ]] && [[ ! -f "$checkpoint_path/config.json" ]] && [[ ! -f "$checkpoint_path/model.safetensors" ]]; then
            echo "Warning: Checkpoint not found or invalid: $checkpoint_path"
            continue
        fi
    fi
    
    for dataset_entry in "${DATASETS[@]}"; do
        IFS='|' read -r data_id split <<< "$dataset_entry"
        
        for config_type in "${CONFIGS_TO_RUN[@]}"; do
            # Check if results already exist
            if [[ "$SKIP_EXISTING" == true ]] && check_existing_results "$checkpoint_path" "$data_id" "$config_type" "$OUTPUT_BASE"; then
                echo "Skipping existing results for $(get_checkpoint_name "$checkpoint_path") - $data_id - $config_type"
                continue
            fi
            
            # Add job to the queue
            # We expand the config params here because associative arrays cannot be exported to the xargs subshell
            IFS=' ' read -r temperature top_p n <<< "${CONFIGS[$config_type]}"
            JOBS+=("$checkpoint_path $data_id $split $config_type $temperature $top_p $n")
        done
    done
done

# Main evaluation loop - execute jobs in parallel
total_jobs=${#JOBS[@]}
echo "Total evaluation jobs to run: $total_jobs"
echo "Jobs will dynamically acquire GPUs from pool: ${GPU_IDS[*]}"
echo ""

# Execute jobs with dynamic GPU allocation
# Each job will acquire an available GPU when it starts and release it when done
for job in "${JOBS[@]}"; do
    echo "$job $OUTPUT_BASE $GPU_POOL_DIR"
done | xargs -n 9 -P "$NUM_GPUS" bash -c 'run_single_evaluation "$@"' _

echo "=========================================="
echo "All evaluations completed!"
echo "Creating summary report..."
echo "=========================================="

# Create summary report
create_summary_report "$OUTPUT_BASE"

echo "Results organized in: $OUTPUT_BASE"
echo "Summary available at: $OUTPUT_BASE/checkpoint_comparison.md"
