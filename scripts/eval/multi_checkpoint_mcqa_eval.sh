#!/bin/bash
set -e
set -u

# =============================================================================
# Multi-Checkpoint MCQA Evaluation Script
# Evaluates multiple model checkpoints on MCQA benchmarks (Science QA, GPQA Diamond)
# Parallel execution across GPUs with dynamic GPU allocation.
#
# USAGE:
#   bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
#       --checkpoints /path/to/training/run/ \
#       --gpus "0,1,2,3" \
#       --dataset all
#
# EXAMPLES:
#   # Evaluate all checkpoints in a training directory
#   bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
#       --checkpoints exp/qwen3-0.6b/science_qa/grpo/repeat_sampler_grpo_8rollouts/ \
#       --gpus "0,1,2,3" --dataset all
#
#   # Evaluate latest 5 checkpoints on Science QA only
#   bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
#       --checkpoints /path/to/run/ --latest 5 --dataset science_qa
#
#   # Evaluate specific checkpoints from a file
#   bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
#       --checkpoints checkpoints.txt --dataset gpqa_diamond
#
#   # Evaluate HuggingFace models
#   bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
#       --checkpoints "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B" --dataset all
#
#   # Evaluate with thinking enabled
#   bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
#       --checkpoints /path/to/run/ --enable-thinking --dataset all
# =============================================================================

# Defaults
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DEFAULT_MAX_MODEL_LEN="4096"
DEFAULT_MAX_NEW_TOKENS="3072"

# MCQA datasets: "name|data_path|answer_key|category_keys"
declare -A MCQA_DATASETS
MCQA_DATASETS["science_qa"]="data/science_qa/test.json|solution|source"
MCQA_DATASETS["gpqa_diamond"]="data/gpqa_diamond/test.json|solution|domain"

# Evaluation configs: "temperature top_p n"
declare -A CONFIGS
CONFIGS["sampling"]="0.6 0.95 8"
CONFIGS["greedy"]="0.0 1.0 1"

# =============================================================================
# Argument parsing
# =============================================================================
CHECKPOINTS=""
DATASET="all"
CONFIG_TYPE="sampling"
GPUS="0"
OUTPUT_BASE=""
SKIP_EXISTING=false
PATTERN="checkpoint-*"
SORT_BY="step"
LIMIT=0
LATEST=0
ENABLE_THINKING=""
N_OVERRIDE=""
TEMPERATURE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoints) CHECKPOINTS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --config) CONFIG_TYPE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --sort-by) SORT_BY="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --latest) LATEST="$2"; LIMIT="$2"; SORT_BY="step"; shift 2 ;;
        --enable-thinking) ENABLE_THINKING="--enable_thinking True"; shift ;;
        --n) N_OVERRIDE="$2"; shift 2 ;;
        --temperature) TEMPERATURE_OVERRIDE="$2"; shift 2 ;;
        --max-model-len) DEFAULT_MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-new-tokens) DEFAULT_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 --checkpoints <path> --gpus <gpu_ids> [options]"
            echo ""
            echo "Required:"
            echo "  --checkpoints PATH    File, directory, or comma-separated list of checkpoints"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET     science_qa, gpqa_diamond, or all (default: all)"
            echo "  --config TYPE         sampling or greedy (default: sampling)"
            echo "  --gpus IDS            Comma-separated GPU IDs (default: 0)"
            echo "  --output-base DIR     Output base directory"
            echo "  --skip-existing       Skip if results already exist"
            echo "  --pattern PATTERN     Checkpoint dir pattern (default: checkpoint-*)"
            echo "  --sort-by TYPE        Sort by: step, time, name (default: step)"
            echo "  --limit N             Limit to first N checkpoints"
            echo "  --latest N            Evaluate latest N checkpoints"
            echo "  --enable-thinking     Enable Qwen3 thinking mode"
            echo "  --n N                 Override number of samples per question"
            echo "  --temperature T       Override temperature"
            echo "  --max-model-len N     Max context length (default: 4096)"
            echo "  --max-new-tokens N    Max generation tokens (default: 3072)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINTS" ]]; then
    echo "Error: --checkpoints is required"
    exit 1
fi

# =============================================================================
# Checkpoint discovery (reused from multi_checkpoint_eval.sh)
# =============================================================================
parse_checkpoints() {
    local input="$1"
    if [[ -f "$input" ]]; then
        CHECKPOINT_LIST=($(grep -v '^#' "$input" | grep -v '^$' | tr '\n' ' '))
    elif [[ -d "$input" ]]; then
        local found=()
        while IFS= read -r -d '' d; do
            if [[ -f "$d/config.json" ]] || [[ -f "$d/model.safetensors" ]]; then
                found+=("$d")
            fi
        done < <(find "$input" -maxdepth 1 -type d -name "$PATTERN" -print0)
        if [[ ${#found[@]} -eq 0 ]]; then
            echo "Error: No checkpoints found in $input matching '$PATTERN'"
            exit 1
        fi
        CHECKPOINT_LIST=($(printf '%s\n' "${found[@]}" | sort -V))
        if [[ "$LIMIT" -gt 0 && "$LIMIT" -lt ${#CHECKPOINT_LIST[@]} ]]; then
            if [[ "$LATEST" -gt 0 ]]; then
                CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | tail -n "$LATEST"))
            else
                CHECKPOINT_LIST=("${CHECKPOINT_LIST[@]:0:$LIMIT}")
            fi
        fi
    else
        IFS=',' read -ra CHECKPOINT_LIST <<< "$input"
    fi
    echo "Checkpoints to evaluate (${#CHECKPOINT_LIST[@]}):"
    for c in "${CHECKPOINT_LIST[@]}"; do echo "  - $(basename "$c")"; done
    echo ""
}

parse_checkpoints "$CHECKPOINTS"

# =============================================================================
# Determine which datasets to evaluate
# =============================================================================
DATASETS_TO_EVAL=()
if [[ "$DATASET" == "all" ]]; then
    DATASETS_TO_EVAL=("science_qa" "gpqa_diamond")
else
    IFS=',' read -ra DATASETS_TO_EVAL <<< "$DATASET"
fi

# Validate datasets
for ds in "${DATASETS_TO_EVAL[@]}"; do
    if [[ -z "${MCQA_DATASETS[$ds]+x}" ]]; then
        echo "Error: Unknown dataset '$ds'. Available: science_qa, gpqa_diamond, all"
        exit 1
    fi
done

# =============================================================================
# GPU pool management
# =============================================================================
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}
echo "Using $NUM_GPUS GPUs: ${GPU_IDS[*]}"

GPU_POOL_DIR=$(mktemp -d "${REPO_ROOT}/.gpu_pool.XXXXXX")
cleanup() { rm -rf "$GPU_POOL_DIR"; }
trap cleanup EXIT INT TERM

acquire_gpu() {
    local timeout=3600
    local waited=0
    while true; do
        for gid in "${GPU_IDS[@]}"; do
            local lock="${GPU_POOL_DIR}/gpu_${gid}.lock.d"
            if mkdir "$lock" 2>/dev/null; then
                echo "$gid"
                return 0
            fi
        done
        sleep 5
        waited=$((waited + 5))
        if [[ $waited -ge $timeout ]]; then return 1; fi
    done
}

release_gpu() {
    rmdir "${GPU_POOL_DIR}/gpu_${1}.lock.d" 2>/dev/null
}

# =============================================================================
# Run evaluation
# =============================================================================

# Get config params
IFS=' ' read -r temperature top_p n_samples <<< "${CONFIGS[$CONFIG_TYPE]}"
[[ -n "$N_OVERRIDE" ]] && n_samples="$N_OVERRIDE"
[[ -n "$TEMPERATURE_OVERRIDE" ]] && temperature="$TEMPERATURE_OVERRIDE"

run_one() {
    local checkpoint="$1"
    local ds_name="$2"

    IFS='|' read -r data_path answer_key category_keys <<< "${MCQA_DATASETS[$ds_name]}"
    local ckpt_name=$(basename "$checkpoint")

    # Determine output dir
    local thinking_suffix=""
    [[ -n "$ENABLE_THINKING" ]] && thinking_suffix="_thinking"
    local base="${OUTPUT_BASE:-exp/local_eval/${ckpt_name}}"
    local out_dir="${base}/${ds_name}/temp_${temperature}_top_p_${top_p}_n_${n_samples}${thinking_suffix}"

    # Skip if exists
    if [[ "$SKIP_EXISTING" == true ]] && [[ -f "${out_dir}/result.log" ]]; then
        echo "Skipping existing: $ckpt_name / $ds_name"
        return 0
    fi

    # Acquire GPU
    local gpu
    gpu=$(acquire_gpu) || { echo "Failed to acquire GPU for $ckpt_name/$ds_name"; return 1; }

    echo "=========================================="
    echo "Checkpoint: $ckpt_name"
    echo "Dataset:    $ds_name"
    echo "Config:     temp=$temperature top_p=$top_p n=$n_samples"
    echo "GPU:        $gpu"
    echo "Output:     $out_dir"
    echo "=========================================="

    local rc=0
    CUDA_VISIBLE_DEVICES=$gpu python3 mcqa_eval.py \
        --base_model "$checkpoint" \
        --local_data_path "$data_path" \
        --problem_key_override "problem" \
        --answer_key_override "$answer_key" \
        --category_keys_override "$category_keys" \
        --output_dir "$out_dir" \
        --n "$n_samples" \
        --temperature "$temperature" \
        --top_p "$top_p" \
        --max_model_len "$DEFAULT_MAX_MODEL_LEN" \
        --max_new_tokens "$DEFAULT_MAX_NEW_TOKENS" \
        --bf16 \
        $ENABLE_THINKING || rc=$?

    release_gpu "$gpu"

    if [[ $rc -eq 0 ]]; then
        echo "✓ Done: $ckpt_name / $ds_name [GPU $gpu]"
    else
        echo "✗ Failed: $ckpt_name / $ds_name [GPU $gpu] (exit $rc)"
    fi
    return $rc
}

# Build job list and execute in parallel
total=0
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    for ds_name in "${DATASETS_TO_EVAL[@]}"; do
        total=$((total + 1))
    done
done
echo "Total jobs: $total"
echo ""

# Run jobs with background processes, limited by GPU count
pids=()
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    for ds_name in "${DATASETS_TO_EVAL[@]}"; do
        # Wait if we've hit the GPU limit
        while [[ ${#pids[@]} -ge $NUM_GPUS ]]; do
            # Wait for any child to finish
            wait -n 2>/dev/null || true
            # Clean up finished pids
            new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        done

        run_one "$checkpoint" "$ds_name" &
        pids+=($!)
    done
done

# Wait for all remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "All MCQA evaluations completed!"
echo "=========================================="

# Print summary
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    ckpt_name=$(basename "$checkpoint")
    echo ""
    echo "--- $ckpt_name ---"
    for ds_name in "${DATASETS_TO_EVAL[@]}"; do
        thinking_suffix=""
        [[ -n "$ENABLE_THINKING" ]] && thinking_suffix="_thinking"
        base="${OUTPUT_BASE:-exp/local_eval/${ckpt_name}}"
        result_file="${base}/${ds_name}/temp_${temperature}_top_p_${top_p}_n_${n_samples}${thinking_suffix}/result.log"
        if [[ -f "$result_file" ]]; then
            echo "[$ds_name]"
            grep "Overall:" "$result_file" | head -2
        fi
    done
done
