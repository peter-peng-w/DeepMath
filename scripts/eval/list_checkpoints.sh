#!/bin/bash
set -e
set -u

# =============================================================================
# Checkpoint Discovery Utility
# Lists checkpoints in a directory without running evaluation
# =============================================================================

show_help() {
    cat << EOF
Checkpoint Discovery Utility

USAGE:
    $0 CHECKPOINT_DIR [OPTIONS]

ARGUMENTS:
    CHECKPOINT_DIR         Directory containing checkpoint subdirectories

OPTIONS:
    --pattern PATTERN      Checkpoint directory pattern (default: checkpoint-*)
    --sort-by TYPE         Sort checkpoints by: name|step|time (default: step)
    --limit N              Limit to first N checkpoints after sorting
    --latest N             Get the latest N checkpoints (equivalent to --sort-by step --limit N --reverse)
    --full-path            Show full paths instead of just names
    --help                 Show this help message

EXAMPLES:
    # List all checkpoints
    $0 /path/to/training/run/

    # List only recent 5 checkpoints
    $0 /path/to/training/run/ --limit 5

    # List latest 3 checkpoints
    $0 /path/to/training/run/ --latest 3

    # List with full paths
    $0 /path/to/training/run/ --full-path

EOF
}

# Default values
PATTERN="checkpoint-*"
SORT_BY="step"
LIMIT=0
FULL_PATH=false
LATEST=0

# Parse arguments
if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

CHECKPOINT_DIR="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --full-path)
            FULL_PATH=true
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

# Handle --latest option
if [[ "$LATEST" -gt 0 ]]; then
    LIMIT="$LATEST"
    SORT_BY="step"
fi

# Validate directory
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: Directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Discovering checkpoints in: $CHECKPOINT_DIR"
echo "Pattern: $PATTERN"

# Find all matching directories
found_checkpoints=()
while IFS= read -r -d '' checkpoint_dir; do
    # Verify it's a valid checkpoint
    if [[ -f "$checkpoint_dir/config.json" ]] || [[ -f "$checkpoint_dir/pytorch_model.bin" ]] || [[ -f "$checkpoint_dir/model.safetensors" ]]; then
        found_checkpoints+=("$checkpoint_dir")
    fi
done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "$PATTERN" -print0)

if [[ ${#found_checkpoints[@]} -eq 0 ]]; then
    echo "No valid checkpoints found matching pattern '$PATTERN'"
    exit 1
fi

echo "Found ${#found_checkpoints[@]} checkpoints"

# Sort checkpoints
case "$SORT_BY" in
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
if [[ "$LIMIT" -gt 0 && "$LIMIT" -lt ${#CHECKPOINT_LIST[@]} ]]; then
    if [[ "$LATEST" -gt 0 ]]; then
        echo "Showing latest $LATEST checkpoints"
        # Get the last N checkpoints (reverse the array and take first N)
        CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | tac | head -n "$LATEST"))
        # Reverse back to show in ascending order
        CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | tac))
    else
        echo "Showing first $LIMIT checkpoints after sorting by $SORT_BY"
        CHECKPOINT_LIST=("${CHECKPOINT_LIST[@]:0:$LIMIT}")
    fi
else
    echo "Sorted by: $SORT_BY"
fi

echo ""
echo "Checkpoints:"
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    if [[ "$FULL_PATH" == true ]]; then
        echo "  $checkpoint"
    else
        echo "  $(basename "$checkpoint")"
    fi
done

echo ""
echo "Total: ${#CHECKPOINT_LIST[@]} checkpoints"
