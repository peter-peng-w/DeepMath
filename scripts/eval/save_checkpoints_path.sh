# Define the parent directory
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted"
#### Pure GRPO (Original GRPO Loss), with different number of rollouts per question
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_2rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_4rollouts"

#### Group GRPO
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_2rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_16gcombo"
CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo"

#### Cache the checkpoint list file
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_2rollouts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_16gcombo/checkpoints_to_eval.txt"
OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoints_to_eval.txt"

# Create the output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Add the base model directory to the file
echo "$CHECKPOINT_DIR" > "$OUTPUT_FILE"

# Find and append all checkpoint subdirectories, sorted by step number
find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V >> "$OUTPUT_FILE"

echo "Checkpoint list saved to $OUTPUT_FILE"