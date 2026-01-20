################################################
####     Parent directory of the Ckpts      ####
################################################
# 
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted"
#### Pure GRPO (Original GRPO Loss), with different number of rollouts per question
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts"

#### Group GRPO
### Repeat Sampler

### James-Stein Estimator
# Batch-level JSPO
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std"

### Proportional Cluster Sampling GRPO
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_cluster_sampler_semantic_group_cosine_weighted_8rollouts_8prompts"
CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std"


################################################
####     Cache the checkpoint list file     ####
################################################

## Vanilla GRPO
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt"

## James-Stein Estimator
# OUTPUT_FILE='./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt'

## Proportional Cluster Sampling GRPO
OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"

# Create the output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Add the base model directory to the file
echo "$CHECKPOINT_DIR" > "$OUTPUT_FILE"

# Find and append all checkpoint subdirectories, sorted by step number
find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V >> "$OUTPUT_FILE"

echo "Checkpoint list saved to $OUTPUT_FILE"