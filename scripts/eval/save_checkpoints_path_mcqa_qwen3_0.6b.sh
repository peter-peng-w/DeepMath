# Define the parent directory
#### Pure GRPO (Original GRPO Loss), with different number of rollouts per question
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs"

#### REINFORCE++ / Batch-level Baseline
CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen3-0.6b/science_qa/grpo/repeat_sampler_10epochs_grpo_8rollouts"

#### JSPO: James-Stein Estimator + Batch-level Shrinkage
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen3-0.6b/science_qa/group_grpo/repeat_sampler_batch_james_stein_8rollouts_no_std"

#### SCPO: James-Stein Estimator + Group-level Shrinkage
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_on_lambda_no_drift"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_mix_lambda_no_drift"

#### Cache the checkpoint list file
## GRPO
OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/grpo/repeat_sampler_10epochs_grpo_8rollouts/checkpoints_to_eval.txt"

## JSPO
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/repeat_sampler_batch_james_stein_8rollouts_no_std/checkpoints_to_eval.txt"

## SCPO
### On-policy Lambda + On-policy Target
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"
### On-policy Lambda + Mixed-Policy Target
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_on_lambda_no_drift/checkpoints_to_eval.txt"
### Mixed-policy Lambda + Mixed-Policy Target
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_mix_lambda_no_drift/checkpoints_to_eval.txt"


## Create the output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Add the base model directory to the file
echo "$CHECKPOINT_DIR" > "$OUTPUT_FILE"

# Find and append all checkpoint subdirectories, sorted by step number
find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V >> "$OUTPUT_FILE"

echo "Checkpoint list saved to $OUTPUT_FILE"