# Define the parent directory
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted"
#### Pure GRPO (Original GRPO Loss), with different number of rollouts per question
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_2rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_4rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_with_std_repeat_sampler_group_mean_8rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts"

#### Group GRPO
### Repeat Sampler
## Compositional Group Numerical Mean
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_mean"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_mean_4rollouts_32gcombo"

## Compositional Group Cosine Similarity Mean
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_2rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_16gcombo"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted"

### Proportional Cluster Sampling GRPO
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_cluster_sampler_semantic_group_cosine_weighted_8rollouts_8prompts"

### James-Stein Estimator

## Repeat Sampler ##
CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_2rollouts_8prompts_no_std"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_run3"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std"

# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std"
# CHECKPOINT_DIR="/project/shenresearchgroup/Peng/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_max_entropy_weighted_cluster_sampler_compositional_james_stein_8rolloΩuts_8prompts_no_std"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_1.0"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_2.0"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0_alpha_0.0"

########################################################################################
####   We relabel the skill of each problem by removing some too general skills.    ####
########################################################################################

#### Proportional sampler selects the clusters based on the size ####
## Standard Cluster-JSPO
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std"

## Proportional Difficulty Aware Cluster ##
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts"
# CHECKPOINT_DIR="/scratch/pw7nc/LLM_reasoning/off-policy-rl/replay-think/exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std"




#### Cache the checkpoint list file
## Vanilla GRPO
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_with_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt"

## Compositional Group Numerical Mean
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_mean_8rollouts_64combo/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_mean_4rollouts_32gcombo/checkpoints_to_eval.txt"\\

## Compositional Group Cosine Similarity Mean
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_2rollouts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_16gcombo/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoints_to_eval.txt"

## James-Stein Estimator
OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_2rollouts_8prompts_no_std/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_run3/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_max_entropy_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std/checkpoints_to_eval.txt"

# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_1.0/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_2.0/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0/checkpoints_to_eval.txt"
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0_alpha_0.0/checkpoints_to_eval.txt"

# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt"

## Proportional Cluster Sampling GRPO
# OUTPUT_FILE="./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_cluster_sampler_semantic_group_cosine_weighted_8rollouts_8prompts/checkpoints_to_eval.txt"

# Create the output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Add the base model directory to the file
echo "$CHECKPOINT_DIR" > "$OUTPUT_FILE"

# Find and append all checkpoint subdirectories, sorted by step number
find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V >> "$OUTPUT_FILE"

echo "Checkpoint list saved to $OUTPUT_FILE"