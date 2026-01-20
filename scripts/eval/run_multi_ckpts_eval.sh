#### Base Model ####
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "Qwen/Qwen2.5-Math-1.5B" \
#     --gpus "0,1" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/base \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "Qwen/Qwen2.5-Math-1.5B-Instruct" \
#     --gpus "2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/instruct \
#     --skip-existing

#### Basline GRPO ####

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_with_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_with_std_repeat_sampler_group_mean_8rollouts \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts \
#     --skip-existing


#### Semantic Group Numerical Mean Baseline Estimator ####


#### Semantic Group Cosine Weighted Mean Baseline Estimator ####

### Repeat Sampler, batch-level per-prompt KNN grouping ###
## G=2, K=8
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_2rollouts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_2rollouts \
#     --skip-existing

## G=4, K=2
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_8gcombo \
#     --skip-existing

## G=4, K=4
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_16gcombo/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_16gcombo \
#     --skip-existing

## G=4, K=8
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo \
#     --skip-existing

## G=8, K=8
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3,4,5" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted \
#     --skip-existing

### Proportional Cluster Sampler ###
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_cluster_sampler_semantic_group_cosine_weighted_8rollouts_8prompts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_cluster_sampler_semantic_group_cosine_weighted_8rollouts_8prompts \
#     --skip-existing

#### James-Stein Estimator ####

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_run3/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std_run3 \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

./scripts/eval/multi_checkpoint_eval.sh \
    --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_2rollouts_8prompts_no_std/checkpoints_to_eval.txt \
    --gpus "0,1,2,3" \
    --config sampling \
    --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_batch_james_stein_2rollouts_8prompts_no_std \
    --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts \
#     --skip-existing

### Compositional JSPO ###
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_proportional_difficulty_aware_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_max_entropy_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_max_entropy_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_1.0/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_1.0 \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_2.0/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_2.0 \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0 \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0_alpha_0.0/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_skills_20epochs_128effectbs_maxent_only_size_prior_weighted_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/lambda_8.0_alpha_0.0 \
#     --skip-existing


# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing