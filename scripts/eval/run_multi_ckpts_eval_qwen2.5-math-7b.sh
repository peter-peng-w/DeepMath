#### Base Model ####
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "Qwen/Qwen2.5-Math-7B" \
#     --gpus "2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/base \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "Qwen/Qwen2.5-Math-7B-Instruct" \
#     --gpus "0,1" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/instruct \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "sail/Qwen2.5-Math-7B-Oat-Zero" \
#     --gpus "2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/Oat-Zero \
#     --skip-existing

#### Basline GRPO ####

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_8epochs_128effectbs_no_std_repeat_sampler_group_mean_4rollouts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_8epochs_128effectbs_no_std_repeat_sampler_group_mean_4rollouts \
#     --skip-existing


# ./scripts/eval/multi_checkpoint_eval_avg_4.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_mean_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_mean_8rollouts_8prompts_no_std \
#     --skip-existing

#### Semantic Group Numerical Mean Baseline Estimator ####


#### Semantic Group Cosine Weighted Mean Baseline Estimator ####

### Repeat Sampler, batch-level per-prompt KNN grouping ###

#### James-Stein Estimator ####

# ./scripts/eval/multi_checkpoint_eval_avg_4.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_4rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_4rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_emb_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_emb_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_emb_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_emb_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_4rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_off_policy_value/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_off_policy_value \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_policy_value_no_drift/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_policy_value_no_drift \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_math_subject_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_onpolicy_target_onpolicy_lambda_no_drift/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_math_subject_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_onpolicy_target_onpolicy_lambda_no_drift \
#     --skip-existing


#### Ablation: Grouping Strategy

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_accuracy_binning_uniform_8rollouts_8prompts_no_std_mix_target_onpolicy_lambda_no_drift/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_accuracy_binning_uniform_8rollouts_8prompts_no_std_mix_target_onpolicy_lambda_no_drift \
#     --skip-existing

./scripts/eval/multi_checkpoint_eval.sh \
    --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_accuracy_binning_max_entropy_8rollouts_8prompts_no_std_mix_target_onpolicy_lambda_no_drift/checkpoints_to_eval.txt \
    --gpus "0,1,2,3" \
    --config sampling \
    --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_accuracy_binning_max_entropy_8rollouts_8prompts_no_std_mix_target_onpolicy_lambda_no_drift \
    --skip-existing