#### Base Model ####
# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "Qwen/Qwen2.5-Math-7B" \
#     --gpus "2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/base \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "Qwen/Qwen2.5-Math-7B-Instruct" \
#     --gpus "2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/instruct \
#     --skip-existing

#### Basline GRPO ####

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/math500/pure_grpo/lvl3to5_8k_20epochs_128effectbs \
#     --skip-existing

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/grpo/lvl3to5_8k_20epochs_128effectbs_no_std_repeat_sampler_group_mean_8rollouts \
#     --skip-existing


#### Semantic Group Numerical Mean Baseline Estimator ####


#### Semantic Group Cosine Weighted Mean Baseline Estimator ####

### Repeat Sampler, batch-level per-prompt KNN grouping ###

#### James-Stein Estimator ####

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_repeat_sampler_batch_james_stein_8rollouts_8prompts_no_std \
#     --skip-existing

./scripts/eval/multi_checkpoint_eval.sh \
    --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval.txt \
    --gpus "0,1,2,3" \
    --config sampling \
    --output-base ./exp/qwen2.5-math-7b/math500/group_grpo/lvl3to5_8k_8epochs_128effectbs_skill_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
    --skip-existing