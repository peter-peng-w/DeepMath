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

# ./scripts/eval/multi_checkpoint_eval.sh \
#     --checkpoints "sail/Qwen2.5-Math-1.5B-Oat-Zero" \
#     --gpus "0,1" \
#     --config sampling \
#     --output-base ./exp/qwen2.5-math-1.5b/Oat-Zero \
#     --skip-existing


#### Basline GRPO ####

# ./scripts/eval/multi_checkpoint_mcqa_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/grpo/repeat_sampler_10epochs_grpo_8rollouts/checkpoints_to_eval-split2.txt \
#     --gpus "0,1,2,3" \
#     --output-base ./exp/qwen3-0.6b/science_qa/grpo/repeat_sampler_10epochs_grpo_8rollouts \
#     --enable-thinking \
#     --skip-existing


#### REINFORCE++ / Batch-level Baseline ####


#### James-Stein Estimator - JSPO ####

./scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints ./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/repeat_sampler_batch_james_stein_8rollouts_no_std/checkpoints_to_eval-split1.txt \
    --gpus "0,1,2,3" \
    --output-base ./exp/qwen3-0.6b/science_qa/group_grpo/repeat_sampler_batch_james_stein_8rollouts_no_std \
    --enable-thinking \
    --skip-existing

#### SCPO ####

# On-policy Lambda + On-policy Shrinkage Target
# ./scripts/eval/multi_checkpoint_mcqa_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std/checkpoints_to_eval-split2.txt \
#     --gpus "0,1,2,3" \
#     --output-base ./exp/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std \
#     --enable-thinking \
#     --skip-existing

## On-policy Lambda + Mix-policy Shrinkage Target
# ./scripts/eval/multi_checkpoint_mcqa_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_on_lambda_no_drift/checkpoints_to_eval.txt \
#     --gpus "0,1,2,3" \
#     --output-base ./exp/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_on_lambda_no_drift \
#     --enable-thinking \
#     --skip-existing

## Mixed-Policy Lambda + Mixed-Policy Shrinkage Target
# ./scripts/eval/multi_checkpoint_mcqa_eval.sh \
#     --checkpoints ./scripts/eval/cache/ckpt_lists/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_mix_lambda_no_drift/checkpoints_to_eval-split1.txt \
#     --gpus "0,1,2,3" \
#     --output-base ./exp/qwen3-0.6b/science_qa/group_grpo/skill_grouping_proportional_cluster_sampler_compositional_james_stein_8rollouts_8prompts_no_std_mix_target_mix_lambda_no_drift \
#     --enable-thinking \
#     --skip-existing