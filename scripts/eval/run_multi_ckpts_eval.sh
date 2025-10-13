./scripts/eval/multi_checkpoint_eval.sh \
    --checkpoints ./scripts/eval/cache/ckpt_lists/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo/checkpoints_to_eval.txt \
    --gpus "0,1,2,3" \
    --config sampling \
    --output-base ./exp/qwen2.5-math-1.5b/math500/group_grpo/lvl3to5_8k_20epochs_128effectbs_repeat_sampler_semantic_group_cosine_weighted_4rollouts_32gcombo \
    --skip-existing