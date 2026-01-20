set -e
set -u

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --chat_template_name default \
    --system_prompt_name disabled \
    --output_dir ./exp/minerva/qwen2.5-math-1.5b/temperature_0.6_top_p_0.95 \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id zwhe99/simplerl-minerva-math \
    --split test \
    --max_model_len 3072 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 16

# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
#     --base_model Qwen/Qwen2.5-Math-1.5B \
#     --chat_template_name default \
#     --system_prompt_name disabled \
#     --output_dir ./exp/amc23/qwen2.5-math-1.5b/greedy \
#     --bf16 True \
#     --tensor_parallel_size 4 \
#     --data_id zwhe99/amc23 \
#     --split test \
#     --max_model_len 3072 \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --n 2