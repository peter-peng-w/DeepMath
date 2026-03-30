"""MCQA Evaluation Script.

A streamlined evaluation script for multiple-choice QA tasks using vLLM.
Separate from uni_eval.py to keep math evaluation clean.

Supports local JSON/JSONL files with configurable field names.
Outputs generation.jsonl + result.log + config.json (same format as uni_eval.py).

Usage:
    python mcqa_eval.py \
        --base_model Qwen/Qwen3-0.6B \
        --local_data_path data/science_qa/test.json \
        --problem_key_override problem \
        --answer_key_override solution \
        --category_keys_override "source" \
        --output_dir exp/eval/science_qa \
        --n 8 --temperature 0.6 --bf16

    python mcqa_eval.py \
        --base_model Qwen/Qwen3-0.6B \
        --local_data_path data/gpqa_diamond/test.json \
        --problem_key_override problem \
        --answer_key_override solution \
        --category_keys_override "domain" \
        --output_dir exp/eval/gpqa_diamond \
        --n 8 --temperature 0.6 --bf16
"""

import os
import re
import math
import json
import fire
import torch
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
from datasets import load_dataset

from vllm import LLM, SamplingParams

from utils.data_utils import write_jsonl, write_json, read_jsonl
from utils.chat_template import CHAT_TEMPLATE, SYSTEM_PROMPT


# ============================================================================
# Metrics
# ============================================================================

def pass_at_k(correct_lst: list[bool], k: int) -> float:
    assert k > 0
    assert k <= len(correct_lst)
    num_samples = len(correct_lst)
    num_correct = sum(correct_lst)
    if num_correct == 0:
        return 0.0
    elif (num_samples - num_correct) < k:
        return 1.0
    else:
        log_ratio = 0.0
        for i in range(k):
            log_ratio += math.log(num_samples - num_correct - i) - math.log(num_samples - i)
        return 1.0 - math.exp(log_ratio)


def mean_at_k(correct_lst: list[bool], k: int) -> float:
    assert k > 0
    assert k <= len(correct_lst)
    return sum(correct_lst[:k]) / k


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_mcqa_answer(resp: str) -> str | None:
    """Extract a single letter answer (A-E) from an MCQA response.

    Strips <think>...</think> blocks, then searches for common answer patterns.
    """
    text = resp.split("</think>")[-1].strip() if "</think>" in resp else resp
    for pattern in [
        r"[Tt]he best answer is\s*[\(\[\{]?\s*([A-Ea-e])\s*[\)\]\}]?",
        r"[Tt]he answer is:?\s*[\(\[\{]?\s*([A-Ea-e])\s*[\)\]\}]?",
        r"\\boxed\{([A-Ea-e])\}",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    # Fallback: last standalone answer-range letter
    fallback = re.findall(r"\b([A-E])\b", text)
    if fallback:
        return fallback[-1]
    return None


def check_mcqa_correct(resp: str, gt_answer: str) -> bool:
    """Check if an MCQA response is correct."""
    extracted = extract_mcqa_answer(resp)
    if extracted is not None:
        return extracted == gt_answer.strip().upper()
    return False


# ============================================================================
# Main Evaluation
# ============================================================================

def eval(
    # required
    base_model: str = None,
    local_data_path: str = None,
    output_dir: str = None,

    # field names
    problem_key_override: str = "problem",
    answer_key_override: str = "solution",
    category_keys_override: str | list = None,

    # chat template
    chat_template_name: str = "default",
    system_prompt_name: str = "disabled",
    enable_thinking: bool = False,

    # model
    bf16: bool = False,
    fp16: bool = False,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.8,

    # data
    start_idx: int = None,
    end_idx: int = None,

    # generation
    max_model_len: int = 4096,
    max_new_tokens: int = 3072,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    n: int = 1,

    seed: int = 42,
):
    """Evaluate an LLM on an MCQA dataset using vLLM."""

    assert base_model is not None, "base_model is required"
    assert local_data_path is not None, "local_data_path is required"
    assert output_dir is not None, "output_dir is required"

    os.makedirs(output_dir, exist_ok=True)
    generation_file = os.path.join(output_dir, "generation.jsonl")
    result_file = os.path.join(output_dir, "result.log")
    config_file = os.path.join(output_dir, "config.json")

    # Parse category_keys from comma-separated string if needed
    if isinstance(category_keys_override, str):
        category_keys = [k.strip() for k in category_keys_override.split(",") if k.strip()]
    elif category_keys_override is not None:
        category_keys = list(category_keys_override)
    else:
        category_keys = []

    problem_key = problem_key_override
    answer_key = answer_key_override

    # Save config
    config_to_save = {k: v for k, v in locals().items() if k != 'config_to_save'}
    write_json(config_file, config_to_save)

    # Load data
    print(f"Loading data from: {local_data_path}")
    if not os.path.exists(local_data_path):
        raise FileNotFoundError(f"Data file not found: {local_data_path}")
    test_dataset = load_dataset('json', data_files=local_data_path, split='train')
    print(f"Loaded {len(test_dataset)} samples")

    # Subset selection
    if start_idx is not None and end_idx is not None:
        assert end_idx > start_idx
        si = max(0, start_idx)
        ei = min(end_idx, len(test_dataset))
        test_dataset = test_dataset.select(range(si, ei))
        print(f"Selected samples [{si}, {ei}): {len(test_dataset)} samples")

    # Load model
    print(f"Loading model: {base_model}")
    llm = LLM(
        model=base_model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32),
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if chat_template_name is not None and chat_template_name != "default":
        tokenizer.chat_template = CHAT_TEMPLATE[chat_template_name]

    # Build prompts
    system_message = []
    if system_prompt_name != "disabled":
        system_message = [{"role": "system", "content": SYSTEM_PROMPT[system_prompt_name]}]

    # Build chat template kwargs
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    prompts = [
        tokenizer.apply_chat_template(
            conversation=system_message + [
                {"role": "user", "content": td[problem_key]}
            ],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for td in test_dataset
    ]

    prompt_lens = [
        len(tokenizer.apply_chat_template(
            conversation=system_message + [
                {"role": "user", "content": td[problem_key]}
            ],
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        ))
        for td in test_dataset
    ]

    print(f"Formulated {len(prompts)} prompts")
    print(f"First prompt:\n{prompts[0][:300]}...")
    print(f"First prompt length: {prompt_lens[0]} tokens")

    # Duplicate prompts n times with different seeds
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            n=1,
            seed=seed + i,
        ) for p in prompts for i in range(n)
    ]
    prompts_dup = [p for p in prompts for _ in range(n)]
    print(f"Generated {len(prompts_dup)} prompts ({n} samples each)")

    # Generate
    if os.path.exists(generation_file) and os.path.getsize(generation_file) > 0:
        print(f"Loading cached generations from {generation_file}")
        generations = read_jsonl(generation_file)
    else:
        generations = []
        outputs = llm.generate(prompts_dup, sampling_params)
        assert len(outputs) == len(prompts_dup)

        for tdi, td in enumerate(test_dataset):
            local_outputs = outputs[tdi * n: (tdi + 1) * n]

            new_td = deepcopy(td)
            new_td["prompt"] = prompts[tdi]
            new_td["prompt_length"] = prompt_lens[tdi]
            new_td["response"] = [lo.outputs[0].text for lo in local_outputs]
            new_td["response_length"] = [len(lo.outputs[0].token_ids) for lo in local_outputs]
            new_td["finish_reason"] = [lo.outputs[0].finish_reason for lo in local_outputs]
            generations.append(new_td)

        write_jsonl(generation_file, generations)

    # Compute correctness
    ks = [2 ** e for e in range(0, 10)]
    ks_pass = [k for k in ks if (2 * k) <= n or k == 1]
    ks_mean = [k for k in ks if k <= n or k == 1]

    for g in tqdm(generations, desc="Computing correctness"):
        gt_answer = str(g[answer_key]).strip()
        g["correct"] = [check_mcqa_correct(resp, gt_answer) for resp in g["response"]]
        g["extracted_answer"] = [extract_mcqa_answer(resp) for resp in g["response"]]

        for k in ks_pass:
            g[f"pass@{k}"] = pass_at_k(g["correct"], k)
        for k in ks_mean:
            g[f"mean@{k}"] = mean_at_k(g["correct"], k)

    write_jsonl(generation_file, generations)

    # Report metrics
    with open(result_file, "w") as f:
        for k in ks_pass:
            f.write(f"pass@{k} >>>\n")
            if category_keys:
                for ck in category_keys:
                    all_cate = sorted(set(str(g.get(ck, "unknown")) for g in generations))
                    for cate in all_cate:
                        vals = [g[f"pass@{k}"] for g in generations if str(g.get(ck, "unknown")) == cate]
                        f.write(f"  {cate}: {sum(vals) / len(vals) * 100:.1f}% ({len(vals)} samples)\n")
            vals = [g[f"pass@{k}"] for g in generations]
            f.write(f"  Overall: {sum(vals) / len(vals) * 100:.1f}% ({len(vals)} samples)\n\n")

        for k in ks_mean:
            f.write(f"mean@{k} >>>\n")
            if category_keys:
                for ck in category_keys:
                    all_cate = sorted(set(str(g.get(ck, "unknown")) for g in generations))
                    for cate in all_cate:
                        vals = [g[f"mean@{k}"] for g in generations if str(g.get(ck, "unknown")) == cate]
                        f.write(f"  {cate}: {sum(vals) / len(vals) * 100:.1f}% ({len(vals)} samples)\n")
            vals = [g[f"mean@{k}"] for g in generations]
            f.write(f"  Overall: {sum(vals) / len(vals) * 100:.1f}% ({len(vals)} samples)\n\n")

    with open(result_file) as f:
        print(f.read())


if __name__ == "__main__":
    fire.Fire(eval)
