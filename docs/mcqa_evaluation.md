# MCQA Evaluation Pipeline

Evaluation pipeline for Multiple-Choice QA tasks, separate from the math reasoning evaluation (`uni_eval.py`). Used to evaluate SCPO/JSPO/GRPO checkpoints trained on Science QA.

## Benchmarks

| Dataset | Source | # Examples | Categories |
|---------|--------|-----------|------------|
| **Science QA** | ARC-Challenge + OpenBookQA | 1,672 | `source`: arc_challenge (1,172), openbookqa (500) |
| **GPQA Diamond** | PhD-level science MCQA | 198 | `domain`: Physics (86), Chemistry (93), Biology (19) |

Both datasets use the same prompt format:
```
Given the following question and 4 candidate answers (A, B, C and D), choose the best answer.
Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Your response should end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of A, B, C, D.
```

Choices are shuffled with a deterministic seed to prevent position bias.

## Files

```
DeepMath/
├── mcqa_eval.py                                    # Single-model MCQA evaluation (vLLM-based)
├── scripts/
│   ├── prepare_gpqa_diamond.py                     # Download & format GPQA Diamond
│   └── eval/
│       ├── mcqa_eval.sh                            # Single-model wrapper
│       └── multi_checkpoint_mcqa_eval.sh           # Multi-checkpoint batch evaluation
└── data/
    ├── science_qa/
    │   └── test.json                               # Science QA test split (1,672 examples)
    └── gpqa_diamond/
        └── test.json                               # GPQA Diamond (198 examples)
```

## Quick Start

### 1. Prepare data (one-time)

Science QA test data is copied from `replay-think/data/science_qa/test.json`.

GPQA Diamond needs to be downloaded and formatted:
```bash
python scripts/prepare_gpqa_diamond.py
```

### 2. Evaluate a single model

```bash
# Science QA only
bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset science_qa --gpu 0

# GPQA Diamond only
bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset gpqa_diamond --gpu 0

# Both benchmarks
bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset all --gpu 0
```

### 3. Evaluate multiple checkpoints (batch)

The `multi_checkpoint_mcqa_eval.sh` script automatically discovers checkpoints in a training directory, evaluates them in parallel across GPUs, and prints a summary.

```bash
# Evaluate all checkpoints in a training run
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints exp/qwen3-0.6b/science_qa/grpo/repeat_sampler_grpo_8rollouts/ \
    --gpus "0,1,2,3" --dataset all

# Latest 5 checkpoints only
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints exp/qwen3-0.6b/science_qa/grpo/repeat_sampler_grpo_8rollouts/ \
    --gpus "0,1,2,3" --latest 5

# Skip already-evaluated checkpoints (useful for resuming)
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints exp/qwen3-0.6b/science_qa/grpo/repeat_sampler_grpo_8rollouts/ \
    --gpus "0,1,2,3" --skip-existing

# Evaluate HuggingFace models
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B" \
    --gpus "0,1" --dataset all

# Evaluate checkpoints listed in a file
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints checkpoints.txt --gpus "0,1,2,3"

# With thinking enabled
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints exp/qwen3-0.6b/... --gpus "0,1,2,3" --enable-thinking
```

The `--checkpoints` argument accepts three formats:
- **Directory**: auto-discovers subdirectories matching `--pattern` (default `checkpoint-*`) that contain `config.json` or `model.safetensors`
- **Text file**: one checkpoint path per line (lines starting with `#` are ignored)
- **Comma-separated list**: inline paths or HuggingFace model IDs

## Options

### multi_checkpoint_mcqa_eval.sh

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoints` | (required) | Directory, file, or comma-separated checkpoint paths |
| `--dataset` | `all` | `science_qa`, `gpqa_diamond`, or `all` |
| `--gpus` | `0` | Comma-separated GPU IDs for parallel execution |
| `--config` | `sampling` | `sampling` (temp=0.6, n=8) or `greedy` (temp=0, n=1) |
| `--n` | `8` | Override number of samples per question |
| `--temperature` | `0.6` | Override sampling temperature |
| `--skip-existing` | (off) | Skip checkpoints with existing `result.log` |
| `--pattern` | `checkpoint-*` | Glob pattern for checkpoint directory discovery |
| `--sort-by` | `step` | Sort discovered checkpoints by: `step`, `time`, `name` |
| `--limit` | (none) | Limit to first N checkpoints after sorting |
| `--latest` | (none) | Evaluate only the latest N checkpoints |
| `--enable-thinking` | (off) | Enable Qwen3 thinking mode |
| `--max-model-len` | `4096` | Max context window |
| `--max-new-tokens` | `3072` | Max generation tokens |
| `--output-base` | `exp/local_eval/{ckpt}` | Override output base directory |

### mcqa_eval.sh (single model)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Model path or HuggingFace ID |
| `--dataset` | `all` | `science_qa`, `gpqa_diamond`, or `all` |
| `--gpu` | `0` | GPU device ID |
| `--n` | `8` | Number of samples per question |
| `--temperature` | `0.6` | Sampling temperature |
| `--top-p` | `0.95` | Top-p sampling |
| `--max-model-len` | `4096` | Max context window |
| `--max-new-tokens` | `3072` | Max generation tokens |
| `--enable-thinking` | (off) | Enable Qwen3 thinking mode |
| `--output-base` | `exp/local_eval/{model}` | Output directory base |

### mcqa_eval.py (direct usage)

```bash
python mcqa_eval.py \
    --base_model Qwen/Qwen3-0.6B \
    --local_data_path data/science_qa/test.json \
    --problem_key_override problem \
    --answer_key_override solution \
    --category_keys_override "source" \
    --output_dir exp/eval/science_qa \
    --n 8 \
    --temperature 0.6 \
    --bf16
```

| Parameter | Description |
|-----------|-------------|
| `--base_model` | Model path or HuggingFace ID |
| `--local_data_path` | Path to JSON/JSONL eval data |
| `--problem_key_override` | Field name for the question (default: `problem`) |
| `--answer_key_override` | Field name for the correct answer letter (default: `solution`) |
| `--category_keys_override` | Comma-separated fields for per-category breakdown (e.g., `"source"`, `"domain"`) |
| `--enable_thinking` | `True`/`False` — controls Qwen3 thinking mode in chat template |
| `--n` | Number of samples per question for pass@k/mean@k |
| `--bf16` | Use bfloat16 precision |

## Sampling & Metrics

Default sampling config: `temperature=0.6`, `top_p=0.95`, `n=8`.

With `n=8` samples per question, the following metrics are computed:
- **mean@k** for k = 1, 2, 4, 8 — average correctness of first k samples
- **pass@k** for k = 1, 2, 4 — probability that at least one of k samples is correct (requires `2k <= n`)

## Thinking Mode

The `enable_thinking` parameter controls Qwen3's chat template behavior:

| Setting | Chat template effect | Use when |
|---------|---------------------|----------|
| `False` (default) | Pre-fills empty `<think></think>` — model answers directly | Model trained with `enable_thinking: false` (no_thinking configs) |
| `True` | Model can freely generate `<think>` blocks | Model trained with `enable_thinking: true` (thinking configs) |

**The eval setting must match the training setting** to avoid prompt format mismatch.

## Output

Each evaluation produces three files in the output directory:

### `generation.jsonl`
Per-question results with all responses and correctness:
```json
{
    "problem": "Given the following question...",
    "solution": "C",
    "prompt": "<full formatted prompt>",
    "response": ["The best answer is C", "The best answer is A", ...],
    "correct": [true, false, ...],
    "extracted_answer": ["C", "A", ...],
    "pass@1": 1.0,
    "mean@1": 0.625,
    "source": "arc_challenge"
}
```

### `result.log`
Aggregate metrics with per-category breakdown:
```
pass@1 >>>
  arc_challenge: 72.5% (1172 samples)
  openbookqa: 65.2% (500 samples)
  Overall: 70.3% (1672 samples)

mean@1 >>>
  arc_challenge: 72.5% (1172 samples)
  openbookqa: 65.2% (500 samples)
  Overall: 70.3% (1672 samples)
```

### `config.json`
Full configuration used for the evaluation run.

## Answer Extraction

The answer extractor strips `<think>...</think>` blocks (if present), then searches for:

1. `"The best answer is [X]"` or `"The best answer is X"` (primary format)
2. `"The answer is X"` or `"The answer is: X"` (common variant)
3. `\boxed{X}` (math-style fallback)
4. Last standalone letter A-E (final fallback)

If no answer is extracted, the response is marked as incorrect.
