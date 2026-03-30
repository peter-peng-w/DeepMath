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
├── mcqa_eval.py                          # Main evaluation script (vLLM-based)
├── scripts/
│   ├── prepare_gpqa_diamond.py           # Download & format GPQA Diamond
│   └── eval/
│       └── mcqa_eval.sh                  # Wrapper script for MCQA evaluation
└── data/
    ├── science_qa/
    │   └── test.json                     # Science QA test split (1,672 examples)
    └── gpqa_diamond/
        └── test.json                     # GPQA Diamond (198 examples)
```

## Quick Start

### 1. Prepare data (one-time)

Science QA test data is copied from `replay-think/data/science_qa/test.json`.

GPQA Diamond needs to be downloaded and formatted:
```bash
python scripts/prepare_gpqa_diamond.py
```

### 2. Evaluate a model

```bash
# Science QA only
bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset science_qa --gpu 0

# GPQA Diamond only
bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset gpqa_diamond --gpu 0

# Both benchmarks
bash scripts/eval/mcqa_eval.sh --model Qwen/Qwen3-0.6B --dataset all --gpu 0
```

### 3. Evaluate a training checkpoint

```bash
bash scripts/eval/mcqa_eval.sh \
    --model /path/to/checkpoint \
    --dataset all \
    --gpu 0 \
    --n 8 \
    --temperature 0.6
```

## Options

### mcqa_eval.sh

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
| `--enable-thinking` | (off) | Enable Qwen3 thinking mode (`enable_thinking=True` in chat template) |
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

Key parameters:

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
