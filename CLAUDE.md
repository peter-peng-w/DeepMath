# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research codebase built on top of the DeepMath-103K project. Trains LLMs (Qwen2.5-Math, Qwen3) with RL variants (GRPO, group/JSPO/SCPO, off-/mixed-policy value estimation) implemented inside the vendored `verl/` submodule, and evaluates the resulting checkpoints on math reasoning and MCQA benchmarks.

Two largely independent surfaces:
- **Training**: Ray + `verl.trainer.main_ppo` driven by Hydra-style `key=value` overrides.
- **Evaluation**: vLLM-based offline batch inference (`uni_eval.py` for math, `mcqa_eval.py` for MCQA) wrapped by shell orchestrators that fan checkpoints × datasets × configs over a GPU pool.

`verl/` is a git submodule with local modifications (the actual RL algorithms — samplers, advantage estimators, value heads — live there). Always `git clone --recurse-submodules` and `pip install -e verl`.

## Environment

### Full environment (training + evaluation)
```bash
conda create -y -n deepmath python=3.12.2 && conda activate deepmath
pip3 install ray[default] torch==2.5.1 vllm==0.7.3 \
    flash-attn==2.7.4.post1 --no-build-isolation
pip3 install omegaconf==2.4.0.dev3 hydra-core==1.4.0.dev1 antlr4-python3-runtime==4.11.0
pip3 install math-verify[antlr4_11_0]==0.7.0 fire deepspeed transformers==4.49.0
pip3 install -e verl
pip3 install langdetect==1.0.9 pebble==5.1.0 word2number
```

### Inference-only environment
If only running `uni_eval.py` / `mcqa_eval.py` (no training), `verl` and its training-only deps can be dropped:

```bash
pip install torch==2.5.1 vllm==0.7.3 transformers==4.49.0 datasets \
    fire tqdm numpy sympy tiktoken regex langdetect==1.0.9 word2number \
    math-verify[antlr4_11_0]==0.7.0 antlr4-python3-runtime==4.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation   # optional, install after torch
```

Safe to drop for inference: `verl` (only `verl.trainer.main_ppo` uses it), `ray[default]` (used by `utils/reward_utils/reward_func*.py` reward and Ray cluster launch), `deepspeed`, `tensorboardX`, `prettytable`, `pebble`, `omegaconf`, `hydra-core`.

Watch out: `langdetect`, `word2number`, `regex`, and `tiktoken` are imported **eagerly** at module top by `utils/polymath/*` and `utils/data_utils.py`, which `uni_eval.py` imports unconditionally — so they're required even when not evaluating PolyMath.

### Required vLLM env vars
Every eval script exports these:
```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Common commands

### Math evaluation — single checkpoint, single dataset
```bash
python3 uni_eval.py \
    --base_model <hf_id_or_path> \
    --chat_template_name default \
    --system_prompt_name simplerl \   # see system-prompt table below
    --output_dir ./exp/<run>/<ckpt>/<dataset>/<config>/ \
    --bf16 True --tensor_parallel_size 8 \
    --data_id zwhe99/MATH --split math500 \
    --max_model_len 32768 --temperature 0.6 --top_p 0.95 --n 16
```

System prompt **must match training**:

| Model family | `--system_prompt_name` |
|---|---|
| DeepMath-Zero-7B / -Math-7B | `simplerl` |
| DeepMath-1.5B / -Omn-1.5B | `disabled` |

Datasets are declared in `DATASET_INFO` at the top of `uni_eval.py` (math500, amc23, aime90, aime25, OlympiadBench, MinervaMath, gpqa_diamond_mc, polymath, etc.). Per-dataset judges (`OBJudge` for OlympiadBench, `pm_judge` for PolyMath) are dispatched in the correctness loop.

### Math evaluation — multi-checkpoint sweep
The orchestrator handles checkpoint discovery, GPU-pool locking, and per-config output paths:
```bash
./scripts/eval/multi_checkpoint_eval.sh \
    --checkpoints <ckpt_list.txt | dir | ckpt1,ckpt2 | hf_id> \
    --gpus "0,1,2,3" \
    --config sampling \                # sampling | greedy | both
    --output-base ./exp/<run_name> \
    --skip-existing                    # idempotent reruns
```
- `DATASETS=(...)` and `CONFIGS["sampling"]="0.6 0.95 64"` are edited at the top of the script — change them there, not via flags.
- Output layout is fixed: `<output-base>/<ckpt_name>/<dataset_name>/temperature_<t>_top_p_<p>_rollouts_<n>/{generation.jsonl,result.log,config.json}`.
- After the sweep finishes, `create_summary_report` writes `<output-base>/checkpoint_comparison.md` aggregating every `result.log`.

The `run_multi_ckpts_eval*.sh` files are not parameterized scripts — they are commit-tracked logs of every sweep that has been launched (most lines commented). To launch a new sweep, **uncomment the relevant block**; do not rewrite the whole file.

`save_checkpoints_path*.sh` works the same way: edit the active `CHECKPOINT_DIR` and `OUTPUT_FILE` at the bottom, run it, and it writes a `checkpoints_to_eval.txt` cache under `scripts/eval/cache/ckpt_lists/...`.

`./scripts/eval/list_checkpoints.sh <dir>` is a read-only checkpoint discoverer (same logic as the dispatcher, no eval).

### Local-dataset evaluation (math)
`scripts/eval/local_dataset_eval.sh` runs `uni_eval.py` on a JSON/JSONL file with **automatic data sharding across GPUs** and a separate merge step:
```bash
# Shard across 4 GPUs
./scripts/eval/local_dataset_eval.sh --parallel --gpus "0,1,2,3" \
    --model <path> --data <file>.json \
    --problem-key problem --answer-key answer \
    --category-keys "level,subject" --n 16

# Merge shard_<start>_<end>/generation.jsonl into a single result.log
./scripts/eval/local_dataset_eval.sh --merge --output-dir <BASE_OUTPUT_DIR>
```
The merge step recomputes pass@k / mean@k from the union of shards.

### MCQA evaluation
Separate from math eval to keep judges clean. Datasets live as local JSON in `data/science_qa/test.json` and `data/gpqa_diamond/test.json` with the choice prompt pre-baked into the `problem` field; `solution` is a single letter A-D.

Single model:
```bash
bash scripts/eval/mcqa_eval.sh --model <path> --dataset all --gpu 0
```
Multi-checkpoint:
```bash
bash scripts/eval/multi_checkpoint_mcqa_eval.sh \
    --checkpoints <dir> --gpus "0,1,2,3" \
    --dataset all --skip-existing [--enable-thinking]
```
`--enable-thinking` toggles Qwen3's chat-template thinking mode; **must match training** (no_thinking vs thinking configs).

GPQA Diamond is regenerated from HF (with deterministic choice shuffling) by `python scripts/prepare_gpqa_diamond.py`.

### Training
Multi-node Ray + verl. Start the cluster, then submit a job from the head node:
```bash
ray start --head --port=6379 --node-ip-address=$HEAD_ADDR --num-gpus=8
ray start --address=$HEAD_ADDR:6379 --node-ip-address=$WORKER_ADDR --num-gpus=8  # workers

bash scripts/train/deepmath-zero-7b.sh   # or deepmath-1.5b / -omn-1.5b / -zero-math-7b
```
Training scripts are `ray job submit` wrappers that launch `python -m verl.trainer.main_ppo` with hundreds of `key=value` overrides. The custom reward function is wired via `custom_reward_function.path=$WORK_DIR/utils/reward_utils/reward_func.py`. Training data is prepared by `python3 verl/examples/data_preprocess/deepmath_103k.py --local_dir $DATA_DIR` (parquet output).

## Architecture you need to know to be productive

### Eval data flow (both `uni_eval.py` and `mcqa_eval.py`)
1. Build prompts via `tokenizer.apply_chat_template`. `uni_eval.py` supports overriding the chat template (see `CHAT_TEMPLATE` in `utils/chat_template.py`) and adding system / prefix / suffix prompts. `mcqa_eval.py` keeps it minimal but forwards `enable_thinking` to the template.
2. Duplicate each prompt `n` times with offset seeds (`seed=base_seed+i`) — vLLM treats each as an independent sample.
3. **Generation cache**: if `<output_dir>/generation.jsonl` exists and is non-empty, skip inference and reload it. To force re-inference, delete that file. `--skip-existing` in the orchestrators checks for `result.log` instead, so it skips even the cache-rebuild.
4. Per-response correctness — math eval `OR`s several judges (`process_results` from OpenMathInst with two extractors, `math_verify.verify`, plus dataset-specific judges); MCQA eval extracts a single letter via a regex cascade.
5. Compute `pass@k` (HumanEval unbiased estimator, requires `2k ≤ n`) and `mean@k` (mean of first k flags). Per-category breakdown is driven by `category_keys` (declared in `DATASET_INFO` or via `--category_keys_override`).
6. Write `generation.jsonl` (full per-sample), `result.log` (aggregate text), `config.json` (full run config).

### Multi-checkpoint orchestrator
The pattern in `multi_checkpoint_eval.sh` and `multi_checkpoint_mcqa_eval.sh`:
- Checkpoint input is polymorphic: file / directory / comma-list / HF model ID. Directories are filtered to subdirs containing `config.json` or `model.safetensors` and matching `--pattern` (default `checkpoint-*`).
- GPU pool is implemented with **`mkdir`-based atomic lockfiles** under a `mktemp -d` directory (each job acquires `gpu_<id>.lock.d`, releases via `trap` on EXIT). Stale-lock cleanup checks PID liveness with `kill -0`.
- Math orchestrator dispatches with `xargs -P NUM_GPUS`; MCQA orchestrator uses bg `&` + `wait -n` capped at `NUM_GPUS`. MCQA also redirects per-job stdout/stderr to `<out_dir>/eval.log` to avoid interleaved logs (recent fix).
- Both compose the output path as `<output-base>/<ckpt_name>/<dataset>/<config_subdir>/...`. **The checkpoint name is required in the path** — without it, sweeps over multiple checkpoints clobber each other (this was a recently fixed skip-existing bug — see `bc91ede`).

### Where the algorithms live
The RL methodology (samplers, advantage / baseline / James-Stein / SCPO estimators, value heads, off-policy mixing) is in the vendored `verl/` submodule, not this top-level repo. The shell scripts under `scripts/train/` are thin Ray-job wrappers that pass `algorithm.adv_estimator=...` and many other `actor_rollout_ref.*` overrides to `verl.trainer.main_ppo`. When investigating training behavior, search inside `verl/` for the sampler / estimator name referenced in the script.

`utils/reward_utils/reward_func.py` is the custom reward function the trainer imports — it wraps `openmathinst_utils.{extract_answer, math_equal}`. `utils/openmathinst_utils.py` and `utils/polymath/` are the in-repo answer judges used by both training rewards and eval correctness scoring.

### Output directory convention
All evaluation outputs go under `exp/`. The standard layout is:
```
exp/
└── <model_family>/<benchmark>/<run_name>/
    └── <ckpt_name>/<dataset_name>/<config_subdir>/
        ├── generation.jsonl   # cached responses + per-sample correct/pass@k/mean@k
        ├── result.log         # aggregate text report (per-category + Overall)
        └── config.json        # full run config
```
`<config_subdir>` is `temperature_0.6_top_p_0.95_rollouts_64` (math) or `temp_0.6_top_p_0.95_n_4` (MCQA). The format diverged between the two scripts — don't try to unify them or existing `--skip-existing` checks will break.

### Notebooks & plotting
`tsne/*.ipynb` are analysis notebooks (reward visualizations, t-SNE, training dynamics) that consume the `exp/` outputs and `data/` artifacts. They are research scratch — treat them as read-mostly unless asked.

## Conventions and pitfalls

- **Never edit the run-log scripts as if they were parameterized.** `run_multi_ckpts_eval*.sh` and `save_checkpoints_path*.sh` are append-only logs of past invocations with everything commented out. To launch a new sweep, uncomment the latest block (or add a new one at the bottom) — don't strip the history.
- **System prompt and `enable_thinking` must match training.** Mismatches silently degrade scores. Check the training script's `actor_rollout_ref.*` settings before picking eval flags.
- **Generation cache is keyed only by output_dir.** Changing `n`, `temperature`, model weights, or the prompt template without changing the output dir will reuse a stale `generation.jsonl`. Either include those in the dir name (the orchestrators do) or delete the file.
- **`pass@k` requires `2k ≤ n`** (HumanEval estimator stability). The eval scripts only emit pass@k for k satisfying this.
- **`xargs` cannot inherit bash arrays.** That's why the orchestrators export `GPU_IDS_STR` as a comma-string and re-parse it inside subshells; don't try to `export -a GPU_IDS`.
- **Use `find . -maxdepth 1 ...`** when discovering checkpoints — `find /` will scan the entire filesystem and exhaust resources on this cluster.
