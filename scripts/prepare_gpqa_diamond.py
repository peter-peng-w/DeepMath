#!/usr/bin/env python3
"""Prepare GPQA Diamond dataset for MCQA evaluation.

Downloads from HuggingFace, shuffles choices with deterministic seed,
and formats to match the Science QA MCQA schema.

Usage:
    python scripts/prepare_gpqa_diamond.py [--output_dir ./data/gpqa_diamond] [--seed 42]
"""

import argparse
import hashlib
import json
import random
import string
from pathlib import Path

from datasets import load_dataset


LETTERS = list(string.ascii_uppercase)

PROMPT_TEMPLATE = (
    "Given the following question and {n} candidate answers ({letter_list}), "
    "choose the best answer.\n"
    "Question: {question}\n"
    "{choices_text}\n"
    'Your response should end with "The best answer is [the_answer_letter]" '
    "where the [the_answer_letter] is one of {letter_list}."
)


def shuffle_choices(choices: list[str], correct_idx: int, seed_str: str) -> tuple[list[str], int]:
    """Shuffle choices and return (shuffled_choices, new_correct_idx)."""
    indexed = list(enumerate(choices))
    rng = random.Random(hashlib.md5(seed_str.encode()).hexdigest())
    rng.shuffle(indexed)
    new_correct_idx = next(i for i, (orig_idx, _) in enumerate(indexed) if orig_idx == correct_idx)
    shuffled = [text for _, text in indexed]
    return shuffled, new_correct_idx


def format_problem(question: str, choices: list[str]) -> str:
    """Format a question + choices into the standard MCQA prompt."""
    n = len(choices)
    letter_list = ", ".join(LETTERS[:n - 1]) + " and " + LETTERS[n - 1]
    choices_text = "\n".join(f"{LETTERS[i]}. {choices[i]}" for i in range(n))
    return PROMPT_TEMPLATE.format(
        n=n,
        letter_list=letter_list,
        question=question,
        choices_text=choices_text,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare GPQA Diamond dataset")
    parser.add_argument("--output_dir", type=Path, default=Path("./data/gpqa_diamond"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPQA Diamond from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print(f"Loaded {len(ds)} examples")

    output = []
    for idx, item in enumerate(ds):
        question = item["Question"]
        correct_answer = item["Correct Answer"]
        choices_raw = [
            correct_answer,
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        correct_idx = 0  # correct answer is always first before shuffling

        record_id = item.get("Record ID", str(idx))
        shuffled, new_correct_idx = shuffle_choices(choices_raw, correct_idx, f"{record_id}_{args.seed}")
        solution = LETTERS[new_correct_idx]

        problem = format_problem(question, shuffled)

        output.append({
            "id": record_id,
            "problem": problem,
            "solution": solution,
            "question": question,
            "choices": [f"{LETTERS[i]}. {shuffled[i]}" for i in range(len(shuffled))],
            "domain": item.get("High-level domain", "unknown"),
            "subdomain": item.get("Subdomain", "unknown"),
        })

    output_path = args.output_dir / "test.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(output)} examples to {output_path}")

    # Print answer distribution
    from collections import Counter
    answer_counts = Counter(item["solution"] for item in output)
    print(f"\nAnswer distribution: {dict(sorted(answer_counts.items()))}")

    domain_counts = Counter(item["domain"] for item in output)
    print(f"Domain distribution: {dict(sorted(domain_counts.items()))}")


if __name__ == "__main__":
    main()
