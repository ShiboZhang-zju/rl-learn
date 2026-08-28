#!/usr/bin/env python3
"""Print the exact chat-template/token/label boundary for one SFT sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-file", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=220)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    row = read_jsonl(args.data_file)[args.index]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = row["prompt"]
    full = prompt + row["completion"]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)

    # Explicitly separate the two operations so the learning path is visible:
    # messages -> chat template string -> tokenizer -> input_ids/attention_mask.
    prompt_encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_encoded["input_ids"][0].tolist()
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()

    # Cross-check the explicit tokenizer call against Transformers' shortcut.
    template_prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    template_full_ids = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
    if isinstance(template_prompt_ids, dict):
        template_prompt_ids = template_prompt_ids["input_ids"]
    if isinstance(template_full_ids, dict):
        template_full_ids = template_full_ids["input_ids"]
    if prompt_ids != template_prompt_ids or input_ids != template_full_ids:
        print("WARNING: explicit tokenizer output differs from apply_chat_template(tokenize=True).")

    common_prefix = 0
    for left, right in zip(prompt_ids, input_ids):
        if left != right:
            break
        common_prefix += 1
    labels = [-100] * common_prefix + list(input_ids[common_prefix:])

    print("=== messages ===")
    print(json.dumps(full, ensure_ascii=False, indent=2))
    print("\n=== prompt chat template ===")
    print(prompt_text)
    print("\n=== full chat template ===")
    print(full_text)
    print(f"\n=== lengths: prompt={len(prompt_ids)}, full={len(input_ids)}, common_prefix={common_prefix} ===")
    if common_prefix != len(prompt_ids):
        print("WARNING: prompt is not an exact prefix of prompt+completion after tokenization.")
    print("\n=== tokenizer tensors ===")
    print(f"input_ids ({len(input_ids)}): {input_ids}")
    print(f"attention_mask ({len(attention_mask)}): {attention_mask}")
    print(f"labels ({len(labels)}): {labels}")
    print("\n=== token table ===")
    for index, token_id in enumerate(input_ids[: args.max_tokens]):
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).replace("\n", "\\n")
        label = labels[index]
        mask = attention_mask[index]
        participation = "LOSS" if label != -100 else "MASK"
        print(
            f"{index:04d}  id={token_id:>6}  attn={mask}  "
            f"label={label:>6}  {participation:<4}  {token!r}"
        )


if __name__ == "__main__":
    main()
