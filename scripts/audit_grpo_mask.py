#!/usr/bin/env python3
"""Audit GRPO loss masking / logprob / ratio on one real sample.

Replicates the exact structure TRL 0.23 uses inside GRPOTrainer:
  - full sequence = left-padded prompt (with chat template + generation prompt) + completion
  - completion_mask marks assistant completion tokens up to and including EOS,
    excluding padding and any token after EOS
  - old_log_probs computed with torch.no_grad() (detached), new_log_probs with grad
  - ratio = exp(new_log_probs - old_log_probs) on masked positions only

Writes outputs/grpo_v1/audit/debug_sample.json for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/sft_v2_5k_p800/checkpoint-1252"))
    parser.add_argument("--data-file", type=Path, default=Path("data/processed/grpo_v1_train.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_v1/audit/debug_sample.json"))
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = read_jsonl(args.data_file)
    row = rows[args.index]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_length).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    expanded = {k: v.repeat_interleave(args.num_generations, dim=0) for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **expanded,
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=pad_id,
            use_cache=True,
        )
    completion_ids_all = generated[:, prompt_len:]
    completion_texts = tokenizer.batch_decode(completion_ids_all, skip_special_tokens=True)

    # Inspect generation index 0 in detail.
    completion_ids = completion_ids_all[0]
    # TRL-style completion_mask: up to and including first EOS (excludes padding / anything after EOS).
    seq_indices = torch.arange(completion_ids.shape[0], device=completion_ids.device)
    eos_positions = (completion_ids == eos_id).nonzero(as_tuple=False)
    if eos_positions.numel() > 0:
        eos_idx = eos_positions[0].item()
    else:
        eos_idx = completion_ids.shape[0] - 1
    completion_mask = (seq_indices <= eos_idx).int()

    full_ids = torch.cat([inputs["input_ids"], completion_ids.unsqueeze(0)], dim=1)
    prompt_attn = inputs["attention_mask"]
    comp_attn = completion_mask.unsqueeze(0).float()
    full_attn = torch.cat([prompt_attn, comp_attn], dim=1)
    token_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    token_mask[0, prompt_len : prompt_len + int(completion_mask.sum())] = True

    def logprobs(use_grad: bool):
        if use_grad:
            logits = model(input_ids=full_ids, attention_mask=full_attn).logits
        else:
            with torch.no_grad():
                logits = model(input_ids=full_ids, attention_mask=full_attn).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        gathered = lp.gather(2, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return gathered, token_mask[:, 1:].bool()

    old_lp, loss_mask = logprobs(False)
    old_lp = old_lp.detach()
    new_lp, loss_mask = logprobs(True)
    ratio = torch.exp(new_lp - old_lp)

    loss_positions = [p for p in range(loss_mask.shape[1]) if bool(loss_mask[0, p])]
    masked_ratio = ratio[loss_mask]
    token_strings = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())
    assert loss_positions, "loss mask must be non-empty"
    for p in loss_positions:
        assert token_mask[0, p + 1], "loss position must predict a completion token"
    assert int(completion_mask.sum()) > 0, "completion must not be empty"

    result = {
        "model": args.model,
        "adapter": str(args.adapter),
        "sample": {"id": row["id"], "index": args.index},
        "prompt_text": prompt_text,
        "completion_text": completion_texts[0],
        "prompt_len": prompt_len,
        "completion_len": int(completion_mask.sum()),
        "input_ids": full_ids[0].tolist(),
        "token_strings": token_strings,
        "completion_mask": [bool(m) for m in token_mask[0].tolist()],
        "loss_token_positions": loss_positions,
        "old_log_probs_detached": True,
        "new_log_probs_requires_grad": bool(new_lp.requires_grad),
        "ratio_mean_on_mask": float(masked_ratio.mean()),
        "ratio_min_on_mask": float(masked_ratio.min()),
        "ratio_max_on_mask": float(masked_ratio.max()),
        "note": (
            "loss position p means the model predicts token p+1; p+1 is an assistant completion "
            "token. System/user prompt tokens, padding, and tokens after EOS are excluded. "
            "Replicates TRL 0.23 GRPOTrainer masking for manual inspection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
