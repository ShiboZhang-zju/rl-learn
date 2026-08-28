#!/usr/bin/env python3
"""Mac-friendly LoRA SFT entry point using the current Hugging Face TRL API."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/sft_val.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-lora", action="store_true", help="Use full-parameter training only for tiny debug runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = choose_device(args.device)
    output_dir = args.output_dir or Path(cfg["output_dir"])
    train_rows = read_jsonl(args.train_file)
    eval_rows = read_jsonl(args.eval_file)
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(int(cfg.get("seed", 42)))
    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Keep raw puzzle/expr metadata in JSONL for auditing, but do not send it
    # through Arrow: nested expressions intentionally have different shapes
    # (person_is/same/not/and/or), which Arrow cannot infer as one struct type.
    train_dataset = Dataset.from_list([{key: row[key] for key in ("prompt", "completion")} for row in train_rows])
    eval_dataset = Dataset.from_list([{key: row[key] for key in ("prompt", "completion")} for row in eval_rows])
    sft_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "do_train": True,
        "do_eval": bool(eval_rows),
        "eval_strategy": cfg.get("eval_strategy", "steps" if eval_rows else "no"),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_ratio": float(cfg["warmup_ratio"]),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "logging_steps": int(cfg["logging_steps"]),
        "save_strategy": cfg.get("save_strategy", "steps"),
        "save_steps": int(cfg["save_steps"]),
        "eval_steps": int(cfg["eval_steps"]),
        "save_total_limit": int(cfg["save_total_limit"]),
        "max_length": int(cfg["max_length"]),
        "pad_to_multiple_of": int(cfg.get("pad_to_multiple_of", 1)),
        "packing": bool(cfg["packing"]),
        "completion_only_loss": bool(cfg["completion_only_loss"]),
        "report_to": cfg.get("report_to", "none"),
        "seed": int(cfg.get("seed", 42)),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
        # MPS on the current local PyTorch (2.3.x) must stay in float32.
        "bf16": bool(cfg.get("bf16", False)),
        "fp16": bool(cfg.get("fp16", False)),
    }
    max_steps = args.max_steps if args.max_steps is not None else int(cfg.get("max_steps", -1))
    if max_steps > 0:
        sft_kwargs["max_steps"] = max_steps
    # Keep this script usable across nearby Transformers/TRL versions.
    supported = set(inspect.signature(SFTConfig).parameters)
    sft_kwargs = {key: value for key, value in sft_kwargs.items() if key in supported}
    if device == "mps" and "use_mps_device" in supported:
        sft_kwargs["use_mps_device"] = True
    if device == "cpu" and "use_cpu" in supported:
        sft_kwargs["use_cpu"] = True
    training_args = SFTConfig(**sft_kwargs)

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=list(cfg["lora_target_modules"]),
            bias="none",
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_rows else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    print(f"device={device}")
    print(f"train_examples={len(train_dataset)} eval_examples={len(eval_dataset)}")
    print(f"trainable_parameters={trainable}/{total} ({100 * trainable / total:.4f}%)")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved adapter/model to {output_dir}")


if __name__ == "__main__":
    main()
