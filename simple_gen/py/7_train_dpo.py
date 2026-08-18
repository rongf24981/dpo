#!/usr/bin/env python3
"""Minimal TRL preference trainer for simple_gen repair preference pairs.

Expected input JSONL schema:

```
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

This is the conversational preference format supported by Hugging Face TRL's
DPOTrainer. Extra columns such as ``metadata`` are removed before training.

The script defaults to standard DPO, but can request other DPOTrainer loss
variants such as SimPO if the installed TRL version exposes those arguments.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def keep_supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def load_jsonl_dataset(path: Path):
    ds = load_dataset("json", data_files=str(path), split="train")
    keep = {"prompt", "chosen", "rejected"}
    remove = [c for c in ds.column_names if c not in keep]
    if remove:
        ds = ds.remove_columns(remove)
    return ds


def maybe_split_eval(train_ds, eval_jsonl: Optional[Path], eval_ratio: float, seed: int):
    if eval_jsonl is not None:
        return train_ds, load_jsonl_dataset(eval_jsonl)
    if eval_ratio <= 0:
        return train_ds, None
    split = train_ds.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPO/SimPO-style preference optimization on repair pairs.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="sigmoid",
        help="TRL DPOTrainer loss type. Use 'sigmoid' for standard DPO; use 'simpo' only if supported by the installed TRL version.",
    )
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--simpo-gamma", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable activation checkpointing to reduce memory use.",
    )
    parser.add_argument(
        "--deepspeed",
        type=Path,
        default=None,
        help="Optional DeepSpeed config path. Launch with accelerate/deepspeed.",
    )
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=-1,
        help="Injected by DeepSpeed/torchrun; accepted here so argparse does not fail.",
    )
    args = parser.parse_args()

    from trl import DPOConfig, DPOTrainer

    dpo_config_params = inspect.signature(DPOConfig.__init__).parameters
    if args.loss_type != "sigmoid" and "loss_type" not in dpo_config_params:
        raise RuntimeError(
            f"Installed TRL DPOConfig does not expose loss_type, so loss_type={args.loss_type!r} "
            "cannot be requested safely."
        )
    if args.loss_type == "simpo" and "simpo_gamma" not in dpo_config_params and args.simpo_gamma is not None:
        raise RuntimeError(
            "Installed TRL DPOConfig does not expose simpo_gamma. Upgrade TRL or use a SimPO-capable trainer."
        )

    train_ds = load_jsonl_dataset(args.train_jsonl)
    train_ds, eval_ds = maybe_split_eval(train_ds, args.eval_jsonl, args.eval_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    dpo_config_kwargs = {
        "output_dir": str(args.output_dir),
        "loss_type": args.loss_type,
        "beta": args.beta,
        "simpo_gamma": args.simpo_gamma,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if eval_ds is not None else None,
        "eval_strategy": "steps" if eval_ds is not None else "no",
        "evaluation_strategy": "steps" if eval_ds is not None else "no",
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "deepspeed": str(args.deepspeed) if args.deepspeed is not None else None,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
    }
    dpo_args = DPOConfig(**keep_supported_kwargs(DPOConfig, dpo_config_kwargs))

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = DPOTrainer(**keep_supported_kwargs(DPOTrainer, trainer_kwargs))
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
