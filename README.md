# DPO / ProSec-Style Preference Training

This repository contains the lightweight preference-training pipeline used for the Abhinav-side secure-code experiments.

The current shared code supports:

- generating preference pairs from known correct/secure Python reference solutions and generated insecure rewrites;
- converting ProSec preference data into TRL conversational preference JSONL;
- LoRA DPO training with ProSec-style hyperparameters;
- requesting SimPO-style loss through TRL if the installed TRL version supports `loss_type="simpo"` and `simpo_gamma`.

No generated data, model weights, checkpoints, API keys, or benchmark outputs are included.

## Main Files

```text
simple_gen/py/6_generate_revision_pairs.py
simple_gen/py/8_generate_reference_negative_pairs.py
simple_gen/py/prepare_prosec_preferences.py
simple_gen/py/7_train_dpo.py
simple_gen/py/configs/deepspeed_zero3_bf16.json
```

## Prepare ProSec Python Preferences

```bash
cd simple_gen/py

python prepare_prosec_preferences.py \
  --dataset-id prosecalign/prosec-mixed-clm7b-inst \
  --revision d4f17919b3d946bcd393d87c15dfecfa13aaf566 \
  --split train \
  --language python \
  --output-jsonl data/prosec_python_mixed_qwen25coder7b.jsonl
```

Expected full Python count from the current dataset revision: `19079` pairs.

## ProSec-Style DPO Run

This uses the ProSec paper's DPO-style hyperparameters where possible:

- LoRA, not full fine-tuning;
- LoRA rank `r=8`;
- LoRA alpha `16`;
- total batch size `64`;
- learning rate `5e-6`;
- DPO beta `0.05`.

For two GPUs with per-device batch size 1:

```text
1 * 2 GPUs * gradient_accumulation_steps 32 = total batch size 64
```

Example:

```bash
cd simple_gen/py

deepspeed --num_gpus=2 7_train_dpo.py \
  --train-jsonl data/prosec_python_mixed_qwen25coder7b.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir outputs/qwen25coder_7b_prosec_python_dpo_400step \
  --loss-type sigmoid \
  --beta 0.05 \
  --learning-rate 5e-6 \
  --max-steps 400 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --max-length 4096 \
  --max-prompt-length 2048 \
  --logging-steps 10 \
  --save-steps 50 \
  --eval-steps 50 \
  --eval-ratio 0.02 \
  --bf16 \
  --gradient-checkpointing \
  --deepspeed configs/deepspeed_zero3_bf16.json \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05
```

This is a 400-step capped DPO comparison. If the goal is exact ProSec DPO from Table 6, use `--max-steps 800`.

## SimPO-Style Run

The ProSec Table 6 SimPO setting for non-Phi models is:

- learning rate `5e-6`;
- beta `1.5`;
- gamma `0.5`;
- steps `400`;
- LoRA `r=8`, alpha `16`;
- total batch size `64`.

This command requests SimPO through TRL's `DPOTrainer` only if the installed TRL version supports it:

```bash
cd simple_gen/py

deepspeed --num_gpus=2 7_train_dpo.py \
  --train-jsonl data/prosec_python_mixed_qwen25coder7b.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir outputs/qwen25coder_7b_prosec_python_simpo_400step \
  --loss-type simpo \
  --beta 1.5 \
  --simpo-gamma 0.5 \
  --learning-rate 5e-6 \
  --max-steps 400 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --max-length 4096 \
  --max-prompt-length 2048 \
  --logging-steps 10 \
  --save-steps 50 \
  --eval-steps 50 \
  --eval-ratio 0.02 \
  --bf16 \
  --gradient-checkpointing \
  --deepspeed configs/deepspeed_zero3_bf16.json \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05
```

If this fails with a TRL `loss_type` or `simpo_gamma` error, the installed TRL version does not support SimPO through `DPOTrainer`; in that case use the DPO command above or upgrade/replace the trainer with a SimPO-capable implementation.

## Our Execution-Filtered Preference Data

The reference-negative generator creates pairs where:

- `chosen` is the known correct and secure reference solution;
- `rejected` is a generated variant that is functional but fails at least one security test.

Example:

```bash
cd simple_gen/py

python 8_generate_reference_negative_pairs.py \
  --dataset-id AetherPrior/py_cwe_GRPO \
  --split train \
  --dpo-output-jsonl data/python_refneg_dpo_full_qwen25coder14b.jsonl \
  --all-output-jsonl data/python_refneg_dpo_full_qwen25coder14b_all.jsonl \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --api-kind chat \
  --max-workers 8 \
  --max-tokens 2048
```

Observed full-pipeline yield from the previous run: `2141` execution-filtered pairs.

## Controlled Comparison

For the next analysis, keep the training script and hyperparameters fixed and vary only the preference dataset:

1. ProSec Python preferences.
2. Our execution-filtered Python preferences.

Then evaluate both trained Qwen2.5-Coder-7B models against the same base model on the same SecCodePLT-Python / CWEval-Python benchmark settings.
