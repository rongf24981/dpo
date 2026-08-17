#!/usr/bin/env python3
"""Convert ProSec preference data into TRL conversational preference JSONL.

Expected ProSec columns:

```
lang
cwe
original_instruction
original_code
fixed_code
benign
```

Output schema:

```
{"prompt": [...], "chosen": [...], "rejected": [...], "metadata": {...}}
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


CODE_FENCE = "```{language}\n{code}\n```"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _assistant_code(code: str, language: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": CODE_FENCE.format(language=language, code=code.strip())}]


def convert_row(row: dict[str, Any], row_index: int, language: str) -> dict[str, Any] | None:
    prompt = _text(row.get("original_instruction"))
    chosen = _text(row.get("fixed_code"))
    rejected = _text(row.get("original_code"))
    if not prompt or not chosen or not rejected:
        return None

    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": _assistant_code(chosen, language),
        "rejected": _assistant_code(rejected, language),
        "metadata": {
            "row_index": row_index,
            "source": "prosecalign/prosec-mixed-clm7b-inst",
            "lang": row.get("lang"),
            "cwe": row.get("cwe"),
            "benign": row.get("benign"),
            "pair_rule": "prosec_fixed_code_chosen_original_code_rejected",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ProSec preference pairs for TRL DPO/SimPO training.")
    parser.add_argument("--dataset-id", default="prosecalign/prosec-mixed-clm7b-inst")
    parser.add_argument("--revision", default="d4f17919b3d946bcd393d87c15dfecfa13aaf566")
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="python")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=-1)
    args = parser.parse_args()

    ds = load_dataset(args.dataset_id, split=args.split, revision=args.revision)
    rows: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(ds):
        if _text(row.get("lang")).lower() != args.language.lower():
            continue
        converted = convert_row(dict(row), idx, args.language)
        if converted is None:
            skipped += 1
            continue
        rows.append(converted)
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"dataset: {args.dataset_id}@{args.revision}")
    print(f"split: {args.split}")
    print(f"language: {args.language}")
    print(f"pairs: {len(rows)}")
    print(f"skipped_missing_fields: {skipped}")
    print(f"output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
