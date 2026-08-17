#!/usr/bin/env python3
"""Build DPO pairs from known-good reference code and generated insecure rewrites.

This is a fallback/preference-data construction path for the Abhinav-side
simple_gen work:

1. Load Python simple_gen / VeRL-style rows.
2. Recover the known Correct & Secure reference solution.
3. Ask a model to rewrite that solution into a subtly insecure variant.
4. Keep the pair only if:
   - chosen/reference code passes functionality and security tests
   - rejected/rewrite passes functionality tests
   - rejected/rewrite fails at least one security test

The DPO output uses the same conversational preference schema as
``7_train_dpo.py``:

``{"prompt": [...], "chosen": [...], "rejected": [...]}``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm


HERE = Path(__file__).resolve().parent
REPAIR_SCRIPT = HERE / "6_generate_revision_pairs.py"
CODE_FENCE = "```python\n{code}\n```"


def _load_repair_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("repair_pair_helpers", REPAIR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {REPAIR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def reference_code_from_row(row: dict[str, Any]) -> str:
    """Recover the reference solution from common simple_gen / VeRL fields."""
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}

    reward_model = row.get("reward_model")
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else {}
    if not isinstance(ground_truth, dict):
        ground_truth = {}

    return _first_text(
        extra.get("generated_code"),
        extra.get("ground_truth_solution"),
        extra.get("fixed_code"),
        extra.get("reference_code"),
        row.get("generated_code"),
        row.get("ground_truth_solution"),
        row.get("fixed_code"),
        row.get("reference_code"),
        row.get("code"),
        ground_truth.get("generated_code"),
        ground_truth.get("ground_truth_solution"),
        ground_truth.get("fixed_code"),
        ground_truth.get("reference_code"),
        ground_truth.get("code"),
    )


def build_negative_prompt(
    *,
    task_prompt: str,
    reference_code: str,
    reference_eval: dict[str, Any],
    attempt: int,
) -> str:
    return f"""Create one Python implementation to use as the rejected side of a security-preference pair.

The code must preserve the original task contract, function name, and function signature.
It should still pass the normal/functionality tests, but it should contain a realistic security weakness so that at least one security test fails.

Do not make the code obviously broken. Do not add prose. Return only one fenced Python code block.

## Original task and harness
{task_prompt}

## Known Correct & Secure reference solution
```python
{reference_code}
```

## Reference harness result
```json
{json.dumps(reference_eval, ensure_ascii=False, indent=2)}
```

Rewrite attempt: {attempt}

Produce the intentionally insecure but functionally plausible implementation now.
"""


def make_dpo_row(
    *,
    prompt: str,
    chosen_code: str,
    rejected_code: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": CODE_FENCE.format(code=chosen_code.strip())}],
        "rejected": [{"role": "assistant", "content": CODE_FENCE.format(code=rejected_code.strip())}],
        "metadata": metadata,
    }


def _passes_negative_filter(eval_result: dict[str, Any], *, min_functionality_rate: float, max_security_rate: float) -> bool:
    try:
        functionality_rate = float(eval_result.get("functionality_rate", 0.0))
    except Exception:
        functionality_rate = 0.0
    try:
        security_rate = float(eval_result.get("security_rate", 0.0))
    except Exception:
        security_rate = 0.0
    return functionality_rate + 1e-9 >= min_functionality_rate and security_rate < max_security_rate


def process_row(
    *,
    idx: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    repair: Any,
    helpers: Any,
    run_unittest: Any,
) -> dict[str, Any]:
    source_row = helpers.task_row_for_filter(row)
    task_prompt = helpers.build_solution_instruction(source_row)
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}

    out: dict[str, Any] = {
        "row_index": idx,
        "cwe": source_row.get("cwe") or row.get("cwe") or extra_info.get("cwe"),
        "function_name": source_row.get("function_name"),
        "prompt": task_prompt,
        "reference": {},
        "negative_attempts": [],
        "dpo_pair": False,
        "skip_reason": None,
    }

    ref_code = reference_code_from_row(row)
    if not ref_code:
        out["skip_reason"] = "missing_reference_code"
        return out

    try:
        ref_code = helpers.extract_python_from_response(ref_code)
        ref_eval = repair.evaluate_code(ref_code, row, helpers, run_unittest)
        out["reference"] = {"code": ref_code, "eval": ref_eval}
    except Exception as exc:
        out["skip_reason"] = f"reference_eval_failed: {type(exc).__name__}: {exc}"
        return out

    if not out["reference"]["eval"].get("correct_secure"):
        out["skip_reason"] = "reference_not_correct_secure"
        return out

    cache_dir: Optional[Path] = None if args.cache_dir is None else args.cache_dir
    last_reason = "no_negative_attempts"
    for attempt in range(1, max(1, args.num_negative_attempts) + 1):
        negative_prompt = build_negative_prompt(
            task_prompt=task_prompt,
            reference_code=out["reference"]["code"],
            reference_eval=out["reference"]["eval"],
            attempt=attempt,
        )
        try:
            raw = repair.call_model(
                args=args,
                stage=f"reference_negative_{attempt}",
                prompt=negative_prompt,
                cache_dir=cache_dir,
            )
            neg_code = helpers.extract_python_from_response(raw)
            neg_eval = repair.evaluate_code(neg_code, row, helpers, run_unittest)
            attempt_record = {"attempt": attempt, "raw_output": raw, "code": neg_code, "eval": neg_eval}
            out["negative_attempts"].append(attempt_record)
        except Exception as exc:
            last_reason = f"negative_generation_failed: {type(exc).__name__}: {exc}"
            out["negative_attempts"].append({"attempt": attempt, "error": last_reason})
            continue

        if not _passes_negative_filter(
            neg_eval,
            min_functionality_rate=args.min_functionality_rate,
            max_security_rate=args.max_security_rate,
        ):
            last_reason = "negative_did_not_match_filter"
            continue

        out["dpo_pair"] = True
        out["skip_reason"] = None
        out["dpo"] = make_dpo_row(
            prompt=task_prompt,
            chosen_code=out["reference"]["code"],
            rejected_code=neg_code,
            metadata={
                "row_index": idx,
                "cwe": out.get("cwe"),
                "function_name": out.get("function_name"),
                "reference_eval": out["reference"]["eval"],
                "rejected_eval": neg_eval,
                "negative_attempt": attempt,
                "pair_rule": "reference_correct_secure_rejected_functional_insecure",
            },
        )
        return out

    out["skip_reason"] = last_reason
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO pairs from reference code and insecure rewrites.")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-id", type=str, default="AetherPrior/py_cwe_GRPO")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dpo-output-jsonl", type=Path, required=True)
    parser.add_argument("--all-output-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--api-kind", choices=("chat", "responses"), default="chat")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--harness-timeout-s", type=float, default=10.0)
    parser.add_argument("--num-negative-attempts", type=int, default=1)
    parser.add_argument("--min-functionality-rate", type=float, default=1.0)
    parser.add_argument("--max-security-rate", type=float, default=0.999999)
    parser.add_argument(
        "--require-abhinav-harness",
        action="store_true",
        help="Fail if Abhinav's verl/recipe/security_harness cannot be imported.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/dpo_reference_negative_generations"))
    parser.add_argument("--no-cache-read", action="store_true")
    parser.add_argument("--no-cache-write", action="store_true")
    args = parser.parse_args()

    repair = _load_repair_helpers()
    helpers = repair._load_filter_helpers()
    try:
        run_unittest = helpers._import_run_harness_unittest()
    except Exception as exc:
        if args.require_abhinav_harness:
            raise RuntimeError(
                "Could not import Abhinav's simple_gen execution harness. "
                "This local clone may have an empty security-test-case/verl directory; "
                "run this on the server copy that contains verl/recipe/security_harness, "
                "or omit --require-abhinav-harness to use the fallback harness."
            ) from exc
        print(f"Warning: using fallback harness because Abhinav harness import failed: {type(exc).__name__}: {exc}")
        run_unittest = lambda code, gt: repair.fallback_run_unittest(code, gt, timeout_s=args.harness_timeout_s)

    rows = repair.load_rows(args)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    results: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {
            ex.submit(
                process_row,
                idx=i,
                row=row,
                args=args,
                repair=repair,
                helpers=helpers,
                run_unittest=run_unittest,
            ): i
            for i, row in enumerate(rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="reference->negative"):
            results[futures[future]] = future.result()

    all_rows = [r for r in results if r is not None]
    dpo_rows = [r["dpo"] for r in all_rows if r.get("dpo_pair") and isinstance(r.get("dpo"), dict)]

    _jsonl_write(args.all_output_jsonl, all_rows)
    _jsonl_write(args.dpo_output_jsonl, dpo_rows)

    skip_counts: dict[str, int] = {}
    for row in all_rows:
        reason = str(row.get("skip_reason") or "dpo_pair")
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    reference_evals = [r.get("reference", {}).get("eval", {}) for r in all_rows]
    negative_evals = [
        attempt.get("eval", {})
        for r in all_rows
        for attempt in r.get("negative_attempts", [])
        if isinstance(attempt, dict) and attempt.get("eval")
    ]

    print(f"input rows: {len(rows)}")
    print(f"dpo pairs: {len(dpo_rows)} -> {args.dpo_output_jsonl}")
    print(f"all outputs: {len(all_rows)} -> {args.all_output_jsonl}")
    print("skip / outcome counts:")
    for k, v in sorted(skip_counts.items()):
        print(f"  {k}: {v}")
    if reference_evals:
        print(f"reference correct&secure: {sum(bool(e.get('correct_secure')) for e in reference_evals)}")
    if negative_evals:
        print(
            "negative functional+insecure: "
            f"{sum(_passes_negative_filter(e, min_functionality_rate=args.min_functionality_rate, max_security_rate=args.max_security_rate) for e in negative_evals)}"
        )


if __name__ == "__main__":
    main()
