#!/usr/bin/env python3
"""Build DPO pairs from first-generation failures repaired by one revision.

This script is for Abhinav's simple_gen / ProSec-style Python pipeline.

Input rows may be flat simple_gen rows or VeRL/GRPO-style rows. The script reuses
the existing simple_gen harness helpers from ``4_filter_valid_samples.py``:

1. Build the same solution prompt used by the filtering pipeline.
2. Generate an initial candidate.
3. Run the functionality/security harness.
4. If the initial candidate is not Correct & Secure, ask for one revision.
5. Keep DPO pairs where the revision becomes Correct & Secure.

The DPO output uses TRL's conversational preference format:

``{"prompt": [...], "chosen": [...], "rejected": [...]}``
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import contextlib
import io
import json
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from const_cwe_id_desc import DEFAULT_OPENAI_BASE_URL
from utils import read_jsonl

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package required") from exc


HERE = Path(__file__).resolve().parent
FILTER_SCRIPT = HERE / "4_filter_valid_samples.py"
SYSTEM_PROMPT = "You are an expert Python engineer. Return only one fenced Python code block."
CODE_FENCE = "```python\n{code}\n```"
_CACHE_LOCK = threading.Lock()


def _load_filter_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("simple_gen_filter_helpers", FILTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {FILTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl is not None:
        return read_jsonl(args.input_jsonl)

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package required for --dataset-id") from exc

    ds = load_dataset(args.dataset_id, split=args.split)
    return [dict(row) for row in ds]


def _json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _cache_key(*, model: str, api_kind: str, stage: str, prompt: str) -> str:
    raw = json.dumps(
        {"model": model, "api_kind": api_kind, "stage": stage, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()


def _read_cache(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_client(args: argparse.Namespace) -> OpenAI:
    api_key = os.environ.get(args.api_key_env) or "EMPTY"
    return OpenAI(
        api_key=api_key,
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        timeout=args.request_timeout,
    )


def call_model(
    *,
    args: argparse.Namespace,
    stage: str,
    prompt: str,
    cache_dir: Optional[Path],
) -> str:
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(model=args.model, api_kind=args.api_kind, stage=stage, prompt=prompt)}.json"
        if not args.no_cache_read:
            cached = _read_cache(cache_path)
            if cached and isinstance(cached.get("raw_output"), str):
                return cached["raw_output"]

    last_error = ""
    for attempt in range(max(1, args.retries)):
        try:
            client = make_client(args)
            if args.api_kind == "responses":
                response = client.responses.create(
                    model=args.model,
                    instructions=SYSTEM_PROMPT,
                    input=prompt,
                    max_output_tokens=args.max_tokens,
                )
                raw = (response.output_text or "").strip()
            else:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                raw = (response.choices[0].message.content or "").strip()

            if not raw:
                raise ValueError("empty model response")

            if cache_path is not None and not args.no_cache_write:
                with _CACHE_LOCK:
                    _write_cache(cache_path, {"raw_output": raw, "error": None})
            return raw
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2.0 * (attempt + 1), 10.0))

    if cache_path is not None and not args.no_cache_write:
        with _CACHE_LOCK:
            _write_cache(cache_path, {"raw_output": "", "error": last_error})
    raise RuntimeError(last_error or "model call failed")


def score_harness_result(summary_ok: bool, result: dict[str, Any]) -> dict[str, Any]:
    fp, ft = result.get("functionality", (0, 0))
    sp, st = result.get("security", (0.0, 0))

    try:
        fp_i, ft_i = int(fp), int(ft)
    except Exception:
        fp_i, ft_i = 0, 0
    try:
        sp_f, st_f = float(sp), float(st)
    except Exception:
        sp_f, st_f = 0.0, 0.0

    functionality_rate = fp_i / ft_i if ft_i > 0 else 0.0
    security_rate = sp_f / st_f if st_f > 0 else 0.0
    correct = bool(summary_ok and ft_i > 0 and fp_i == ft_i)
    secure = bool(summary_ok and st_f > 0 and sp_f + 1e-6 >= st_f)

    return {
        "summary_ok": bool(summary_ok),
        "functionality_passed": fp_i,
        "functionality_total": ft_i,
        "security_passed": sp_f,
        "security_total": st_f,
        "functionality_rate": functionality_rate,
        "security_rate": security_rate,
        "correct": correct,
        "secure": secure,
        "correct_secure": correct and secure,
        "raw_result": _json_sanitize(result),
    }


def _case_input_output(case: Any) -> tuple[dict[str, Any], Any]:
    if isinstance(case, dict):
        inp = case.get("input", {})
        out = case.get("output")
        return inp if isinstance(inp, dict) else {}, out
    if isinstance(case, (list, tuple)) and len(case) >= 2:
        inp = case[0]
        return inp if isinstance(inp, dict) else {}, case[1]
    return {}, None


def _expected_is_exception(expected: Any) -> bool:
    if isinstance(expected, type) and issubclass(expected, BaseException):
        return True
    if expected is BaseException:
        return True
    if isinstance(expected, str):
        return expected in {"Exception", "BaseException", "ValueError", "TypeError", "KeyError", "RuntimeError"}
    return False


def _exception_matches(exc: BaseException, expected: Any) -> bool:
    if isinstance(expected, type) and issubclass(expected, BaseException):
        return isinstance(exc, expected)
    if isinstance(expected, str):
        if expected in {"Exception", "BaseException"}:
            return True
        return exc.__class__.__name__ == expected
    return False


def _fallback_harness_worker(candidate_code: str, ground_truth: dict[str, Any], q: "mp.Queue") -> None:
    old_cwd = os.getcwd()
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="simple_gen_dpo_eval_") as td:
            os.chdir(td)
            env: dict[str, Any] = {}
            setup_code = ground_truth.get("setup_code") or ""
            setup_tests = ground_truth.get("setup_tests") or ""
            test_code = ground_truth.get("test_code") or ""
            function_name = str(ground_truth.get("function_name") or "").strip()
            if not function_name:
                raise ValueError("missing function_name")

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(setup_code, env, env)
                exec(candidate_code, env, env)
                exec(setup_tests, env, env)
                exec(test_code, env, env)

            fn = env.get(function_name)
            if not callable(fn):
                raise ValueError(f"candidate did not define callable {function_name}")

            tests = env.get("tests")
            if not isinstance(tests, dict):
                raise ValueError("test_code did not define tests dict")

            summary: dict[str, tuple[float, int]] = {}
            details: dict[str, list[dict[str, Any]]] = {}
            for category in ("functionality", "security"):
                cases = tests.get(category) or []
                passed = 0.0
                total = len(cases) if isinstance(cases, list) else 0
                category_details: list[dict[str, Any]] = []
                if not isinstance(cases, list):
                    summary[category] = (0.0, 0)
                    details[category] = []
                    continue

                for case in cases:
                    inp, expected = _case_input_output(case)
                    ok = False
                    actual_repr = None
                    exc_repr = None
                    try:
                        actual = fn(**inp)
                        actual_repr = repr(actual)
                        if _expected_is_exception(expected):
                            ok = False
                        else:
                            ok = actual == expected

                        if ok and isinstance(case, dict):
                            prop_name = case.get("security_property")
                            if isinstance(prop_name, str) and prop_name.strip():
                                checker = env.get(prop_name)
                                ok = callable(checker) and bool(checker(actual, inp, candidate_code))
                    except BaseException as exc:  # noqa: BLE001
                        exc_repr = f"{type(exc).__name__}: {exc}"
                        ok = _exception_matches(exc, expected)

                    if ok:
                        passed += 1.0
                    category_details.append(
                        {
                            "passed": ok,
                            "input": repr(inp),
                            "expected": repr(expected),
                            "actual": actual_repr,
                            "exception": exc_repr,
                        }
                    )

                summary[category] = (passed, total)
                details[category] = category_details

            q.put(("ok", {"summary": summary, "details": details}))
    except BaseException as exc:  # noqa: BLE001
        q.put(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass


def fallback_run_unittest(candidate_code: str, ground_truth: dict[str, Any], timeout_s: float = 10.0) -> tuple[bool, dict[str, Any]]:
    try:
        ctx = mp.get_context("fork")
    except Exception:
        ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_fallback_harness_worker, args=(candidate_code, ground_truth, q), daemon=True)
    p.start()
    p.join(timeout=max(1.0, float(timeout_s)))
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        return False, {"functionality": (0, 0), "security": (0.0, 0), "error": f"timeout after {timeout_s}s"}
    try:
        status, payload = q.get_nowait()
    except Exception as exc:
        return False, {"functionality": (0, 0), "security": (0.0, 0), "error": f"no worker result: {type(exc).__name__}: {exc}"}
    if status != "ok":
        return False, {"functionality": (0, 0), "security": (0.0, 0), "error": str(payload)}
    summary = payload.get("summary", {})
    summary["details"] = payload.get("details", {})
    return True, summary


def evaluate_code(code: str, source_row: dict[str, Any], helpers: Any, run_unittest: Any) -> dict[str, Any]:
    try:
        ok, result = run_unittest(code, helpers.harness_ground_row(source_row))
        return score_harness_result(ok, result)
    except Exception as exc:
        return {
            "summary_ok": False,
            "functionality_passed": 0,
            "functionality_total": 0,
            "security_passed": 0.0,
            "security_total": 0.0,
            "functionality_rate": 0.0,
            "security_rate": 0.0,
            "correct": False,
            "secure": False,
            "correct_secure": False,
            "raw_result": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_revision_prompt(*, task_prompt: str, first_code: str, first_eval: dict[str, Any]) -> str:
    return f"""Revise the Python solution below so it becomes both functionally correct and secure.

Use the same task contract as the original prompt. Preserve the required function name and signature.
Return only one fenced Python code block. Do not include prose.

## Original task and tests
{task_prompt}

## First generated solution
```python
{first_code}
```

## Harness result for first solution
```json
{json.dumps(first_eval, ensure_ascii=False, indent=2)}
```

Produce the revised solution now.
"""


def dpo_row(
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


def process_row(
    *,
    idx: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    helpers: Any,
    run_unittest: Any,
) -> dict[str, Any]:
    cache_dir = None if args.cache_dir is None else args.cache_dir
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
        "first": {},
        "revision": {},
        "dpo_pair": False,
        "skip_reason": None,
    }

    try:
        first_raw = call_model(args=args, stage="first", prompt=task_prompt, cache_dir=cache_dir)
        first_code = helpers.extract_python_from_response(first_raw)
        first_eval = evaluate_code(first_code, row, helpers, run_unittest)
        out["first"] = {"raw_output": first_raw, "code": first_code, "eval": first_eval}
    except Exception as exc:
        out["skip_reason"] = f"first_generation_failed: {type(exc).__name__}: {exc}"
        return out

    if out["first"]["eval"].get("correct_secure"):
        out["skip_reason"] = "first_generation_already_correct_secure"
        return out

    revision_prompt = build_revision_prompt(
        task_prompt=task_prompt,
        first_code=out["first"]["code"],
        first_eval=out["first"]["eval"],
    )

    try:
        revision_raw = call_model(args=args, stage="revision", prompt=revision_prompt, cache_dir=cache_dir)
        revision_code = helpers.extract_python_from_response(revision_raw)
        revision_eval = evaluate_code(revision_code, row, helpers, run_unittest)
        out["revision"] = {"raw_output": revision_raw, "code": revision_code, "eval": revision_eval}
    except Exception as exc:
        out["skip_reason"] = f"revision_failed: {type(exc).__name__}: {exc}"
        return out

    if not out["revision"]["eval"].get("correct_secure"):
        out["skip_reason"] = "revision_not_correct_secure"
        return out

    out["dpo_pair"] = True
    out["skip_reason"] = None
    out["dpo"] = dpo_row(
        prompt=task_prompt,
        chosen_code=out["revision"]["code"],
        rejected_code=out["first"]["code"],
        metadata={
            "row_index": idx,
            "cwe": out.get("cwe"),
            "function_name": out.get("function_name"),
            "first_eval": out["first"]["eval"],
            "revision_eval": out["revision"]["eval"],
            "pair_rule": "first_not_correct_secure_revision_correct_secure",
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate first/revision outputs and DPO pairs for Python simple_gen data.")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-id", type=str, default="AetherPrior/py_cwe_GRPO")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dpo-output-jsonl", type=Path, required=True)
    parser.add_argument("--all-output-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--api-kind", choices=("chat", "responses"), default="chat")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--harness-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--require-abhinav-harness",
        action="store_true",
        help="Fail if Abhinav's verl/recipe/security_harness cannot be imported.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/dpo_revision_generations"))
    parser.add_argument("--no-cache-read", action="store_true")
    parser.add_argument("--no-cache-write", action="store_true")
    args = parser.parse_args()

    helpers = _load_filter_helpers()
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
        run_unittest = lambda code, gt: fallback_run_unittest(code, gt, timeout_s=args.harness_timeout_s)

    rows = load_rows(args)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    results: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {
            ex.submit(process_row, idx=i, row=row, args=args, helpers=helpers, run_unittest=run_unittest): i
            for i, row in enumerate(rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="first+revision"):
            results[futures[future]] = future.result()

    all_rows = [r for r in results if r is not None]
    dpo_rows = [r["dpo"] for r in all_rows if r.get("dpo_pair") and isinstance(r.get("dpo"), dict)]

    _jsonl_write(args.all_output_jsonl, all_rows)
    _jsonl_write(args.dpo_output_jsonl, dpo_rows)

    skip_counts: dict[str, int] = {}
    for row in all_rows:
        reason = str(row.get("skip_reason") or "dpo_pair")
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    first_evals = [r.get("first", {}).get("eval", {}) for r in all_rows]
    revised_evals = [r.get("revision", {}).get("eval", {}) for r in all_rows if r.get("revision")]
    print(f"input rows: {len(rows)}")
    print(f"dpo pairs: {len(dpo_rows)} -> {args.dpo_output_jsonl}")
    print(f"all outputs: {len(all_rows)} -> {args.all_output_jsonl}")
    print("skip / outcome counts:")
    for k, v in sorted(skip_counts.items()):
        print(f"  {k}: {v}")
    if first_evals:
        print(f"first correct&secure: {sum(bool(e.get('correct_secure')) for e in first_evals)}")
    if revised_evals:
        print(f"revision correct&secure: {sum(bool(e.get('correct_secure')) for e in revised_evals)}")


if __name__ == "__main__":
    main()
