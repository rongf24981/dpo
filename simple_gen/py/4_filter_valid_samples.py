#!/usr/bin/env python3
"""Generate candidate solutions with an LLM, then keep rows whose code passes the simple_gen harness.

Reads either **flat** simple_gen JSONL rows or **VeRL / GRPO** rows produced by
``4_gather_rl_datasets_per_cwe.py`` (``prompt`` as chat list, harness under
``reward_model.ground_truth``). It asks ``gpt-5.4-mini`` (configurable) to emit code, then evaluates with the
same subprocess harness as ``verl/recipe/security_harness/safe_k_reward.py`` (``run_tests``).

Output rows keep the **same top-level schema** as the input (e.g. full RL record plus ``generated_code`` / harness fields).

Phase 1 uses a thread pool like ``2_harness_alignment.py``. Phase 2 runs the harness on each generation
and writes only passing rows to the primary ``--output-jsonl``; optionally writes every attempt to
``--all-output-jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from const_cwe_id_desc import DEFAULT_OPENAI_BASE_URL
from utils import parse_tests, read_jsonl

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package required") from exc

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_CACHE_WRITE_LOCK = threading.Lock()


def load_dotenv_file(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def load_dotenv_for_script(script_path: str) -> None:
    base = Path(script_path).resolve().parent
    for candidate in (base / ".env", Path.cwd() / ".env"):
        load_dotenv_file(candidate)


def _verl_root() -> Path:
    # This file lives at `.../security-test-case/simple_gen/py/5_filter_valid_samples.py`
    # and `verl/` lives at `.../security-test-case/verl/`.
    return Path(__file__).resolve().parents[2] / "verl"


def _import_run_harness_unittest():
    root = _verl_root()
    if not root.is_dir():
        raise RuntimeError(f"expected verl root at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from recipe.security_harness.execution_harness import (
        run_simple_gen_harness_unittest,
    )

    return run_simple_gen_harness_unittest


def extract_python_from_response(raw: str) -> str:
    m = CODE_BLOCK_RE.search(raw or "")
    if m:
        return m.group(1).strip()
    return (raw or "").strip()


def _generation_cache_key(*, model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    # Some upstream rows may contain lone surrogate code points.
    # `errors="surrogatepass"` keeps the encoding deterministic for caching.
    return hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()


def _read_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _flatten_chat_prompt(prompt: Any) -> str:
    """Turn GRPO-style ``[{"role","content"}, ...]`` or a plain string into one instruction string."""
    if isinstance(prompt, list):
        parts: List[str] = []
        for m in prompt:
            if not isinstance(m, dict):
                continue
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
            elif c is not None and not isinstance(c, str):
                parts.append(json.dumps(c, ensure_ascii=False, default=str))
        return "\n\n".join(parts)
    if isinstance(prompt, str):
        return prompt
    return ""


def _rl_ground_truth_bundle(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """If ``row`` is VeRL-style, return a flat harness dict; else ``None``."""
    rm = row.get("reward_model")
    if not isinstance(rm, dict):
        return None
    gt = rm.get("ground_truth")
    if not isinstance(gt, dict):
        return None
    fn = gt.get("function_name")
    tc = gt.get("test_code")
    if not (isinstance(fn, str) and fn.strip() and isinstance(tc, str) and tc.strip()):
        return None
    return {
        "function_name": fn.strip(),
        "test_code": tc,
        "setup_tests": gt["setup_tests"] if isinstance(gt.get("setup_tests"), str) else "",
        "setup_code": gt["setup_code"] if isinstance(gt.get("setup_code"), str) else "",
    }


def _infer_arguments_from_tests(task_row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Infer ``arguments`` (names) from testcase input keys when missing.

    RL/VeRL rows often omit an explicit arguments schema; however the harness contract
    requires ``func(**case['input'])`` so the keys strongly indicate parameter names.
    """
    try:
        tests = parse_tests(task_row.get("setup_tests"), task_row.get("test_code"))
    except Exception:
        return []

    seen: List[str] = []
    for category in ("functionality", "security"):
        cases = tests.get(category) or []
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            inp = case.get("input")
            if not isinstance(inp, dict):
                continue
            for k in inp.keys():
                if isinstance(k, str) and k and k not in seen:
                    seen.append(k)

    return [{"name": k, "type": "Any", "description": ""} for k in seen]


def task_row_for_filter(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flat simple_gen-like dict for ``build_solution_instruction`` / ``parse_tests``."""
    bundled = _rl_ground_truth_bundle(row)
    if bundled is not None:
        task: Dict[str, Any] = {**bundled}
        task["prompt"] = _flatten_chat_prompt(row.get("prompt"))
        args = row.get("arguments")
        task["arguments"] = args if isinstance(args, list) else []
        if not task["arguments"]:
            task["arguments"] = _infer_arguments_from_tests(task)
        ret = row.get("return")
        task["return"] = ret if ret is not None else ""
        raises = row.get("raises")
        task["raises"] = raises if isinstance(raises, str) else (raises or "")
        ei = row.get("extra_info")
        if isinstance(ei, dict) and ei.get("cwe"):
            task["cwe"] = ei["cwe"]
        return task
    out = dict(row)
    if isinstance(out.get("prompt"), list):
        out["prompt"] = _flatten_chat_prompt(out.get("prompt"))
    return out


def harness_ground_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal row for ``run_simple_gen_harness_unittest``."""
    bundled = _rl_ground_truth_bundle(row)
    if bundled is not None:
        return bundled
    return {
        "function_name": row.get("function_name"),
        "test_code": row.get("test_code"),
        "setup_tests": row["setup_tests"] if isinstance(row.get("setup_tests"), str) else "",
        "setup_code": row["setup_code"] if isinstance(row.get("setup_code"), str) else "",
    }


def _tests_preview(row: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        tests = parse_tests(row.get("setup_tests"), row.get("test_code"))
        return tests, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _security_property_names(security: Any) -> List[str]:
    names: List[str] = []
    if not isinstance(security, list):
        return names
    for case in security:
        if not isinstance(case, dict):
            continue
        sp = case.get("security_property")
        if isinstance(sp, str) and sp.strip():
            names.append(sp.strip())
    return sorted(set(names))


def _json_sanitize(obj: Any, *, _seen: Optional[set[int]] = None) -> Any:
    """Best-effort conversion to JSON-serializable types, robust to cycles.

    This script embeds parsed testcase payloads into the LLM prompt. Some harnesses
    can include objects with self-references; Python's json encoder raises
    ``ValueError: Circular reference detected`` in that case. We prefer a stable,
    readable blob over failing the whole row.
    """
    if _seen is None:
        _seen = set()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("utf-8", errors="replace")

    oid = id(obj)
    if oid in _seen:
        return "<circular>"

    if isinstance(obj, dict):
        _seen.add(oid)
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            try:
                key = k if isinstance(k, str) else str(k)
            except Exception:
                key = "<unprintable key>"
            out[key] = _json_sanitize(v, _seen=_seen)
        _seen.remove(oid)
        return out

    if isinstance(obj, (list, tuple, set)):
        _seen.add(oid)
        out_list = [_json_sanitize(v, _seen=_seen) for v in obj]
        _seen.remove(oid)
        return out_list

    # Fallback for arbitrary objects (including exceptions/classes)
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def build_solution_instruction(row: Dict[str, Any]) -> str:
    tests, parse_err = _tests_preview(row)
    if tests is not None:
        functionality = tests.get("functionality", [])
        security = tests.get("security", [])
        prop_names = _security_property_names(security)
        cases_blob = json.dumps(
            _json_sanitize({"functionality": functionality, "security": security}),
            ensure_ascii=False,
            indent=2,
        )
        if prop_names:
            prop_line = (
                "These ``security_property`` strings appear on security testcases; each must map to a "
                f"callable in ``security_properties`` and return True when invoked as "
                f"``checker(result, case_input, generated_source)`` with your source: {prop_names}."
            )
        else:
            prop_line = ""
    else:
        cases_blob = f"<could not parse tests: {parse_err}>"
        prop_line = "Testcases could not be parsed from harness code; rely on the fenced ``test_code`` above."

    arguments = json.dumps(row.get("arguments", []), ensure_ascii=False, indent=2)
    ret = json.dumps(row.get("return"), ensure_ascii=False)
    raises = row.get("raises", "")
    setup_code = row.get("setup_code") or ""

    wo_test_cases = f"""You write a single code block that implements the task below.

The evaluator will:
1. Execute ``setup_code`` (environment / fixtures available to your implementation).
2. Load your code, then run ``function_name(**case["input"])`` for each testcase.
3. For test cases with an ``output`` field, require the return value to equal ``output`` (or expect an exception when ``output`` is Exception).


## Task

**function_name:** {row.get("function_name", "")}
**arguments (hint):**
{arguments}
**return:** {ret}
**raises:** {raises}

**setup_code (runs before your implementation is loaded):**
```
{setup_code}
```

**Prompt (what to implement):**
{row.get("prompt", "")}

## Output rules
- Return **only** one markdown fenced code block (``` ... ```) containing the full solution source.
- Define ``{row.get("function_name", "solution")}`` exactly as specified; no prose outside the fence.
- Your implementation must be self-contained aside from ``setup_code`` already executed by the harness.

"""
    w_test_cases = f"""You write a single code block that implements the task below.

The evaluator will:
1. Execute ``setup_code`` (environment / fixtures available to your implementation).
2. Load your code, then run ``function_name(**case["input"])`` for each testcase.
3. For test cases with an ``output`` field, require the return value to equal ``output`` (or expect an exception when ``output`` is Exception).


## Task

**function_name:** {row.get("function_name", "")}
**arguments (hint):**
{arguments}
**return:** {ret}
**raises:** {raises}

**setup_code (runs before your implementation is loaded):**
```
{setup_code}
```

**Prompt (what to implement):**
{row.get("prompt", "")}

**Test harness — setup_tests (imports / globals for tests only):**
```
{row.get("setup_tests") or ""}
```

**Test harness — test_code (test cases ``tests = {{...}}``):**
```
{row.get("test_code") or ""}
```

**Parsed testcase payloads (must satisfy every entry when combined with your code):**
```json
{cases_blob}
```

---

## Output rules
- If the test cases are rewarding insecure behavior, simply output "REJECT" and nothing else.
- Otherwise, return **only** one markdown fenced code block (``` ... ```) containing the full solution source.
- Define ``{row.get("function_name", "solution")}`` exactly as specified; no prose outside the fence.
- Your implementation must be self-contained aside from ``setup_code`` already executed by the harness.
"""
    if row.get("cwe") in ["CWE-78", "CWE-79", "CWE-200", "CWE-352", "CWE-601"]:
        return wo_test_cases
    else:
        return w_test_cases
# 4. For **security** cases that include a non-empty ``security_property`` string, the harness loads ``security_properties`` from the test bundle (``setup_tests`` / merged ``test_code``), looks up that name, and **must** get ``True`` from
#    ``checker(result, case_input, generated_source)`` where ``result`` is your function's return value, ``case_input`` is the testcase ``input`` dict, and ``generated_source`` is your full solution source. If the checker is missing, not callable, raises, or returns false, that security case fails. When both ``output`` and ``security_property`` are present, both the value check and the property check must succeed for that case to score fully.


def harness_fully_passes(summary_ok: bool, ut_result: Dict[str, Any]) -> bool:
    """True only when the reward harness reports all cases passed.

    ``safe_k_reward.run_tests`` already folds ``security_property`` checks into the
    per-case score (checker must return True when that key is present); requiring
    ``security_passed == security_total`` therefore includes property tests.
    """
    if not summary_ok:
        return False
    fp, ft = ut_result.get("functionality", (0, 0))
    sp, st = ut_result.get("security", (0.0, 0))
    if ft <= 0 or st <= 0:
        return False
    if int(fp) != int(ft):
        return False
    try:
        return float(sp) + 1e-6 >= float(st)
    except (TypeError, ValueError):
        return False


def process_row_generate(
    client: OpenAI,
    model: str,
    source_row: Dict[str, Any],
    task_row: Dict[str, Any],
    max_retries: int,
    *,
    cache_dir: Optional[Path],
    cache_read: bool,
    cache_write: bool,
) -> Dict[str, Any]:
    out = dict(source_row)
    out["mini_gen_error"] = None
    out["generated_code"] = ""
    out["mini_gen_cache_hit"] = False

    fn = task_row.get("function_name")
    if not (isinstance(fn, str) and fn.strip()):
        out["mini_gen_error"] = "skipped: missing function_name"
        return out
    if not (task_row.get("test_code") or "").strip():
        out["mini_gen_error"] = "skipped: empty test_code"
        return out

    try:
        prompt = build_solution_instruction(task_row)
    except Exception as e:
        out["mini_gen_error"] = f"error building solution instruction: {type(e).__name__}: {e}"
        return out
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        key = _generation_cache_key(model=model, prompt=prompt)
        cache_path = cache_dir / f"{key}.json"
        if cache_read:
            cached = _read_cache(cache_path)
            if isinstance(cached, dict):
                cached_code = cached.get("generated_code")
                cached_err = cached.get("mini_gen_error")
                if isinstance(cached_code, str) and cached_code.strip() and not cached_err:
                    out["generated_code"] = cached_code
                    out["mini_gen_error"] = None
                    out["mini_gen_cache_hit"] = True
                    return out

    last_err: Optional[str] = None
    for _ in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=(
                    "You are an expert Python engineer. "
                    "Follow the output format exactly: one ```python fenced block only."
                ),
                input=prompt,
                service_tier="flex",
            )
            raw = (resp.output_text or "").strip()
            code = extract_python_from_response(raw)
            if not code.strip():
                raise ValueError("empty code after extraction")
            out["generated_code"] = code
            out["mini_gen_error"] = None
            if cache_path is not None and cache_write:
                with _CACHE_WRITE_LOCK:
                    _atomic_write_json(
                        cache_path,
                        {
                            "model": model,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8", errors="surrogatepass")
                            ).hexdigest(),
                            "generated_code": code,
                            "mini_gen_error": None,
                        },
                    )
            return out
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    out["mini_gen_error"] = last_err or "no valid output after retries"
    if cache_path is not None and cache_write:
        with _CACHE_WRITE_LOCK:
            _atomic_write_json(
                cache_path,
                {
                    "model": model,
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                    "generated_code": "",
                    "mini_gen_error": out["mini_gen_error"],
                },
            )
    return out


def process_row_use_existing(source_row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(source_row)
    out["mini_gen_error"] = None
    code = out.get("generated_code")
    if not (isinstance(code, str) and code.strip()):
        out["mini_gen_error"] = "skipped: missing generated_code (use generation instead)"
        out["generated_code"] = ""
    out["mini_gen_cache_hit"] = False
    return out


def evaluate_harness(row_with_code: Dict[str, Any], run_unittest: Any) -> Dict[str, Any]:
    out = dict(row_with_code)
    code = out.get("generated_code") or ""
    if out.get("mini_gen_error"):
        out["harness_passed"] = False
        out["harness_summary_ok"] = False
        out["harness_ut_result"] = {}
        return out

    base_row = harness_ground_row(out)
    ok, ut_result = run_unittest(code, base_row)
    out["harness_summary_ok"] = ok
    out["harness_ut_result"] = {k: list(v) if isinstance(v, tuple) else v for k, v in ut_result.items()}
    out["harness_passed"] = harness_fully_passes(ok, ut_result)
    return out


def _strip_generation_and_harness_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Emit only original keys + generated_code."""
    out = dict(row)
    code_val = out.get("generated_code", "")
    for k in (
        "mini_gen_error",
        "mini_gen_cache_hit",
        "harness_passed",
        "harness_summary_ok",
        "harness_ut_result",
    ):
        out.pop(k, None)
    out["generated_code"] = code_val if isinstance(code_val, str) else str(code_val)
    return out


def main() -> None:
    load_dotenv_for_script(__file__)
    run_unittest = _import_run_harness_unittest()

    parser = argparse.ArgumentParser(
        description="LLM-generate solutions for simple_gen JSONL rows; keep harness-passing rows"
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Rows where harness_passed is true")
    parser.add_argument(
        "--all-output-jsonl",
        type=Path,
        default=None,
        help="Write every row after generation + harness eval (default: no file)",
    )
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help=(
            "Timeout (seconds) for each OpenAI API call. "
            "Prevents a single stuck request from hanging the whole run."
        ),
    )
    parser.add_argument(
        "--use-existing-generated-code",
        action="store_true",
        help="Skip LLM generation and evaluate the existing `generated_code` field on each row.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/filter_valid_samples"),
        help="Directory for on-disk generation cache (keyed by model+prompt).",
    )
    parser.add_argument(
        "--no-cache-read",
        action="store_true",
        help="Disable reading from cache.",
    )
    parser.add_argument(
        "--no-cache-write",
        action="store_true",
        help="Disable writing to cache.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    # Set an explicit per-request timeout; otherwise a stuck network call can block a worker thread
    # indefinitely, which makes large batches appear "hung" near the end.
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        timeout=args.request_timeout,
    )

    gen_rows: List[Optional[Dict[str, Any]]] = [None] * len(rows)

    def _gen_one(i: int, r: Dict[str, Any]) -> Dict[str, Any]:
        if args.use_existing_generated_code:
            return process_row_use_existing(r)
        task = task_row_for_filter(r)
        return process_row_generate(
            client,
            args.model,
            r,
            task,
            args.retries,
            cache_dir=args.cache_dir,
            cache_read=not args.no_cache_read,
            cache_write=not args.no_cache_write,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(_gen_one, i, rows[i]): i for i in range(len(rows))}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generating"):
            i = futures[fut]
            gen_rows[i] = fut.result()

    partial: List[Dict[str, Any]] = [r for r in gen_rows if r is not None]
    merged: List[Optional[Dict[str, Any]]] = [None] * len(partial)

    def _eval_one(i: int, r: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_harness(r, run_unittest)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(_eval_one, i, partial[i]): i for i in range(len(partial))}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="harness"):
            i = futures[fut]
            merged[i] = fut.result()

    final_rows: List[Dict[str, Any]] = [r for r in merged if r is not None]

    if args.all_output_jsonl is not None:
        args.all_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.all_output_jsonl.open("w", encoding="utf-8") as f:
            for r in final_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    passed = [r for r in final_rows if r.get("harness_passed")]
    slim = [_strip_generation_and_harness_fields(r) for r in passed]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in slim:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_gen_err = sum(1 for r in final_rows if r.get("mini_gen_error"))
    n_cache_hit = sum(1 for r in final_rows if r.get("mini_gen_cache_hit"))
    n_h = sum(1 for r in final_rows if r.get("harness_passed"))
    print(f"total rows: {len(final_rows)}")
    print(f"generation errors: {n_gen_err}")
    print(f"cache hits: {n_cache_hit}")
    print(f"harness passed: {n_h} -> {args.output_jsonl}")
    if args.all_output_jsonl:
        print(f"wrote all evaluated rows to {args.all_output_jsonl}")


if __name__ == "__main__":
    main()
