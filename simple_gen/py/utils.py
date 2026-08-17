import ast
import contextlib
import io
import json
import multiprocessing as mp
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


MD_BLOCK_RE = re.compile(
    r"##\s*test setup\s*```python(.*?)```.*?##\s*tests\s*```python(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

def extract_blocks(raw: str):
    m = MD_BLOCK_RE.search(raw)
    if not m:
        raise ValueError("missing required markdown sections")

    return m.group(1).strip(), m.group(2).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def parse_blocks(setup_code: str, test_code: str):
    try:
        setup_ast = ast.parse(setup_code or "")
        test_ast = ast.parse(test_code or "")
    except SyntaxError as e:
        raise ValueError(f"invalid python syntax: {e}")

    return setup_ast, test_ast

def get_test_cases_node(test_ast: ast.Module) -> ast.Dict:
    for node in test_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "tests":
                    if isinstance(node.value, ast.Dict):
                        return node.value
                    raise ValueError("tests must be a dict")

    raise ValueError("tests assignment not found")

def validate_test_list(node: ast.AST):
    if not isinstance(node, ast.List) or len(node.elts) < 2:
        raise ValueError("each test list must have >= 2 items")

    for item in node.elts:
        if not isinstance(item, ast.Dict):
            raise ValueError("each test case must be a dict")

        seen_keys = set()
        input_node = None

        for k, v in zip(item.keys, item.values):
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                raise ValueError("test case keys must be strings")

            seen_keys.add(k.value)

            if k.value == "input":
                input_node = v

        if "input" not in seen_keys or "output" not in seen_keys:
            raise ValueError("test case must contain input and output")

        if not isinstance(input_node, ast.Dict):
            raise ValueError("input must be a dict")
        
def validate_test_cases_dict(node: ast.Dict):
    keys = []
    for k in node.keys:
        if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
            raise ValueError("tests keys must be strings")
        keys.append(k.value)

    for required in ("functionality", "security"):
        if required not in keys:
            raise ValueError(f"missing key: {required}")

    for key_node, val_node in zip(node.keys, node.values):
        key = key_node.value
        if key in ("functionality", "security"):
            validate_test_list(val_node)

def parse_and_validate(raw: str):
    setup_code, test_code = extract_blocks(raw)
    setup_ast, test_ast = parse_blocks(setup_code, test_code)

    test_cases_node = get_test_cases_node(test_ast)
    validate_test_cases_dict(test_cases_node)

    return {
        "test_setup": setup_code,
        "test_code": test_code,
    }


def _exec_source_fragment(x: Any) -> str:
    """Normalize harness source strings for ``exec``.

    JSON may set ``setup_tests`` / ``test_code`` / ``setup_code`` to ``null``; then
    ``row.get("setup_tests", "")`` is still ``None`` because the key exists.
    """

    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, str):
        return x
    return str(x)


def _jsonable(obj: Any, *, _seen: Optional[set[int]] = None) -> Any:
    """Convert to JSON/pickle-friendly primitives (cycle-safe)."""
    if _seen is None:
        _seen = set()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    oid = id(obj)
    if oid in _seen:
        return "<circular>"

    if isinstance(obj, dict):
        _seen.add(oid)
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _jsonable(v, _seen=_seen)
        _seen.remove(oid)
        return out

    if isinstance(obj, (list, tuple, set)):
        _seen.add(oid)
        out_list = [_jsonable(v, _seen=_seen) for v in obj]
        _seen.remove(oid)
        return out_list

    return str(obj)


def _exec_parse_tests_worker(setup_tests_src: str, test_code_src: str, q: "mp.Queue") -> None:
    try:
        env: Dict[str, Any] = {}
        # Suppress harness prints/warnings from polluting parent stdout.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(setup_tests_src, env, env)
            exec(test_code_src, env, env)

        tests = env.get("tests")
        if not isinstance(tests, dict):
            raise ValueError("test_code did not define a dict named 'tests'")

        # Ensure payload is always picklable; Queue failures can otherwise happen
        # asynchronously in the feeder thread and the parent sees an empty queue.
        q.put(("ok", _jsonable(tests)))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def parse_tests(setup_tests: Any, test_code: Any, *, timeout_s: float = 2.0) -> Dict[str, Any]:
    """Exec harness code in an isolated process with a hard timeout.

    This prevents hangs from infinite loops / slow imports inside harness strings.
    """
    setup_tests_src = _exec_source_fragment(setup_tests)
    test_code_src = _exec_source_fragment(test_code)

    # Prefer fork on Linux for speed and to avoid requiring an importable __main__ file.
    # Fall back to spawn on platforms where fork isn't available (e.g., Windows).
    try:
        ctx = mp.get_context("fork")
    except Exception:
        ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_exec_parse_tests_worker, args=(setup_tests_src, test_code_src, q), daemon=True)
    p.start()
    p.join(timeout=max(0.0, float(timeout_s)))
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        raise TimeoutError(f"parse_tests timed out after {timeout_s}s")

    try:
        status, payload = q.get_nowait()
    except Exception as e:
        raise ValueError(f"parse_tests failed: no result from worker ({type(e).__name__}: {e})")

    if status != "ok":
        raise ValueError(str(payload))
    if not isinstance(payload, dict):
        raise ValueError("test_code did not define a dict named 'tests'")
    return payload

# def parse_tests_without_exec(setup_tests: Any, test_code: Any) -> Dict[str, Any]:
#     setup_tests = _exec_source_fragment(setup_tests)
#     test_code = _exec_source_fragment(test_code)
#     return parse_tests(setup_tests, test_code)