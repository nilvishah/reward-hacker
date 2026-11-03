import ast
import math
import multiprocessing as mp
import sys

ALLOWED_IMPORTS = {"math"}
BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help",
    "memoryview", "type", "object", "super",
}
REQUIRED_ARGS = ["obs", "prev_obs", "action", "base_reward", "terminated", "truncated"]


def static_check(code: str) -> list:
    """Inspect the code without running it. Returns a list of problems (empty = safe)."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    found_fn = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    problems.append(f"blocked import: {name}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("blocked: global/nonlocal state")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda, ast.With, ast.Try)):
            problems.append(f"blocked construct: {type(node).__name__}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            problems.append(f"blocked name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            problems.append(f"blocked private attribute: {node.attr}")
        elif isinstance(node, ast.While):
            problems.append("blocked: while loops (could run forever)")
        elif isinstance(node, ast.FunctionDef) and node.name == "compute_bonus":
            found_fn = True
            args = [a.arg for a in node.args.args]
            if args != REQUIRED_ARGS:
                problems.append(f"wrong signature: {args}")

    if not found_fn:
        problems.append("missing compute_bonus function")
    return problems


def _smoke_worker(code: str, queue):
    try:
        namespace = {}
        exec(compile(code, "<reward>", "exec"), namespace)
        fn = namespace["compute_bonus"]
        samples = [
            [0.0, 1.4, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.2, 0.4, -1.2, 0.5, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [-1.0, 0.5, -2.0, 1.0, -1.5, -1.0, 1.0, 0.0],
        ]
        values = []
        for obs in samples:
            for action in range(4):
                values.append(fn(obs, samples[0], action, 0.0, False, False))
        queue.put(("ok", values))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def smoke_test(code: str, timeout: float = 5.0) -> list:
    """Run the function on sample inputs in a separate process with a time limit."""
    queue = mp.Queue()
    proc = mp.Process(target=_smoke_worker, args=(code, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        return [f"timed out after {timeout}s"]
    if queue.empty():
        return ["crashed without a result"]
    status, payload = queue.get()
    if status == "error":
        return [f"runtime error: {payload}"]
    problems = []
    for v in payload:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"returned a non-finite or non-number value: {v!r}")
            break
        if abs(v) > 10:
            problems.append(f"bonus too large: {v} (limit 10 per frame)")
            break
    return problems


def validate(code: str) -> list:
    problems = static_check(code)
    if problems:
        return problems
    return smoke_test(code)


if __name__ == "__main__":
    code = open(sys.argv[1]).read()
    problems = validate(code)
    if problems:
        print("❌ BLOCKED")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("✅ PASSED")
