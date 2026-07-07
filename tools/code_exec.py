from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


_SANDBOX_PREAMBLE = textwrap.dedent("""\
import builtins as __builtins__
import os as __os
import sys as __sys

_root = __os.path.abspath(".")

_orig_open = __builtins__.open
def _safe_open(*args, **kwargs):
    path = args[0]
    if isinstance(path, (str, bytes)):
        resolved = __os.path.abspath(
            __os.path.join(__os.getcwd(), path) if not __os.path.isabs(path) else path
        )
        if not resolved.startswith(_root):
            raise PermissionError(
                f"Filesystem access denied outside sandbox: {path}"
            )
    return _orig_open(*args, **kwargs)
__builtins__.open = _safe_open

import socket as __socket
class _SandboxSocket:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Network access is disabled in code execution")
    def __getattr__(self, name):
        raise RuntimeError("Network access is disabled in code execution")
__socket.socket = _SandboxSocket
__socket.create_connection = _SandboxSocket
""")


def _wrap_bare_expression(code: str) -> str:
    """REPL-style auto-capture: print the last bare expression if present.

    If the last top-level statement in *code* is a bare expression (e.g.
    ``847 * 293 / 17``), wraps it so the result is printed.  Non-string
    constants only — docstring-like strings are left alone.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    if not tree.body:
        return code

    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code

    # Skip string constants (likely docstrings or comments) and
    # function calls (already producing side effects like print())
    if isinstance(last.value, ast.Constant) and isinstance(last.value.value, str):
        return code
    if isinstance(last.value, ast.Call):
        return code

    # Replace the bare expression with: _auto_result = <expr>; print(_auto_result)
    expr_ast = last.value
    assign = ast.Assign(
        targets=[ast.Name(id="_auto_result", ctx=ast.Store())],
        value=expr_ast,
    )
    print_stmt = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[ast.Name(id="_auto_result", ctx=ast.Load())],
            keywords=[],
        ),
    )
    tree.body[-1] = assign
    tree.body.append(print_stmt)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def run_python(code: str, timeout: int = 10) -> dict[str, Any]:
    """Execute *code* in an isolated subprocess and return the result.

    The code runs in a temporary directory with a 10-second timeout.
    Captures stdout, stderr, and any runtime errors.

    Returns a dict with keys:
        success (bool)
        stdout (str)
        stderr (str)
        error (str | None)           # subprocess-level error message
        warning (str | None)         # set when success=True but stdout is empty
    """
    wrapped = _wrap_bare_expression(code)
    full_code = _SANDBOX_PREAMBLE + "\n" + wrapped

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="gaia_code_exec_")
        script_path = Path(tmpdir) / "_exec.py"
        script_path.write_text(full_code, encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": tmpdir,
            },
        )

        result: dict[str, Any] = {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": None,
            "warning": None,
        }
        if result["success"] and not result["stdout"] and not result["stderr"]:
            result["warning"] = (
                "Code executed successfully but produced no output — "
                "did you forget to print() the result?"
            )
        return result
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"Execution timed out after {timeout}s",
            "warning": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "warning": None,
        }
    finally:
        if tmpdir is not None:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
