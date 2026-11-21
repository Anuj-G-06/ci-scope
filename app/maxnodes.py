"""Scaled stand-in for tests/test_parser.py::TestParser::test_max_nodes, run on a given sqlglot package.

    python -I -B maxnodes.py <dir holding the sqlglot package>

The real test parses ~2M-node SQL to hit max_nodes limits (28.8 s recorded). This runs the same assertions
at 1/100 of the width (same nesting depth, same code paths, limits scaled to match), then times and
measures the peak memory of a full ~80k-node parse, so the caller can compare them with the pinned
code's: a mutation that slows parsing down or makes it hungrier keeps the real test.
Prints {"ok": bool, "seconds": float, "peak": int, "why": str}.
"""

import json
import os
import subprocess
import sys
import time
import tracemalloc

TEST = "tests/test_parser.py::TestParser::test_max_nodes"
CHECK_MS = 1500  # CPU of one run of this script, rounded up
CPU_FACTOR = 2.0
SLOWER, HUNGRIER = 3.0, 1.25  # patched/pinned limits on the full parse's time and peak memory
BUDGET_SECONDS = 30.0


class MaxNodes:
    """Drops test_max_nodes when the scaled run passes and parses no slower and no hungrier than pinned."""

    def __init__(self):
        self.spent = 0.0
        self.pinned = None

    def worth(self, behavioral_files, saving_ms: int) -> bool:
        code_only = bool(behavioral_files) and all(p.startswith("sqlglot/") and p.endswith(".py") for p in behavioral_files)
        return code_only and saving_ms > CPU_FACTOR * CHECK_MS * (2 if self.pinned is None else 1)

    def _run(self, root: str, log):
        remaining = BUDGET_SECONDS - self.spent
        if remaining < 5:
            log("max_nodes check skipped: time budget used")
            return None
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", os.path.abspath(__file__), root],
                env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")},
                capture_output=True, text=True, timeout=remaining,
            )
            return json.loads(result.stdout) if result.returncode == 0 else None
        except Exception as error:  # noqa: BLE001
            log(f"max_nodes check error {error!r}")
            return None
        finally:
            self.spent += time.monotonic() - started

    def passes(self, workspace, patched: dict, log) -> bool:
        if self.pinned is None:
            root = workspace.apply({})
            self.pinned = (self._run(root, log) if root is not None else None) or {"ok": False}
        if not self.pinned.get("ok"):
            return False
        root = workspace.apply(patched)
        try:
            got = self._run(root, log) if root is not None else None
        finally:
            workspace.restore()
        if not got or not got.get("ok"):
            log(f"max_nodes check: kept ({(got or {}).get('why', 'no result')})")
            return False
        slower = got["seconds"] > SLOWER * self.pinned["seconds"] + 0.05
        hungrier = got["peak"] > HUNGRIER * self.pinned["peak"]
        if slower or hungrier:
            log(f"max_nodes check: kept (time {got['seconds']:.3f} vs {self.pinned['seconds']:.3f}, peak {got['peak']} vs {self.pinned['peak']})")
            return False
        return True


def main(package_dir: str) -> None:
    sys.path.insert(0, package_dir)
    import logging

    logging.disable(logging.CRITICAL)
    from sqlglot import exp, parse_one
    from sqlglot.errors import ParseError

    def raises(sql, max_nodes, needle=None):
        try:
            parse_one(sql, max_nodes=max_nodes)
        except ParseError as error:
            if needle is not None and needle not in str(error):
                raise AssertionError(f"message lacks {needle!r}")
            return
        raise AssertionError(f"no ParseError at max_nodes={max_nodes}")

    out = {"ok": False, "seconds": 0.0, "peak": 0, "why": ""}
    try:
        assert isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select)
        cols = [f"col{i}" for i in range(500)]
        assert isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1_000_000), exp.Select)
        expr = "x" + "=x" * 100  # the real test's depth
        wide = f"SELECT {','.join([expr] * 100)}"  # 1/100 of its width
        tracemalloc.start()
        started = time.process_time()
        raises(wide, 10_000, "Maximum number of AST nodes")  # builds 10k nodes, like the real test's 1M
        out["seconds"] = time.process_time() - started
        out["peak"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        raises(wide, 5_000, "(5000)")
        raises("SELECT " + ",".join(f"x={i}" for i in range(100)), 5)
        assert isinstance(parse_one("SELECT 1, 2, 3"), exp.Select)
        out["ok"] = True
    except Exception as error:  # noqa: BLE001 - any failure keeps the real test
        out["why"] = f"{type(error).__name__}: {error}"[:200]
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
