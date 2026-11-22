"""Runtime use of tpc_exact.py: drop test_execute_tpcds when the cases a patch can reach all still pass.

Cases a patch can reach come from the per-case line map (real data, one process per case). A change
that runs at import time, or any behavioral file we have no line data for, reaches every case.

Two ways to clear a case:
  - exact: execute it on the real tables with the patched code and compare with DuckDB's result;
  - tree (any patch that changes only runtime lines): planning and execution get nothing from the
    test's parse, pickle round trip and optimizer but the optimized tree, so a case whose optimized
    tree and DuckDB SQL equal the pinned ones, and whose planning and execution run none of the changed
    lines, behaves exactly as on the pinned commit. Cases that fail the tree test fall back to the exact
    check. Cases the test only parses and pickles are re-parsed and pickled, which is all it does.
The check only runs when its CPU estimate (with a safety factor for slower machines) is below the
expected scheduled seconds it saves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
# The test's class setup also loads tables through Table: a change there is outside any case.
NOT_ELIGIBLE = {"sqlglot/executor/table.py"}
CPU_FACTOR = 1.3  # board CPU has tracked ours at ~1.1-1.3x (v7-v10 board runs)
# CPU of the check subprocess, measured in the container: interpreter + sqlglot import; per tree case
# (parse, pickle, optimize, fingerprint, DuckDB SQL); per parsed-only case; pandas + expected results;
# per executed case.
BASE_MS, TREE_MS, PARSED_MS, PANDAS_MS, EXECUTED_MS = 300, 40, 3, 700, 170
PASS_RATE = 0.6  # how often a selected TPC-DS test passed on our labeled commits (22 of 37)
BUDGET_SECONDS = float(os.environ.get("SHRAPNEL_TPC_BUDGET", "60"))  # total wall time for all checks in a run


class Workspace:
    """The pinned sqlglot package in a temp dir; `apply(patched)` writes a patch in, `restore()` undoes it."""

    def __init__(self, source):
        self.source = source
        self.root = None
        self.written: list[str] = []

    def apply(self, patched: dict[str, str]) -> str | None:
        if self.root is None:
            root = tempfile.mkdtemp(prefix="shrapnel-")
            for name in self.source.names():
                if name.startswith("sqlglot/") and name.endswith(".py"):
                    target = os.path.join(root, name)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "wb") as file:
                        file.write(self.source.read(name))
            self.root = root
        for path, text in patched.items():
            if not path.startswith("sqlglot/"):
                self.restore()
                return None
            with open(os.path.join(self.root, path), "w", encoding="utf-8", newline="") as file:
                file.write(text)
            self.written.append(path)
        return self.root

    def restore(self) -> None:
        for path in self.written:
            with open(os.path.join(self.root, path), "wb") as file:
                file.write(self.source.read(path))
        self.written = []


class TPC:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.spent = 0.0
        self._loaded = False  # the case list and the line maps load only when a check is considered

    def _load(self) -> None:
        if self._loaded:
            return
        root = os.path.join(self.data_dir, "tpc")
        with open(os.path.join(root, "cases.json"), encoding="utf-8") as file:
            cases = json.load(file)["cases"]
        self.count = len(cases)
        self.executed = {i for i, c in enumerate(cases) if c["execute"]}

        def read(name):
            path = os.path.join(root, name)
            if not os.path.exists(path):
                return None
            with open(path, "rb") as file:
                return json.loads(zlib.decompress(file.read()))

        self._lines = read("lines.json.z")
        self._post = read("post_lines.json.z")  # lines that run after the optimizer returns
        trees = os.path.join(root, "trees.json")  # written at image build from the pinned code
        self.trees = None
        if os.path.exists(trees):
            with open(trees, encoding="utf-8") as file:
                self.trees = {int(k): v for k, v in json.load(file).items()}
        closure = os.path.join(root, "closure.json")  # written at image build: see build_structure.py
        self.closure = None
        if os.path.exists(closure):
            with open(closure, encoding="utf-8") as file:
                self.closure = set(json.load(file))
        self._loaded = True

    def _before_planning(self, path: str) -> bool:
        """An import-time change in this file reaches a case only before the optimizer returns: neither
        planning nor execution imports it, directly or not (the DuckDB modules the test's reference SQL
        comes from, say). The tree check then sees all of it: the parse, the optimized tree, the DuckDB SQL."""
        return self.closure is not None and path.startswith("sqlglot/") and path not in self.closure

    def eligible(self, code_files) -> bool:
        """sqlglot code the check can run patched (other changed files are ones the test never reads)."""
        return bool(code_files) and all(
            p.startswith("sqlglot/") and p.endswith(".py") and p not in NOT_ELIGIBLE for p in code_files
        )

    def reached(self, behavioral_files, touched) -> set[int]:
        self._load()
        covered = {path for path, *_ in touched}
        if any(path not in covered for path in behavioral_files):
            return set(range(self.count))  # a change we have no line data for: assume everything
        cases: set[int] = set()
        for path, first, last, runtime in touched:
            if not runtime:
                return set(range(self.count))
            per_line = self._lines.get(path, {})
            for line in range(first, last + 1):
                cases.update(per_line.get(str(line), ()))
        return cases

    def plan(self, behavioral_files, touched) -> tuple[set[int], set[int]]:
        """(exact cases, tree cases) to clear. Exact cases include those the test only parses and pickles."""
        cases = self.reached(behavioral_files, touched)
        covered = {path for path, *_ in touched}
        if (
            self.trees is None or self._post is None or any(path not in covered for path in behavioral_files)
            or not all(runtime or self._before_planning(path) for path, _first, _last, runtime in touched)
        ):
            return cases, set()
        post: set[int] = set()  # cases whose planning or execution runs a changed line
        for path, first, last, _runtime in touched:
            per_line = self._post.get(path, {})
            for line in range(first, last + 1):
                post.update(per_line.get(str(line), ()))
        executed = cases & self.executed
        return (cases - self.executed) | (post & executed), executed - post

    def _cost(self, exact: set[int], tree: set[int]) -> int:
        executed = len(exact & self.executed)
        cost = BASE_MS + TREE_MS * len(tree) + PARSED_MS * (len(exact) - executed)
        return cost + (PANDAS_MS + EXECUTED_MS * executed if executed else 0)

    def worth(self, exact: set[int], tree: set[int], saving_ms: int) -> bool:
        self._load()
        if not exact and not tree:
            return saving_ms > 0  # nothing to check: passes() clears it without a subprocess
        return PASS_RATE * saving_ms > CPU_FACTOR * self._cost(exact, tree)

    def fallbacks(self, exact: set[int], tree: set[int], saving_ms: int) -> int:
        """How many tree cases that differ from the pinned trees the check may still execute exactly
        before its expected CPU outweighs the expected saving."""
        spare = PASS_RATE * saving_ms / CPU_FACTOR - self._cost(exact, tree)
        if not exact & self.executed:
            spare -= PANDAS_MS
        return max(0, min(len(tree), int(spare // EXECUTED_MS)))

    def passes(self, workspace: str, exact: set[int], tree: set[int], log, fallbacks: int = 6) -> bool:
        if not exact and not tree:
            return True
        remaining = BUDGET_SECONDS - self.spent
        if remaining < 10:
            log("tpc check skipped: time budget used")
            return False
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", os.path.join(HERE, "tpc_exact.py"), workspace, self.data_dir,
                 ",".join(str(c) for c in sorted(exact)), ",".join(str(c) for c in sorted(tree)), str(fallbacks)],
                env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")},
                capture_output=True, text=True, timeout=remaining,
            )
            verdict = json.loads(result.stdout) if result.returncode == 0 else {"verdict": "error"}
        except Exception as error:  # noqa: BLE001
            log(f"tpc check error {error!r}; keeping the test")
            return False
        finally:
            self.spent += time.monotonic() - started
        if verdict.get("verdict") != "pass":
            log(f"tpc check: {verdict.get('verdict')} at case {verdict.get('case')} ({verdict.get('why', '')[:120]})")
        return verdict.get("verdict") == "pass"
