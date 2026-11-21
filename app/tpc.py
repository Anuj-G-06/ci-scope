"""Runtime use of tpc_exact.py: drop test_execute_tpcds when the cases a patch can reach all still pass.

Cases a patch can reach come from the per-case line map (real data, one process per case). A change
that runs at import time, or any behavioral file we have no line data for, reaches every case.

Two ways to clear a case:
  - exact: execute it on the real tables with the patched code and compare with DuckDB's result;
  - tree (optimizer-only patches): the executor gets nothing from the optimizer but the optimized tree,
    so a case whose optimized tree equals the pinned one, and whose planning and execution run none of
    the changed lines, behaves exactly as on the pinned commit. Cases that fail the tree test fall back
    to the exact check.
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
CPU_FACTOR = 2.0  # board CPU has measured up to ~2x ours
FIXED_MS, EXECUTED_MS, PARSED_MS = 1300, 250, 10  # exact check: pandas + sqlglot imports, per case
TREE_FIXED_MS, TREE_MS = 900, 80  # tree check: sqlglot import, per case
PASS_RATE = 0.7  # how often a selected-but-checkable TPC-DS test passed on our labeled mutants
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
        self._loaded = True

    def eligible(self, behavioral_files) -> bool:
        return bool(behavioral_files) and all(
            p.startswith("sqlglot/") and p.endswith(".py") and p not in NOT_ELIGIBLE for p in behavioral_files
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
        """(exact cases, tree cases) to clear."""
        cases = self.reached(behavioral_files, touched)
        optimizer_only = all(p.startswith("sqlglot/optimizer/") for p in behavioral_files)
        if not (optimizer_only and self.trees is not None and self._post is not None and all(r for *_x, r in touched)):
            return cases, set()
        # The parser is unchanged, so cases the test only parses and pickles are cleared already.
        exact: set[int] = set()
        for path, first, last, _runtime in touched:
            per_line = self._post.get(path, {})
            for line in range(first, last + 1):
                exact.update(per_line.get(str(line), ()))
        executed = cases & self.executed
        exact &= executed
        return exact, executed - exact

    def worth(self, exact: set[int], tree: set[int], saving_ms: int) -> bool:
        self._load()
        cost = 0
        if tree:
            cost += TREE_FIXED_MS + TREE_MS * len(tree)
        if exact:
            executed = len(exact & self.executed)
            cost += FIXED_MS + EXECUTED_MS * executed + PARSED_MS * (len(exact) - executed)
        return PASS_RATE * saving_ms > CPU_FACTOR * cost

    def passes(self, workspace: str, exact: set[int], tree: set[int], log) -> bool:
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
                 ",".join(str(c) for c in sorted(exact)), ",".join(str(c) for c in sorted(tree))],
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
