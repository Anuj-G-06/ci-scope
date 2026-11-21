"""Runtime use of proxy.py: drop the TPC-H / TPC-DS executor tests when a patch provably leaves them alone.

Applies only when every file the patch changes behaviorally is in a module whose effect proxy.py's
digests capture in full (parsing, optimizing and generating). The executor's runtime code, the planner,
expressions, helpers, schema, serde and anything under tests/ always keep the tests. Any error,
timeout or unexpected output keeps them too, and a total time budget protects the run's time limit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = {
    "tpcds": ("tests/test_executor.py::TestExecutor::test_execute_tpcds",),
    "tpch": ("tests/test_executor.py::TestExecutor::test_execute_tpch",
             "tests/test_executor.py::TestExecutor::test_optimized_tpch"),
}
ELIGIBLE_PREFIXES = (
    "sqlglot/parser.py", "sqlglot/parsers/", "sqlglot/generator.py", "sqlglot/generators/",
    "sqlglot/dialects/", "sqlglot/typing/", "sqlglot/optimizer/", "sqlglot/transforms.py",
    "sqlglot/time.py", "sqlglot/tokens.py", "sqlglot/tokenizer_core.py", "sqlglot/jsonpath.py",
    "sqlglot/trie.py",
)
NOT_ELIGIBLE = {"sqlglot/optimizer/scope.py"}  # the executor calls build_scope while it runs
BUDGET_SECONDS = float(os.environ.get("SHRAPNEL_STANDIN_BUDGET", "90"))
# CPU seconds one stand-in run costs per suite (measured in the scoring image, rounded up). A suite is
# only checked when dropping its tests would save more scheduled seconds than this.
CPU_COST_MS = {"tpch": 1500, "tpcds": 7000}


def eligible(paths) -> bool:
    return bool(paths) and all(p.startswith(ELIGIBLE_PREFIXES) and p not in NOT_ELIGIBLE for p in paths)


class StandIn:
    def __init__(self, data_dir: str, source):
        self.spec = os.path.join(data_dir, "proxy_spec.json")
        with open(os.path.join(data_dir, "proxy_baseline.json"), encoding="utf-8") as file:
            self.baseline = json.load(file)
        self.source = source  # pack.Pack of the pinned files
        self.workspace = None
        self.spent = 0.0

    def _prepare(self) -> str:
        if self.workspace is None:
            root = tempfile.mkdtemp(prefix="shrapnel-")
            for name in self.source.names():
                if name.startswith("sqlglot/") and name.endswith(".py"):
                    target = os.path.join(root, name)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "wb") as file:
                        file.write(self.source.read(name))
            self.workspace = root
        return self.workspace

    def unchanged_suites(self, patched: dict[str, str], log, suites=("tpch", "tpcds")) -> set[str]:
        """Suites whose every digest equals the pinned commit's under this patch."""
        if self.spent >= BUDGET_SECONDS:
            log("stand-in skipped: time budget used")
            return set()
        started = time.monotonic()
        root = self._prepare()
        written = []
        try:
            for path, text in patched.items():
                if not path.startswith("sqlglot/"):
                    return set()
                target = os.path.join(root, path)
                with open(target, "w", encoding="utf-8", newline="") as file:
                    file.write(text)
                written.append(path)
            remaining = max(5.0, BUDGET_SECONDS - self.spent)
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", os.path.join(HERE, "proxy.py"), root, self.spec, ",".join(suites)],
                env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")},
                capture_output=True, text=True, timeout=remaining,
            )
            if result.returncode != 0:
                log(f"stand-in failed (exit {result.returncode}); keeping the tests")
                return set()
            digests = json.loads(result.stdout)
            return {suite for suite in suites if digests.get(suite) == self.baseline.get(suite)}
        except Exception as error:  # noqa: BLE001 - never let the stand-in cost a miss
            log(f"stand-in error {error!r}; keeping the tests")
            return set()
        finally:
            for path in written:  # restore the pinned file
                with open(os.path.join(root, path), "wb") as file:
                    file.write(self.source.read(path))
            self.spent += time.monotonic() - started
