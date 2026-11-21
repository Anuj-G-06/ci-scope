"""Runtime use of tpc_exact.py: drop test_execute_tpcds when the cases a patch can reach all still pass.

Cases a patch can reach come from the per-case line map (real data, one process per case). A change
that runs at import time, or any behavioral file we have no line data for, reaches every case. The
check only runs when it is worth it: its CPU estimate (with a safety factor for slower machines) must
be below the expected scheduled seconds it saves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
# The test's class setup also loads tables through Table: a change there is outside any case.
NOT_ELIGIBLE = {"sqlglot/executor/table.py"}
CPU_FACTOR = 2.0  # board CPU has measured up to ~2x ours
FIXED_MS, EXECUTED_MS, PARSED_MS = 1300, 250, 10
PASS_RATE = 0.7  # how often a selected-but-checkable TPC-DS test passed on our labeled mutants
BUDGET_SECONDS = float(os.environ.get("SHRAPNEL_TPC_BUDGET", "60"))  # total wall time for all checks in a run


class TPC:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.spent = 0.0
        self._loaded = False  # the case list and the 5 MB line map load only when a check is considered

    def _load(self) -> None:
        if self._loaded:
            return
        root = os.path.join(self.data_dir, "tpc")
        with open(os.path.join(root, "cases.json"), encoding="utf-8") as file:
            cases = json.load(file)["cases"]
        self.count = len(cases)
        self.executed = {i for i, c in enumerate(cases) if c["execute"]}
        with open(os.path.join(root, "lines.json.z"), "rb") as file:
            self._lines = json.loads(zlib.decompress(file.read()))
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

    def worth(self, cases: set[int], saving_ms: int) -> bool:
        self._load()
        executed = len(cases & self.executed)
        cost = CPU_FACTOR * (FIXED_MS + EXECUTED_MS * executed + PARSED_MS * (len(cases) - executed))
        return PASS_RATE * saving_ms > cost

    def passes(self, workspace: str, cases: set[int], log) -> bool:
        if not cases:
            return True
        remaining = BUDGET_SECONDS - self.spent
        if remaining < 10:
            log("tpc check skipped: time budget used")
            return False
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", os.path.join(HERE, "tpc_exact.py"), workspace, self.data_dir,
                 ",".join(str(c) for c in sorted(cases))],
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
