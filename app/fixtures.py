"""Fixture files: which reader tests a change to a TPC fixture can affect.

Every TestExecutor test reads both TPC fixtures (setUpClass loads them), but each test uses only some
fields of the loaded (meta, sql, expected) pairs. The others need nothing but a loader that succeeds.
"""

from __future__ import annotations

from patch import PatchError, apply

TPC_FIXTURES = {
    "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql": (
        ("tests/test_optimizer.py::TestOptimizer::test_tpcds", (0, 1, 2)),  # check_file: all of it
        ("tests/test_executor.py::TestExecutor::test_execute_tpcds", (0, 1)),  # (meta, sql, _)
    ),
    "tests/fixtures/optimizer/tpc-h/tpc-h.sql": (
        ("tests/test_optimizer.py::TestOptimizer::test_tpch", (0, 1, 2)),
        ("tests/test_executor.py::TestExecutor::test_execute_tpch", (0, 1)),
        ("tests/test_executor.py::TestExecutor::test_optimized_tpch", (1, 2)),  # (_, sql, optimized)
    ),
}
FIXTURE_LOADER = "tests/helpers.py"


def _filter_comments(s):
    """tests/helpers.py, verbatim."""
    return "\n".join([line for line in s.splitlines() if line and not line.startswith("--")])


def _extract_meta(sql):
    """tests/helpers.py, verbatim (it raises on a '#' line without ':' and on a block of only '#' lines)."""
    meta = {}
    sql_lines = sql.split("\n")
    i = 0
    while sql_lines[i].startswith("#"):
        key, val = sql_lines[i].split(":", maxsplit=1)
        meta[key.lstrip("#").strip()] = val.strip()
        i += 1
    sql = "\n".join(sql_lines[i:])
    return sql, meta


def fixture_pairs(text: str) -> list:
    """tests/helpers.py load_sql_fixture_pairs, on the text open(path, encoding="utf-8") returns."""
    statements = _filter_comments(text.replace("\r\n", "\n").replace("\r", "\n")).split(";")
    pairs = []
    for i in range(0, len(statements) - 1, 2):
        sql, meta = _extract_meta(statements[i].strip())
        pairs.append((meta, sql, statements[i + 1].strip()))
    return pairs


def tpc_fixture_tests(path: str, before: str, hunks, readers: int, bit_of, notes: list[str]) -> int | None:
    """The readers whose fields of the loaded pairs change. None means every reader."""
    try:
        after, _blocks = apply(before, hunks)
        after.encode("utf-8")  # a lone surrogate: bytes that open(encoding="utf-8") cannot decode
    except (PatchError, UnicodeError):
        return None
    old = fixture_pairs(before)
    try:
        new = fixture_pairs(after)
    except Exception:  # noqa: BLE001 - the loader raises, so setUpClass fails for every reader
        notes.append(f"{path}: the fixture loader fails")
        return None
    bits, names = 0, []
    for nodeid, fields in TPC_FIXTURES[path]:
        if len(old) != len(new) or any(tuple(o[k] for k in fields) != tuple(n[k] for k in fields) for o, n in zip(old, new)):
            bit = bit_of(nodeid)
            if bit is None or not bit & readers:
                return None
            bits |= bit
            names.append(nodeid.rsplit("::", 1)[1])
    notes.append(f"{path}: changed pair fields -> {', '.join(names) or 'no test'}")
    return bits
