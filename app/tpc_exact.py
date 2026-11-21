"""Exact runtime check of test_execute_tpcds on the cases a patch can reach, run on the patched sqlglot.

    python -I -B tpc_exact.py <dir holding the sqlglot package> <data_dir> <case,case,...>

For each case it does what the test does: parse in the test process, pickle the expression to a worker
and unpickle it, find its tables, execute it on the real TPC-DS tables with the patched executor, and
compare with the DuckDB result using the test's own assert_frame_equal call. The DuckDB result was
computed from the pinned commit, so it is only valid while the patch leaves the case's DuckDB SQL
unchanged; otherwise the verdict is "unknown" and the caller keeps the test.

Prints {"verdict": "pass" | "fail" | "unknown", "case": i, "why": ...}. "pass" means every given case
behaves as the test requires; cases the patch cannot reach are unchanged by construction.
"""

import ast
import csv
import gzip
import json
import os
import pickle
import sys


def open_file(file_name):
    """tests/test_executor.py's open_file, verbatim in behavior."""
    with open(file_name, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    if gzipped:
        return gzip.open(file_name, "rt", newline="")
    return open(file_name, encoding="utf-8", newline="")


def load_table(Table, path, columns):
    """tests/test_executor.py's setUpClass setup(), Python side only."""
    reader = csv.reader(open_file(path), delimiter="|")
    rows = []
    ctypes = []
    next(reader)
    for row in reader:
        if not ctypes:
            for i, v in enumerate(row):
                try:
                    ctypes.append(type(ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ctypes.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
    return Table(columns=columns, rows=rows)


def main(package_dir: str, data_dir: str, wanted: str) -> None:
    sys.path.insert(0, package_dir)
    import logging

    logging.disable(logging.CRITICAL)
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor import execute
    from sqlglot.executor.table import Table

    root = os.path.join(data_dir, "tpc")
    with open(os.path.join(root, "cases.json"), encoding="utf-8") as file:
        spec = json.load(file)
    schema, cases = spec["schema"], spec["cases"]
    indices = [int(x) for x in wanted.split(",") if x != ""]

    def result(verdict, case=None, why=""):
        sys.stdout.write(json.dumps({"verdict": verdict, "case": case, "why": why[:300]}))

    expected = None
    tables = {}
    for i in indices:
        case = cases[i]
        try:
            expression = parse_one(case["sql"])  # the test process parses every case
            expression = pickle.loads(pickle.dumps(expression))  # ...and pickles it to a worker
        except Exception as error:  # noqa: BLE001
            return result("fail", i, f"parse/pickle {type(error).__name__}: {error}")
        if not case["execute"]:
            continue  # mp_execute returns None: nothing else runs for this case
        try:
            duckdb_sql = transpile(case["sql"], write="duckdb")[0]
        except Exception as error:  # noqa: BLE001
            return result("fail", i, f"duckdb transpile {type(error).__name__}")
        if duckdb_sql != case["duckdb"]:
            return result("unknown", i, "the DuckDB SQL changed, so the pinned expected result does not apply")
        if expected is None:
            with open(os.path.join(root, "expected.pkl"), "rb") as file:
                expected = pickle.load(file)
        try:
            names = [t.name for t in find_tables(expression)]
            for name in names:
                if name not in tables:
                    tables[name] = load_table(Table, os.path.join(root, "tables", f"{name}.csv.gz"), schema[name])
            table = execute(expression, schema=schema, tables={name: tables[name] for name in names})
        except Exception as error:  # noqa: BLE001
            return result("fail", i, f"execute {type(error).__name__}: {error}")
        try:
            a = expected[str(i)] if str(i) in expected else expected[i]
            b = pd.DataFrame(((np.nan if c is None else c for c in r) for r in table.rows), columns=table.columns)
            assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
        except AssertionError as error:
            return result("fail", i, f"result differs: {error}")
        except Exception as error:  # noqa: BLE001
            return result("fail", i, f"compare {type(error).__name__}: {error}")
    result("pass")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
