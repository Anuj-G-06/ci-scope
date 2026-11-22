"""Runtime check of test_execute_tpcds on the cases a patch can reach, run on the patched sqlglot.

    python -I -B tpc_exact.py <dir holding the sqlglot package> <data_dir> <exact cases> [<tree cases> [<max fallbacks>]]
    python -I -B tpc_exact.py <dir holding the pinned sqlglot> <data_dir> --trees > trees.json

Exact cases do what the test does: parse in the test process, pickle the expression to a worker and
unpickle it, find its tables, execute it on the real TPC-DS tables, and compare with the DuckDB result
using the test's own assert_frame_equal call. The DuckDB result was computed from the pinned commit, so
it is only valid while the patch leaves the case's DuckDB SQL unchanged; otherwise the verdict is
"unknown" and the caller keeps the test.

Tree cases (patches that change only runtime lines the planner and executor do not run for that case)
first check that the case's DuckDB SQL is unchanged, then parse, pickle and run the real execute() up to
the planner, and fingerprint the optimized tree it would plan: every node, argument, type, comment, meta
value and cached hash, plus its parent links and any shared nodes. A case whose fingerprint equals the
pinned one is cleared; any other goes to the exact check.

Prints {"verdict": "pass" | "fail" | "unknown", "case": i, "why": ...}.
"""

import ast
import csv
import gzip
import hashlib
import json
import os
import pickle
import sys

MAX_FALLBACKS = 6  # default: tree cases that differ and may be executed; more and the test is kept


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


def fingerprint(root, exp) -> str:
    """Everything the planner can read from a tree. Iterative, so it never adds stack depth."""
    out, stack, seen = [], [(root, None, None, None)], {}
    while stack:
        value, parent, key, index = stack.pop()
        if isinstance(value, str) and parent is None and key == "mark":
            out.append(value)
            continue
        if isinstance(value, exp.Expr):
            if id(value) in seen:
                out.append(f"S:{seen[id(value)]}")  # the same node object in two places
                continue
            seen[id(value)] = len(seen)
            out.append("E:" + type(value).__name__)
            if getattr(value, "_hash", None) is not None:
                out.append("H")  # a cached hash the planner's lookups would trust, stale or not
            if parent is not None and (value.parent is not parent or value.arg_key != key or value.index != index):
                out.append("P!")  # a stale parent link
            if value.comments:
                out.append("C:" + json.dumps(value.comments))
            meta = getattr(value, "_meta", None)
            if meta:
                out.append("M:" + json.dumps(sorted(meta.items(), key=lambda kv: kv[0]), default=str))
            kind = value.type
            if kind is not None:
                try:  # types can refer back to themselves: their SQL text, not a walk
                    out.append("T:" + kind.sql())
                except Exception as error:  # noqa: BLE001
                    out.append("T!" + type(error).__name__)
            stack.append((")", None, "mark", None))
            for k, item in sorted(value.args.items(), key=lambda kv: kv[0], reverse=True):
                if isinstance(item, list):
                    stack.append(("]", None, "mark", None))
                    stack.extend((child, value, k, i) for i, child in reversed(list(enumerate(item))))
                    stack.append(("[", None, "mark", None))
                else:
                    stack.append((item, value, k, None))
                stack.append(("K:" + k, None, "mark", None))
        else:
            out.append("V:" + repr(value))
    return hashlib.sha256("\x02".join(out).encode("utf-8", "surrogatepass")).hexdigest()


def main(package_dir: str, data_dir: str, wanted: str, tree_wanted: str = "", max_fallbacks: str = "") -> None:
    sys.path.insert(0, package_dir)
    import logging

    logging.disable(logging.CRITICAL)
    import sqlglot.executor as executor_module
    from sqlglot import exp, find_tables, parse_one, transpile
    from sqlglot.executor import execute
    from sqlglot.executor.table import Table

    root = os.path.join(data_dir, "tpc")
    with open(os.path.join(root, "cases.json"), encoding="utf-8") as file:
        spec = json.load(file)
    schema, cases = spec["schema"], spec["cases"]

    def result(verdict, case=None, why=""):
        sys.stdout.write(json.dumps({"verdict": verdict, "case": case, "why": why[:300]}))

    def duckdb_changed(i):
        """(verdict, why) when the test's DuckDB side differs from the pinned one, so the pinned expected
        result no longer applies; None when it is the same SQL."""
        try:
            sql = transpile(cases[i]["sql"], write="duckdb")[0]
        except Exception as error:  # noqa: BLE001
            return "fail", f"duckdb transpile {type(error).__name__}"
        return None if sql == cases[i]["duckdb"] else ("unknown", "the DuckDB SQL changed, so the pinned expected result does not apply")

    class Planned(Exception):
        pass

    real_plan = executor_module.Plan

    def optimized_fingerprint(i):
        """execute() exactly as the test calls it, stopped where the planner would take the tree."""
        expression = pickle.loads(pickle.dumps(parse_one(cases[i]["sql"])))
        empty = {t.name: Table(columns=schema[t.name], rows=[]) for t in find_tables(expression)}
        captured = []

        def plan(tree):
            captured.append(fingerprint(tree, exp))
            raise Planned

        executor_module.Plan = plan  # execute() looks Plan up in its module globals
        try:
            execute(expression, schema=schema, tables=empty)
        except Planned:
            pass
        finally:
            executor_module.Plan = real_plan
        return captured[0] if captured else None

    if wanted == "--trees":
        sys.stdout.write(json.dumps({i: optimized_fingerprint(i) for i, c in enumerate(cases) if c["execute"]}))
        return

    indices = [int(x) for x in wanted.split(",") if x != ""]
    tree_indices = [int(x) for x in tree_wanted.split(",") if x != ""]
    if tree_indices:
        with open(os.path.join(root, "trees.json"), encoding="utf-8") as file:
            pinned = json.load(file)
        for i in tree_indices:  # cheap, and it settles most patches that change the result early
            changed = duckdb_changed(i)
            if changed:
                return result(changed[0], i, changed[1])
        fallbacks = []
        for i in tree_indices:
            try:
                same = optimized_fingerprint(i) == pinned.get(str(i))
            except Exception:  # noqa: BLE001 - the exact check will see the error, if it is real
                same = False
            if not same:
                fallbacks.append(i)
                if len(fallbacks) > (int(max_fallbacks) if max_fallbacks else MAX_FALLBACKS):
                    return result("unknown", i, "too many optimized trees differ from the pinned ones")
        indices = sorted(set(indices) | set(fallbacks))

    expected = None
    tables = {}
    pd = np = assert_frame_equal = None
    for i in indices:
        case = cases[i]
        try:
            expression = parse_one(case["sql"])  # the test process parses every case
            expression = pickle.loads(pickle.dumps(expression))  # ...and pickles it to a worker
        except Exception as error:  # noqa: BLE001
            return result("fail", i, f"parse/pickle {type(error).__name__}: {error}")
        if not case["execute"]:
            continue  # mp_execute returns None: nothing else runs for this case
        changed = duckdb_changed(i)
        if changed:
            return result(changed[0], i, changed[1])
        if expected is None:
            import numpy as np
            import pandas as pd
            from pandas.testing import assert_frame_equal

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
    main(*sys.argv[1:6])
