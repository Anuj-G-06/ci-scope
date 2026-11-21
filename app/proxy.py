"""Differential stand-in for sqlglot's TPC-H / TPC-DS tests, run in a subprocess on the patched sqlglot.

    python -I -S -B proxy.py <dir holding the sqlglot package> <proxy_spec.json>

For every query those tests use, it hashes everything the tests' outcome depends on, apart from data
and executor runtime code:
  - the DuckDB SQL each test runs for its expected result (`transpile(sql, write="duckdb")`),
  - the parse -> pickle -> unpickle round trip the executor test does across processes,
  - find_tables, and the real `sqlglot.executor.execute()` call on empty tables, capturing the
    optimized tree (serde, with types), the plan, and every Python code string the executor generates.

If every digest matches the pinned commit's, the real run gets the same inputs, plans and generated
code over unchanged data and unchanged executor code, so it has the same outcome. The caller only
uses this for patches confined to modules whose effect these digests capture in full.
Prints one JSON object: {"tpch": [digest, ...], "tpcds": [digest, ...]}.
"""

import hashlib
import json
import pickle
import re
import sys

_ADDRESS = re.compile(r" \(\d+\)|0x[0-9a-fA-F]+")


def _canonical_plan(step, memo) -> str:
    """A plan with no memory addresses and with dependencies in a fixed order."""
    key = id(step)
    if key not in memo:
        dependencies = sorted(_canonical_plan(d, memo) for d in step.dependencies)
        context = [_ADDRESS.sub("", line) for line in step._to_s("")]
        memo[key] = json.dumps([
            step.type_name, step.name, [e.sql() for e in step.projections],
            step.condition.sql() if step.condition else None, str(step.limit), str(step.offset),
            context, dependencies,
        ])
    return memo[key]


def main(package_dir: str, spec_path: str, suites: str = "tpch,tpcds") -> None:
    sys.path.insert(0, package_dir)
    import logging

    logging.disable(logging.CRITICAL)
    from sqlglot import exp, find_tables, parse_one, transpile
    from sqlglot.executor import execute, python
    from sqlglot.executor.table import Table

    with open(spec_path, encoding="utf-8") as file:
        spec = json.load(file)

    mark = object()

    def canon(root) -> str:
        """A tree's content with each node's arguments sorted by key (the parser fills argument dicts
        in an order that varies between runs and carries no meaning). Iterative on purpose: the
        stand-in must not use stack depth that the real test would not."""
        out, stack = [], [root]
        while stack:
            value = stack.pop()
            if type(value) is tuple and len(value) == 2 and value[0] is mark:
                out.append(value[1])
                continue
            if isinstance(value, exp.Expr):
                out.append("E:" + type(value).__name__)
                if value.comments:
                    out.append("C:" + json.dumps(value.comments))
                meta = getattr(value, "_meta", None)
                if meta:
                    out.append("M:" + json.dumps(sorted(meta.items(), key=lambda kv: kv[0]), default=str))
                stack.append((mark, ")"))
                for key, item in sorted(value.args.items(), key=lambda kv: kv[0], reverse=True):
                    stack.append(item)
                    stack.append((mark, "K:" + key))
                kind = value.type
                if kind is not None:
                    # A type as its SQL text: type nodes can refer back to themselves, so walking
                    # them as trees could loop.
                    try:
                        out.append("T:" + kind.sql())
                    except Exception as error:  # noqa: BLE001
                        out.append("T!" + type(error).__name__)
            elif isinstance(value, list):
                out.append("[")
                stack.append((mark, "]"))
                stack.extend(reversed(value))
            else:
                out.append("V:" + repr(value))
        return "\x02".join(out)

    errors = 0

    generated: list[str] = []
    original_generate = python.PythonExecutor.generate

    def generate(self, expression):
        if expression:  # the real generate() returns None for a missing expression
            try:
                generated.append(self.generator.generate(expression))
            except Exception as error:  # noqa: BLE001
                generated.append(f"!{type(error).__name__}")
        return original_generate(self, expression)

    python.PythonExecutor.generate = generate
    planned: list[str] = []
    import sqlglot.executor as executor_module

    real_plan = executor_module.Plan

    codegen = python.Python().generator(identify=True, comments=False)  # PythonExecutor's settings

    def expressions_in(value, found, seen):
        if isinstance(value, exp.Expr):
            found.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                expressions_in(key, found, seen)
                expressions_in(item, found, seen)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                expressions_in(item, found, seen)

    def step_code(root) -> list[str]:
        """Python code for every expression every plan step holds. The executor generates some of
        it only when rows exist (e.g. after grouping), so running on empty tables is not enough."""
        out, stack, visited = [], [root], set()
        while stack:
            step = stack.pop()
            if id(step) in visited:
                continue
            visited.add(id(step))
            stack.extend(step.dependencies)
            found = []
            for name, value in vars(step).items():
                if name not in ("dependencies", "dependents"):
                    expressions_in(value, found, visited)
            for expression in found:
                try:
                    out.append(codegen.generate(expression.copy()))
                except Exception as error:  # noqa: BLE001
                    out.append(f"!{type(error).__name__}")
        return sorted(out)

    def plan(expression):
        planned.append(canon(expression))
        result = real_plan(expression)
        planned.append(_canonical_plan(result.root, {}))
        planned.extend(step_code(result.root))
        return result

    executor_module.Plan = plan  # execute() looks Plan up in its module globals

    out = {}
    for suite in suites.split(","):
        schema = spec[suite]["schema"]
        empty = {name: Table(columns=columns, rows=[]) for name, columns in schema.items()}
        digests = []
        for query in spec[suite]["queries"]:
            digest = hashlib.sha256()

            def add(tag, value):
                nonlocal errors
                if tag.endswith("!") or (tag == "code" and str(value).startswith("!")):
                    errors += 1
                digest.update(f"{tag}\x00{value}\x01".encode("utf-8", "surrogatepass"))

            for key in ("sql", "optimized"):
                if key == "optimized" and suite != "tpch":
                    continue
                try:
                    add(f"duckdb:{key}", transpile(query[key], write="duckdb")[0])
                except Exception as error:  # noqa: BLE001
                    add(f"duckdb:{key}!", type(error).__name__)
            # Every query is parsed in the test process and pickled to a worker, executed or not.
            try:
                expression = pickle.loads(pickle.dumps(parse_one(query["sql"])))
                add("roundtrip", canon(expression))
            except Exception as error:  # noqa: BLE001
                expression = None
                add("roundtrip!", f"{type(error).__name__}:{_ADDRESS.sub('', str(error))[:200]}")
            if query["execute"] and expression is not None:
                generated.clear()
                planned.clear()
                try:
                    names = sorted({t.name for t in find_tables(expression)})  # a set: order is arbitrary
                    add("tables", ",".join(names))
                    tables = {name: empty[name] for name in names if name in empty}
                    result = execute(expression, schema=schema, tables=tables)
                    add("columns", ",".join(map(str, result.columns)))
                except Exception as error:  # noqa: BLE001
                    add("execute!", f"{type(error).__name__}:{_ADDRESS.sub('', str(error))[:200]}")
                for item in planned:
                    add("plan", item)
                for item in sorted(generated):  # steps run in a set-dependent order: hash the multiset
                    add("code", item)
            digests.append(digest.hexdigest())
        out[suite] = digests
    sys.stderr.write(json.dumps({"errors": errors}) + "\n")
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:4])
