"""Image build step: precompute every source file's structure so the selector never parses whole files,
and the pinned optimized-tree fingerprints of the TPC-DS cases (in the image's own Python).

    python build_structure.py <data_dir>     reads source.pack, writes structure.pack and tpc/trees.json
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pack  # noqa: E402
from structure import analyze  # noqa: E402


def main(data_dir: str) -> None:
    source = pack.Pack(os.path.join(data_dir, "source.pack"))
    target = os.path.join(data_dir, "structure.pack")
    members = {}
    for name in sorted(source.names()):
        if name.endswith(".py"):
            members[name] = json.dumps(analyze(source.read(name).decode("utf-8")), separators=(",", ":")).encode("utf-8")
    pack.write(target, members, level=6)
    count = len(members)
    print(f"structure for {count} files -> {target}")


def build_trees(data_dir: str) -> None:
    cases_path = os.path.join(data_dir, "tpc", "cases.json")
    if not os.path.exists(cases_path):
        return
    from tpc import Workspace

    import shutil

    root = Workspace(pack.Pack(os.path.join(data_dir, "source.pack"))).apply({})
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            [sys.executable, "-I", "-B", os.path.join(here, "tpc_exact.py"), root, data_dir, "--trees"],
            env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, check=True,
        ).stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)
    trees = json.loads(out)
    with open(cases_path, encoding="utf-8") as file:
        executed = sum(1 for c in json.load(file)["cases"] if c["execute"])
    if len(trees) != executed or not all(trees.values()):
        sys.exit("could not fingerprint every executed TPC-DS case on the pinned code")
    with open(os.path.join(data_dir, "tpc", "trees.json"), "w", encoding="utf-8") as file:
        json.dump(trees, file)
    print(f"tree fingerprints for {len(trees)} TPC-DS cases")


def build_closure(data_dir: str) -> None:
    """tpc/closure.json: every sqlglot file the modules that run after the optimizer import, directly or
    not. An import-time change outside it cannot reach planning or execution."""
    import ast
    import zlib

    post_path = os.path.join(data_dir, "tpc", "post_lines.json.z")
    if not os.path.exists(post_path):
        return
    with open(post_path, "rb") as file:
        post = json.loads(zlib.decompress(file.read()))
    source = pack.Pack(os.path.join(data_dir, "source.pack"))

    def module(path: str) -> str:
        name = path[:-3].replace("/", ".")
        return name[: -len(".__init__")] if name.endswith(".__init__") else name

    paths = {module(p): p for p in source.names() if p.startswith("sqlglot/") and p.endswith(".py")}

    def imports(path: str) -> set[str]:
        base = module(path).split(".")
        if not path.endswith("__init__.py"):
            base = base[:-1]
        found = set()
        for node in ast.walk(ast.parse(source.read(path))):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    package = ".".join(base[: len(base) - node.level + 1])
                    target = f"{package}.{node.module}" if node.module else package
                found.add(target)
                found.update(f"{target}.{alias.name}" for alias in node.names)  # `from package import module`
        names = set()
        for name in found:  # importing a.b.c imports a and a.b too
            parts = name.split(".")
            names.update(".".join(parts[:i]) for i in range(1, len(parts) + 1) if ".".join(parts[:i]) in paths)
        return names

    closure, todo = set(), [module(p) for p in post]
    while todo:
        name = todo.pop()
        if name in closure or name not in paths:
            continue
        closure.add(name)
        todo.extend(imports(paths[name]))
    with open(os.path.join(data_dir, "tpc", "closure.json"), "w", encoding="utf-8") as file:
        json.dump(sorted(paths[name] for name in closure), file)
    print(f"planner/executor import closure: {len(closure)} of {len(paths)} sqlglot files")


if __name__ == "__main__":
    main(sys.argv[1])
    build_trees(sys.argv[1])
    build_closure(sys.argv[1])
