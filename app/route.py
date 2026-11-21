"""The decision tree: which tests can a patch affect?

Test sets are Python ints used as bitsets over the map's test index. Anything the tree cannot reason
about precisely raises Everything, and the caller runs every test for that commit.
"""

from __future__ import annotations

import ast
import json
import posixpath
import os
import re

import infect
from pack import Pack
from patch import Block, FilePatch, PatchError, apply, parse, splice
from structure import Structure

# Files no collected test imports or reads, and that pytest does not read as configuration.
INERT_FILES = {
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md", "LICENSE",
    "sqlglot.png", "Makefile", "MANIFEST.in", ".gitignore", ".gitpod.yml",
    ".pre-commit-config.yaml", ".gitmodules", "sqlglot-integration-tests", "setup.py",
    "tests/leakcheck.py", "tests/gen_fixtures.py", "tests/fixtures/jsonpath/LICENSE",
    "sqlglot/py.typed",
}
INERT_PREFIXES = ("posts/", ".github/", "benchmarks/", "pdoc/")
CONFIG_FILES = {"pyproject.toml": ("tool.ruff",), "setup.cfg": ("mypy",)}
PYTEST_FILES = {"conftest.py", "pytest.ini", "tox.ini"}
LAZY_LOAD_TEST = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"


class Everything(Exception):
    """This commit could affect anything we cannot bound: run every test."""


def _comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _snippet(lines: list[str]) -> str:
    text = "\n".join(lines)
    if lines and lines[0][:1] in (" ", "\t"):
        return "if 1:\n" + text
    return text


def _module_name(path: str) -> str:
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _same_code(a, b) -> bool:
    """Trivial compiler equivalence: equal bytecode, ignoring line numbers and file names."""
    if type(a) is not type(b):
        return False
    if hasattr(a, "co_code"):
        fields = (
            "co_code", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_argcount",
            "co_posonlyargcount", "co_kwonlyargcount", "co_flags", "co_nlocals", "co_stacksize",
            "co_exceptiontable", "co_name",
        )
        if any(getattr(a, f, None) != getattr(b, f, None) for f in fields):
            return False
        if len(a.co_consts) != len(b.co_consts):
            return False
        return all(_same_code(x, y) for x, y in zip(a.co_consts, b.co_consts))
    if isinstance(a, (tuple, frozenset)):
        return len(a) == len(b) and all(_same_code(x, y) for x, y in zip(sorted(a, key=repr), sorted(b, key=repr)))
    return a == b and repr(a) == repr(b)


class Data:
    """Runtime maps. Everything is loaded lazily per touched file and cached for the batch."""

    def __init__(self, root: str):
        self.root = root

        def load(name):
            with open(os.path.join(root, name), "rb") as file:
                return file.read()

        self.tests: list[str] = json.loads(load("tests.json"))
        meta = json.loads(load("meta.json"))
        self.width: int = meta["set_width_bytes"]
        self.sets = load("sets.bin")
        users = json.loads(load("users.json"))
        self.class_users: dict[str, int] = users["class"]
        self.file_readers: dict[str, int] = users["file"]
        self.module_users: dict[str, int] = users["module"]
        self.import_lines: dict[str, list[int]] = users.get("import_lines", {})
        self.repo_files = set(json.loads(load("repo_files.json")))
        self._lines = Pack(os.path.join(root, "lines.pack"))
        self._source = Pack(os.path.join(root, "source.pack"))
        self._structure = Pack(os.path.join(root, "structure.pack"))
        self._line_names = self._lines.index
        self._source_names = self._source.index
        self._cache_lines: dict[str, dict[int, int]] = {}
        self._cache_source: dict[str, str] = {}
        self._cache_structure: dict[str, Structure] = {}
        self.by_file: dict[str, int] = {}
        self.lazy_load = 0  # this test runs `import sqlglot` in a subprocess that no map can see
        for i, nodeid in enumerate(self.tests):
            file = nodeid.split("::", 1)[0]
            self.by_file[file] = self.by_file.get(file, 0) | (1 << i)
            if nodeid == LAZY_LOAD_TEST:
                self.lazy_load = 1 << i
        self._open_values()

    def bits(self, sid: int) -> int:
        offset = sid * self.width
        return int.from_bytes(self.sets[offset:offset + self.width], "little")

    def has_source(self, path: str) -> bool:
        return path in self._source_names

    def source(self, path: str) -> str:
        text = self._cache_source.get(path)
        if text is None:
            text = self._cache_source[path] = self._source.read(path).decode("utf-8")
        return text

    def structure(self, path: str) -> Structure:
        found = self._cache_structure.get(path)
        if found is None:
            found = self._cache_structure[path] = Structure(json.loads(self._structure.read(path)))
        return found

    def line_sets(self, path: str) -> dict[int, int]:
        found = self._cache_lines.get(path)
        if found is None:
            if path in self._line_names:
                raw = json.loads(self._lines.read(path))
                found = {int(k): v for k, v in raw.items()}
            else:
                found = {}
            self._cache_lines[path] = found
        return found

    def tests_on(self, path: str, first: int, last: int) -> int:
        per_line = self.line_sets(path)
        bits = 0
        for line in range(first, last + 1):
            sid = per_line.get(line)
            if sid is not None:
                bits |= self.bits(sid)
        return bits

    def users_of_class(self, path: str, qualname: str) -> int | None:
        sid = self.class_users.get(f"{_module_name(path)}:{qualname}")
        return None if sid is None else self.bits(sid)

    def users_of_module(self, path: str) -> int:
        sid = self.module_users.get(path)
        return 0 if sid is None else self.bits(sid)

    # ----- recorded comparison states (see infect.py) -----
    def has_values(self, path: str) -> bool:
        return self._values is not None and path in self._value_names

    def values_at(self, path: str, site: str):
        found = self._cache_values.get(path)
        if found is None:
            found = self._cache_values[path] = json.loads(self._values.read(path))
        return found["sites"].get(site), found["overflow"].get(site)

    def _open_values(self) -> None:
        target = os.path.join(self.root, "values.pack")
        self._values = Pack(target) if os.path.exists(target) else None
        self._value_names = self._values.index if self._values is not None else {}
        self._cache_values: dict[str, dict] = {}


class Router:
    def __init__(self, data: Data):
        self.data = data
        self._units = 0  # behavioral changes in the current patch
        self._candidate: int | None = None  # tests proven unaffected by the only change, if any

    # ----- entry point -----
    def route(self, patch_text: str, notes: list[str]) -> int:
        """Bitset of tests that can be affected. Raises Everything when it cannot be bounded."""
        try:
            files = parse(patch_text)
        except PatchError as error:
            raise Everything(f"unparseable patch: {error}")
        if not files:
            raise Everything("patch names no files")
        self._units, self._candidate = 0, None
        bits = 0
        for file_patch in files:
            bits |= self.route_file(file_patch, notes)
        # Recorded states describe the pinned program. They stay valid only when the comparison is
        # the patch's one behavioral change, so pruning is applied in that case alone.
        if self._units == 1 and self._candidate:
            pruned = bits & self._candidate
            bits &= ~self._candidate
            notes.append(f"infection check removed {bin(pruned).count('1')} tests")
        return bits

    def route_file(self, fp: FilePatch, notes: list[str]) -> int:
        path = fp.path
        name = posixpath.basename(path)
        if fp.renamed or fp.binary:
            if fp.binary and path.startswith("tests/fixtures/") and path in self.data.file_readers:
                notes.append(f"{path}: binary fixture -> readers")
                self._units += 1
                return self.data.bits(self.data.file_readers[path])
            raise Everything(f"{path}: rename or binary change")
        if fp.old_path is None:  # new file
            if name in PYTEST_FILES or (name.startswith("test_") and name.endswith(".py")) or (
                path.startswith("tests/") and name == "__init__.py"
            ):
                raise Everything(f"{path}: new file pytest would collect or load")
            notes.append(f"{path}: new file nothing imports")
            return 0
        if fp.new_path is None:
            raise Everything(f"{path}: deleted file")
        if path in INERT_FILES or path.startswith(INERT_PREFIXES):
            notes.append(f"{path}: inert")
            return 0
        if path.startswith("sqlglotc/"):
            notes.append(f"{path}: sqlglotc -> test_sqlglotc")
            self._units += 1
            return self.data.by_file.get("tests/test_sqlglotc.py", 0)
        if name in PYTEST_FILES:
            raise Everything(f"{path}: pytest configuration")
        if path in CONFIG_FILES:
            return self.route_config(fp, notes)
        if path.startswith("tests/fixtures/"):
            return self.route_fixture(fp, notes)
        if path.endswith(".py") and self.data.has_source(path):
            return self.route_python(fp, notes)
        if path not in self.data.repo_files:
            raise Everything(f"{path}: not a file of the pinned commit")
        raise Everything(f"{path}: no route for this kind of file")

    # ----- configuration -----
    def route_config(self, fp: FilePatch, notes: list[str]) -> int:
        allowed = CONFIG_FILES[fp.path]
        before = self.data.source(fp.path).split("\n")
        try:
            _after, blocks = apply("\n".join(before), fp.hunks)
        except PatchError as error:
            raise Everything(f"{fp.path}: {error}")
        for block in blocks:
            lines = list(range(block.first, block.last + 1)) if not block.insertion else [block.after]
            for line in lines:
                section = ""
                for i in range(line, 0, -1):
                    match = re.match(r"^\s*\[([^\]]+)\]", before[i - 1])
                    if match:
                        section = match.group(1).strip()
                        if i == line and not block.insertion:
                            raise Everything(f"{fp.path}: section header changed")
                        break
                if not section.startswith(allowed):
                    raise Everything(f"{fp.path}: change in [{section}]")
            if any(re.match(r"^\s*\[", text) for text in block.added + block.removed):
                raise Everything(f"{fp.path}: section header added or removed")
        notes.append(f"{fp.path}: tool-only config section")
        return 0

    # ----- fixtures -----
    def route_fixture(self, fp: FilePatch, notes: list[str]) -> int:
        sid = self.data.file_readers.get(fp.path)
        if sid is None:
            raise Everything(f"{fp.path}: fixture with no recorded readers")
        if fp.path.endswith(".sql"):
            changed = [line for hunk in fp.hunks for line in hunk.lines if line[:1] in "+-"]
            # tests/helpers.py drops empty lines and lines starting with "--" before parsing.
            if changed and all(line[1:] == "" or line[1:].startswith("--") for line in changed):
                notes.append(f"{fp.path}: only lines the fixture loader drops")
                return 0
        notes.append(f"{fp.path}: fixture -> readers")
        self._units += 1
        return self.data.bits(sid)

    # ----- python -----
    def route_python(self, fp: FilePatch, notes: list[str]) -> int:
        path = fp.path
        before = self.data.source(path)
        try:
            after, blocks = apply(before, fp.hunks)
        except PatchError as error:
            raise Everything(f"{path}: {error}")
        if not blocks:
            return 0
        s = self.data.structure(path)
        before_lines = before.split("\n")

        pending = [b for b in blocks if not self._inert_text_block(b, s)]
        if not pending:
            notes.append(f"{path}: comments or blank lines only")
            return 0

        bits = 0
        for group, first, last in self._regions(pending, s):
            bits |= self._route_region(path, s, before, before_lines, after, group, first, last, notes)
        return bits

    @staticmethod
    def _inert_text_block(block: Block, s: Structure) -> bool:
        """A change made only of comment or blank lines, outside strings, away from line 1-2 cookies
        and backslash continuations, cannot change the program."""
        if not all(_comment_or_blank(line) for line in block.removed + block.added):
            return False
        if block.after < 2:  # an encoding declaration lives on line 1 or 2
            return False
        if block.insertion:
            return not s.boundary_inside_string(block.after) and not s.is_continued(block.after)
        return not any(s.inside_string(line) for line in range(block.first, block.last + 1)) and not s.is_continued(
            block.first - 1
        )

    @staticmethod
    def _regions(blocks: list[Block], s: Structure) -> list[tuple[list[Block], int, int]]:
        """Groups blocks into the smallest whole statements that contain them, merging overlaps until
        stable. A group at (0, 0) lies at module level, between or around top-level statements."""
        spans: list[tuple[int, int, list[Block]]] = []
        for block in blocks:
            if block.insertion:
                # Insertions before line 1 or after the last line are module level by definition.
                if block.after < 1 or block.after >= s.lines:
                    spans.append((0, 0, [block]))
                    continue
                spans.append((block.after, block.after + 1, [block]))
            else:
                spans.append((block.first, block.last, [block]))
        while True:
            resolved = []
            for first, last, group in spans:
                index = s.common(first, last) if first else -1
                if index == -1:
                    resolved.append((0, 0, group))
                else:
                    a, b = s.span(index)
                    resolved.append((a, b, group))
            resolved.sort(key=lambda r: (r[0], r[1]))
            merged: list[tuple[int, int, list[Block]]] = []
            for first, last, group in resolved:
                if merged:
                    p_first, p_last, p_group = merged[-1]
                    same_module_level = first == 0 and p_first == 0
                    overlap = first != 0 and p_first != 0 and first <= p_last
                    if same_module_level or overlap:
                        merged[-1] = (p_first, max(p_last, last), p_group + group)
                        continue
                merged.append((first, last, group))
            if len(merged) == len(spans):
                return [(group, first, last) for first, last, group in merged]
            spans = merged

    LARGE_REGION = 300  # lines; above this, an insertion of real code is routed without a region parse

    def _route_region(self, path, s, before, before_lines, after, group, first, last, notes) -> int:
        if first == 0:  # module level, between top-level statements
            return self._route_module_level(path, s, before, after, group, notes)
        if last - first > self.LARGE_REGION and all(block.insertion for block in group):
            return self._route_inserted_code(path, s, before, before_lines, after, group, notes)
        old_text = before_lines[first - 1:last]
        new_text = splice(before_lines, first, last, group)
        try:
            old_tree = ast.parse(_snippet(old_text))
        except SyntaxError:
            return self._route_whole_file(path, s, before, after, group, notes)
        try:
            new_tree = ast.parse(_snippet(new_text))
        except SyntaxError:
            return self._route_whole_file(path, s, before, after, group, notes)
        if infect.same(old_tree, new_tree):
            notes.append(f"{path}:{first}-{last} same syntax tree")
            return 0
        # Compile in the real function context, so return/yield/break/nonlocal are checked.
        outer = s.outermost_def(s.common(first, last))
        if outer != -1:
            c_first, c_last = s.span(outer)
            c_old, c_new = before_lines[c_first - 1:c_last], splice(before_lines, c_first, c_last, group)
        else:
            c_first, c_last, c_old, c_new = first, last, old_text, new_text
        try:
            new_code = compile(_snippet(c_new), path, "exec")
        except (SyntaxError, ValueError) as error:
            raise Everything(f"{path}: patched code does not compile ({error.__class__.__name__})")
        try:
            old_code = compile(_snippet(c_old), path, "exec")
            if _same_code(old_code, new_code):
                notes.append(f"{path}:{first}-{last} same bytecode")
                return 0
        except (SyntaxError, ValueError):
            pass
        self._units += 1
        bits = 0
        runtime_only = True
        imported = self._import_lines(path)
        for block in group:
            lines = [block.after, block.after + 1] if block.insertion else range(block.first, block.last + 1)
            for line in lines:
                if 1 <= line <= s.lines:
                    if s.kind(line) != "runtime" or line in imported:
                        runtime_only = False
                    bits |= self._route_line(path, s, line, notes)
        if runtime_only and self.data.has_values(path):
            pairs = infect.changed_comparisons(old_tree, new_tree)
            if pairs:
                wrapped = bool(old_text) and old_text[0][:1] in (" ", "\t")
                offset = first - (2 if wrapped else 1)
                self._candidate = self._proven(path, pairs, offset, notes)
        return bits

    def _proven(self, path: str, pairs, offset: int, notes: list[str]) -> int:
        """Tests whose every recorded state gives the same outcome for every changed comparison."""
        proven = None
        for old_expr, new_expr in pairs:
            site = infect.core(old_expr)
            key = f"{site.lineno + offset}:{site.col_offset}:{site.end_lineno + offset}:{site.end_col_offset}"
            entries, overflow = self.data.values_at(path, key)
            if not entries:
                notes.append(f"{path}:{key} no recorded states")
                return 0
            recorded = infected = 0
            for left, right, same, sid in entries:
                bits = self.data.bits(sid)
                recorded |= bits
                if left == "?":
                    infected |= bits
                    continue
                try:
                    old_outcome = infect.evaluate(old_expr, site, left, right, bool(same))
                    new_outcome = infect.evaluate(new_expr, site, left, right, bool(same))
                    if old_outcome != new_outcome:
                        infected |= bits
                except infect.Unknown:
                    infected |= bits
            if overflow is not None:
                bits = self.data.bits(overflow)
                recorded |= bits
                infected |= bits
            clean = recorded & ~infected
            proven = clean if proven is None else proven & clean
            notes.append(f"{path}:{key} states {len(entries)}, clean tests {bin(clean).count('1')}")
        return proven or 0

    def _route_line(self, path: str, s: Structure, line: int, notes: list[str]) -> int:
        kind = s.kind(line)
        bits = 0
        if kind in ("runtime", "both"):
            first, last = s.coverage_region(line)
            got = self.data.tests_on(path, first, last)
            if line in self._import_lines(path):
                got |= self.data.users_of_module(path)
                notes.append(f"{path}:{line} runs at import too")
            notes.append(f"{path}:{line} runtime {bin(got).count('1')}")
            bits |= got
        if kind == "header":
            body = s.def_body(line)
            got = self.data.tests_on(path, body[0], body[1]) if body else 0
            if s.decorator_line(line):
                got |= self._import_users(path, s, line, notes)
            notes.append(f"{path}:{line} def header {bin(got).count('1')}")
            bits |= got
        if kind in ("import", "both"):
            bits |= self._import_users(path, s, line, notes)
        if path.startswith("sqlglot/") and (kind != "runtime" or line in self._import_lines(path)):
            bits |= self.data.lazy_load  # code that runs during `import sqlglot`
        return bits

    def _import_lines(self, path: str) -> set[int]:
        return set(self.data.import_lines.get(path, ()))

    def _import_users(self, path: str, s: Structure, line: int, notes: list[str]) -> int:
        if path.startswith("tests/"):
            raise Everything(f"{path}:{line} import-time code in a test module")
        qualname = s.class_name(line)
        if qualname:
            users = self.data.users_of_class(path, qualname)
            if users:
                notes.append(f"{path}:{line} class {qualname} {bin(users).count('1')}")
                return users
        users = self.data.users_of_module(path)
        notes.append(f"{path}:{line} module {bin(users).count('1')}")
        return users

    def _route_inserted_code(self, path, s, before, before_lines, after, group, notes) -> int:
        """Code inserted into a big statement (say, a 9,000-line class): check it compiles in its
        function, then route its neighbours. Inserted statements always change the program."""
        self._units += 1
        bits = 0
        for block in group:
            index = s.common(block.after, block.after + 1)
            outer = s.outermost_def(index) if index != -1 else -1
            if outer != -1:
                c_first, c_last = s.span(outer)
                text = splice(before_lines, c_first, c_last, [block])
            else:
                text = block.added
            try:
                compile(_snippet(text), path, "exec")
            except (SyntaxError, ValueError):
                return self._route_whole_file(path, s, before, after, group, notes)
            for line in (block.after, block.after + 1):
                if 1 <= line <= s.lines:
                    bits |= self._route_line(path, s, line, notes)
        return bits

    def _route_module_level(self, path, s, before, after, group, notes) -> int:
        """Changes between top-level statements: parse the added text on its own."""
        for block in group:
            added = "\n".join(block.added)
            try:
                tree = ast.parse(added) if added.strip() else ast.parse("")
            except SyntaxError:
                return self._route_whole_file(path, s, before, after, group, notes)
            if tree.body or not all(_comment_or_blank(line) for line in block.removed):
                return self._import_route_or_everything(path, notes)
        notes.append(f"{path}: module-level comments only")
        return 0

    def _import_route_or_everything(self, path: str, notes: list[str]) -> int:
        if path.startswith("tests/"):
            raise Everything(f"{path}: module-level change in a test module")
        self._units += 1
        users = self.data.users_of_module(path) | self.data.lazy_load
        notes.append(f"{path}: module-level change -> module users {bin(users).count('1')}")
        return users

    def _route_whole_file(self, path, s, before, after, group, notes) -> int:
        """Rare fallback: compare the whole files."""
        try:
            new_tree = ast.parse(after)
        except SyntaxError:
            raise Everything(f"{path}: patched file does not parse")
        try:
            compile(after, path, "exec")
        except (SyntaxError, ValueError):
            raise Everything(f"{path}: patched file does not compile")
        if infect.same(ast.parse(before), new_tree):
            notes.append(f"{path}: same syntax tree (whole file)")
            return 0
        self._units += 1
        bits = 0
        for block in group:
            lines = [block.after, block.after + 1] if block.insertion else range(block.first, block.last + 1)
            for line in lines:
                if 1 <= line <= s.lines:
                    bits |= self._route_line(path, s, line, notes)
        return bits
