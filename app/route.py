"""The decision tree: which tests can a patch affect?

Test sets are Python ints used as bitsets over the map's test index. Anything the tree cannot reason
about precisely raises Everything, and the caller runs every test for that commit.
"""

from __future__ import annotations

import posixpath
import re

from fixtures import FIXTURE_LOADER, TPC_FIXTURES, tpc_fixture_tests
from maps import Data
from patch import FilePatch, PatchError, apply, parse
from pyroute import Everything, PythonRoutes

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


class Router(PythonRoutes):
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
        self._paths = {f.path for f in files}
        self.behavioral_files: set[str] = set()  # files whose change can alter behavior
        self.touched: list[tuple[str, int, int, bool]] = []  # (path, first, last, runtime-only) per changed line
        self.patched: dict[str, str] = {}  # python path -> patched text
        bits = 0
        for file_patch in files:
            before_units = self._units
            bits |= self.route_file(file_patch, notes)
            if self._units > before_units:
                self.behavioral_files.add(file_patch.path)
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
            parts = path.split("/")
            if (len(parts) == 1 and name.endswith((".py", ".pth"))) or (len(parts) == 2 and name == "__init__.py"):
                # The checkout is on sys.path ahead of the standard library and site-packages: a new
                # top-level module or package (json.py, sitecustomize.py, pandas/__init__.py) can
                # shadow what every test imports.
                raise Everything(f"{path}: new top-level module")
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
        if fp.path in TPC_FIXTURES and FIXTURE_LOADER not in self._paths:
            found = None
            if self.data.has_source(fp.path):
                readers = self.data.bits(sid)
                found = tpc_fixture_tests(fp.path, self.data.source(fp.path), fp.hunks, readers, self.data.bit_of, notes)
            if found is not None:
                self._units += 1 if found else 0
                return found
        if fp.path.endswith(".sql"):
            changed = [line for hunk in fp.hunks for line in hunk.lines if line[:1] in "+-"]
            # tests/helpers.py drops empty lines and lines starting with "--" before parsing.
            if changed and all(line[1:] == "" or line[1:].startswith("--") for line in changed):
                notes.append(f"{fp.path}: only lines the fixture loader drops")
                return 0
        notes.append(f"{fp.path}: fixture -> readers")
        self._units += 1
        return self.data.bits(sid)
