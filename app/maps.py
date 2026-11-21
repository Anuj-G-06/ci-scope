"""The runtime maps built from the pinned commit, loaded lazily per touched file and cached for the batch.

Test sets are Python ints used as bitsets over the map's test index (tests.json); sets.bin stores each
distinct set once and the other maps refer to it by number.
"""

from __future__ import annotations

import json
import os

from pack import Pack
from structure import Structure

LAZY_LOAD_TEST = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"


def module_name(path: str) -> str:
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


class Data:
    def __init__(self, root: str):
        self.root = root

        def load(name):
            with open(os.path.join(root, name), "rb") as file:
                return json.loads(file.read())

        self.tests: list[str] = load("tests.json")
        self.width: int = load("meta.json")["set_width_bytes"]
        with open(os.path.join(root, "sets.bin"), "rb") as file:
            self.sets = file.read()
        users = load("users.json")
        self.class_users: dict[str, int] = users["class"]
        self.file_readers: dict[str, int] = users["file"]
        self.module_users: dict[str, int] = users["module"]
        self.import_lines: dict[str, list[int]] = users.get("import_lines", {})
        loaded = users.get("import_sqlglot")
        self.import_sqlglot = set(loaded) if loaded else None  # None: assume every module
        self.repo_files = set(load("repo_files.json"))
        self._lines = Pack(os.path.join(root, "lines.pack"))
        self._source = Pack(os.path.join(root, "source.pack"))
        self._structure = Pack(os.path.join(root, "structure.pack"))
        values = os.path.join(root, "values.pack")
        self._values = Pack(values) if os.path.exists(values) else None
        dicts = os.path.join(root, "dicts.pack")
        self._dicts = Pack(dicts) if os.path.exists(dicts) else None
        self._cache: dict[tuple[str, str], object] = {}
        self.by_file: dict[str, int] = {}
        self._bit_of: dict[str, int] | None = None
        self.lazy_load = 0  # this test runs `import sqlglot` in a subprocess that no map can see
        for i, nodeid in enumerate(self.tests):
            file = nodeid.split("::", 1)[0]
            self.by_file[file] = self.by_file.get(file, 0) | (1 << i)
            if nodeid == LAZY_LOAD_TEST:
                self.lazy_load = 1 << i
        self.attr_read: dict[str, int] = {}
        self.attr_import: set[str] = set()
        if os.path.exists(os.path.join(root, "attrs.json")):
            attrs = load("attrs.json")
            self.attr_read, self.attr_import = attrs["read"], set(attrs["import"])

    def _cached(self, kind: str, path: str, make):
        key = (kind, path)
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def lazy_load_for(self, path: str) -> int:
        """test_lazy_load runs a bare `import sqlglot`: only the files that import loads can affect it."""
        if self.import_sqlglot is None or path in self.import_sqlglot:
            return self.lazy_load
        return 0

    def bits(self, sid: int) -> int:
        offset = sid * self.width
        return int.from_bytes(self.sets[offset:offset + self.width], "little")

    def bit_of(self, nodeid: str) -> int | None:
        if self._bit_of is None:
            self._bit_of = {test: 1 << i for i, test in enumerate(self.tests)}
        return self._bit_of.get(nodeid)

    def has_source(self, path: str) -> bool:
        return path in self._source.index

    def source(self, path: str) -> str:
        return self._cached("source", path, lambda: self._source.read(path).decode("utf-8"))

    def structure(self, path: str) -> Structure:
        return self._cached("structure", path, lambda: Structure(json.loads(self._structure.read(path))))

    def line_sets(self, path: str) -> dict[int, int]:
        def make():
            if path not in self._lines.index:
                return {}
            return {int(k): v for k, v in json.loads(self._lines.read(path)).items()}

        return self._cached("lines", path, make)

    def tests_on(self, path: str, first: int, last: int) -> int:
        per_line = self.line_sets(path)
        bits = 0
        for line in range(first, last + 1):
            sid = per_line.get(line)
            if sid is not None:
                bits |= self.bits(sid)
        return bits

    def users_of_class(self, path: str, qualname: str) -> int | None:
        sid = self.class_users.get(f"{module_name(path)}:{qualname}")
        return None if sid is None else self.bits(sid)

    def users_of_module(self, path: str) -> int:
        sid = self.module_users.get(path)
        return 0 if sid is None else self.bits(sid)

    def attr_readers(self, owner: str, name: str) -> int | None:
        """Tests that read owner.NAME at runtime (resolved to owner), or None when it is also read at
        import time (it feeds derived state) or no runtime read was ever recorded."""
        key = f"{owner}|{name}"
        if key in self.attr_import or key not in self.attr_read:
            return None
        return self.bits(self.attr_read[key])

    def dict_reads(self, tag: str) -> dict | None:
        """Recorded per-key reads of a class-level dict ("module:Class|NAME"), or None if not mapped."""
        module = tag.split(":", 1)[0]
        if self._dicts is None or module not in self._dicts.index:
            return None
        return self._cached("dicts", module, lambda: json.loads(self._dicts.read(module))).get(tag)

    # ----- recorded comparison states (see infect.py) -----
    def has_values(self, path: str) -> bool:
        return self._values is not None and path in self._values.index

    def values_at(self, path: str, site: str):
        found = self._cached("values", path, lambda: json.loads(self._values.read(path)))
        return found["sites"].get(site), found["overflow"].get(site)
