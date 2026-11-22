"""Routing a change to a Python file: comment-only edits, same syntax tree or bytecode, class-body
attributes (classattr.py), recorded comparison states (infect.py), then line coverage and import users."""

from __future__ import annotations

import ast
import re

import infect
from classattr import dict_route, field_route
from equiv import comment_or_blank, same_code, snippet
from patch import Block, FilePatch, PatchError, apply, splice
from structure import Structure
from testmod import test_change

_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")  # PEP 263, lines 1-2
LARGE_REGION = 300  # lines; above this, an insertion of real code is routed without a region parse


class Everything(Exception):
    """This commit could affect anything we cannot bound: run every test."""


def _inert_text_block(block: Block, s: Structure) -> bool:
    """A change made only of comment or blank lines, outside strings, away from line 1-2 cookies and
    backslash continuations, cannot change the program."""
    if not all(comment_or_blank(line) for line in block.removed + block.added) or block.after < 2:
        return False  # (an encoding declaration lives on line 1 or 2)
    if block.insertion:
        return not s.boundary_inside_string(block.after) and not s.is_continued(block.after)
    inside = any(s.inside_string(line) for line in range(block.first, block.last + 1))
    return not inside and not s.is_continued(block.first - 1)


def _regions(blocks: list[Block], s: Structure) -> list[tuple[list[Block], int, int]]:
    """Groups blocks into the smallest whole statements that contain them, merging overlaps until stable.
    A group at (0, 0) lies at module level, between or around top-level statements."""
    spans = []
    for block in blocks:
        if block.insertion and (block.after < 1 or block.after >= s.lines):
            spans.append((0, 0, [block]))  # before line 1 or after the last line: module level
        else:
            spans.append((block.after, block.after + 1, [block]) if block.insertion else (block.first, block.last, [block]))
    while True:
        resolved = []
        for first, last, group in spans:
            index = s.common(first, last) if first else -1
            resolved.append((0, 0, group) if index == -1 else (*s.span(index), group))
        resolved.sort(key=lambda r: (r[0], r[1]))
        merged = []
        for first, last, group in resolved:
            if merged:
                p_first, p_last, p_group = merged[-1]
                if (first == 0 and p_first == 0) or (first != 0 and p_first != 0 and first <= p_last):
                    merged[-1] = (p_first, max(p_last, last), p_group + group)
                    continue
            merged.append((first, last, group))
        if len(merged) == len(spans):
            return [(group, first, last) for first, last, group in merged]
        spans = merged


def _lines(blocks: list[Block], s: Structure) -> list[int]:
    return [line for block in blocks for line in block.lines if 1 <= line <= s.lines]


class PythonRoutes:
    """Mixed into route.Router, which provides data, notes plumbing and the per-patch state."""

    def route_python(self, fp: FilePatch, notes: list[str]) -> int:
        path = fp.path
        before = self.data.source(path)
        try:
            after, blocks = apply(before, fp.hunks)
        except PatchError as error:
            raise Everything(f"{path}: {error}")
        self.patched[path] = after
        try:
            after.encode("utf-8")
        except UnicodeEncodeError:  # bytes that are not UTF-8: the module no longer imports
            raise Everything(f"{path}: the patched file is not UTF-8")
        old_head, new_head = before.split("\n")[:2], after.split("\n")[:2]
        if old_head != new_head and any(_COOKIE.match(line) for line in old_head + new_head):
            raise Everything(f"{path}: encoding declaration changed")  # how Python decodes the file
        if not blocks:
            return 0
        s = self.data.structure(path)
        pending = [b for b in blocks if not _inert_text_block(b, s)]
        if not pending:
            notes.append(f"{path}: comments or blank lines only")
            return 0
        self._test_local, self._test_methods = False, None
        if path.startswith("tests/"):
            self._test_local, fresh, self._test_methods = test_change(before, after, pending)
            if fresh and all(self.data.bit_of(f"{path}::{cls}::{name}") is None for cls, name in fresh):
                notes.append(f"{path}: only new test methods ({', '.join(name for _cls, name in fresh)})")
                return 0
        bits, before_lines = 0, before.split("\n")
        for group, first, last in _regions(pending, s):
            bits |= self._route_region(path, s, before, before_lines, after, group, first, last, notes)
        return bits

    def _route_region(self, path, s, before, before_lines, after, group, first, last, notes) -> int:
        if first == 0:  # module level, between top-level statements
            return self._route_module_level(path, s, before, after, group, notes)
        if last - first > LARGE_REGION and all(block.insertion for block in group):
            return self._route_inserted_code(path, s, before, before_lines, after, group, notes)
        old_text, new_text = before_lines[first - 1:last], splice(before_lines, first, last, group)
        try:
            old_tree, new_tree = ast.parse(snippet(old_text)), ast.parse(snippet(new_text))
        except SyntaxError:
            return self._route_whole_file(path, s, before, after, group, notes)
        if infect.same(old_tree, new_tree):
            notes.append(f"{path}:{first}-{last} same syntax tree")
            return 0
        # Compile in the real function context, so return/yield/break/nonlocal are checked.
        c_old, c_new = old_text, new_text
        outer = s.outermost_def(s.common(first, last))
        if outer != -1:
            c_first, c_last = s.span(outer)
            c_old, c_new = before_lines[c_first - 1:c_last], splice(before_lines, c_first, c_last, group)
        try:
            new_code = compile(snippet(c_new), path, "exec")
        except (SyntaxError, ValueError) as error:
            raise Everything(f"{path}: patched code does not compile ({error.__class__.__name__})")
        try:
            if same_code(compile(snippet(c_old), path, "exec"), new_code):
                notes.append(f"{path}:{first}-{last} same bytecode")
                return 0
        except (SyntaxError, ValueError):
            pass
        self._units += 1
        lines = _lines(group, s)
        # A changed line inside a function or lambda body runs at call time: its own coverage is
        # more precise than every reader of the enclosing class attribute.
        if not any(s.kind(line) == "runtime" for line in lines):
            for route in (dict_route, field_route):
                found = route(self.data, path, s, first, last, old_tree, new_tree, notes)
                if found is not None:
                    self.touched.append((path, first, last, False))  # runs at import: the TPC check must know
                    return found
        bits = self._route_lines(path, s, lines, notes)
        imported = self._import_lines(path)
        if self.data.has_values(path) and all(s.kind(line) == "runtime" and line not in imported for line in lines):
            pairs = infect.changed_comparisons(old_tree, new_tree)
            if pairs:
                wrapped = bool(old_text) and old_text[0][:1] in (" ", "\t")
                self._candidate = infect.proven(self.data, path, pairs, first - (2 if wrapped else 1), notes)
        return bits

    def _route_lines(self, path: str, s: Structure, lines, notes: list[str]) -> int:
        bits = 0
        for line in lines:
            bits |= self._route_line(path, s, line, notes)
        return bits

    def _route_line(self, path: str, s: Structure, line: int, notes: list[str]) -> int:
        kind, imported = s.kind(line), line in self._import_lines(path)
        if kind == "runtime" and not imported:
            first, last = s.coverage_region(line)
            self.touched.append((path, first, last, True))
        else:
            self.touched.append((path, line, line, False))  # runs at import: reaches everything
        bits = 0
        if kind in ("runtime", "both"):
            first, last = s.coverage_region(line)
            got = self.data.tests_on(path, first, last)
            if imported:
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
        if path.startswith("sqlglot/") and (kind != "runtime" or imported):
            bits |= self.data.lazy_load_for(path)  # code that runs during `import sqlglot`
        return bits

    def _import_lines(self, path: str) -> set[int]:
        return set(self.data.import_lines.get(path, ()))

    def _test_module_users(self, path: str, local: bool, notes: list[str], where: str) -> int:
        """A test module's import-time change that only binds its own names reaches the tests whose
        process imported it: its own tests and those of modules importing it."""
        if not local or path not in self.data.module_users:
            raise Everything(f"{where} import-time code in a test module")
        users = self.data.users_of_module(path)
        methods = getattr(self, "_test_methods", None)
        if methods:  # only test methods' defs changed: their own tests and inherited copies
            users &= self.data.named(methods)
            notes.append(f"{where} test methods {', '.join(sorted(methods))}: {bin(users).count('1')}")
            return users
        notes.append(f"{where} test module users {bin(users).count('1')}")
        return users

    def _import_users(self, path: str, s: Structure, line: int, notes: list[str]) -> int:
        if path.startswith("tests/"):
            return self._test_module_users(path, getattr(self, "_test_local", False), notes, f"{path}:{line}")
        qualname = s.class_name(line)
        users = self.data.users_of_class(path, qualname) if qualname else None
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
            text = splice(before_lines, *s.span(outer), [block]) if outer != -1 else block.added
            try:
                compile(snippet(text), path, "exec")
            except (SyntaxError, ValueError):
                return self._route_whole_file(path, s, before, after, group, notes)
            bits |= self._route_lines(path, s, _lines([block], s), notes)
        return bits

    def _route_module_level(self, path, s, before, after, group, notes) -> int:
        """Changes between top-level statements: parse the added text on its own."""
        for block in group:
            added = "\n".join(block.added)
            try:
                tree = ast.parse(added) if added.strip() else ast.parse("")
            except SyntaxError:
                return self._route_whole_file(path, s, before, after, group, notes)
            if tree.body or not all(comment_or_blank(line) for line in block.removed):
                return self._import_route_or_everything(path, notes)
        notes.append(f"{path}: module-level comments only")
        return 0

    def _import_route_or_everything(self, path: str, notes: list[str]) -> int:
        if path.startswith("tests/"):
            self._units += 1
            return self._test_module_users(path, getattr(self, "_test_local", False), notes, f"{path}: module-level change;")
        self._units += 1
        users = self.data.users_of_module(path) | self.data.lazy_load_for(path)
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
        return self._route_lines(path, s, _lines(group, s), notes)
