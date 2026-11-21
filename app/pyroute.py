"""Routing a change to a Python file: comment-only edits, same syntax tree or bytecode, field-level class
attributes, recorded comparison states, then line coverage and import-time users."""

from __future__ import annotations

import ast
import re

import infect
from equiv import comment_or_blank, same_code, snippet
from maps import module_name
from patch import Block, FilePatch, PatchError, apply, splice
from structure import Structure


_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")  # PEP 263, lines 1-2


class Everything(Exception):
    """This commit could affect anything we cannot bound: run every test."""


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
        if not all(comment_or_blank(line) for line in block.removed + block.added):
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
            old_tree = ast.parse(snippet(old_text))
        except SyntaxError:
            return self._route_whole_file(path, s, before, after, group, notes)
        try:
            new_tree = ast.parse(snippet(new_text))
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
            new_code = compile(snippet(c_new), path, "exec")
        except (SyntaxError, ValueError) as error:
            raise Everything(f"{path}: patched code does not compile ({error.__class__.__name__})")
        try:
            old_code = compile(snippet(c_old), path, "exec")
            if same_code(old_code, new_code):
                notes.append(f"{path}:{first}-{last} same bytecode")
                return 0
        except (SyntaxError, ValueError):
            pass
        changed = [ln for b in group for ln in ([b.after, b.after + 1] if b.insertion else range(b.first, b.last + 1))]
        # A changed line inside a function or lambda body runs at call time: its own coverage is
        # more precise than every reader of the enclosing class attribute.
        field = None
        if not any(1 <= ln <= s.lines and s.kind(ln) == "runtime" for ln in changed):
            field = self._field_route(path, s, first, last, old_tree, new_tree, notes)
        if field is not None:
            self._units += 1
            return field
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

    @staticmethod
    def _single_target(tree) -> tuple[str, ast.AST] | None:
        body = tree.body
        if len(body) == 1 and isinstance(body[0], ast.If) and isinstance(body[0].test, ast.Constant):
            body = body[0].body  # the "if 1:" wrapper of an indented snippet
        if len(body) != 1:
            return None
        node = body[0]
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id, node
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            return node.target.id, node
        return None

    def _field_route(self, path, s, first, last, old_tree, new_tree, notes) -> int | None:
        """A change confined to one class-body `NAME = value` reaches only tests that read that
        attribute at runtime, resolved to this class (not a subclass override)."""
        if not path.startswith("sqlglot/") or not self.data.attr_read:
            return None
        index = s.common(first, last)
        if index == -1 or s.stmts[index][3] != "simple" or s.span(index) != (first, last):
            return None
        parent = s.stmts[index][2]
        if parent == -1 or s.stmts[parent][3] != "class":
            return None
        old, new = self._single_target(old_tree), self._single_target(new_tree)
        if old is None or new is None or old[0] != new[0]:
            return None
        name, qualname = old[0], s.stmts[parent][6]
        if name in s.class_refs.get(parent, ()):
            return None  # the class body derives something from it at creation: invisible to the recorder
        readers = self.data.attr_readers(f"{module_name(path)}:{qualname}", name)
        if readers is None:
            return None
        notes.append(f"{path}:{first} field {qualname}.{name} read by {bin(readers).count('1')}")
        return readers | self.data.lazy_load_for(path)

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
        if kind == "runtime" and line not in self._import_lines(path):
            first, last = s.coverage_region(line)
            self.touched.append((path, first, last, True))
        else:
            self.touched.append((path, line, line, False))  # runs at import: reaches everything
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
            bits |= self.data.lazy_load_for(path)  # code that runs during `import sqlglot`
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
                compile(snippet(text), path, "exec")
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
            if tree.body or not all(comment_or_blank(line) for line in block.removed):
                return self._import_route_or_everything(path, notes)
        notes.append(f"{path}: module-level comments only")
        return 0

    def _import_route_or_everything(self, path: str, notes: list[str]) -> int:
        if path.startswith("tests/"):
            raise Everything(f"{path}: module-level change in a test module")
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
        bits = 0
        for block in group:
            lines = [block.after, block.after + 1] if block.insertion else range(block.first, block.last + 1)
            for line in lines:
                if 1 <= line <= s.lines:
                    bits |= self._route_line(path, s, line, notes)
        return bits
