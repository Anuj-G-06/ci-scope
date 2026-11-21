"""Where a line sits in a Python file, computed once at image build time, looked up at runtime.

analyze() parses a pinned source file and returns plain lists, so the build can store them as JSON.
The lookups are O(1) per line plus O(nesting depth), so the selector never parses a whole file
just to learn what a changed line belongs to.

Kinds of statement: "def" (function or method, decorators included), "class", "compound"
(any other statement with a body) and "simple".
"""

from __future__ import annotations

import ast
import io
import tokenize

DEF, CLASS, COMPOUND, SIMPLE = "def", "class", "compound", "simple"


def analyze(source: str) -> dict:
    """Statement spans, line-to-innermost-statement index, lambda bodies, class names, strings."""
    tree = ast.parse(source)
    line_count = source.count("\n") + 1
    stmts: list[list] = []  # [start, end, parent, kind, body_start, def_line, qualname]

    def visit(node: ast.AST, parent: int, qual: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                decorators = getattr(child, "decorator_list", None) or []
                start = min([d.lineno for d in decorators] + [child.lineno])
                body = getattr(child, "body", None)
                body_start = body[0].lineno if isinstance(body, list) and body else 0
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind, name = DEF, f"{qual}{child.name}"
                    inner_qual = f"{name}.<locals>."
                elif isinstance(child, ast.ClassDef):
                    kind, name = CLASS, f"{qual}{child.name}"
                    inner_qual = f"{name}."
                else:
                    kind = COMPOUND if body_start else SIMPLE
                    name, inner_qual = "", qual
                index = len(stmts)
                stmts.append([start, child.end_lineno, parent, kind, body_start, child.lineno, name])
                visit(child, index, inner_qual)
            else:
                visit(child, parent, qual)

    visit(tree, -1, "")

    line_stmt = [-1] * (line_count + 2)
    for index, (start, end, *_rest) in enumerate(stmts):  # parents come before children
        for line in range(start, end + 1):
            line_stmt[line] = index

    lambdas = [
        [node.lineno, node.body.lineno, node.body.end_lineno]
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda)
    ]
    lambdas.sort()

    strings = []  # multi-line string tokens: [first_line, last_line]
    continued = []  # lines ending in a backslash continuation
    lines = source.split("\n")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.start[0] != token.end[0] and token.type in _STRING_TYPES:
            strings.append([token.start[0], token.end[0]])
    for number, text in enumerate(lines, start=1):
        if text.endswith("\\") and not _inside(strings, number):
            continued.append(number)
    return {
        "stmts": stmts,
        "line_stmt": line_stmt,
        "lambdas": lambdas,
        "strings": strings,
        "continued": continued,
        "lines": line_count,
    }


_STRING_TYPES = {tokenize.STRING}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    if hasattr(tokenize, _name):
        _STRING_TYPES.add(getattr(tokenize, _name))


def _inside(spans: list, line: int) -> bool:
    """True when a line begins inside a multi-line string (it starts after the string's first line)."""
    return any(first < line <= last for first, last in spans)


class Structure:
    """Runtime view over analyze() output for one file."""

    def __init__(self, data: dict):
        self.stmts = data["stmts"]
        self.line_stmt = data["line_stmt"]
        self.lambdas = data["lambdas"]
        self.strings = data["strings"]
        self.continued = set(data["continued"])
        self.lines = data["lines"]

    # ----- statements -----
    def innermost(self, line: int) -> int:
        if 0 < line < len(self.line_stmt):
            return self.line_stmt[line]
        return -1

    def ancestors(self, index: int) -> list[int]:
        chain = []
        while index != -1:
            chain.append(index)
            index = self.stmts[index][2]
        return chain

    def common(self, first: int, last: int) -> int:
        """Innermost statement containing every line in [first, last], or -1 for module level."""
        a = self.innermost(first)
        if a == -1:
            return -1
        for index in self.ancestors(a):
            start, end = self.stmts[index][0], self.stmts[index][1]
            if start <= first and last <= end:
                return index
        return -1

    def span(self, index: int) -> tuple[int, int]:
        return self.stmts[index][0], self.stmts[index][1]

    def outermost_def(self, index: int) -> int:
        """The outermost function containing a statement, or -1. Compiling it keeps return/yield/
        break/nonlocal in their real context."""
        found = -1
        for i in self.ancestors(index):
            if self.stmts[i][3] == DEF:
                found = i
        return found

    def coverage_region(self, line: int) -> tuple[int, int]:
        """Lines whose execution means the statement holding `line` ran: a simple statement's whole
        span, a compound statement's header, or a whole compound for else:/except:/finally: lines."""
        index = self.innermost(line)
        if index == -1:
            return line, line
        start, end, _parent, kind, body_start, _def_line, _name = self.stmts[index]
        if kind == SIMPLE:
            return start, end
        if body_start and line < body_start:
            return start, body_start - 1
        return start, end

    # ----- when a line runs -----
    def kind(self, line: int) -> str:
        """'runtime' (inside a function or lambda body), 'header' (a def's signature or decorators),
        'import' (class body or module level) or 'both' (a lambda body sharing a line with code
        that runs at import)."""
        for index in self.ancestors(self.innermost(line)):
            start, end, _parent, kind, body_start, def_line, _name = self.stmts[index]
            if kind == DEF:
                if body_start and line >= body_start:
                    return "runtime"
                return "header"
        for lambda_line, body_first, body_last in self.lambdas:
            if body_first <= line <= body_last:
                return "runtime" if line > lambda_line else "both"
        return "import"

    def class_name(self, line: int) -> str:
        """Qualified name of the innermost class whose body holds the line ('' at module level)."""
        for index in self.ancestors(self.innermost(line)):
            if self.stmts[index][3] == CLASS:
                start, end, _parent, _kind, body_start, _def_line, name = self.stmts[index]
                return name
            if self.stmts[index][3] == DEF:
                return ""
        return ""

    def def_body(self, line: int) -> tuple[int, int] | None:
        """Body lines of the innermost function whose header holds `line`."""
        for index in self.ancestors(self.innermost(line)):
            start, end, _parent, kind, body_start, _def_line, _name = self.stmts[index]
            if kind == DEF:
                return (body_start, end) if body_start else None
        return None

    def decorator_line(self, line: int) -> bool:
        for index in self.ancestors(self.innermost(line)):
            start, _end, _parent, kind, _body_start, def_line, _name = self.stmts[index]
            if kind in (DEF, CLASS):
                return start <= line < def_line
        return False

    # ----- text hazards -----
    def inside_string(self, line: int) -> bool:
        return _inside(self.strings, line)

    def boundary_inside_string(self, after_line: int) -> bool:
        """True when a line inserted after `after_line` would land inside a multi-line string."""
        return any(first <= after_line < last for first, last in self.strings)

    def is_continued(self, line: int) -> bool:
        return line in self.continued
