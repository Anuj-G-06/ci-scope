"""Test modules: which import-time edits stay local to the module, and which only add new tests.

A test module's import-time code (class bodies, decorators, defaults, module level) reaches other tests
only through the names it binds, as long as it calls nothing (bar unittest and mock decorators) and
assigns into no other object. Then the tests that can see the change are the module's users: its own
tests and those of modules importing it. New test methods are not numbered tests, so adding them under
fresh names reaches nothing.
"""

from __future__ import annotations

import ast

from patch import Block

TEST_DECORATORS = {"skip", "skipIf", "skipUnless", "expectedFailure", "patch", "object", "dict", "multiple"}


def _local(node, decorator: bool = False) -> bool:
    """Code that, run at import, only binds names in its own module or class: no calls (bar unittest and
    mock decorators), no assignment into other objects. Function bodies run only when called."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        evaluated = args.defaults + [d for d in args.kw_defaults if d is not None]
        decorators = getattr(node, "decorator_list", [])
        return all(_local(d, True) for d in decorators) and all(_local(d) for d in evaluated)
    if isinstance(node, ast.If) and _main_guard(node.test):
        return True  # `if __name__ == "__main__":` never runs under pytest
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if not (decorator and name in TEST_DECORATORS):
            return False
        return all(_local(a) for a in node.args) and all(_local(k.value) for k in node.keywords)
    targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else (
        [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
    if any(not isinstance(t, ast.Name) for t in targets) or isinstance(node, (ast.Global, ast.Nonlocal)):
        return False
    return all(_local(child) for child in ast.iter_child_nodes(node))


def _main_guard(test) -> bool:
    return (
        isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
        and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__"
    )


def module_local(text: str) -> bool:
    """A whole module that compiles and, imported, only binds its own names (see _local)."""
    try:
        compile(text, "<new>", "exec", dont_inherit=True)
        return _local(ast.parse(text))
    except (SyntaxError, ValueError):
        return False


def _span(node) -> tuple[int, int]:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]), node.end_lineno


def _hits(lines: set[int], first: int, last: int) -> bool:
    return any(first <= line <= last for line in lines)


def _new_lines(blocks: list[Block]) -> set[int]:
    """Line numbers, in the patched file, of every added line."""
    lines, offset = set(), 0
    for block in sorted(blocks, key=lambda b: b.first):
        start = (block.after + 1 if block.insertion else block.first) + offset
        lines.update(range(start, start + len(block.added)))
        offset += len(block.added) - len(block.removed)
    return lines


def _import_statements(tree, lines: set[int]) -> list:
    """Module- and class-level statements (decorators included) holding any of `lines`; a function's
    body runs only when called, so a change there counts as its def statement."""
    found, stack = [], list(tree.body)
    while stack:
        node = stack.pop()
        first, last = _span(node)
        if not _hits(lines, first, last):
            continue
        if isinstance(node, ast.ClassDef) and _hits(lines, node.body[0].lineno, last):
            stack.extend(node.body)
            if _hits(lines, first, node.body[0].lineno - 1):  # the header itself: bases, keywords, decorators
                found.append(ast.ClassDef(name=node.name, bases=node.bases, keywords=node.keywords, body=[],
                                          decorator_list=node.decorator_list, type_params=[]))
        else:
            found.append(node)
    return found


def _bound(node) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
    return [t.id for t in targets if isinstance(t, ast.Name)]


def test_change(before: str, after: str, blocks: list[Block]):
    """(local, fresh, methods) for a test module's change.
    local: every module- or class-level statement it touches, old and new, only binds names in its own
    module or class. fresh: when it only inserts new test methods into existing classes, under names they
    do not define, their (class, name). methods: when every statement it touches is a test method's def,
    their names (a test method runs only as its own test, or as a subclass's inherited copy)."""
    try:
        old_tree, new_tree = ast.parse(before), ast.parse(after)
    except SyntaxError:
        return False, None, None
    removed = {line for b in blocks for line in range(b.first, b.last + 1)}
    added = _new_lines(blocks)
    old_stmts, new_stmts = _import_statements(old_tree, removed), _import_statements(new_tree, added)
    touched = old_stmts + new_stmts
    if not all(_local(st) for st in touched):
        return False, None, None
    owner = {id(n): c.name for tree in (old_tree, new_tree) for c in ast.walk(tree) if isinstance(c, ast.ClassDef) for n in c.body}

    def test_method(st) -> bool:
        return isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name.startswith("test") and id(st) in owner

    methods = {st.name for st in touched} if touched and all(test_method(st) for st in touched) else None
    if removed or not new_stmts:
        return True, None, methods
    defined = {(c.name, name) for c in ast.walk(old_tree) if isinstance(c, ast.ClassDef) for n in c.body for name in _bound(n)}
    fresh, covered = [], set()
    for st in new_stmts:
        first, last = _span(st)
        if not (test_method(st) and (owner[id(st)], st.name) not in defined and set(range(first, last + 1)) <= added):
            return True, None, methods
        fresh.append((owner[id(st)], st.name))
        covered.update(range(first, last + 1))
    text = after.split("\n")
    if any(i not in covered and text[i - 1].strip() and not text[i - 1].strip().startswith("#") for i in added):
        return True, None, methods
    return True, fresh, methods
