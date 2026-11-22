"""Class-body attribute routes: a change confined to one `NAME = value` statement in a class body reaches
only the tests that read that attribute at runtime, or, for a dict literal, the changed keys."""

from __future__ import annotations

import ast
import functools

import infect
from maps import module_name


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


def _class_assignment(path, s, first, last, old_tree, new_tree):
    """(name, class qualname, class index, old value, new value) when the region is exactly one simple
    class-body statement assigning the same single name before and after."""
    if not path.startswith("sqlglot/"):
        return None
    index = s.common(first, last)
    if index == -1 or s.stmts[index][3] != "simple" or s.span(index) != (first, last):
        return None
    parent = s.stmts[index][2]
    if parent == -1 or s.stmts[parent][3] != "class":
        return None
    old, new = _single_target(old_tree), _single_target(new_tree)
    if old is None or new is None or old[0] != new[0]:
        return None
    return old[0], s.stmts[parent][6], parent, old[1].value, new[1].value


def field_route(data, path, s, first, last, old_tree, new_tree, notes) -> int | None:
    """Tests that read the attribute at runtime, resolved to this class (not a subclass override)."""
    found = data.attr_read and _class_assignment(path, s, first, last, old_tree, new_tree)
    if not found:
        return None
    name, qualname, parent, _old, _new = found
    if name in s.class_refs.get(parent, ()):
        return None  # the class body derives something from it at creation: invisible to the recorder
    readers = data.attr_readers(f"{module_name(path)}:{qualname}", name)
    if readers is None:
        return None
    notes.append(f"{path}:{first} field {qualname}.{name} read by {bin(readers).count('1')}")
    return readers | data.lazy_load_for(path)


def dict_route(data, path, s, first, last, old_tree, new_tree, notes) -> int | None:
    """A change to the values of some entries of a `NAME = {...}` literal reaches the tests that looked
    those keys up at runtime or read the whole dict, provided nothing read those values at import (into
    derived state or a subclass's copy) and the new values are evaluated without side effects."""
    found = _class_assignment(path, s, first, last, old_tree, new_tree)
    if not found or not isinstance(found[3], ast.Dict) or not isinstance(found[4], ast.Dict):
        return None
    name, qualname, _parent, old_dict, new_dict = found
    if len(old_dict.keys) != len(new_dict.keys):
        return None
    tag = f"{module_name(path)}:{qualname}|{name}"
    reads = data.dict_reads(tag)
    if reads is None or reads["iV"]:
        return None  # not mapped, or its values feed something built at import
    changed = []
    for old_key, old_value, new_key, new_value in zip(old_dict.keys, old_dict.values, new_dict.keys, new_dict.values):
        if (old_key is None) != (new_key is None) or (old_key is not None and not infect.same(old_key, new_key)):
            return None  # a key or a ** entry changed
        if infect.same(old_value, new_value):
            continue
        if old_key is None or not (isinstance(old_key, ast.Constant) and type(old_key.value) in (str, int)):
            return None
        changed.append((old_key.value, old_value, new_value))
    if not changed:
        return None
    bits = data.bits(reads["V"]) if reads["V"] is not None else 0
    for key, old_value, new_value in changed:
        rep = f"s:{key}" if type(key) is str else f"i:{key}"
        if rep in reads["iv"] or not _safe_new_value(data, path, old_value, new_value):
            return None
        sid = reads["v"].get(rep)
        if sid is not None:
            bits |= data.bits(sid)
    keys = ", ".join(repr(k) for k, _o, _n in changed)
    notes.append(f"{path}:{first} dict entry {qualname}|{name}[{keys}] read by {bin(bits).count('1')}")
    return bits | data.lazy_load_for(path)


def _refs(node) -> set[str]:
    """Dotted names (x, x.y.z) a value expression reads."""
    found = set()
    for sub in ast.walk(node):
        parts = []
        while isinstance(sub, ast.Attribute):
            parts.append(sub.attr)
            sub = sub.value
        if isinstance(sub, ast.Name):
            found.add(".".join([sub.id, *reversed(parts)]))
    return found


def _safe_new_value(data, path, old_value, new_value) -> bool:
    """The new value cannot raise or have side effects when the class is created: built from constants,
    containers, lambdas and calls to closure factories, reading no name the old value did not read."""
    if not _refs(new_value) <= _refs(old_value):
        return False

    def safe(node) -> bool:
        if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
            return True
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(not isinstance(e, ast.Starred) and safe(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return all(k is not None and safe(k) and safe(v) for k, v in zip(node.keys, node.values))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
                return isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)
            return isinstance(node.op, ast.Not) and safe(node.operand)
        if isinstance(node, ast.Lambda):
            return all(safe(d) for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None])
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name) and _closure_factory(data, path, node.func.id)
                and all(not isinstance(a, ast.Starred) and safe(a) for a in node.args)
                and all(k.arg is not None and safe(k.value) for k in node.keywords)
            )
        return False

    return safe(new_value)


@functools.lru_cache(maxsize=None)
def _closure_factory(data, path: str, name: str, depth: int = 0) -> bool:
    """`name`, as seen from `path`, is a plain function whose body only defines and returns a closure."""
    if depth > 2 or not data.has_source(path):
        return False
    for node in ast.parse(data.source(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]  # a docstring
            inner = {n.name for n in body[:-1] if isinstance(n, ast.FunctionDef)}
            returned = body[-1].value if body and isinstance(body[-1], ast.Return) else None
            return (
                not node.decorator_list and len(inner) == len(body) - 1
                and (isinstance(returned, ast.Lambda) or (isinstance(returned, ast.Name) and returned.id in inner))
            )
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    base = node.module.replace(".", "/")
                    for candidate in (f"{base}.py", f"{base}/__init__.py"):
                        if data.has_source(candidate):
                            return _closure_factory(data, candidate, alias.name, depth + 1)
                    return False
    return False
